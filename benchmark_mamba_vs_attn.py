"""
Benchmark: Mamba2 vs Attention in ForecastingEngine

Compares forward/backward speed, memory usage, and parameter counts
for different block types and configurations across sequence lengths
corresponding to HealPix levels 4-7.

Usage:
    srun --partition=gpu --account=ab0995 --gpus=1 --time=00:15:00 \
        uv run python benchmark_mamba_vs_attn.py
"""

import argparse
import json
import sys
import time

# Ensure output is visible immediately
sys.stdout.reconfigure(line_buffering=True)
print("DEBUG: benchmark_mamba_vs_attn.py starting...", flush=True)

import torch
import torch.nn as nn

sys.path.insert(0, "/work/ab0995/a270225/WeatherGenerator2/src")


def count_params(module):
    return sum(p.numel() for p in module.parameters())


def benchmark_block(block, seq_len, dim, x_lens=None, batch_size=1, input_x=None, num_warmup=5, num_runs=30, dtype=torch.bfloat16):
    """Benchmark a single block: forward time, backward time, peak memory, params."""
    device = "cuda"
    block = block.to(device).to(dtype)
    block.train()

    if input_x is None:
        input_x = torch.randn(batch_size, seq_len, dim, device=device, dtype=dtype, requires_grad=True)
    else:
        input_x = input_x.to(device).to(dtype).detach().clone().requires_grad_(True)

    x = input_x

    def run_fwd(input_x):
        if x_lens is not None:
            return block(input_x, x_lens=x_lens)
        return block(input_x)

    # Warmup
    for _ in range(num_warmup):
        y = run_fwd(x)
        loss = y.sum()
        loss.backward()
        x.grad = None

    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()

    # Forward timing
    fwd_start = torch.cuda.Event(enable_timing=True)
    fwd_end = torch.cuda.Event(enable_timing=True)

    fwd_start.record()
    for _ in range(num_runs):
        y = run_fwd(x)
    fwd_end.record()
    torch.cuda.synchronize()
    fwd_ms = fwd_start.elapsed_time(fwd_end) / num_runs

    # Backward timing
    bwd_start = torch.cuda.Event(enable_timing=True)
    bwd_end = torch.cuda.Event(enable_timing=True)

    bwd_start.record()
    for _ in range(num_runs):
        y = run_fwd(x)
        loss = y.sum()
        loss.backward()
        x.grad = None
    bwd_end.record()
    torch.cuda.synchronize()
    bwd_ms = bwd_start.elapsed_time(bwd_end) / num_runs

    peak_mb = torch.cuda.max_memory_allocated() / (1024 ** 2)
    params = count_params(block)

    return {
        "fwd_ms": round(fwd_ms, 2),
        "bwd_ms": round(bwd_ms, 2),
        "peak_mb": round(peak_mb, 1),
        "params": params,
        "params_m": round(params / 1e6, 2),
    }


def create_attention_block(dim, num_heads=16):
    """Create a MultiSelfAttentionHeadVarlen block (the FE's current default)."""
    from weathergen.model.attention import MultiSelfAttentionHeadVarlen
    return MultiSelfAttentionHeadVarlen(
        dim_embed=dim,
        num_heads=num_heads,
        dropout_rate=0.0,
        with_qk_lnorm=True,
        with_flash=True,
        norm_type="LayerNorm",
        norm_eps=1e-5,
        attention_dtype=torch.bfloat16,
    )


def create_bimamba2_block(dim, d_state=64, d_conv=4, expand=2):
    """Create a BiMamba2Block (bidirectional Mamba-2)."""
    from weathergen.model.mamba_block import BiMamba2Block
    return BiMamba2Block(
        dim_embed=dim,
        d_state=d_state,
        d_conv=d_conv,
        expand=expand,
        dropout_rate=0.0,
        norm_type="LayerNorm",
        norm_eps=1e-5,
    )


def create_mamba2_block(dim, d_state=64, d_conv=4, expand=2, reverse=False):
    """Create a Mamba2Block (unidirectional)."""
    from weathergen.model.mamba_block import Mamba2Block
    return Mamba2Block(
        dim_embed=dim,
        d_state=d_state,
        d_conv=d_conv,
        expand=expand,
        dropout_rate=0.0,
        norm_type="LayerNorm",
        norm_eps=1e-5,
        reverse=reverse,
    )


def run_single_block_benchmark(dim=512, batch_size=1, expand=2):
    """Benchmark individual blocks at different sequence lengths."""
    # Sequence lengths corresponding to HealPix levels
    hp_levels = {
        "H4 (3072)": 3072,
        "H5 (12288)": 12288,
        "H6 (49152)": 49152,
        "H7 (196608)": 196608,
        "H8 (786432)": 786432,
    }

    print("\n" + "=" * 120)
    print(f"{'Single Block Benchmark':^120}")
    print(f"{'dim_embed=' + str(dim) + ', batch_size=' + str(batch_size):^120}")
    print("=" * 120)
    print(f"{'HP Level':<16} | {'Block Type':<20} | {'Fwd (ms)':>10} | {'Bwd (ms)':>10} | {'Peak VRAM (MB)':>15} | {'Params (M)':>10} | {'Status':>8}")
    print("-" * 120)

    results = {}

    for hp_name, seq_len in hp_levels.items():
        block_configs = [
            ("Attention", lambda: create_attention_block(dim)),
            ("BiMamba2", lambda: create_bimamba2_block(dim, expand=expand)),
            ("Mamba2 (uni)", lambda: create_mamba2_block(dim, expand=expand)),
        ]

        for block_name, block_fn in block_configs:
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats()

            try:
                block = block_fn()
                
                # Setup x_lens and shape for Varlen
                x_lens = None
                input_x = torch.randn(batch_size, seq_len, dim, device="cuda", dtype=torch.bfloat16, requires_grad=True)
                if "Varlen" in type(block).__name__:
                    # Prepend 0 so that attention.py's cumsum creates [0, seq1, seq1+seq2, ...]
                    x_lens = torch.tensor([0] + [seq_len] * batch_size, device="cuda", dtype=torch.int32)
                    input_x = input_x.view(-1, dim) # Flatten for Varlen
                
                r = benchmark_block(block, seq_len, dim, x_lens=x_lens, batch_size=batch_size, input_x=input_x)
                status = "OK"
                print(
                    f"{hp_name:<16} | {block_name:<20} | {r['fwd_ms']:>10.2f} | {r['bwd_ms']:>10.2f} | "
                    f"{r['peak_mb']:>15.1f} | {r['params_m']:>10.2f} | {status:>8}"
                )
                results[f"{hp_name}_{block_name}"] = r
            except torch.cuda.OutOfMemoryError:
                print(
                    f"{hp_name:<16} | {block_name:<20} | {'---':>10} | {'---':>10} | "
                    f"{'---':>15} | {'---':>10} | {'OOM':>8}"
                )
                results[f"{hp_name}_{block_name}"] = {"status": "OOM"}
            except Exception as e:
                err_msg = str(e)[:30]
                print(
                    f"{hp_name:<16} | {block_name:<20} | {'---':>10} | {'---':>10} | "
                    f"{'---':>15} | {'---':>10} | {err_msg:>8}"
                )
                results[f"{hp_name}_{block_name}"] = {"status": f"FAIL: {err_msg}"}

            # Cleanup
            if block is not None:
                del block
            torch.cuda.empty_cache()

    return results


def run_engine_benchmark(dim=512, num_blocks=1, batch_size=1, expand=2):
    """Benchmark a simulated ForecastingEngine with different configs."""
    from weathergen.model.layers import MLP
    from weathergen.model.attention import MultiSelfAttentionHeadVarlen

    # Target HealPix Level 7 for full-scale verification
    seq_len = 196608
    num_heads = max(1, dim // 128) # Scale heads for headdim=128

    print("\n" + "=" * 120)
    print(f"{'ForecastingEngine-Level Benchmark (SCALE: H7)':^120}")
    print(f"{'dim=' + str(dim) + ', num_blocks=' + str(num_blocks) + ', seq_len=' + str(seq_len) + ', batch=' + str(batch_size):^120}")
    print("=" * 120)

    configs = {
        "Pure Attention": {"mamba_indices": set()},  # all attention
        "Hybrid (33% attn)": {"mamba_indices": {0, 1, 3, 4}},  # keep 2,5 as attention
        "Hybrid (17% attn)": {"mamba_indices": {0, 1, 2, 3, 4}},  # keep 5 as attention
        "Pure Mamba2": {"mamba_indices": {0, 1, 2, 3, 4, 5}},  # all mamba
    }

    print(f"{'Config':<24} | {'Fwd (ms)':>10} | {'Bwd (ms)':>10} | {'Peak VRAM (MB)':>15} | {'Params (M)':>10} | {'Status':>8}")
    print("-" * 120)

    results = {}
    for config_name, cfg in configs.items():
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()

        try:
            # Build a simple ModuleList simulating the FE
            blocks = nn.ModuleList()
            for i in range(num_blocks):
                if i in cfg["mamba_indices"]:
                    blocks.append(create_bimamba2_block(dim, expand=expand))
                else:
                    # Use Varlen version for engine benchmark at H7
                    blocks.append(MultiSelfAttentionHeadVarlen(dim, num_heads=num_heads, dropout_rate=0.0))
                blocks.append(
                    MLP(dim, dim, with_residual=True, dropout_rate=0.0,
                        norm_type="LayerNorm", norm_eps=1e-5)
                )

            blocks = blocks.to("cuda").to(torch.bfloat16)
            blocks.train()

            # Input for Varlen/Engine should be flattened
            input_tokens = torch.randn(batch_size * seq_len, dim, device="cuda", dtype=torch.bfloat16,
                             requires_grad=True)
            # Match attention.py expectations: [0, seq_len, 2*seq_len, ...]
            x_lens = torch.tensor([0] + [seq_len] * batch_size, device="cuda", dtype=torch.int32)
            
            def engine_forward(tokens):
                from torch.utils.checkpoint import checkpoint
                for layer in blocks:
                    if isinstance(layer, MultiSelfAttentionHeadVarlen):
                        # Attention Varlen takes x, x_lens, ada_ln_aux, coords
                        tokens = checkpoint(layer, tokens, x_lens, None, None, use_reentrant=False)
                    elif "BiMamba2Block" in type(layer).__name__ or "Mamba2Block" in type(layer).__name__:
                        # Mamba blocks take x, coords, ada_ln_aux
                        tokens = checkpoint(layer, tokens, None, None, use_reentrant=False)
                    else:
                        # MLP and others
                        tokens = checkpoint(layer, tokens, None, use_reentrant=False)
                return tokens

            # Warmup
            for _ in range(2):
                y = engine_forward(input_tokens)
                y.sum().backward()
                input_tokens.grad = None

            torch.cuda.synchronize()
            torch.cuda.reset_peak_memory_stats()
            
            fwd_start = torch.cuda.Event(enable_timing=True)
            fwd_end = torch.cuda.Event(enable_timing=True)
            
            fwd_start.record()
            for _ in range(5): # Fewer runs for full FE at H7
                y = engine_forward(input_tokens)
            fwd_end.record()
            torch.cuda.synchronize()
            fwd_ms = fwd_start.elapsed_time(fwd_end) / 5

            bwd_summary = 0
            # Skip full backward if it takes too long or OOMs
            try:
                bwd_start = torch.cuda.Event(enable_timing=True)
                bwd_end = torch.cuda.Event(enable_timing=True)
                bwd_start.record()
                for _ in range(3):
                    y = engine_forward(input_tokens)
                    y.sum().backward()
                    input_tokens.grad = None
                bwd_end.record()
                torch.cuda.synchronize()
                bwd_summary = bwd_start.elapsed_time(bwd_end) / 3
            except:
                bwd_summary = 0

            peak_mb = torch.cuda.max_memory_allocated() / (1024 ** 2)
            params_m = count_params(blocks) / 1e6
            
            print(f"{config_name:<24} | {fwd_ms:>10.2f} | {bwd_summary:>10.2f} | {peak_mb:>15.1f} | {params_m:>10.2f} | OK")
            results[config_name] = {"fwd_ms": fwd_ms, "bwd_ms": bwd_summary, "peak_mb": peak_mb}

        except torch.cuda.OutOfMemoryError:
            print(f"{config_name:<24} | {'---':>10} | {'---':>10} | {'---':>15} | {'---':>10} | OOM")
            results[config_name] = {"status": "OOM"}
        except Exception as e:
            msg = str(e)[:30]
            print(f"{config_name:<24} | {'---':>10} | {'---':>10} | {'---':>15} | {'---':>10} | FAIL: {msg}")
            results[config_name] = {"status": f"FAIL: {msg}"}
        finally:
            if 'blocks' in locals():
                del blocks
            if 'input_tokens' in locals():
                del input_tokens
            torch.cuda.empty_cache()

    return results


def main():
    parser = argparse.ArgumentParser(description="Benchmark Mamba2 vs Attention")
    parser.add_argument("--dim", type=int, default=512, help="Embedding dimension")
    parser.add_argument("--batch-size", type=int, default=1, help="Batch size")
    parser.add_argument("--num-blocks", type=int, default=1, help="Number of FE blocks")
    parser.add_argument("--expand", type=int, default=2, help="Mamba2 expansion factor")
    parser.add_argument("--output", type=str, default=None, help="Path to save JSON results")
    parser.add_argument("--block-only", action="store_true", help="Only run single-block benchmark")
    parser.add_argument("--engine-only", action="store_true", help="Only run engine benchmark")
    args = parser.parse_args()

    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"VRAM: {torch.cuda.get_device_properties(0).total_memory / (1024**3):.1f} GB")
    print(f"PyTorch: {torch.__version__}")

    all_results = {}

    if not args.engine_only:
        all_results["single_block"] = run_single_block_benchmark(
            dim=args.dim, batch_size=args.batch_size, expand=args.expand
        )

    if not args.block_only:
        all_results["engine"] = run_engine_benchmark(
            dim=args.dim, num_blocks=args.num_blocks, batch_size=args.batch_size, expand=args.expand
        )

    if args.output:
        with open(args.output, "w") as f:
            json.dump(all_results, f, indent=2)
        print(f"\nResults saved to {args.output}")

    print("\n" + "=" * 120)
    print("Benchmark complete.")


if __name__ == "__main__":
    main()
