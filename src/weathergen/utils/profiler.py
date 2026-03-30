import time
import logging
from contextlib import contextmanager
from typing import Dict, Any

logger = logging.getLogger(__name__)

class Profiler:
    def __init__(self, enabled: bool = False):
        self.enabled = enabled
        self.stats: Dict[str, Dict[str, Any]] = {}

    @contextmanager
    def time_block(self, name: str, synchronize: bool = False):
        if not self.enabled:
            yield
            return

        if synchronize:
            self._sync()

        start = time.perf_counter()
        try:
            yield
        finally:
            if synchronize:
                self._sync()
            end = time.perf_counter()
            duration = end - start
            if name not in self.stats:
                self.stats[name] = {"total_time": 0.0, "count": 0, "avg_time": 0.0}
            
            self.stats[name]["total_time"] += duration
            self.stats[name]["count"] += 1
            self.stats[name]["avg_time"] = self.stats[name]["total_time"] / self.stats[name]["count"]

    def _sync(self):
        try:
            import torch
            if torch.cuda.is_available():
                torch.cuda.synchronize()
        except ImportError:
            pass

    def reset(self):
        self.stats = {}

    def add_stat(self, name: str, duration: float):
        if not self.enabled:
            return
        if name not in self.stats:
            self.stats[name] = {"total_time": 0.0, "count": 0, "avg_time": 0.0}
        
        self.stats[name]["total_time"] += duration
        self.stats[name]["count"] += 1
        self.stats[name]["avg_time"] = self.stats[name]["total_time"] / self.stats[name]["count"]

    def get_stats(self) -> Dict[str, Dict[str, Any]]:
        return self.stats

    def log_stats(self, rank: int = 0, world_size: int = 1):
        if not self.enabled or not self.stats:
            return
        
        try:
            import torch
            import torch.distributed as dist
            if dist.is_available() and dist.is_initialized():
                if rank == 0:
                    logger.info(f"--- Performance Profiler Stats (across {world_size} ranks) ---")
                
                for name, data in self.stats.items():
                    # Collect avg_time from all ranks
                    val = data["avg_time"]
                    tensor = torch.tensor([val], device=f"cuda:{torch.cuda.current_device()}")
                    gather_list = [torch.zeros(1, device=f"cuda:{torch.cuda.current_device()}") for _ in range(world_size)]
                    dist.all_gather(gather_list, tensor)
                    
                    if rank == 0:
                        vals = [t.item() for t in gather_list]
                        v_min, v_max, v_mean = min(vals), max(vals), sum(vals)/len(vals)
                        
                        msg = f"{name:25s}: Mean={v_mean:.4f}s, Min={v_min:.4f}s, Max={v_max:.4f}s"
                        if world_size <= 8:
                            # Show individual ranks for small clusters
                            ranks_str = ", ".join([f"R{i}:{v:.4f}" for i, v in enumerate(vals)])
                            msg += f" | {ranks_str}"
                        logger.info(msg)
                
                if rank == 0:
                    logger.info("---------------------------------------------------------------")
                return
        except (ImportError, RuntimeError):
            pass

        # Fallback for non-distributed or error
        logger.info("--- Performance Profiler Stats ---")
        for name, data in self.stats.items():
            logger.info(f"{name}: total={data['total_time']:.4f}s, count={data['count']}, avg={data['avg_time']:.4f}s")
        logger.info("----------------------------------")
