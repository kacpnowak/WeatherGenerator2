"""Area-weighted score kernel.

Computes RMSE, bias and ACC for a single (sample, stream, forecast-step, model) block,
given the per-channel wet mask and area weights produced by
``meshtools.masks_and_weights``. Pure numpy, vectorized over channels.
"""

import numpy as np

#: Row order of the array returned by `score_block`.
METRICS = ("rmse", "bias", "acc")


def score_block(
    pred: np.ndarray,  # (npts, nchan) float32   forecast field
    target: np.ndarray,  # (npts, nchan) float32   truth field
    clim: np.ndarray | None,  # (npts, nchan) float32   day-of-year climatology, or None
    mask: np.ndarray,  # (nchan, npts) bool      wet mask (from meshtools)
    w: np.ndarray,  # (nchan, npts) float32   area weights (from meshtools)
) -> np.ndarray:  # (3, nchan) float64      rows: rmse, bias, acc
    """Area-weighted RMSE, bias and ACC per channel.

    Weights are ``W = w * mask`` (per channel); all sums are over points. With
    ``err = pred - target``::

        rmse = sqrt( sum(W * err**2) / sum(W) )
        bias = sum(W * err) / sum(W)
        acc  = sum(W * f' * o') / sqrt( sum(W * f'**2) * sum(W * o'**2) )

    where ``f' = pred - clim`` and ``o' = target - clim``.

    If `clim` is None, the acc row is all-NaN. If the acc denominator is 0 for a
    channel (e.g. ``pred`` is identical to `clim`), that channel's acc is NaN.
    """
    # (npts, nchan) -> (nchan, npts), accumulate in float64.
    pred64 = pred.T.astype(np.float64)
    target64 = target.T.astype(np.float64)

    W = w.astype(np.float64) * mask

    sum_w = W.sum(axis=1)
    err = pred64 - target64

    # A channel can be entirely masked out (fully-dry channel), making sum_w == 0
    # and, for acc, the denominator 0 too (e.g. pred is identical to clim). Both are
    # legitimate 0/0 -> NaN results, computed silently (brief's requirement).
    with np.errstate(invalid="ignore", divide="ignore"):
        rmse = np.sqrt((W * err**2).sum(axis=1) / sum_w)
        bias = (W * err).sum(axis=1) / sum_w

        if clim is None:
            acc = np.full(pred64.shape[0], np.nan, dtype=np.float64)
        else:
            clim64 = clim.T.astype(np.float64)
            f = pred64 - clim64
            o = target64 - clim64
            num = (W * f * o).sum(axis=1)
            den = np.sqrt((W * f**2).sum(axis=1) * (W * o**2).sum(axis=1))
            acc = num / den

    return np.stack([rmse, bias, acc])


if __name__ == "__main__":
    import math

    rng = np.random.default_rng(0)
    npts, nchan = 200, 4
    pred = rng.normal(size=(npts, nchan)).astype(np.float32)
    target = rng.normal(size=(npts, nchan)).astype(np.float32)
    clim = rng.normal(size=(npts, nchan)).astype(np.float32)
    mask = rng.random((nchan, npts)) > 0.3
    w = rng.random((nchan, npts)).astype(np.float32) + 0.1

    out = score_block(pred, target, clim, mask, w)
    assert out.dtype == np.float64
    assert out.shape == (3, nchan)

    # reference: naive loops
    for c in range(nchan):
        ws = we = we2 = wfo = wf2 = wo2 = 0.0
        for p in range(npts):
            if not mask[c, p]:
                continue
            W = float(w[c, p])
            e = float(pred[p, c]) - float(target[p, c])
            f = float(pred[p, c]) - float(clim[p, c])
            o = float(target[p, c]) - float(clim[p, c])
            ws += W
            we += W * e
            we2 += W * e * e
            wfo += W * f * o
            wf2 += W * f * f
            wo2 += W * o * o
        assert abs(out[0, c] - math.sqrt(we2 / ws)) < 1e-9
        assert abs(out[1, c] - we / ws) < 1e-9
        assert abs(out[2, c] - wfo / math.sqrt(wf2 * wo2)) < 1e-9
    print("hand-computation check: PASS")

    # clim=None -> acc row all-NaN, rmse/bias unchanged
    out_noclim = score_block(pred, target, None, mask, w)
    assert np.all(np.isnan(out_noclim[2]))
    assert np.array_equal(out_noclim[0], out[0])
    assert np.array_equal(out_noclim[1], out[1])
    print("clim=None check: PASS")

    # pred == clim -> acc denominator 0/0 -> NaN, silently (no warnings)
    with np.errstate(all="raise"):
        out_degenerate = score_block(clim, target, clim, mask, w)
    assert np.all(np.isnan(out_degenerate[2]))
    print("degenerate (pred==clim) 0/0 check: PASS (no warnings raised)")

    # fully-masked channel -> sum_w == 0 -> rmse/bias/acc all NaN, silently
    mask_fully_dry = mask.copy()
    mask_fully_dry[0, :] = False
    with np.errstate(all="raise"):
        out_dry = score_block(pred, target, clim, mask_fully_dry, w)
    assert np.all(np.isnan(out_dry[:, 0]))
    print("fully-masked channel check: PASS (no warnings raised) ->", out_dry[:, 0])

    print("dtype:", out.dtype, "shape:", out.shape)
    print("All checks passed.")
