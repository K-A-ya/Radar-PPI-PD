"""The DSP toolbox: MTI, CFAR, peak picking."""
import math
import numpy as np


def mti(x, order=2):
    """Moving target indicator. order 2 = three-pulse canceller (1, -2, 1).
    Anything that looks the same pulse after pulse (hills, buildings) subtracts to ~zero."""
    for _ in range(order):
        x = x[1:] - x[:-1]
    return x


def mti_noise_gain(order):
    return float(sum(math.comb(order, k) ** 2 for k in range(order + 1)))   # 2 for order 1, 6 for order 2


def n_training_cells(guard, train):
    (gr, gd), (tr, td) = guard, train
    return (2 * (gr + tr) + 1) * (2 * (gd + td) + 1) - (2 * gr + 1) * (2 * gd + 1)


def cfar_alpha(pfa, n_train, n_int=1):
    """Threshold multiplier for cell-averaging CFAR so false alarms happen with probability pfa.
    n_int = how many pulses were averaged in each cell (1 for coherent / FFT data).
    Exact result: Pfa = sum_k C(K+k-1, k) p^K (1-p)^k with K = n_train*n_int, p = 1/(1+alpha/n_train)."""
    K = n_train * n_int

    def pf(c):
        lp, lq = -math.log1p(c), math.log(c) - math.log1p(c)
        return sum(math.exp(math.lgamma(K + k) - math.lgamma(k + 1) - math.lgamma(K) + K * lp + k * lq)
                   for k in range(n_int))
    lo, hi = 1e-6, 1e4
    for _ in range(80):                                     # bisection on a log scale
        mid = math.sqrt(lo * hi)
        lo, hi = (mid, hi) if pf(mid) > pfa else (lo, mid)
    return n_train * math.sqrt(lo * hi)


def box_sum(a, hr, hd=0):
    """Sum of a over a (2*hd+1) x (2*hr+1) window around every cell, via a summed-area table (fast!).
    Doppler axis wraps around (it's circular), range axis repeats its edge."""
    a = np.atleast_2d(a)
    p = np.pad(a, ((hd, hd), (0, 0)), mode="wrap") if hd else a
    p = np.pad(p, ((0, 0), (hr, hr)), mode="edge")
    S = np.zeros((p.shape[0] + 1, p.shape[1] + 1))
    S[1:, 1:] = p.cumsum(0).cumsum(1)
    H, W = 2 * hd + 1, 2 * hr + 1
    return S[H:, W:] - S[:-H, W:] - S[H:, :-W] + S[:-H, :-W]


def ca_cfar(power, guard, train, alpha):
    """Cell-averaging CFAR. Each cell is compared with alpha * (average of its neighbours, skipping guard cells).
    guard/train are (range, doppler) cells on each side. Works on 1D or 2D data. Returns (hits, threshold)."""
    (gr, gd), (tr, td) = guard, train
    ring = box_sum(power, gr + tr, gd + td) - box_sum(power, gr, gd)    # big window minus the guard window
    thr = alpha * ring / n_training_cells(guard, train)
    return power > thr, thr


def local_peaks(power, hits):
    """Keep only the strongest cell in each blob of detections, so one target = one dot."""
    pad = np.pad(power, 1, mode="constant", constant_values=-np.inf)
    keep = hits.copy()
    for dy in (-1, 0, 1):
        for dx in (-1, 0, 1):
            if dy or dx:
                keep &= power >= pad[1 + dy:1 + dy + power.shape[0], 1 + dx:1 + dx + power.shape[1]]
    return keep


def to_db(x):
    return 10 * np.log10(np.maximum(x, 1e-12))
