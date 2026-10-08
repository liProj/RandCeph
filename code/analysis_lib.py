"""Helpers shared by analysis.py and figures.py."""
import os

import common as C
import numpy as np

rng = np.random.default_rng(0)


def preds(tag, train, key, what="err", ttc=False):
    """Stack over the seeds that exist: (n_seeds, N, 19[, 2])."""
    out = []
    for s in (0, 1, 2):
        p = f"{C.run_dir(tag, train, s)}/" + ("ttc.npz" if ttc else "preds.npz")
        if os.path.exists(p):
            z = np.load(p)
            k = f"{what}_ens_{key}" if ttc else f"{what}_{key}"
            if k in z:
                out.append(z[k])
    return np.array(out)


def boot(e, fn, B=2000):
    n = e.shape[0]
    v = [fn(e[rng.integers(0, n, n)]) for _ in range(B)]
    return float(np.percentile(v, 2.5)), float(np.percentile(v, 97.5))


def angle(a, b, c):
    """Angle at b between rays b->a and b->c, degrees.  Arrays (..., 2)."""
    u, v = a - b, c - b
    cs = (u * v).sum(-1) / (np.linalg.norm(u, axis=-1) * np.linalg.norm(v, axis=-1))
    return np.degrees(np.arccos(np.clip(cs, -1, 1)))


def ceph(P):
    S, N, A, B = (P[..., C.COMMON.index(k), :] for k in ("S", "N", "A", "B"))
    sna, snb = angle(S, N, A), angle(S, N, B)
    return sna, snb, sna - snb


def skel(anb):
    return np.where(anb < 0, 3, np.where(anb > 4, 2, 1))
