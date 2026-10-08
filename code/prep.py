"""Decode every cephalogram once and store (a) the authors' input: grayscale, squashed to 512x512
with PIL's default resampling; (b) a larger grayscale copy with the long side 1024 and the aspect
ratio preserved, for experiments at higher resolution.  data/prep/<set>_<split>.npz"""
import os
import sys
from multiprocessing import Pool

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import common as C                                    # noqa: E402
import numpy as np                                    # noqa: E402
from PIL import Image                                 # noqa: E402

Image.MAX_IMAGE_PIXELS = None
L = 1024


def load(p):
    im = Image.open(p).convert("L")
    w, h = im.size
    a = np.asarray(im.resize((512, 512)), np.uint8)
    s = L / max(w, h)
    b = np.zeros((L, L), np.uint8)
    r = np.asarray(im.resize((max(1, round(w * s)), max(1, round(h * s))), Image.BICUBIC), np.uint8)
    b[:r.shape[0], :r.shape[1]] = r
    return a, b, w, h


if __name__ == "__main__":
    os.makedirs(C.PREP, exist_ok=True)
    for name, fn, splits in (("isbi", C.isbi_items, ("train", "test1", "test2")), ("aariz", C.aariz_items, ("train", "valid", "test"))):
        for sp in splits:
            it = fn(sp)
            with Pool(10) as pool:
                R = pool.map(load, [d["path"] for d in it], chunksize=8)
            np.savez(f"{C.PREP}/{name}_{sp}.npz", x512=np.stack([r[0] for r in R]), x1024=np.stack([r[1] for r in R]),
                     w=np.array([r[2] for r in R]), h=np.array([r[3] for r in R]), lm=np.stack([d["lm"] for d in it]),
                     senior=np.stack([d["senior"] for d in it]), junior=np.stack([d["junior"] for d in it]),
                     mm=np.array([d["mm"] for d in it], np.float32), device=np.array([d["device"] for d in it]), ids=np.array([d["id"] for d in it]))
            w, h = np.array([r[2] for r in R]), np.array([r[3] for r in R])
            mm = np.array([d["mm"] for d in it])
            print(f"{name} {sp}: {len(it)} images | size median {int(np.median(w))}x{int(np.median(h))} (min {w.min()}x{h.min()}, max {w.max()}x{h.max()}) | "
                  f"mm/px {mm.min():.3f}-{mm.max():.3f} | field of view median {np.median(w * mm):.0f}x{np.median(h * mm):.0f} mm | NaN landmarks {int(np.isnan(np.stack([d['lm'] for d in it])).any(-1).sum())}", flush=True)
