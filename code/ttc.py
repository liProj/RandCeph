"""Test-time self-canonicalisation (no retraining, no labels of the test image).

The first pass uses the training view (letterbox).  From its own prediction the model estimates
where the anatomy is (centroid of the 19 landmarks) and how large it is (their RMS radius), and
the image is re-rendered so that both equal the training-set averages; this is repeated --iters
times.  Optionally the final pass is an ensemble over a few scales around the canonical one
(coordinate-wise median).

  python ttc.py --run s1_letter/isbi_s0 --iters 2 --scales 0.8,1,1.25
Output: results/runs/<tag>/<run>/ttc.npz + ttc.json (does not touch metrics.json)
"""
import argparse
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import common as C                                    # noqa: E402
import numpy as np                                    # noqa: E402
import torch                                          # noqa: E402
import train_lm as T                                  # noqa: E402


def canon_M(c, r, c0, r0, s_mul=1.0):
    """Input px -> original px so that centroid c (orig px) lands on c0 (input px) and radius r on r0."""
    s = r / r0 * s_mul
    M = torch.zeros(len(c), 2, 3, device="cuda")
    M[:, 0, 0] = M[:, 1, 1] = s
    M[:, 0, 2], M[:, 1, 2] = c[:, 0] - s * c0[0], c[:, 1] - s * c0[1]
    return M


@torch.no_grad()
def predict(model, D, a, idx, M):
    x = (T.render(D, idx, M, a) - 0.5) / 0.25
    with torch.autocast("cuda", dtype=torch.bfloat16, enabled=True):
        hm = model(x)
    return T.to_orig(T.soft_argmax(hm, a.H, a.W), M)


@torch.no_grad()
def train_stats(tr, a):
    idx = torch.arange(tr.n, device="cuda")
    y = T.to_input(tr.lm, T.view_matrix(tr, idx, a, False))
    c = y.mean(1)
    r = torch.sqrt(((y - c[:, None]) ** 2).sum(-1).mean(1))
    return c.mean(0), float(r.mean())


@torch.no_grad()
def run(model, D, a, c0, r0, iters, scales, bs=8):
    out = {k: [] for k in range(iters + 1)}
    ens = []
    for i in range(0, D.n, bs):
        idx = torch.arange(i, min(i + bs, D.n), device="cuda")
        P = predict(model, D, a, idx, T.view_matrix(D, idx, a, False))
        out[0].append(P)
        for it in range(1, iters + 1):
            c = P.mean(1)
            r = torch.sqrt(((P - c[:, None]) ** 2).sum(-1).mean(1))
            if it == iters and len(scales) > 1:
                Q = torch.stack([predict(model, D, a, idx, canon_M(c, r, c0, r0, s)) for s in scales])
                ens.append(Q.median(0).values)
            P = predict(model, D, a, idx, canon_M(c, r, c0, r0))
            out[it].append(P)
    res = {f"it{k}": torch.cat(v).cpu().numpy() for k, v in out.items()}
    if ens:
        res["ens"] = torch.cat(ens).cpu().numpy()
    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True)
    ap.add_argument("--iters", type=int, default=2)
    ap.add_argument("--scales", default="0.8,1,1.25")
    q = ap.parse_args()
    d = f"{C.RUNS}/{q.run}"
    m = json.load(open(f"{d}/metrics.json"))
    a = argparse.Namespace(**m["args"])
    a.pretrained = 0
    model = T.build(a)
    model.load_state_dict(torch.load(f"{d}/model.pt"))
    model.eval()
    if a.train.startswith("lodo:"):
        tr, d_in, d_out = T.lodo_split(a.train[5:], a.seed)
        ev = {"lodo_in": d_in, "lodo_out": d_out, "isbi_test1": T.Data("isbi", "test1")}
    else:
        tr = {"aariz": lambda: T.Data("aariz", "train"), "isbi": lambda: T.Data("isbi", "train"),
              "joint": lambda: T.cat([T.Data("aariz", "train"), T.Data("isbi", "train")])}[a.train]()
        ev = {"aariz_valid": T.Data("aariz", "valid"), "isbi_test1": T.Data("isbi", "test1"), "aariz_test": T.Data("aariz", "test"), "isbi_test2": T.Data("isbi", "test2")}
    c0, r0 = train_stats(tr, a)
    scales = [float(s) for s in q.scales.split(",")]
    res, save = dict(c0=[float(v) for v in c0], r0=r0, iters=q.iters, scales=scales), {}
    for k, D in ev.items():
        R = run(model, D, a, c0, r0, q.iters, scales)
        for name, P in R.items():
            mre, e = C.mre_mm(P, D.lm.cpu().numpy(), D.mm.cpu().numpy())
            res[f"mre_{name}_{k}"] = mre
            res[f"sdr2_{name}_{k}"] = C.sdr(e, 2.0)
            save[f"pred_{name}_{k}"], save[f"err_{name}_{k}"] = P.astype(np.float32), e.astype(np.float32)
        save[f"ids_{k}"] = D.ids.astype(str)
        print(k, " ".join(f"{n} {res[f'mre_{n}_{k}']:.3f}/{res[f'sdr2_{n}_{k}']:.1f}" for n in R), flush=True)
    np.savez_compressed(f"{d}/ttc.npz", **save)
    json.dump(res, open(f"{d}/ttc.json", "w"), indent=1)


if __name__ == "__main__":
    main()
