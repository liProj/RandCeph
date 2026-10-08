"""Secondary analyses on the saved predictions of the proposed model (no training).

results/analysis/
  boot.csv           image-bootstrap 95% intervals of MRE and SDR@2 per cell (seed-averaged errors)
  device.csv         per-device error in listed mm and in anatomy-calibrated mm (A->A, I->A, joint)
  lodo_cal.csv       leave-one-device-out, listed vs calibrated units, ours and the article's
  offset.csv         per-landmark systematic offset of cross-dataset predictions (shape units -> mm)
  fewshot.csv        MRE after correcting per-landmark offsets estimated from k labelled target images
  angles.csv         SNA / SNB / ANB errors (degrees) and skeletal-class agreement
  cdf.csv            cumulative error distributions (0-10 mm)
  scale.csv          error vs apparent anatomy size in the network input (I->A)
  facts.json         scalars quoted in the text
"""
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import common as C                                    # noqa: E402
import numpy as np                                    # noqa: E402
import pandas as pd                                   # noqa: E402
from scipy import stats                               # noqa: E402

R = f"{C.ROOT}/results"
O = f"{R}/analysis"
MAIN = "main"
CELLS = [("A->A", "aariz", "aariz_valid"), ("A->I", "aariz", "isbi_test1"), ("I->I", "isbi", "isbi_test1"), ("I->A", "isbi", "aariz_valid")]
os.makedirs(O, exist_ok=True)
rng = np.random.default_rng(0)
IMG = pd.read_csv(f"{R}/audit/images.csv").set_index("id")
PREP = {n: np.load(f"{C.PREP}/{n}.npz", allow_pickle=True) for n in ("aariz_train", "aariz_valid", "aariz_test", "isbi_train", "isbi_test1", "isbi_test2")}
F = {}


from analysis_lib import angle, boot, ceph, preds, skel                    # noqa: E402,F401


# ------------------------------------------------------------------ bootstrap intervals, CDF
rows, cdf = [], []
grid = np.arange(0, 10.01, 0.1)
for tag, lab in ((MAIN, "RandCeph"), ("pilot_squash", "Same network, article's input")):
    for cell, tr, te in CELLS:
        e = preds(tag, tr, te)
        if len(e) == 0:
            continue
        em = e.mean(0)                                   # seed-averaged error per image and landmark
        lo, hi = boot(em, lambda x: np.nanmean(x))
        l2, h2 = boot(em, lambda x: 100 * np.nanmean(x <= 2.0))
        rows.append(dict(tag=tag, cell=cell, n_seeds=len(e), n_img=em.shape[1 - 1], mre=float(np.nanmean(em)), lo=lo, hi=hi, median=float(np.nanmedian(e)),
                         p90=float(np.nanpercentile(e, 90)), sdr2=float(100 * np.nanmean(e <= 2)), sdr2_lo=l2, sdr2_hi=h2,
                         img_over_4mm=float(100 * np.mean(np.nanmean(e, 2) > 4)), x_human=float(np.nanmean(e) / C.INTEROBS)))
        for g in grid:
            cdf.append(dict(tag=tag, cell=cell, mm=float(g), frac=float(100 * np.nanmean(e <= g))))
pd.DataFrame(rows).to_csv(f"{O}/boot.csv", index=False)
pd.DataFrame(cdf).to_csv(f"{O}/cdf.csv", index=False)

# ------------------------------------------------------------------ per device, listed and calibrated mm
ids = PREP["aariz_valid"]["ids"].astype(str)
ratio = IMG.loc[ids, "cal_ratio"].values                 # calibrated mm = listed mm x ratio
dev = PREP["aariz_valid"]["device"].astype(str)
rows = []
for tag in (MAIN, "pilot_squash"):
    for tr in ("aariz", "isbi", "joint"):
        e = preds(tag, tr, "aariz_valid")
        if len(e) == 0:
            continue
        img = np.nanmean(e, (0, 2))
        for d in C.DEVICES + ["all"]:
            m = np.ones(len(dev), bool) if d == "all" else dev == d
            rows.append(dict(tag=tag, train=tr, device=d, n=int(m.sum()), listed=float(img[m].mean()), calibrated=float((img * ratio)[m].mean()),
                             sdr2_listed=float(100 * np.nanmean(e[:, m] <= 2)), sdr2_cal=float(100 * np.nanmean(e[:, m] * ratio[m][None, :, None] <= 2)),
                             ratio=float(np.median(ratio[m]))))
DV = pd.DataFrame(rows)
DV.to_csv(f"{O}/device.csv", index=False)
io_all = []
for sp in ("train", "valid", "test"):
    z = PREP[f"aariz_{sp}"]
    io = np.nanmean(np.linalg.norm(z["senior"] - z["junior"], axis=-1), 1)
    io_all.append(pd.DataFrame(dict(device=z["device"].astype(str), listed=io * z["mm"], calibrated=io * z["mm"] * IMG.loc[z["ids"].astype(str), "cal_ratio"].values)))
IO = pd.concat(io_all).groupby("device").mean().reset_index()
IO.to_csv(f"{O}/interobs_device.csv", index=False)
F["interobs_listed"], F["interobs_cal"] = float(pd.concat(io_all).listed.mean()), float(pd.concat(io_all).calibrated.mean())

# ------------------------------------------------------------------ leave-one-device-out in both units
SC = pd.read_csv(f"{R}/audit/scale_by_device.csv").set_index("device")
rows = []
for d in C.DEVICES:
    n, pin, pout, pgd = C.PUBLISHED_LODO[d]
    r = dict(device=d, n=n, ratio=float(SC.loc[d, "ratio"]), pub_out=pout[0], pub_out_cal=pout[0] * float(SC.loc[d, "ratio"]))
    e = preds(MAIN, f"lodo:{d}", "lodo_out")
    if len(e):
        z = np.load(f"{C.run_dir(MAIN, 'lodo:' + d, 0)}/preds.npz")
        rt = IMG.loc[z["ids_lodo_out"].astype(str), "cal_ratio"].values
        img = np.nanmean(e, 2)                           # (seeds, N)
        r.update(ours_out=float(img.mean()), ours_out_sd=float(img.mean(1).std(ddof=1)) if len(e) > 1 else np.nan, ours_out_cal=float((img * rt).mean()),
                 ours_sdr2=float(100 * np.nanmean(e <= 2)), ours_sdr2_cal=float(100 * np.nanmean(e * rt[None, :, None] <= 2)), n_seeds=len(e))
    rows.append(r)
LC = pd.DataFrame(rows)
LC.to_csv(f"{O}/lodo_cal.csv", index=False)
rho = stats.spearmanr(LC.pub_out, 1 / LC.ratio)
F["lodo_pub_rho_units"], F["lodo_pub_p_units"] = float(rho.statistic), float(rho.pvalue)
F["lodo_pub_best"], F["lodo_pub_worst"] = LC.sort_values("pub_out").device.iloc[0], LC.sort_values("pub_out").device.iloc[-1]
F["lodo_pub_best_cal"], F["lodo_pub_worst_cal"] = LC.sort_values("pub_out_cal").device.iloc[0], LC.sort_values("pub_out_cal").device.iloc[-1]
rk = stats.spearmanr(LC.pub_out, LC.pub_out_cal)
F["lodo_pub_rank_rho"] = float(rk.statistic)
if "ours_out" in LC and LC.ours_out.notna().all():
    r2 = stats.spearmanr(LC.ours_out, 1 / LC.ratio)
    F["lodo_ours_rho_units"], F["lodo_ours_p_units"] = float(r2.statistic), float(r2.pvalue)
    F["lodo_ours_mean"], F["lodo_ours_mean_cal"] = float(LC.ours_out.mean()), float(LC.ours_out_cal.mean())
    F["lodo_ours_cv"], F["lodo_ours_cv_cal"] = float(LC.ours_out.std() / LC.ours_out.mean()), float(LC.ours_out_cal.std() / LC.ours_out_cal.mean())
    F["lodo_pub_cv"], F["lodo_pub_cv_cal"] = float(LC.pub_out.std() / LC.pub_out.mean()), float(LC.pub_out_cal.std() / LC.pub_out_cal.mean())

# ------------------------------------------------------------------ systematic offsets and few-shot correction
def frame(P):
    c = P.mean(-2, keepdims=True)
    r = np.sqrt(((P - c) ** 2).sum(-1).mean(-1))[..., None, None]
    return r


rows, fs = [], []
for cell, tr, te in (("I->A", "isbi", "aariz_valid"), ("A->I", "aariz", "isbi_test1")):
    # Offsets are estimated on labelled images of the TARGET evaluation set and applied to the remaining ones
    # (cross-fitted), so no image is corrected with its own annotation.
    P = preds(MAIN, tr, te, "pred")
    if len(P) == 0:
        continue
    G, mm = PREP[te]["lm"], PREP[te]["mm"]
    N = P.shape[1]
    U = (G[None] - P) / frame(P)                                           # residual in units of the predicted shape radius
    rmm = float((frame(P)[..., 0, 0] * mm[None]).mean())                   # mean radius in listed mm, to express offsets in mm
    e0 = np.linalg.norm(P - G[None], axis=-1) * mm[None, :, None]
    e_full = np.zeros_like(e0)
    half = rng.permutation(N)
    for fa, fb in ((half[:N // 2], half[N // 2:]), (half[N // 2:], half[:N // 2])):
        d = U[:, fa].mean(1, keepdims=True)
        e_full[:, fb] = np.linalg.norm(P[:, fb] + d * frame(P[:, fb]) - G[None, fb], axis=-1) * mm[None, fb, None]
    for k, s in enumerate(C.COMMON):
        off = U[:, :, k].mean((0, 1)) * rmm
        rows.append(dict(cell=cell, symbol=s, off_x=float(off[0]), off_y=float(off[1]), off_norm=float(np.linalg.norm(off)), mre=float(np.nanmean(e0[:, :, k])),
                         mre_corrected=float(np.nanmean(e_full[:, :, k])), random_sd=float(np.sqrt((U[:, :, k].var(1).sum(-1)).mean()) * rmm)))
    F[f"offset_frac_{cell}"] = float(1 - np.nanmean(e_full) / np.nanmean(e0))
    F[f"offset_mre_{cell}"], F[f"offset_mre0_{cell}"] = float(np.nanmean(e_full)), float(np.nanmean(e0))
    for kshot in (0, 1, 2, 5, 10, 20, 50, 75):
        v, s2 = [], []
        for rep in range(1 if kshot == 0 else 200):
            pick = rng.choice(N, kshot, replace=False)
            rest = np.setdiff1d(np.arange(N), pick)
            d = U[:, pick].mean(1, keepdims=True) if kshot else 0.0
            e = np.linalg.norm(P[:, rest] + d * frame(P[:, rest]) - G[None, rest], axis=-1) * mm[None, rest, None]
            v.append(np.nanmean(e, (1, 2)))                                # per seed
            s2.append(100 * np.nanmean(e <= 2))
        v = np.array(v)
        fs.append(dict(cell=cell, k=kshot, mre=float(v.mean()), sd_draw=float(v.mean(1).std()), sd_seed=float(v.mean(0).std(ddof=1)) if v.shape[1] > 1 else np.nan,
                       sdr2=float(np.mean(s2))))
OF = pd.DataFrame(rows)
OF.to_csv(f"{O}/offset.csv", index=False)
pd.DataFrame(fs).to_csv(f"{O}/fewshot.csv", index=False)
if len(OF):
    a, b = OF[OF.cell == "I->A"].set_index("symbol"), OF[OF.cell == "A->I"].set_index("symbol")
    if len(a) and len(b):
        cosv = [(a.loc[s, "off_x"] * b.loc[s, "off_x"] + a.loc[s, "off_y"] * b.loc[s, "off_y"]) / (a.loc[s, "off_norm"] * b.loc[s, "off_norm"]) for s in C.COMMON]
        F["offset_opposed_n"] = int(sum(c < -0.5 for c in cosv))
        F["offset_opposed"] = ", ".join(C.LM_NAMES[s] for s, c in zip(C.COMMON, cosv) if c < -0.5 and min(a.loc[s, "off_norm"], b.loc[s, "off_norm"]) > 1.0)
        F["offset_mean_cos"] = float(np.mean(cosv))
# reference mean shapes of the two datasets (Procrustes, similarity) - annotation/population difference independent of any model
def procrustes(X):
    X = X - X.mean(1, keepdims=True)
    X = X / np.sqrt((X ** 2).sum((1, 2), keepdims=True))
    ref = X[0]
    for _ in range(6):
        out = []
        for x in X:
            u, s, vt = np.linalg.svd(x.T @ ref)
            if np.linalg.det(u @ vt) < 0:
                u[:, -1] *= -1
            out.append(x @ (u @ vt))
        out = np.array(out)
        ref = out.mean(0)
        ref /= np.sqrt((ref ** 2).sum())
    return out


A_, I_ = PREP["aariz_train"]["lm"], PREP["isbi_train"]["lm"]
Pp = procrustes(np.concatenate([A_, I_]))
ma, mi = Pp[:len(A_)].mean(0), Pp[len(A_):].mean(0)
cs_mm = float(np.sqrt(((I_ - I_.mean(1, keepdims=True)) ** 2).sum((1, 2))).mean() * 0.1)
pooled = np.sqrt((Pp[:len(A_)].var(0).sum(-1) + Pp[len(A_):].var(0).sum(-1)) / 2)
MS = pd.DataFrame(dict(symbol=C.COMMON, aariz_x=ma[:, 0], aariz_y=ma[:, 1], isbi_x=mi[:, 0], isbi_y=mi[:, 1], diff_mm=np.linalg.norm(mi - ma, axis=1) * cs_mm,
                       effect=np.linalg.norm(mi - ma, axis=1) / pooled))
MS.to_csv(f"{O}/meanshape.csv", index=False)
F["meanshape_top"] = ", ".join(f"{C.LM_NAMES[s]} ({d:.1f} mm)" for s, d in MS.sort_values("diff_mm", ascending=False)[["symbol", "diff_mm"]].head(3).values)

# ------------------------------------------------------------------ angles and skeletal class
rows = []
for cell, tr, te in CELLS + [("J->A", "joint", "aariz_valid"), ("J->I", "joint", "isbi_test1")]:
    P = preds(MAIN, tr, te, "pred")
    if len(P) == 0:
        continue
    G = PREP[te]["lm"]
    g, p = ceph(G), ceph(P)
    r = dict(cell=cell, n_seeds=len(P))
    for nm, gv, pv in zip(("SNA", "SNB", "ANB"), g, p):
        d = pv - gv[None]
        r[f"{nm}_mae"], r[f"{nm}_bias"], r[f"{nm}_sd"], r[f"{nm}_within2"] = float(np.abs(d).mean()), float(d.mean()), float(d.std()), float(100 * (np.abs(d) <= 2).mean())
    cg, cp = skel(g[2]), skel(p[2])
    r["class_agree"] = float(100 * (cp == cg[None]).mean())
    po = (cp == cg[None]).mean()
    pe = sum(((cp == c).mean()) * ((cg == c).mean()) for c in (1, 2, 3))
    r["class_kappa"] = float((po - pe) / (1 - pe))
    rows.append(r)
pd.DataFrame(rows).to_csv(f"{O}/angles.csv", index=False)
z = PREP["aariz_valid"]
js, jj = ceph(z["senior"]), ceph(z["junior"])
F["interobs_ANB_mae"] = float(np.nanmean(np.abs(js[2] - jj[2])))
F["interobs_class_agree"] = float(100 * np.mean(skel(js[2]) == skel(jj[2])))

# ------------------------------------------------------------------ error vs apparent anatomy size in the input (I->A)
tr_r = []
for n in ("isbi_train", "aariz_train", "aariz_valid", "isbi_test1"):
    z = PREP[n]
    s = np.maximum(z["w"], z["h"]) / 512.0               # letterbox: original px per input px
    lm = z["lm"]
    r = np.sqrt(((lm - lm.mean(1, keepdims=True)) ** 2).sum(-1).mean(-1)) / s
    tr_r.append(pd.DataFrame(dict(set=n, id=z["ids"].astype(str), device=z["device"].astype(str), radius_letter=r,
                                  radius_squash=np.sqrt((((lm - lm.mean(1, keepdims=True)) / np.stack([z["w"], z["h"]], 1)[:, None] * 512) ** 2).sum(-1).mean(-1)),
                                  aspect=z["w"] / z["h"])))
SCA = pd.concat(tr_r)
for tag in (MAIN, "pilot_squash", "abl_letter_noaug", "abl_squash_aug"):
    e = preds(tag, "isbi", "aariz_valid")
    if len(e):
        SCA.loc[SCA.set == "aariz_valid", f"err_{tag}"] = np.nanmean(e, (0, 2))
SCA.to_csv(f"{O}/scale.csv", index=False)
F["radius_isbi_train"] = float(SCA[SCA.set == "isbi_train"].radius_letter.mean())
F["radius_isbi_sd"] = float(SCA[SCA.set == "isbi_train"].radius_letter.std())
av = SCA[SCA.set == "aariz_valid"]
F["radius_aariz_min"], F["radius_aariz_max"] = float(av.radius_letter.min()), float(av.radius_letter.max())
F["aspect_min"], F["aspect_max"] = float(SCA.aspect.min()), float(SCA.aspect.max())
for tag in (MAIN, "pilot_squash"):
    if f"err_{tag}" in av:
        rr = stats.spearmanr(np.abs(np.log(av.radius_letter / F["radius_isbi_train"])), av[f"err_{tag}"])
        F[f"scale_rho_{tag}"], F[f"scale_p_{tag}"] = float(rr.statistic), float(rr.pvalue)
json.dump(F, open(f"{O}/facts.json", "w"), indent=1)
pd.set_option("display.width", 250)
for f in ("boot", "device", "lodo_cal", "fewshot", "angles"):
    print(f"--- {f}")
    print(pd.read_csv(f"{O}/{f}.csv").round(2).to_string(index=False))
print(json.dumps(F, indent=1))
