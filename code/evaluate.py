"""Collect every finished run and set it against the numbers PRINTED in the article.

results/runs.csv            one row per run
results/cells.csv           method x cell (A->A, A->I, I->I, I->A): mean, SD, n seeds, per-seed values
results/table2.csv          the article's Table 2 with our rows added; Welch test vs each published row
results/gd.csv              Generalization Drop (per seed, then averaged - the article's convention)
results/joint.csv           the article's Table 5 with our rows
results/lodo.csv            the article's Table 4 with our rows
results/landmark.csv        the article's Table S1 with our columns
results/ablation.csv        component ablation
results/sdr.csv             SDR at 2 / 2.5 / 3 / 4 mm
"""
import glob
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import common as C                                    # noqa: E402
import numpy as np                                    # noqa: E402
import pandas as pd                                   # noqa: E402
from scipy import stats                               # noqa: E402

R = f"{C.ROOT}/results"
MAIN = "main"
CELLS = [("A->A", "aariz", "aariz_valid"), ("A->I", "aariz", "isbi_test1"), ("I->I", "isbi", "isbi_test1"), ("I->A", "isbi", "aariz_valid")]
LABEL = {"main": "RandCeph", "main+ttc": "RandCeph + test-time canonicalisation", "s1_letter": "RandCeph, 60 passes over the training set", "s1_long": "RandCeph, equal number of steps", "pilot_squash": "Same network, article's input",
         "abl_letter_noaug": "RandCeph without augmentation", "abl_squash_aug": "Article's input + our augmentation", "abl_scratch": "RandCeph, no ImageNet weights",
         "abl_convnext": "RandCeph, ConvNeXt-T encoder", "s2_canon": "Second-stage canonical frame", "s1_uda": "RandCeph + unlabelled target images"}


def load_runs():
    rows = []
    for p in sorted(glob.glob(f"{C.RUNS}/*/*/metrics.json")):
        m = json.load(open(p))
        row = {k: v for k, v in m.items() if not isinstance(v, (dict, list))}
        row["run"] = os.path.basename(os.path.dirname(p))
        rows.append(row)
        t = p.replace("metrics.json", "ttc.json")
        if os.path.exists(t):                           # test-time canonicalisation of the same weights
            j = json.load(open(t))
            r2 = dict(tag=m["tag"] + "+ttc", train=m["train"], seed=m["seed"], run=row["run"], train_sec=m["train_sec"], params=m["params"])
            for k, v in j.items():
                if k.startswith(f"mre_{TTC}_"):
                    r2["mre_" + k[len(f"mre_{TTC}_"):]] = v
                if k.startswith(f"sdr2_{TTC}_"):
                    r2["sdr2_" + k[len(f"sdr2_{TTC}_"):]] = v
            rows.append(r2)
    D = pd.DataFrame(rows)
    extra = []
    for sfx in ("", "+ttc"):                            # the proposed method = one configuration per training set
        for tr in D.train.unique():
            d = D[(D.tag == C.real_tag("main", tr) + sfx) & (D.train == tr)].copy()
            d["tag"] = "main" + sfx
            extra.append(d)
    return pd.concat([D] + extra, ignore_index=True)


TTC = "ens"


def ms(v):
    v = np.asarray(v, float)
    return float(v.mean()), (float(v.std(ddof=1)) if len(v) > 1 else float("nan")), len(v)


def welch(m1, s1, n1, m2, s2, n2):
    """Welch t-test from summary statistics (ours vs a published mean, SD over 3 seeds)."""
    if not (np.isfinite(s1) and np.isfinite(s2)) or n1 < 2 or n2 < 2:
        return float("nan")
    return float(stats.ttest_ind_from_stats(m1, s1, n1, m2, s2, n2, equal_var=False).pvalue)


def main():
    D = load_runs()
    D.to_csv(f"{R}/runs.csv", index=False)
    # ---------------- cells
    rows = []
    for tag in sorted(D.tag.unique()):
        for cell, tr, te in CELLS:
            d = D[(D.tag == tag) & (D.train == tr)].sort_values("seed")
            if len(d) == 0 or f"mre_{te}" not in d or d[f"mre_{te}"].isna().all():
                continue
            mu, sd, n = ms(d[f"mre_{te}"])
            r = dict(tag=tag, cell=cell, mre=mu, sd=sd, n=n, seeds=" ".join(f"{v:.3f}" for v in d[f"mre_{te}"]))
            for t in ("2", "2.5", "3", "4"):
                if f"sdr{t}_{te}" in d and d[f"sdr{t}_{te}"].notna().all():
                    r[f"sdr{t}"], r[f"sdr{t}_sd"], _ = ms(d[f"sdr{t}_{te}"])
            rows.append(r)
    cells = pd.DataFrame(rows)
    cells.to_csv(f"{R}/cells.csv", index=False)
    # ---------------- Table 2
    rp = pd.read_csv(f"{R}/repro/table2.csv", names=["net", "dir", "seed", "in", "cross"]) if os.path.exists(f"{R}/repro/table2.csv") else pd.DataFrame(columns=["net", "dir", "seed", "in", "cross"])
    T = []
    for net, v in C.PUBLISHED.items():
        T.append(dict(source="article", method=net, **{c: v[i][0] for i, (c, _, _) in enumerate(CELLS)}, **{c + "_sd": v[i][1] for i, (c, _, _) in enumerate(CELLS)}, n=3))
    for net in C.PUBLISHED:
        a, i = rp[(rp.net == net) & (rp.dir == "A->I")], rp[(rp.net == net) & (rp.dir == "I->A")]
        if len(a) and len(i):
            r = dict(source="rerun of the authors' code", method=net, n=min(len(a), len(i)))
            for c, v in (("A->A", a["in"]), ("A->I", a["cross"]), ("I->I", i["in"]), ("I->A", i["cross"])):
                r[c], r[c + "_sd"], _ = ms(v)
            T.append(r)
    for tag in ("pilot_squash", MAIN, MAIN + "+ttc"):
        d = cells[cells.tag == tag]
        if len(d) == 4:
            r = dict(source="this work", method=LABEL[tag], n=int(d.n.min()))
            for _, q in d.iterrows():
                r[q.cell], r[q.cell + "_sd"] = q.mre, q.sd
            T.append(r)
    T = pd.DataFrame(T)
    T.to_csv(f"{R}/table2.csv", index=False)
    # tests: ours vs every published row, per cell
    W = []
    for tag in (MAIN, MAIN + "+ttc", "pilot_squash"):
        d = cells[cells.tag == tag].set_index("cell")
        for net, v in C.PUBLISHED.items():
            for i, (c, _, _) in enumerate(CELLS):
                if c in d.index:
                    W.append(dict(tag=tag, net=net, cell=c, published=v[i][0], published_sd=v[i][1], ours=d.loc[c, "mre"], ours_sd=d.loc[c, "sd"], n=d.loc[c, "n"],
                                  diff=d.loc[c, "mre"] - v[i][0], ratio=v[i][0] / d.loc[c, "mre"], p=welch(d.loc[c, "mre"], d.loc[c, "sd"], d.loc[c, "n"], v[i][0], v[i][1], 3),
                                  worst_seed_ours=max(float(s) for s in d.loc[c, "seeds"].split()), beats_mean_minus_2sd=bool(max(float(s) for s in d.loc[c, "seeds"].split()) < v[i][0] - 2 * v[i][1])))
    pd.DataFrame(W).to_csv(f"{R}/vs_published.csv", index=False)
    # ---------------- Generalization Drop (per seed, then mean)
    G = []
    for net, v in C.PUBLISHED_GD.items():
        G.append(dict(source="article", method=net, gd_AI=v[0][0], gd_AI_sd=v[0][1], gd_IA=v[1][0], gd_IA_sd=v[1][1]))
    for net in C.PUBLISHED:
        a, i = rp[(rp.net == net) & (rp.dir == "A->I")], rp[(rp.net == net) & (rp.dir == "I->A")]
        if len(a) and len(i):
            ga, gi = ms(100 * (a["cross"] - a["in"]) / a["in"]), ms(100 * (i["cross"] - i["in"]) / i["in"])
            G.append(dict(source="rerun of the authors' code", method=net, gd_AI=ga[0], gd_AI_sd=ga[1], gd_IA=gi[0], gd_IA_sd=gi[1]))
    for tag in sorted(D.tag.unique()):
        a, i = D[(D.tag == tag) & (D.train == "aariz")], D[(D.tag == tag) & (D.train == "isbi")]
        if len(a) and len(i) and "mre_isbi_test1" in a:
            ga, gi = ms(100 * (a.mre_isbi_test1 - a.mre_aariz_valid) / a.mre_aariz_valid), ms(100 * (i.mre_aariz_valid - i.mre_isbi_test1) / i.mre_isbi_test1)
            G.append(dict(source="this work", method=LABEL.get(tag, tag), tag=tag, gd_AI=ga[0], gd_AI_sd=ga[1], gd_IA=gi[0], gd_IA_sd=gi[1], n=min(len(a), len(i))))
    pd.DataFrame(G).to_csv(f"{R}/gd.csv", index=False)
    # ---------------- joint training (Table 5)
    J = [dict(source="article", method=net, aariz=v[0][0], aariz_sd=v[0][1], isbi=v[1][0], isbi_sd=v[1][1], n=3) for net, v in C.PUBLISHED_JOINT.items()]
    for tag in (MAIN, MAIN + "+ttc"):
        d = D[(D.tag == tag) & (D.train == "joint")]
        if len(d):
            a, i = ms(d.mre_aariz_valid), ms(d.mre_isbi_test1)
            r = dict(source="this work", method=LABEL[tag], aariz=a[0], aariz_sd=a[1], isbi=i[0], isbi_sd=i[1], n=len(d))
            for k in ("sdr2_aariz_valid", "sdr2_isbi_test1", "mre_aariz_test", "mre_isbi_test2"):
                if k in d and d[k].notna().all():
                    r[k] = float(d[k].mean())
            J.append(r)
    pd.DataFrame(J).to_csv(f"{R}/joint.csv", index=False)
    # ---------------- leave-one-device-out (Table 4); the article pools per image
    Lr = []
    for dev, (n, pin, pout, pgd) in C.PUBLISHED_LODO.items():
        r = dict(device=dev, n=n, pub_in=pin[0], pub_in_sd=pin[1], pub_out=pout[0], pub_out_sd=pout[1], pub_gd=pgd[0], pub_gd_sd=pgd[1])
        for tag, sfx in ((MAIN, ""), (MAIN + "+ttc", "_ttc")):
            d = D[(D.tag == tag) & (D.train == f"lodo:{dev}")]
            if len(d) and "mre_lodo_out" in d:
                col_in, col_out = ("mre_img_lodo_in", "mre_img_lodo_out") if sfx == "" else ("mre_lodo_in", "mre_lodo_out")
                r["in" + sfx], r["in_sd" + sfx], r["seeds" + sfx] = ms(d[col_in])
                r["out" + sfx], r["out_sd" + sfx], _ = ms(d[col_out])
                r["gd" + sfx], r["gd_sd" + sfx], _ = ms(100 * (d[col_out] - d[col_in]) / d[col_in])
                if sfx == "":
                    r["sdr2_out"] = float(d.sdr2_lodo_out.mean())
                    r["worst_seed_out"] = float(d[col_out].max())
        Lr.append(r)
    pd.DataFrame(Lr).to_csv(f"{R}/lodo.csv", index=False)
    # ---------------- per landmark (Table S1): trained on Aariz; in = Aariz test, cross = ISBI test1
    E = {k: [] for k in ("aariz_test", "isbi_test1", "aariz_valid")}
    EI = {k: [] for k in ("aariz_valid", "isbi_test1")}
    for s in (0, 1, 2):
        p = f"{C.run_dir(MAIN, 'aariz', s)}/preds.npz"
        if os.path.exists(p):
            z = np.load(p)
            for k in E:
                E[k].append(np.nanmean(z[f"err_{k}"], 0))
        p = f"{C.run_dir(MAIN, 'isbi', s)}/preds.npz"
        if os.path.exists(p):
            z = np.load(p)
            for k in EI:
                EI[k].append(np.nanmean(z[f"err_{k}"], 0))
    io = []
    for sp in ("train", "valid", "test"):
        z = np.load(f"{C.PREP}/aariz_{sp}.npz", allow_pickle=True)
        io.append(np.linalg.norm(z["senior"] - z["junior"], axis=-1) * z["mm"][:, None])
    io = np.nanmean(np.concatenate(io), 0)
    Lm = []
    for k, s in enumerate(C.COMMON):
        p = C.PUBLISHED_LM[s]
        r = dict(symbol=s, name=C.LM_NAMES[s], hrnet_in=p[0], hrnet_cross=p[1], vit_in=p[2], vit_cross=p[3], interobs=float(io[k]))
        if E["aariz_test"]:
            a, c = np.array(E["aariz_test"])[:, k], np.array(E["isbi_test1"])[:, k]
            r.update(ours_in=a.mean(), ours_in_sd=a.std(ddof=1) if len(a) > 1 else np.nan, ours_cross=c.mean(), ours_cross_sd=c.std(ddof=1) if len(c) > 1 else np.nan,
                     ours_gd=float(np.mean(100 * (c - a) / a)), n=len(a))
        if EI["aariz_valid"]:
            a, c = np.array(EI["isbi_test1"])[:, k], np.array(EI["aariz_valid"])[:, k]
            r.update(ours_I_in=a.mean(), ours_I_cross=c.mean())
        Lm.append(r)
    pd.DataFrame(Lm).to_csv(f"{R}/landmark.csv", index=False)
    # ---------------- ablation
    A = []
    for tag in ("pilot_squash", "abl_squash_aug", "abl_letter_noaug", "abl_scratch", "s1_letter", "s1_long", MAIN, MAIN + "+ttc", "s2_canon", "abl_convnext", "s1_uda"):
        d = cells[cells.tag == tag]
        if len(d):
            r = dict(tag=tag, method=LABEL[tag])
            for _, q in d.iterrows():
                r[q.cell], r[q.cell + "_sd"], r[q.cell + "_n"] = q.mre, q.sd, q.n
            A.append(r)
    pd.DataFrame(A).to_csv(f"{R}/ablation.csv", index=False)
    cells[["tag", "cell", "n"] + [c for c in cells.columns if c.startswith("sdr")]].to_csv(f"{R}/sdr.csv", index=False)
    pd.set_option("display.width", 250)
    print(T.round(2).to_string(index=False))
    print(pd.DataFrame(A).round(2).to_string(index=False))
    print(pd.DataFrame(J).round(2).to_string(index=False))
    print(pd.DataFrame(Lr).round(2).to_string(index=False))


if __name__ == "__main__":
    main()
