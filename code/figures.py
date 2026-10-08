"""All figures of the paper, drawn from the result files only.  A figure whose inputs are missing is skipped
(and reported), so the script can be run while training is still in progress."""
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import common as C                                    # noqa: E402
import figstyle as S                                  # noqa: E402
import numpy as np                                    # noqa: E402
import pandas as pd                                   # noqa: E402
from figstyle import clean, plt, save                 # noqa: E402

R = f"{C.ROOT}/results"
A = f"{R}/analysis"
FIG = f"{C.ROOT}/paper/figures"
MAIN = "main"
CELLS = ["A->A", "A->I", "I->I", "I->A"]
CELL_T = {"A->A": "Aariz → Aariz (in-domain)", "A->I": "Aariz → ISBI (cross)", "I->I": "ISBI → ISBI (in-domain)", "I->A": "ISBI → Aariz (cross)"}
NETS = ["HRNet", "Cascade", "ViT", "ResNet50"]
SHORT = {"ART Plus": "ART Plus", "Veraviewepocs 2D": "Veraview. 2D", "Hyperion X5": "Hyperion X5", "ProMax with ProTouch": "ProMax ProTouch",
         "Rotograph EVO": "Rotograph EVO", "Smart3D": "Smart3D", "ProMax 2D": "ProMax 2D", "ISBI (CRANEX)": "ISBI"}
PREP = {}
done = []


def prep(n):
    if n not in PREP:
        PREP[n] = np.load(f"{C.PREP}/{n}.npz", allow_pickle=True)
    return PREP[n]


def fig(fn):
    def run():
        try:
            fn()
            done.append(fn.__name__)
        except (FileNotFoundError, KeyError, IndexError, ValueError) as e:
            print(f"  SKIPPED {fn.__name__}: {type(e).__name__} {e}")
            if os.environ.get("STRICT"):
                raise
    return run


def view(z, i, kind, R_=384):
    """The image as the network sees it (numpy, for illustration) and the landmarks in that frame."""
    import cv2
    w, h = float(z["w"][i]), float(z["h"][i])
    s = 1024 / max(w, h)
    im = z["x1024"][i][: int(round(h * s)), : int(round(w * s))]
    lm = z["lm"][i]
    if kind == "squash":
        return cv2.resize(im, (R_, R_), interpolation=cv2.INTER_AREA), lm / np.array([w, h]) * R_
    f = R_ / max(w, h)
    nw, nh = int(round(w * f)), int(round(h * f))
    out = np.zeros((R_, R_), np.uint8)
    x0, y0 = (R_ - nw) // 2, (R_ - nh) // 2
    out[y0:y0 + nh, x0:x0 + nw] = cv2.resize(im, (nw, nh), interpolation=cv2.INTER_AREA)
    return out, lm * f + np.array([x0, y0])


@fig
def fig01_views():
    picks = [("isbi_test1", None), ("aariz_valid", "Hyperion X5"), ("aariz_valid", "ProMax with ProTouch"), ("aariz_valid", "Rotograph EVO")]
    f, ax = plt.subplots(2, 4, figsize=(9.6, 5.0))
    for j, (n, dev) in enumerate(picks):
        z = prep(n)
        i = 0 if dev is None else int(np.where(z["device"] == dev)[0][0])
        for r, kind in enumerate(("squash", "letter")):
            im, lm = view(z, i, kind)
            ax[r, j].imshow(im, cmap="gray", vmin=0, vmax=255)
            ax[r, j].scatter(lm[:, 0], lm[:, 1], s=7, c=S.CAT[3], linewidths=0)
            ax[r, j].set_xticks([])
            ax[r, j].set_yticks([])
            for sp in ax[r, j].spines.values():
                sp.set_color(S.GRID)
        ax[0, j].set_title(f"{'ISBI 2015' if dev is None else SHORT[dev]}\n{int(z['w'][i])} × {int(z['h'][i])} px, listed {float(z['mm'][i]):.3f} mm/px", fontsize=8)
    ax[0, 0].set_ylabel("article's input\n(squashed to a square)", fontsize=8.5, color=S.INK)
    ax[1, 0].set_ylabel(f"{S.NAME} input\n(aspect preserved)", fontsize=8.5, color=S.INK)
    f.tight_layout()
    save(f, f"{FIG}/fig01_views")


@fig
def fig02_audit():
    D = pd.read_csv(f"{R}/audit/images.csv")
    order = ["ISBI (CRANEX)", "ART Plus", "ProMax 2D", "ProMax with ProTouch", "Rotograph EVO", "Veraviewepocs 2D", "Smart3D", "Hyperion X5"]
    f, ax = plt.subplots(1, 2, figsize=(9.6, 3.6), gridspec_kw=dict(width_ratios=[1.5, 1]))
    rng = np.random.default_rng(0)
    ax[0].axhspan(62, 78, color=S.CAT[2], alpha=0.13, lw=0)
    ax[0].text(7.45, 79.5, "anatomically plausible\nsella–nasion length", fontsize=7.5, color=S.CAT[5], ha="right", va="bottom")
    for k, d in enumerate(order):
        v = D[D.device == d].sn_mm.values
        ax[0].scatter(k + rng.uniform(-0.28, 0.28, len(v)), v, s=4, color=S.COLOR["plain"] if d.startswith("ISBI") else S.COLOR["ours"], alpha=0.35, linewidths=0)
        ax[0].plot([k - 0.33, k + 0.33], [np.median(v)] * 2, color=S.INK, lw=1.6)
        ax[0].text(k, np.median(v) + (9 if np.median(v) < 100 else -14), f"{np.median(v):.0f}", ha="center", fontsize=8, color=S.INK)
    ax[0].set_xticks(range(len(order)))
    ax[0].set_xticklabels([f"{SHORT[d]}\n(n={int((D.device == d).sum())})" for d in order], fontsize=7, rotation=30, ha="right")
    ax[0].set_ylabel("sella–nasion distance (listed mm)")
    ax[0].set_title("(a) One anatomical distance, converted with the listed pixel spacing", fontsize=9)
    clean(ax[0])
    g = pd.read_csv(f"{R}/audit/scale_by_device.csv")
    ax[1].plot([0.05, 0.3], [0.05, 0.3], color=S.MUTED, lw=1, ls="--")
    for _, r in g.iterrows():
        ax[1].scatter(r.listed_mm, r.implied_mm, s=18 + r.n / 4, color=S.COLOR["plain"] if r.device.startswith("ISBI") else S.COLOR["ours"], edgecolor="white", linewidth=0.8, zorder=3)
        ax[1].annotate(SHORT[r.device], (r.listed_mm, r.implied_mm), xytext=(5, 3 if r.device not in ("ProMax 2D", "Smart3D") else -9), textcoords="offset points", fontsize=7, color=S.INK2)
    ax[1].set_xlabel("listed pixel spacing (mm/px)")
    ax[1].set_ylabel("spacing implied by anatomy (mm/px)")
    ax[1].set_title("(b) Listed vs implied spacing", fontsize=9)
    ax[1].set_xlim(0.05, 0.3)
    ax[1].set_ylim(0.05, 0.3)
    clean(ax[1], "both")
    f.tight_layout()
    save(f, f"{FIG}/fig02_audit")


def bar_label(ax, x, y, txt, **kw):
    ax.text(x, y, txt, ha="center", va="bottom", fontsize=7, color=S.INK, **kw)


@fig
def fig03_matrix():
    T = pd.read_csv(f"{R}/table2.csv")
    ours = T[(T.source == "this work") & (T.method == "Proposed")].iloc[0]
    plain = T[(T.source == "this work") & (T.method.str.startswith("Same network"))].iloc[0]
    f, ax = plt.subplots(1, 4, figsize=(9.8, 3.3), sharey=True)
    for a, c in zip(ax, CELLS):
        vals = [(n, T[(T.source == "article") & (T.method == n)].iloc[0], S.COLOR[n]) for n in NETS] + [("plain", plain, S.COLOR["plain"]), (S.NAME, ours, S.COLOR["ours"])]
        for k, (n, r, col) in enumerate(vals):
            a.bar(k, r[c], yerr=r[c + "_sd"] if np.isfinite(r[c + "_sd"]) else None, color=col, width=0.72, error_kw=dict(lw=0.8, ecolor=S.INK2, capsize=2))
            bar_label(a, k, r[c] * 1.12 + (r[c + "_sd"] if np.isfinite(r[c + "_sd"]) else 0), f"{r[c]:.2f}", fontweight="bold" if n == S.NAME else None)
        a.axhline(C.INTEROBS, color=S.COLOR["human"], lw=1, ls="--")
        a.set_yscale("log")
        a.set_ylim(0.3, 45)
        a.set_yticks([0.5, 1, 2, 5, 10, 20])
        a.set_yticklabels(["0.5", "1", "2", "5", "10", "20"])
        a.set_xticks(range(6))
        a.set_xticklabels(["HRNet", "Cascade", "ViT", "ResNet50", "same net,\narticle input", S.NAME], rotation=40, ha="right", fontsize=7)
        a.set_title(CELL_T[c], fontsize=9)
        clean(a)
    ax[0].set_ylabel("mean radial error (mm, log scale)")
    ax[0].text(-0.45, 0.33, "dashed: inter-observer 0.53 mm", fontsize=6.5, color=S.COLOR["human"], bbox=dict(facecolor="white", edgecolor="none", pad=0.6, alpha=0.9))
    f.tight_layout()
    save(f, f"{FIG}/fig03_matrix")


@fig
def fig04_gd():
    G = pd.read_csv(f"{R}/gd.csv")
    T = pd.read_csv(f"{R}/table2.csv")
    o = G[(G.source == "this work") & (G.tag == MAIN)].iloc[0]
    ot = T[(T.source == "this work") & (T.method == "Proposed")].iloc[0]
    f, ax = plt.subplots(1, 2, figsize=(9.4, 3.3))
    names = NETS + [S.NAME]
    for j, (k, lab) in enumerate((("gd_AI", "Aariz → ISBI"), ("gd_IA", "ISBI → Aariz"))):
        for i, n in enumerate(names):
            r = o if n == S.NAME else G[(G.source == "article") & (G.method == n)].iloc[0]
            ax[0].bar(i + (j - 0.5) * 0.4, r[k], 0.38, yerr=r[k + "_sd"] if np.isfinite(r[k + "_sd"]) else None, color=(S.COLOR["ours"] if n == S.NAME else S.COLOR[n]),
                      hatch="" if j == 0 else "///", edgecolor="white", error_kw=dict(lw=0.8, ecolor=S.INK2, capsize=2))
            bar_label(ax[0], i + (j - 0.5) * 0.4, r[k] + (r[k + "_sd"] if np.isfinite(r[k + "_sd"]) else 0) + 6, f"{r[k]:.0f}")
    ax[0].set_xticks(range(5))
    ax[0].set_xticklabels(names)
    ax[0].set_ylabel("Generalization Drop (%)")
    ax[0].set_title("(a) Relative drop (plain: A→I, hatched: I→A)")
    clean(ax[0])
    # absolute in-domain vs cross-domain: the drop depends on the denominator
    for n in names:
        r = ot if n == S.NAME else T[(T.source == "article") & (T.method == n)].iloc[0]
        col = S.COLOR["ours"] if n == S.NAME else S.COLOR[n]
        for cin, ccr, mk in (("A->A", "A->I", "o"), ("I->I", "I->A", "s")):
            ax[1].scatter(r[cin], r[ccr], s=46 if n == S.NAME else 30, color=col, marker=mk, edgecolor="white", linewidth=0.7, zorder=3)
            if mk == "o":
                off = {"HRNet": (-30, -3), "Cascade": (6, 3), "ViT": (-20, -3), "ResNet50": (6, -8)}.get(n, (7, 2))
                ax[1].annotate(n, (r[cin], r[ccr]), xytext=off, textcoords="offset points", fontsize=7, color=S.INK2)
    ax[1].plot([0.5, 30], [0.5, 30], color=S.MUTED, lw=1, ls="--")
    ax[1].text(7, 6.2, "no drop", fontsize=7, color=S.MUTED, rotation=38)
    ax[1].set_xscale("log")
    ax[1].set_yscale("log")
    ax[1].set_xlim(0.6, 8)
    ax[1].set_ylim(0.6, 30)
    ax[1].set_xticks([1, 2, 5])
    ax[1].set_yticks([1, 2, 5, 10, 20])
    for a_ in (ax[1].xaxis, ax[1].yaxis):
        a_.set_major_formatter(plt.FuncFormatter(lambda v, _: f"{v:g}"))
        a_.set_minor_formatter(plt.NullFormatter())
    ax[1].set_xlabel("in-domain MRE (mm)")
    ax[1].set_ylabel("cross-domain MRE (mm)")
    ax[1].set_title("(b) Absolute errors (circle: trained on Aariz, square: on ISBI)")
    clean(ax[1], "both")
    f.tight_layout()
    save(f, f"{FIG}/fig04_gd")


@fig
def fig05_ablation():
    Ab = pd.read_csv(f"{R}/ablation.csv").set_index("tag")
    order = [t for t in ("pilot_squash", "abl_squash_aug", "abl_letter_noaug", "abl_scratch", MAIN, MAIN + "+ttc", "s2_canon", "abl_convnext", "s1_uda") if t in Ab.index]
    if C.MAIN_ISBI == "s1_letter" and "s1_long" in Ab.index:
        order.append("s1_long")
    if C.MAIN_ISBI != "s1_letter" and "s1_letter" in Ab.index:
        order.insert(order.index(MAIN), "s1_letter")
    lab = {"pilot_squash": "article's input, no augmentation", "abl_squash_aug": "article's input + augmentation", "abl_letter_noaug": "aspect-preserving input, no augmentation",
           "abl_scratch": f"{S.NAME} without ImageNet weights", "s1_letter": "aspect-preserving input + augmentation, 60 passes", MAIN: f"{S.NAME} (proposed)", MAIN + "+ttc": "+ test-time canonicalisation", "s2_canon": "+ second-stage canonical frame",
           "abl_convnext": "ConvNeXt-T encoder", "s1_uda": "+ unlabelled target images (mean teacher)", "s1_long": "7,020 steps on ISBI"}
    f, ax = plt.subplots(1, 4, figsize=(10.2, 0.42 * len(order) + 1.3), sharey=True)
    for a, c in zip(ax, CELLS):
        for k, t in enumerate(order):
            if c not in Ab.columns or not np.isfinite(Ab.loc[t, c]):
                continue
            v, sd = Ab.loc[t, c], Ab.loc[t, c + "_sd"]
            a.barh(k, v, xerr=sd if np.isfinite(sd) else None, color=S.COLOR["ours"] if t == MAIN else (S.COLOR["plain"] if t == "pilot_squash" else S.SEQ[2]), height=0.7,
                   error_kw=dict(lw=0.8, ecolor=S.INK2, capsize=2))
            a.text(v + (sd if np.isfinite(sd) else 0) + a.get_xlim()[1] * 0.0, k, f" {v:.2f}", va="center", fontsize=7.5, color=S.INK)
        a.set_title(CELL_T[c], fontsize=8.5)
        a.set_xlabel("MRE (mm)")
        a.set_xlim(0, a.get_xlim()[1] * 1.22)
        clean(a, "x")
    ax[0].set_yticks(range(len(order)))
    ax[0].set_yticklabels([lab[t] for t in order], fontsize=8)
    ax[0].invert_yaxis()
    f.tight_layout()
    save(f, f"{FIG}/fig05_ablation")


@fig
def fig06_landmark():
    L = pd.read_csv(f"{R}/landmark.csv")
    L = L.sort_values("ours_cross").reset_index(drop=True)
    f, ax = plt.subplots(1, 2, figsize=(9.8, 4.6), sharey=True)
    y = np.arange(len(L))
    for a, (k, t) in zip(ax, (("in", "(a) Trained on Aariz, tested on Aariz (in-domain)"), ("cross", "(b) Trained on Aariz, tested on ISBI (cross-domain)"))):
        for i in y:
            a.plot([L.loc[i, f"ours_{k}"], max(L.loc[i, f"hrnet_{k}"], L.loc[i, f"vit_{k}"])], [i, i], color=S.GRID, lw=1.2, zorder=1)
        a.scatter(L[f"hrnet_{k}"], y, s=26, color=S.COLOR["HRNet"], marker="s", label="article: HRNet", zorder=3)
        a.scatter(L[f"vit_{k}"], y, s=26, color=S.COLOR["ViT"], marker="^", label="article: ViT", zorder=3)
        a.scatter(L[f"ours_{k}"], y, s=34, color=S.COLOR["ours"], marker="D", label=S.NAME, zorder=4)
        a.scatter(L.interobs, y, s=22, color=S.COLOR["human"], marker="|", label="inter-observer", zorder=3)
        a.axvline(2.0, color=S.MUTED, lw=0.9, ls=":")
        a.set_xscale("log")
        a.set_xlim(0.045, 40)
        a.set_xticks([0.1, 0.3, 0.5, 1, 2, 5, 10, 20])
        a.set_xticklabels(["0.1", "0.3", "0.5", "1", "2", "5", "10", "20"])
        a.set_xlabel("mean radial error (mm, log scale)")
        a.set_title(t, fontsize=9)
        clean(a, "x")
    ax[0].set_yticks(y)
    ax[0].set_yticklabels(L.name, fontsize=8)
    ax[1].text(2.05, len(L) - 0.4, "2 mm", fontsize=7, color=S.MUTED)
    h, l = ax[1].get_legend_handles_labels()
    f.legend(h, l, loc="lower center", ncol=4, fontsize=8, bbox_to_anchor=(0.55, -0.045))
    f.tight_layout()
    save(f, f"{FIG}/fig06_landmark")


@fig
def fig07_lodo():
    L = pd.read_csv(f"{A}/lodo_cal.csv")
    if "ours_out" not in L or L.ours_out.isna().any():
        raise ValueError("leave-one-device-out runs incomplete")
    Lp = pd.read_csv(f"{R}/lodo.csv").set_index("device")
    L = L.sort_values("pub_out").reset_index(drop=True)
    f, ax = plt.subplots(1, 2, figsize=(9.8, 3.6))
    x = np.arange(len(L))
    for a, (kp, ko, t) in zip(ax, (("pub_out", "ours_out", "(a) Held-out device error in listed millimetres (the article's metric)"),
                                   ("pub_out_cal", "ours_out_cal", "(b) The same errors in anatomy-calibrated millimetres"))):
        sd_p = [Lp.loc[d, "pub_out_sd"] for d in L.device] if kp == "pub_out" else None
        sd_o = [Lp.loc[d, "out_sd"] for d in L.device] if ko == "ours_out" and "out_sd" in Lp else None
        a.bar(x - 0.2, L[kp], 0.38, yerr=sd_p, color=S.COLOR["ViT"], label="article (ViT)", error_kw=dict(lw=0.8, ecolor=S.INK2, capsize=2))
        a.bar(x + 0.2, L[ko], 0.38, yerr=sd_o, color=S.COLOR["ours"], label=S.NAME, error_kw=dict(lw=0.8, ecolor=S.INK2, capsize=2))
        for i in x:
            bar_label(a, i - 0.2, L.loc[i, kp] + (sd_p[i] if sd_p else 0) + 0.1, f"{L.loc[i, kp]:.1f}")
            bar_label(a, i + 0.2, L.loc[i, ko] + 0.1, f"{L.loc[i, ko]:.2f}")
        a.set_xticks(x)
        a.set_xticklabels([f"{SHORT[d]}\n×{r:.2f}" if kp.endswith("cal") else SHORT[d] for d, r in zip(L.device, L.ratio)], rotation=35, ha="right", fontsize=7)
        a.set_ylabel("held-out MRE (mm)")
        a.set_title(t, fontsize=8.5)
        clean(a)
    ax[0].legend(loc="upper left")
    f.tight_layout()
    save(f, f"{FIG}/fig07_lodo")


@fig
def fig08_joint():
    J = pd.read_csv(f"{R}/joint.csv")
    T = pd.read_csv(f"{R}/table2.csv")
    o = J[(J.source == "this work") & (J.method == "Proposed")].iloc[0]
    ot = T[(T.source == "this work") & (T.method == "Proposed")].iloc[0]
    f, ax = plt.subplots(1, 2, figsize=(8.6, 3.2), sharey=True)
    names = NETS + [S.NAME]
    for a, (k, cs, t) in zip(ax, (("aariz", "A->A", "(a) Tested on Aariz"), ("isbi", "A->I", "(b) Tested on ISBI"))):
        for i, n in enumerate(names):
            jr = o if n == S.NAME else J[(J.source == "article") & (J.method == n)].iloc[0]
            tr = ot if n == S.NAME else T[(T.source == "article") & (T.method == n)].iloc[0]
            col = S.COLOR["ours"] if n == S.NAME else S.COLOR[n]
            a.bar(i - 0.2, tr[cs], 0.38, color=col, alpha=0.45, yerr=tr[cs + "_sd"] if np.isfinite(tr[cs + "_sd"]) else None, error_kw=dict(lw=0.8, ecolor=S.INK2, capsize=2))
            a.bar(i + 0.2, jr[k], 0.38, color=col, yerr=jr[k + "_sd"] if np.isfinite(jr[k + "_sd"]) else None, error_kw=dict(lw=0.8, ecolor=S.INK2, capsize=2))
            bar_label(a, i - 0.2, tr[cs] * 1.1, f"{tr[cs]:.2f}")
            bar_label(a, i + 0.2, jr[k] * 1.1, f"{jr[k]:.2f}")
        a.axhline(C.INTEROBS, color=S.COLOR["human"], lw=1, ls="--")
        a.set_yscale("log")
        a.set_ylim(0.4, 20)
        a.set_yticks([0.5, 1, 2, 5, 10])
        a.set_yticklabels(["0.5", "1", "2", "5", "10"])
        a.set_xticks(range(5))
        a.set_xticklabels(names, fontsize=8)
        a.set_title(t + " — pale: trained on Aariz only, solid: joint", fontsize=8.5)
        clean(a)
    ax[0].set_ylabel("MRE (mm, log scale)")
    f.tight_layout()
    save(f, f"{FIG}/fig08_joint")


@fig
def fig09_cdf():
    Cd = pd.read_csv(f"{A}/cdf.csv")
    f, ax = plt.subplots(1, 4, figsize=(9.8, 2.9), sharey=True)
    for a, c in zip(ax, CELLS):
        for tag, col, lab in (("pilot_squash", S.COLOR["plain"], "same network, article's input"), (MAIN, S.COLOR["ours"], S.NAME)):
            d = Cd[(Cd.tag == tag) & (Cd.cell == c)]
            a.plot(d.mm, d.frac, color=col, label=lab)
        a.axvline(2, color=S.MUTED, lw=0.9, ls=":")
        if c in ("A->A", "I->I"):
            a.scatter([2] * 4, list(C.PUBLISHED_SDR2_IN.values()), s=16, color=S.COLOR["pub"], marker="o", zorder=4, label="article, in-domain (4 networks, one seed)" if c == "A->A" else None)
        else:
            a.plot([2, 2], C.PUBLISHED_SDR2_CROSS, color=S.COLOR["pub"], lw=3.5, solid_capstyle="butt", zorder=4, label="article, cross-domain range" if c == "A->I" else None)
        a.set_title(CELL_T[c], fontsize=8.5)
        a.set_xlabel("error threshold (mm)")
        a.set_xlim(0, 8)
        a.set_ylim(0, 100)
        clean(a, "both")
    ax[0].set_ylabel("landmarks within threshold (%)")
    h, l = [], []
    for a in ax:
        hh, ll = a.get_legend_handles_labels()
        for x, y in zip(hh, ll):
            if y not in l:
                h.append(x)
                l.append(y)
    f.legend(h, l, loc="lower center", ncol=4, fontsize=7.5, bbox_to_anchor=(0.5, -0.07))
    f.tight_layout()
    save(f, f"{FIG}/fig09_cdf")


@fig
def fig10_offsets():
    O = pd.read_csv(f"{A}/offset.csv")
    Fs = pd.read_csv(f"{A}/fewshot.csv")
    M = pd.read_csv(f"{A}/meanshape.csv").set_index("symbol")
    f, ax = plt.subplots(1, 3, figsize=(10.2, 3.7), gridspec_kw=dict(width_ratios=[1, 1, 1.05]))
    Z = prep("isbi_train")
    lm = Z["lm"]
    base = (lm - lm.mean(1, keepdims=True)).mean(0) * 0.1                      # mean ISBI shape in mm (for drawing only)
    for a, c, t in ((ax[0], "I->A", "(a) Trained on ISBI, tested on Aariz"), (ax[1], "A->I", "(b) Trained on Aariz, tested on ISBI")):
        d = O[O.cell == c].set_index("symbol")
        a.scatter(base[:, 0], base[:, 1], s=10, color=S.MUTED, zorder=2)
        for k, s in enumerate(C.COMMON):
            big = d.loc[s, "off_norm"] > 2.0
            a.annotate("", xy=(base[k, 0] - d.loc[s, "off_x"] * 3, base[k, 1] - d.loc[s, "off_y"] * 3), xytext=(base[k, 0], base[k, 1]),
                       arrowprops=dict(arrowstyle="-|>", color=S.COLOR["plain"] if big else S.SEQ[3], lw=1.3 if big else 0.8, shrinkA=0, shrinkB=0))
            a.text(base[k, 0] + 1.5, base[k, 1] + 1.5, s.replace("`", "′"), fontsize=6.5, color=S.INK2)
        a.set_aspect("equal")
        a.invert_yaxis()
        a.set_xticks([])
        a.set_yticks([])
        a.set_title(t, fontsize=8.5)
        for sp in a.spines.values():
            sp.set_visible(False)
    ax[0].text(0.0, -0.09, "arrows: mean prediction − reference, drawn ×3; orange: offset > 2 mm", transform=ax[0].transAxes, fontsize=6.8, color=S.INK2)
    for c, col, mk in (("I->A", S.COLOR["ours"], "o"), ("A->I", S.COLOR["cal"], "s")):
        d = Fs[Fs.cell == c]
        xs = np.arange(len(d))
        ax[2].errorbar(xs, d.mre, yerr=d.sd_draw, color=col, marker=mk, capsize=2, lw=1.6, label={"I->A": "ISBI → Aariz", "A->I": "Aariz → ISBI"}[c])
        ax[2].set_xticks(xs)
        ax[2].set_xticklabels(d.k.astype(int))
    ax[2].set_xlabel("labelled target images used to estimate the offsets")
    ax[2].set_ylabel("cross-domain MRE (mm)")
    ax[2].set_title("(c) Offset correction from k target images", fontsize=8.5)
    ax[2].set_ylim(0, None)
    ax[2].legend()
    clean(ax[2])
    f.tight_layout()
    save(f, f"{FIG}/fig10_offsets")


@fig
def fig11_qualitative():
    cases = [("A->A", "aariz", "aariz_valid"), ("A->I", "aariz", "isbi_test1"), ("I->I", "isbi", "isbi_test1"), ("I->A", "isbi", "aariz_valid")]
    f, ax = plt.subplots(2, 4, figsize=(9.8, 5.3))
    for j, (c, tr, te) in enumerate(cases):
        z = prep(te)
        for r, tag in enumerate(("pilot_squash", MAIN)):
            p = np.load(f"{C.run_dir(tag, tr, 0)}/preds.npz")
            e = np.nanmean(np.load(f"{C.run_dir(MAIN, tr, 0)}/preds.npz")[f"err_{te}"], 1)
            i = int(np.argsort(e)[len(e) // 2])                               # the median image of the proposed model
            w, h = float(z["w"][i]), float(z["h"][i])
            s = 1024 / max(w, h)
            im = z["x1024"][i][: int(round(h * s)), : int(round(w * s))]
            a = ax[r, j]
            a.imshow(im, cmap="gray", vmin=0, vmax=255)
            g, q = z["lm"][i] * s, p[f"pred_{te}"][i] * s
            for k in range(19):
                a.plot([g[k, 0], q[k, 0]], [g[k, 1], q[k, 1]], color=S.CAT[3], lw=0.9)
            a.scatter(g[:, 0], g[:, 1], s=9, color=S.CAT[2], linewidths=0, label="reference")
            a.scatter(q[:, 0], q[:, 1], s=9, color=S.CAT[7], marker="x", linewidths=0.9, label="prediction")
            a.set_xticks([])
            a.set_yticks([])
            a.set_title(f"{CELL_T[c] if r == 0 else ''}\n{float(np.nanmean(p[f'err_{te}'][i])):.2f} mm", fontsize=8)
            for sp in a.spines.values():
                sp.set_color(S.GRID)
    ax[0, 0].set_ylabel("same network,\narticle's input", fontsize=8.5, color=S.INK)
    ax[1, 0].set_ylabel(S.NAME, fontsize=8.5, color=S.INK)
    ax[0, 0].legend(loc="lower left", fontsize=6.5, frameon=True, facecolor="white", framealpha=0.8)
    f.tight_layout()
    save(f, f"{FIG}/fig11_qualitative")


@fig
def fig12_angles():
    import analysis_lib as AL
    f, ax = plt.subplots(1, 4, figsize=(9.8, 2.9), sharex=True, sharey=True)
    for a, (c, tr, te) in zip(ax, (("A->A", "aariz", "aariz_valid"), ("A->I", "aariz", "isbi_test1"), ("I->I", "isbi", "isbi_test1"), ("I->A", "isbi", "aariz_valid"))):
        P = AL.preds(MAIN, tr, te, "pred")
        g, p = AL.ceph(prep(te)["lm"])[2], AL.ceph(P)[2].mean(0)
        a.axhspan(0, 4, color=S.CAT[2], alpha=0.08, lw=0)
        a.axvspan(0, 4, color=S.CAT[2], alpha=0.08, lw=0)
        a.plot([-10, 14], [-10, 14], color=S.MUTED, lw=0.9, ls="--")
        a.scatter(g, p, s=7, color=S.COLOR["ours"], alpha=0.6, linewidths=0)
        a.set_title(f"{CELL_T[c]}\nMAE {np.abs(p - g).mean():.2f}°", fontsize=8.5)
        a.set_xlabel("reference ANB (°)")
        a.set_xlim(-8, 13)
        a.set_ylim(-8, 13)
        clean(a, "both")
    ax[0].set_ylabel("predicted ANB (°)")
    f.tight_layout()
    save(f, f"{FIG}/fig12_angles")


@fig
def fig13_device():
    D = pd.read_csv(f"{A}/device.csv")
    IO = pd.read_csv(f"{A}/interobs_device.csv").set_index("device")
    d = D[(D.tag == MAIN) & (D.train == "aariz") & (D.device != "all")].set_index("device").loc[C.DEVICES]
    f, ax = plt.subplots(1, 2, figsize=(9.4, 3.3), sharey=True)
    x = np.arange(len(d))
    for a, k, t in ((ax[0], "listed", "(a) Listed millimetres"), (ax[1], "calibrated", "(b) Anatomy-calibrated millimetres")):
        a.bar(x - 0.2, d[k], 0.38, color=S.COLOR["ours"], label=f"{S.NAME}, in-domain")
        a.bar(x + 0.2, IO.loc[C.DEVICES, k], 0.38, color=S.COLOR["human"], alpha=0.75, label="senior vs junior tracing")
        for i in x:
            bar_label(a, i - 0.2, d[k].iloc[i] + 0.02, f"{d[k].iloc[i]:.2f}")
            bar_label(a, i + 0.2, IO.loc[C.DEVICES[i], k] + 0.02, f"{IO.loc[C.DEVICES[i], k]:.2f}")
        a.set_xticks(x)
        a.set_xticklabels([f"{SHORT[v]}\n(n={int(n)})" for v, n in zip(d.index, d.n)], rotation=35, ha="right", fontsize=7)
        a.set_title(t, fontsize=9)
        clean(a)
    ax[0].set_ylabel("mean radial error on Aariz validation (mm)")
    ax[0].legend(loc="upper left", fontsize=7.5)
    f.tight_layout()
    save(f, f"{FIG}/fig13_device")


@fig
def fig14_scale():
    Sc = pd.read_csv(f"{A}/scale.csv")
    tr = Sc[Sc.set == "isbi_train"]
    av = Sc[Sc.set == "aariz_valid"].copy()
    av["cal_ratio"] = pd.read_csv(f"{R}/audit/images.csv").set_index("id").loc[av.id.astype(str), "cal_ratio"].values
    f, ax = plt.subplots(1, 2, figsize=(9.4, 3.4), sharey=True)
    for a, (k, tag, col, t) in zip(ax, (("radius_squash", "pilot_squash", S.COLOR["plain"], "(a) Article's input, no augmentation"), ("radius_letter", MAIN, S.COLOR["ours"], f"(b) {S.NAME}"))):
        lo, hi = tr[k].min(), tr[k].max()
        a.axvspan(lo, hi, color=S.MUTED, alpha=0.18, lw=0)
        a.text((lo + hi) / 2, 55, "training\nrange", ha="center", fontsize=7, color=S.INK2)
        if tag == MAIN:
            a.axvspan(lo / 1.6, hi * 1.6, color=S.COLOR["ours"], alpha=0.07, lw=0)
            a.text(hi * 1.6, 55, "with scale\naugmentation ", ha="right", fontsize=7, color=S.COLOR["ours"])
        for dv, mk in zip(C.DEVICES, ("o", "s", "^", "D", "v", "P", "X")):
            m = av.device == dv
            a.scatter(av[k][m], (av[f"err_{tag}"] * av.cal_ratio)[m], s=13, color=col, marker=mk, alpha=0.75, linewidths=0, label=SHORT[dv])
        a.set_yscale("log")
        a.set_ylim(0.8, 90)
        a.set_yticks([1, 2, 5, 10, 20, 50])
        a.set_yticklabels(["1", "2", "5", "10", "20", "50"])
        a.set_xlabel("anatomy size in the network input (RMS landmark radius, px)", fontsize=8)
        a.set_title(t, fontsize=9)
        clean(a, "both")
    ax[0].set_ylabel("ISBI → Aariz error per image\n(calibrated mm, log scale)")
    ax[1].legend(fontsize=6.5, ncol=2, loc="upper right", bbox_to_anchor=(1.0, 0.86))
    f.tight_layout()
    save(f, f"{FIG}/fig14_scale")


if __name__ == "__main__":
    os.makedirs(FIG, exist_ok=True)
    for fn in (fig01_views, fig02_audit, fig03_matrix, fig04_gd, fig05_ablation, fig06_landmark, fig07_lodo, fig08_joint, fig09_cdf, fig10_offsets, fig11_qualitative,
               fig12_angles, fig13_device, fig14_scale):
        fn()
    json.dump(done, open(f"{FIG}/done.json", "w"))
    print(len(done), "figures")
