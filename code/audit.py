"""Data reconciliation with the article and the pixel-spacing audit.  No model involved.

results/data_check.csv, data_check.md
results/audit/scale_by_device.csv   sella-nasion and nasion-menton distances per device in the
                                    listed millimetres; implied true spacing
results/audit/images.csv            one row per image with its calibration factor
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import common as C                                    # noqa: E402
import numpy as np                                    # noqa: E402
import pandas as pd                                   # noqa: E402

rows = []
for n in ("aariz_train", "aariz_valid", "aariz_test", "isbi_train", "isbi_test1", "isbi_test2"):
    z = np.load(f"{C.PREP}/{n}.npz", allow_pickle=True)
    S, N, Me = (C.COMMON.index(k) for k in ("S", "N", "Me"))
    for i in range(len(z["w"])):
        lm = z["lm"][i]
        rows.append(dict(set=n.split("_")[0], split=n.split("_")[1], id=str(z["ids"][i]), device=str(z["device"][i]), w=int(z["w"][i]), h=int(z["h"][i]),
                         mm=float(z["mm"][i]), sn_px=float(np.linalg.norm(lm[S] - lm[N])), nme_px=float(np.linalg.norm(lm[N] - lm[Me])),
                         interobs_px=float(np.nanmean(np.linalg.norm(z["senior"][i] - z["junior"][i], axis=-1))) if not np.isnan(z["junior"][i]).all() else np.nan))
D = pd.DataFrame(rows)
D["sn_mm"], D["nme_mm"] = D.sn_px * D.mm, D.nme_px * D.mm
ref = float(D[D.device == "ART Plus"].sn_mm.median())           # reference device: largest group, plausible adult S-N
D["cal"] = ref / D.sn_px                                        # anatomy-calibrated mm per pixel of THIS image
D["cal_ratio"] = D.cal / D.mm                                   # factor by which the listed spacing understates length
os.makedirs(f"{C.ROOT}/results/audit", exist_ok=True)
D.to_csv(f"{C.ROOT}/results/audit/images.csv", index=False)
g = D.groupby(["set", "device"]).agg(n=("w", "size"), width=("w", "median"), height=("h", "median"), listed_mm=("mm", "median"), fov_w_mm=("w", lambda v: 0),
                                     sn_mm=("sn_mm", "median"), sn_sd=("sn_mm", "std"), nme_mm=("nme_mm", "median"), ratio=("cal_ratio", "median"),
                                     interobs_mm=("interobs_px", lambda v: np.nan)).reset_index()
for i, r in g.iterrows():
    d = D[(D.set == r.set) & (D.device == r.device)]
    g.loc[i, "fov_w_mm"] = float((d.w * d.mm).median())
    g.loc[i, "interobs_mm"] = float((d.interobs_px * d.mm).mean()) if d.interobs_px.notna().any() else np.nan
g["implied_mm"] = g.listed_mm * g.ratio
g.to_csv(f"{C.ROOT}/results/audit/scale_by_device.csv", index=False)
print(g.round(3).to_string(index=False))
print(f"reference S-N (ART Plus median) = {ref:.1f} mm; images on devices whose listed spacing is off by more than 20%: "
      f"{int(((D.set == 'aariz') & ((D.cal_ratio > 1.2) | (D.cal_ratio < 0.8))).sum())} of {int((D.set == 'aariz').sum())} Aariz images")

# ---- reconciliation with the article
R = []


def add(item, article, ours):
    R.append(dict(item=item, article=article, ours=ours, match="yes" if article == ours else "NO"))


a = D[D.set == "aariz"]
i_ = D[D.set == "isbi"]
add("ISBI 2015 images", 400, len(i_))
add("ISBI train / test1 / test2", "150 / 150 / 100", " / ".join(str(int((i_.split == s).sum())) for s in ("train", "test1", "test2")))
add("ISBI image size", "1935x2400", f"{int(i_.w.median())}x{int(i_.h.median())}")
add("ISBI pixel spacing (mm)", 0.1, round(float(i_.mm.median()), 4))
add("Aariz images", 1000, len(a))
add("Aariz train / valid / test", "700 / 150 / 150", " / ".join(str(int((a.split == s).sum())) for s in ("train", "valid", "test")))
add("Aariz devices", 7, a.device.nunique())
for dev, n in {"ART Plus": 366, "Veraviewepocs 2D": 177, "Hyperion X5": 143, "ProMax with ProTouch": 135, "Rotograph EVO": 79, "Smart3D": 59, "ProMax 2D": 41}.items():
    add(f"Aariz images from {dev}", n, int((a.device == dev).sum()))
add("Shared landmarks", 19, len(C.COMMON))
io = float((a.interobs_px * a.mm).mean())
add("Aariz inter-observer variability (mm, article 0.53)", 0.53, round(io, 2))
T = pd.DataFrame(R)
T.to_csv(f"{C.ROOT}/results/data_check.csv", index=False)
ok = int((T.match == "yes").sum())
with open(f"{C.ROOT}/data_check.md", "w") as fh:
    fh.write("# Data reconciliation - Diagnostics 16(17):2726 vs the two public datasets as downloaded\n\n"
             "Aariz: figshare 27986417, `Aariz.zip`, MD5 `e0bd645bca6759abdae4f199d841bda6` equal to figshare's checksum. "
             "ISBI 2015: the Kaggle export `jiahongqian/cephalometric-landmarks` named in the authors' loader (122,670,555 bytes; archive integrity test passed).\n\n"
             f"**{ok}/{len(T)} items match.**\n\n" + T.to_markdown(index=False) + "\n\n## Pixel-spacing audit\n\n"
             "The same anatomical distance (sella-nasion) measured from the reference annotations, converted with each image's listed spacing:\n\n"
             + g[["set", "device", "n", "width", "height", "listed_mm", "sn_mm", "sn_sd", "implied_mm", "ratio"]].round(3).to_markdown(index=False) + "\n")
print(f"{ok}/{len(T)} items match")
print(T[T.match != "yes"].to_string())
