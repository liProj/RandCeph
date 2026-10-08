"""Shared constants, data access and metrics for round 6 (article 2726, cephalometric landmarks).

Ground truth and evaluation follow the authors' released script exactly: 19 landmarks common to
both datasets; Aariz ground truth = mean of the senior and junior tracings, ISBI = senior tracing
(the Kaggle export has no junior file); error in millimetres = pixel error in the ORIGINAL image
times the image's mm/pixel (ISBI 0.1; Aariz per image from cephalogram_machine_mappings.csv);
in-domain evaluation on Aariz `valid` (150) and ISBI `test1` (150), as in their m8_multiseed.py.
"""
import csv
import glob
import json
import os

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")

import numpy as np

ROOT = os.path.abspath(os.environ.get("RANDCEPH_ROOT", os.path.join(os.path.dirname(__file__), "..")))
AARIZ = os.environ.get("RANDCEPH_AARIZ", f"{ROOT}/data/aariz/Aariz")
ISBI = os.environ.get("RANDCEPH_ISBI", f"{ROOT}/data/isbi")
PREP = f"{ROOT}/data/prep"
RUNS = f"{ROOT}/results/runs"
COMMON = ["S", "N", "Or", "Po", "A", "B", "Pog", "Me", "Gn", "Go", "LIT", "UIT", "Ls", "Li", "Sn", "Pog`", "PNS", "ANS", "Ar"]
# Article, Table 2: MRE in mm, mean and SD over three seeds: (A->A, A->I, I->I, I->A)
PUBLISHED = {"HRNet": ((2.47, 0.12), (11.13, 1.54), (4.12, 0.94), (21.25, 1.83)),
             "Cascade": ((2.75, 0.31), (12.01, 1.82), (4.40, 0.48), (22.15, 1.61)),
             "ViT": ((2.67, 0.09), (8.67, 0.73), (5.11, 0.29), (20.15, 0.26)),
             "ResNet50": ((3.04, 0.12), (7.73, 0.39), (5.55, 0.14), (23.07, 1.30))}
# Article, Table 5: balanced joint training, MRE on (Aariz, ISBI)
PUBLISHED_JOINT = {"HRNet": ((2.21, 0.17), (2.53, 0.13)), "Cascade": ((2.10, 0.11), (2.52, 0.05)),
                   "ViT": ((2.50, 0.04), (3.35, 0.06)), "ResNet50": ((2.75, 0.10), (3.54, 0.17))}


def aariz_items(split):
    amm = {r["cephalogram_id"]: (float(r["pixel_size"]), r["machine"]) for r in csv.DictReader(open(f"{AARIZ}/cephalogram_machine_mappings.csv"))}
    ann = f"{AARIZ}/{split}/Annotations/Cephalometric Landmarks"
    out = []
    for p in sorted(glob.glob(f"{ann}/Senior Orthodontists/*.json")):
        cid = os.path.splitext(os.path.basename(p))[0]

        def rd(q):
            return {l["symbol"]: (l["value"]["x"], l["value"]["y"]) for l in json.load(open(q))["landmarks"]}
        a = rd(p)
        jp = f"{ann}/Junior Orthodontists/{cid}.json"
        j = rd(jp) if os.path.exists(jp) else None
        sym = a if j is None else {k: ((a[k][0] + j[k][0]) / 2, (a[k][1] + j[k][1]) / 2) for k in a if k in j}
        lm = np.array([sym.get(k, (np.nan, np.nan)) for k in COMMON], np.float32)
        sen = np.array([a.get(k, (np.nan, np.nan)) for k in COMMON], np.float32)
        jun = np.array([j.get(k, (np.nan, np.nan)) for k in COMMON], np.float32) if j else np.full((19, 2), np.nan, np.float32)
        out.append(dict(id=cid, path=glob.glob(f"{AARIZ}/{split}/Cephalograms/{cid}.*")[0], lm=lm, senior=sen, junior=jun, mm=amm[cid][0], device=amm[cid][1]))
    return out


def isbi_items(split):
    out = []
    for r in csv.DictReader(open(f"{ISBI}/{split}_senior.csv")):
        lm = np.array([[float(r[f"{k}_x"]), float(r[f"{k}_y"])] for k in range(1, 20)], np.float32)
        out.append(dict(id=r["image_path"], path=f"{ISBI}/cepha400/cepha400/{r['image_path']}", lm=lm, senior=lm, junior=np.full((19, 2), np.nan, np.float32), mm=0.1, device="ISBI (CRANEX)"))
    return sorted(out, key=lambda d: d["id"])


def mre_mm(pred_px, gt_px, mm):
    """pred_px, gt_px: (N, 19, 2) in ORIGINAL-image pixels; mm: (N,).  Mean radial error over all
    valid landmarks (the authors' pooling), plus per-image and per-landmark errors."""
    d = np.linalg.norm(pred_px - gt_px, axis=-1) * np.asarray(mm)[:, None]
    ok = ~np.isnan(d)
    return float(d[ok].mean()), d


def sdr(d, thr):
    ok = ~np.isnan(d)
    return float((d[ok] <= thr).mean() * 100)

# Article, Table 3: Generalization Drop in %, (A->I, I->A), mean and SD of the per-seed values
PUBLISHED_GD = {"HRNet": ((350, 44), (426, 70)), "Cascade": ((338, 52), (408, 74)), "ViT": ((225, 22), (295, 27)), "ResNet50": ((155, 13), (316, 18))}
# Article, Table 2 caption: SDR@2 mm in-domain (one seed); cross-domain range 5.8-17.4 %
PUBLISHED_SDR2_IN = {"HRNet": 63.7, "Cascade": 59.8, "ViT": 46.1, "ResNet50": 37.6}
PUBLISHED_SDR2_CROSS = (5.8, 17.4)
INTEROBS = 0.53
# Article, Table 4 (ViT): device -> (n, in-domain MRE, held-out MRE, GD %)
PUBLISHED_LODO = {"Smart3D": (59, (2.67, 0.02), (1.89, 0.18), (-29, 7)), "Hyperion X5": (143, (2.76, 0.19), (2.86, 0.39), (5, 21)),
                  "ART Plus": (366, (2.83, 0.27), (5.66, 0.07), (101, 21)), "Veraviewepocs 2D": (177, (2.92, 0.12), (4.85, 1.00), (66, 28)),
                  "ProMax 2D": (41, (2.65, 0.15), (4.88, 0.20), (85, 11)), "ProMax with ProTouch": (135, (2.82, 0.16), (7.94, 0.75), (181, 15)),
                  "Rotograph EVO": (79, (2.41, 0.15), (6.55, 0.30), (172, 9))}
PUBLISHED_LODO_MEAN = (2.72, 4.95)                    # text of Section 3.3: in-domain reference, held-out average
# Article, Supplementary Table S1: trained on Aariz; in-domain = Aariz test split, cross = ISBI test1.
# landmark -> (HRNet in, HRNet cross, ViT in, ViT cross), order of COMMON
PUBLISHED_LM = {"S": (4.08, 16.91, 3.40, 11.22), "N": (2.99, 7.67, 3.13, 11.64), "Or": (3.31, 9.42, 3.41, 9.64), "Po": (4.28, 21.16, 3.60, 12.01),
                "A": (2.70, 12.06, 2.61, 11.92), "B": (1.95, 10.57, 2.41, 5.96), "Pog": (1.10, 8.36, 2.30, 5.12), "Me": (0.73, 5.96, 2.13, 4.96),
                "Gn": (0.72, 5.23, 2.20, 4.77), "Go": (5.05, 29.21, 3.90, 8.04), "LIT": (2.61, 10.67, 2.80, 7.75), "UIT": (1.84, 4.75, 2.75, 6.69),
                "Ls": (1.59, 3.89, 3.00, 8.34), "Li": (1.36, 4.36, 2.94, 10.12), "Sn": (2.54, 5.27, 2.89, 9.30), "Pog`": (1.71, 14.41, 2.81, 12.61),
                "PNS": (3.53, 15.69, 3.23, 7.96), "ANS": (3.05, 6.72, 3.04, 7.51), "Ar": (3.93, 19.25, 3.33, 9.16)}
PUBLISHED_LM_ALL = (2.58, 11.13, 2.94, 8.67)
LM_NAMES = {"S": "Sella", "N": "Nasion", "Or": "Orbitale", "Po": "Porion", "A": "A-point", "B": "B-point", "Pog": "Pogonion", "Me": "Menton", "Gn": "Gnathion",
            "Go": "Gonion", "LIT": "Lower incisor tip", "UIT": "Upper incisor tip", "Ls": "Upper lip", "Li": "Lower lip", "Sn": "Subnasale",
            "Pog`": "Soft-tissue pogonion", "PNS": "Posterior nasal spine", "ANS": "Anterior nasal spine", "Ar": "Articulare"}
DEVICES = ["ART Plus", "Veraviewepocs 2D", "Hyperion X5", "ProMax with ProTouch", "Rotograph EVO", "Smart3D", "ProMax 2D"]

# The proposed method ("main") is one configuration per training set; the tag of the ISBI-trained runs is set here.
MAIN_AARIZ = "s1_letter"
MAIN_ISBI = "s1_letter"


def real_tag(tag, train):
    if tag != "main":
        return tag
    return MAIN_ISBI if train == "isbi" else MAIN_AARIZ


def run_dir(tag, train, seed):
    return f"{RUNS}/{real_tag(tag, train)}/{train.replace(':', '_').replace(' ', '_')}_s{seed}"
