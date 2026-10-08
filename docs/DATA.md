# Datasets

Obtain the datasets from their original providers and follow their access and reuse terms.

- Aariz: [Figshare record 27986417](https://doi.org/10.6084/m9.figshare.27986417).
- ISBI 2015: the code expects the [cephalometric-landmarks Kaggle export](https://www.kaggle.com/datasets/jiahongqian/cephalometric-landmarks).

Expected layout:

```text
data/
  aariz/Aariz/
    cephalogram_machine_mappings.csv
    train/
      Cephalograms/
      Annotations/Cephalometric Landmarks/
        Senior Orthodontists/*.json
        Junior Orthodontists/*.json
    valid/ ... same structure ...
    test/  ... same structure ...
  isbi/
    train_senior.csv
    test1_senior.csv
    test2_senior.csv
    cepha400/cepha400/<image_path from CSV>
```

Aariz has 700/150/150 train/valid/test images; ISBI has 150/150/100 train/test1/test2 images. The image caches created by `code/prep.py` contain both 512-pixel square images and 1024-pixel aspect-preserving images.

The shared landmark order is S, N, Or, Po, A, B, Pog, Me, Gn, Go, LIT, UIT, Ls, Li, Sn, soft-tissue Pog, PNS, ANS, Ar. Aariz coordinates average senior and junior annotations when both exist. The ISBI export uses the senior annotation.

Predictions and references are compared in original-image pixels, then multiplied by the image's listed mm/pixel. ISBI uses 0.1 mm/pixel; Aariz reads its device mapping CSV. MRE pools valid landmarks; SDR reports the percentage with radial error at or below the threshold.

Example external storage configuration:

```bash
export RANDCEPH_ROOT=/path/to/experiment
export RANDCEPH_AARIZ=/path/to/Aariz
export RANDCEPH_ISBI=/path/to/isbi
python code/prep.py
```

In PowerShell, use `$env:RANDCEPH_ROOT = 'D:/RandCephData'` and equivalent assignments for the dataset paths. Neither data download nor conversion of arbitrary annotation formats is performed automatically.
