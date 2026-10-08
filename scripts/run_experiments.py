"""Run named RandCeph experiment suites from any working directory."""
import argparse
from pathlib import Path
import shlex
import subprocess
import sys

REPO = Path(__file__).resolve().parents[1]
AUG = ["--aug_scale", "0.6", "--aug_rot", "12", "--aug_shift", "0.10", "--aug_photo", "1"]
DEVICES = ["ART Plus", "Veraviewepocs 2D", "Hyperion X5", "ProMax with ProTouch",
           "Rotograph EVO", "Smart3D", "ProMax 2D"]


def configurations(suite):
    if suite == "main":
        return [("s1_letter", source, "letter", AUG) for source in ("aariz", "isbi")]
    if suite == "controls":
        return [(tag, source, view, []) for tag, view in
                (("pilot_squash", "squash"), ("abl_letter_noaug", "letter"))
                for source in ("aariz", "isbi")]
    if suite == "ablations":
        return [(tag, source, view, extra) for tag, view, extra in
                (("abl_squash_aug", "squash", AUG),
                 ("abl_scratch", "letter", AUG + ["--pretrained", "0"]),
                 ("abl_convnext", "letter", AUG + ["--enc", "convnext_tiny"]))
                for source in ("aariz", "isbi")]
    if suite == "joint":
        return [("s1_letter", "joint", "letter", AUG)]
    if suite == "lodo":
        return [("s1_letter", "lodo:" + device, "letter", AUG) for device in DEVICES]
    raise ValueError(suite)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--suite", choices=["main", "controls", "ablations", "joint", "lodo"], required=True)
    parser.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    for tag, source, view, extra in configurations(args.suite):
        for seed in args.seeds:
            command = [sys.executable, str(REPO / "code" / "train_lm.py"),
                       "--tag", tag, "--train", source, "--view", view, "--seed", str(seed)] + extra
            print(shlex.join(command), flush=True)
            if not args.dry_run:
                subprocess.run(command, cwd=REPO, check=True)


if __name__ == "__main__":
    main()
