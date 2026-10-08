"""Pilot step 1: run the authors' own cross-dataset script (supplement, code/m8_multiseed.py) with
only three kinds of edits, all applied textually here so the supplement stays untouched:
  - data paths (their ~/Downloads/... -> our data directories),
  - device (their Apple MPS / CPU -> CUDA),
  - which architectures and seeds to run (command line), and where the log is written.
Nothing in the model, the loss, the preprocessing or the evaluation is changed.

  python pilot_repro.py HRNet 0        # architecture name as in their CFG, seed list "0" or "0,1,2"
"""
import os
import re
import sys

ROOT = os.path.abspath(os.environ.get("RANDCEPH_ROOT", os.path.join(os.path.dirname(__file__), "..")))
SRC = f"{ROOT}/supp/diagnostics-16-02726-s001/code/m8_multiseed.py"
arch, seeds = sys.argv[1], sys.argv[2]
s = open(SRC).read()


def rep(a, b):
    global s
    assert s.count(a) == 1, (a, s.count(a))
    s = s.replace(a, b)


rep('AARIZ=os.path.expanduser("~/Downloads/Aariz"); ISBI=os.path.expanduser("~/Downloads/archive-2")',
    f'AARIZ="{ROOT}/data/aariz/Aariz"; ISBI="{ROOT}/data/isbi"')
rep('DEV="mps" if torch.backends.mps.is_available() else "cpu"', 'DEV="cuda"')
rep('SEEDS=[0,1,2]', f'SEEDS=[{seeds}]')
rep('R=open(os.path.expanduser("~/Downloads/SONUCLAR.txt"),"a")', f'R=open("{ROOT}/results/pilot/repro_{arch}_s{seeds.replace(",", "")}.txt","a")')
rep('for name,kind,SIZE,build,lr in CFG:', f'for name,kind,SIZE,build,lr in [c for c in CFG if c[0]=="{arch}"]:')
# with a single seed the script's closing summary (SD over seeds, Mann-Whitney over families) is undefined; stop before it
i = s.index("# paradigma karsilastirmasi")
s = s[:i] + "R.close()\n"
s = s.replace('log(f"  -> {name} {tag}: GD {np.mean(gds):.0f}% +/- {np.std(gds,ddof=1):.0f}% (n=3 seed)")',
              'log(f"  -> {name} {tag}: GD mean {np.mean(gds):.0f}% over {len(gds)} seed(s)")')
os.makedirs(f"{ROOT}/results/pilot", exist_ok=True)
exec(compile(s, SRC, "exec"))
