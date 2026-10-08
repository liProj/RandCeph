"""Full reproduction of the article's Table 2 with the authors' own script (m8_multiseed.py).

Edits, applied textually so the supplement stays untouched:
  - data paths and device (as in pilot_repro.py);
  - which architecture / seeds to run and where the log goes;
  - SPEED ONLY: the decoded, grayscale, resized image of each file is kept in memory after its
    first use.  The authors decode every full-size radiograph from disk on every access; the
    cached array is bit-identical to what their two lines produce (same PIL calls), so models,
    loss, augmentation and evaluation are unchanged.
  python repro_cached.py <HRNet|Cascade|ViT|ResNet50> <seeds, e.g. 0,1,2>
"""
import os
import sys

ROOT = os.path.abspath(os.environ.get("RANDCEPH_ROOT", os.path.join(os.path.dirname(__file__), "..")))
SRC = f"{ROOT}/supp/diagnostics-16-02726-s001/code/m8_multiseed.py"
arch, seeds = sys.argv[1], sys.argv[2]
s = open(SRC).read()


def rep(a, b, n=1):
    global s
    assert s.count(a) == n, (a, s.count(a))
    s = s.replace(a, b)


rep('AARIZ=os.path.expanduser("~/Downloads/Aariz"); ISBI=os.path.expanduser("~/Downloads/archive-2")', f'AARIZ="{ROOT}/data/aariz/Aariz"; ISBI="{ROOT}/data/isbi"')
rep('DEV="mps" if torch.backends.mps.is_available() else "cpu"', 'DEV="cuda"')
rep('SEEDS=[0,1,2]', f'SEEDS=[{seeds}]')
rep('R=open(os.path.expanduser("~/Downloads/SONUCLAR.txt"),"a")', f'R=open("{ROOT}/results/repro/{arch}_s{seeds.replace(",", "")}.txt","a")')
rep('for name,kind,SIZE,build,lr in CFG:', f'for name,kind,SIZE,build,lr in [c for c in CFG if c[0]=="{arch}"]:')
# speed-only cache
rep('        img=Image.open(ip).convert("L"); W0,H0=img.size\n        img=img.resize((SIZE,SIZE)); a=np.asarray(img,dtype=np.float32)/255.0',
    '        a,W0,H0=_ld(ip,SIZE)')
rep('        img=Image.open(os.path.join(s.imd,iid)).convert("L"); W0,H0=img.size\n        img=img.resize((SIZE,SIZE)); a=np.asarray(img,dtype=np.float32)/255.0',
    '        a,W0,H0=_ld(os.path.join(s.imd,iid),SIZE)')
rep('class AarizDS(Dataset):', '''_C={}
def _ld(p,SIZE):
    k=(p,SIZE)
    if k not in _C:
        img=Image.open(p).convert("L"); W0,H0=img.size
        _C[k]=(np.asarray(img.resize((SIZE,SIZE)),dtype=np.uint8).copy(),W0,H0)
    u,W0,H0=_C[k]
    return np.asarray(u,dtype=np.float32)/255.0,W0,H0
class AarizDS(Dataset):''')
# also record in-domain and cross-domain MRE per seed in a machine-readable file
rep('            log(f"{name} {tag} seed{sd}: in={i_mre:.2f} cross={c_mre:.2f} GD={gd:.0f}%")',
    '            log(f"{name} {tag} seed{sd}: in={i_mre:.2f} cross={c_mre:.2f} GD={gd:.0f}%")\n'
    f'            open("{ROOT}/results/repro/table2.csv","a").write(f"{{name}},{{tag}},{{sd}},{{i_mre:.4f}},{{c_mre:.4f}}\\n")')
i = s.index("# paradigma karsilastirmasi")
s = s[:i] + "R.close()\n"
s = s.replace('log(f"  -> {name} {tag}: GD {np.mean(gds):.0f}% +/- {np.std(gds,ddof=1):.0f}% (n=3 seed)")', 'log(f"  -> {name} {tag}: GD mean {np.mean(gds):.0f}% over {len(gds)} seed(s)")')
os.makedirs(f"{ROOT}/results/repro", exist_ok=True)
exec(compile(s, SRC, "exec"))
