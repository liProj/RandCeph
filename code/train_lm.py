"""Landmark detector used for the pilot baseline and for the proposed method.

Network: ImageNet-pretrained ResNet-34 encoder (first convolution collapsed to one channel) with
a U-Net decoder up to 1/2 of the input resolution; one heatmap per landmark; coordinates by a
soft-argmax; L1 loss on the coordinates (in input pixels).  This is a standard recipe, not a
contribution.

Input "views" (how an original radiograph is mapped to the network input):
  --view squash   the article's input: the whole image squashed to R x R, aspect ratio ignored
  --view phys     the image resampled to a fixed --mmpp millimetres per pixel using its LISTED
                  pixel spacing (kept as a negative control: the listed spacings are not
                  physically consistent across devices, see code/audit.py)
  --view letter   aspect-preserving letterbox: the whole image scaled isotropically to fit H x W
  --view canon    anatomy-anchored canonical frame: isotropic scale and shift chosen so that the
                  centroid of the 19 landmarks is at the canvas centre and their RMS radius is
                  --r0 pixels.  Training uses the reference landmarks with jitter; evaluation
                  uses the landmarks predicted by a first-stage run (--stage1 <tag>), optionally
                  refined --iters times with the model's own output
Evaluation is the article's: MRE in mm over the 19 shared landmarks on Aariz `valid` and ISBI
`test1`; additionally Aariz `test` and ISBI `test2`.  As in the article, the model after the
last epoch is evaluated (no checkpoint selection, no test-time peeking).

  python train_lm.py --tag base_squash --train aariz --view squash --seed 0
"""
import argparse
import json
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import common as C                                    # noqa: E402
import numpy as np                                    # noqa: E402
import torch                                          # noqa: E402
import torch.nn as nn                                 # noqa: E402
import torch.nn.functional as F                       # noqa: E402
import torchvision                                    # noqa: E402

L = 1024                                              # long side of the cached images


def conv(i, o):
    return nn.Sequential(nn.Conv2d(i, o, 3, 1, 1, bias=False), nn.BatchNorm2d(o), nn.ReLU(inplace=True))


class UNet(nn.Module):
    def __init__(self, n_lm=19, pretrained=True):
        super().__init__()
        r = torchvision.models.resnet34(weights=torchvision.models.ResNet34_Weights.DEFAULT if pretrained else None)
        w = r.conv1.weight.data.sum(1, keepdim=True)
        r.conv1 = nn.Conv2d(1, 64, 7, 2, 3, bias=False)
        r.conv1.weight.data = w
        self.s0 = nn.Sequential(r.conv1, r.bn1, r.relu)              # 1/2, 64
        self.s1 = nn.Sequential(r.maxpool, r.layer1)                 # 1/4, 64
        self.s2, self.s3, self.s4 = r.layer2, r.layer3, r.layer4     # 1/8 128, 1/16 256, 1/32 512
        self.d3 = nn.Sequential(conv(512 + 256, 256), conv(256, 256))
        self.d2 = nn.Sequential(conv(256 + 128, 128), conv(128, 128))
        self.d1 = nn.Sequential(conv(128 + 64, 64), conv(64, 64))
        self.d0 = nn.Sequential(conv(64 + 64, 64), conv(64, 64))
        self.head = nn.Conv2d(64, n_lm, 1)
        self.logt = nn.Parameter(torch.zeros(1))

    def forward(self, x):
        e0 = self.s0(x)
        e1 = self.s1(e0)
        e2 = self.s2(e1)
        e3 = self.s3(e2)
        e4 = self.s4(e3)
        up = lambda a, b: F.interpolate(a, size=b.shape[-2:], mode="bilinear", align_corners=False)   # noqa: E731
        d = self.d3(torch.cat([up(e4, e3), e3], 1))
        d = self.d2(torch.cat([up(d, e2), e2], 1))
        d = self.d1(torch.cat([up(d, e1), e1], 1))
        d = self.d0(torch.cat([up(d, e0), e0], 1))
        return self.head(d) * torch.exp(self.logt)                   # heatmaps at 1/2 resolution


def soft_argmax(hm, H, W):
    B, K, h, w = hm.shape
    p = F.softmax(hm.float().view(B, K, -1), -1).view(B, K, h, w)
    xs = (torch.arange(w, device=hm.device) + 0.5) * (W / w)
    ys = (torch.arange(h, device=hm.device) + 0.5) * (H / h)
    return torch.stack([(p.sum(2) * xs).sum(-1), (p.sum(3) * ys).sum(-1)], -1)     # (B, K, 2) in input px


class UNetCX(nn.Module):
    """Same decoder idea on a ConvNeXt-Tiny encoder (ImageNet weights); heatmaps at 1/2 resolution."""

    def __init__(self, n_lm=19, pretrained=True):
        super().__init__()
        m = torchvision.models.convnext_tiny(weights=torchvision.models.ConvNeXt_Tiny_Weights.DEFAULT if pretrained else None).features
        self.pre = nn.Sequential(conv(1, 32), conv(32, 32))                          # full-resolution stem -> 1/2 skip
        self.s1 = nn.Sequential(m[0], m[1])                                          # 1/4, 96
        self.s2 = nn.Sequential(m[2], m[3])                                          # 1/8, 192
        self.s3 = nn.Sequential(m[4], m[5])                                          # 1/16, 384
        self.s4 = nn.Sequential(m[6], m[7])                                          # 1/32, 768
        self.d3 = nn.Sequential(conv(768 + 384, 256), conv(256, 256))
        self.d2 = nn.Sequential(conv(256 + 192, 128), conv(128, 128))
        self.d1 = nn.Sequential(conv(128 + 96, 64), conv(64, 64))
        self.d0 = nn.Sequential(conv(64 + 32, 64), conv(64, 64))
        self.head = nn.Conv2d(64, n_lm, 1)
        self.logt = nn.Parameter(torch.zeros(1))

    def forward(self, x):
        e0 = F.avg_pool2d(self.pre(x), 2)
        e1 = self.s1(x.expand(-1, 3, -1, -1))
        e2 = self.s2(e1)
        e3 = self.s3(e2)
        e4 = self.s4(e3)
        up = lambda a, b: F.interpolate(a, size=b.shape[-2:], mode="bilinear", align_corners=False)   # noqa: E731
        d = self.d3(torch.cat([up(e4, e3), e3], 1))
        d = self.d2(torch.cat([up(d, e2), e2], 1))
        d = self.d1(torch.cat([up(d, e1), e1], 1))
        d = self.d0(torch.cat([up(d, e0), e0], 1))
        return self.head(d) * torch.exp(self.logt)


def build(a):
    net = {"resnet34": UNet, "convnext_tiny": UNetCX}[getattr(a, "enc", "resnet34")]
    return net(pretrained=bool(a.pretrained)).cuda().to(memory_format=torch.channels_last)


class Data:
    """All images of one split on the GPU (uint8, 1024 cache) with their geometry."""

    def __init__(self, name, split):
        z = np.load(f"{C.PREP}/{name}_{split}.npz", allow_pickle=True)
        self.x = torch.from_numpy(z["x1024"]).cuda()
        self.w = torch.tensor(z["w"], dtype=torch.float32, device="cuda")
        self.h = torch.tensor(z["h"], dtype=torch.float32, device="cuda")
        self.lm = torch.tensor(z["lm"], device="cuda")                            # original px
        self.mm = torch.tensor(z["mm"], device="cuda")
        self.device = z["device"]
        self.ids = z["ids"]
        self.name, self.split, self.n = name, split, len(self.w)


def subset(D, idx, name):
    """Rows `idx` of a Data object as a new Data object."""
    out = object.__new__(Data)
    t = torch.as_tensor(np.asarray(idx), device="cuda")
    out.x, out.w, out.h, out.lm, out.mm = D.x[t], D.w[t], D.h[t], D.lm[t], D.mm[t]
    out.device, out.ids = D.device[np.asarray(idx)], D.ids[np.asarray(idx)]
    out.name, out.split, out.n = name, "subset", len(idx)
    return out


def lodo_split(held, seed):
    """The authors' leave-one-device-out split (m6_lodo_multiseed.py): all 1000 Aariz images grouped
    by device in the order of cephalogram_machine_mappings.csv; the held-out device is the test set;
    the others are pooled, permuted with numpy default_rng(seed), the first 15% form the in-domain
    validation set and the rest the training set."""
    import csv
    A = cat([Data("aariz", "train"), Data("aariz", "valid"), Data("aariz", "test")])
    pos = {cid: i for i, cid in enumerate(A.ids)}
    groups = {}
    for r in csv.DictReader(open(f"{C.AARIZ}/cephalogram_machine_mappings.csv")):
        if r["cephalogram_id"] in pos:
            groups.setdefault(r["machine"], []).append(r["cephalogram_id"])
    order = sorted(groups, key=lambda d: -len(groups[d]))
    assert held in groups, (held, list(groups))
    pool = [c for d in order if d != held for c in groups[d]]
    pool = list(np.random.default_rng(seed).permutation(pool))
    n_val = int(len(pool) * 0.15)
    ix = lambda ids: [pos[c] for c in ids]                                          # noqa: E731
    return subset(A, ix(pool[n_val:]), "lodo_train"), subset(A, ix(pool[:n_val]), "lodo_in"), subset(A, ix(groups[held]), "lodo_out")


def cat(ds):
    out = object.__new__(Data)
    out.x = torch.cat([d.x for d in ds])
    out.w, out.h, out.lm, out.mm = (torch.cat([getattr(d, k) for d in ds]) for k in ("w", "h", "lm", "mm"))
    out.device = np.concatenate([d.device for d in ds])
    out.ids = np.concatenate([d.ids for d in ds])
    out.name, out.split, out.n = "+".join(d.name for d in ds), "train", len(out.w)
    out.src = torch.cat([torch.full((d.n,), i, device="cuda") for i, d in enumerate(ds)])
    return out


def view_matrix(D, idx, a, aug, anchor=None):
    """Affine map from network-input pixel coordinates (x, y, 1) to ORIGINAL-image pixel
    coordinates, per image: (B, 2, 3).  Augmentation = random similarity transform about the
    canvas centre (scale, rotation, shift)."""
    B = len(idx)
    w, h = D.w[idx], D.h[idx]
    H, W = a.H, a.W
    if a.view == "squash":
        sx, sy = w / W, h / H                         # original px per input px
        tx = torch.zeros(B, device="cuda")
        ty = torch.zeros(B, device="cuda")
    elif a.view == "letter":
        s = torch.maximum(w / W, h / H)               # original px per input px, whole image visible
        sx = sy = s
        tx = (w - W * s) / 2
        ty = (h - H * s) / 2
    elif a.view == "canon":
        ref = D.lm[idx] if anchor is None else anchor                 # (B, 19, 2) original px
        c = ref.mean(1)
        r = torch.sqrt(((ref - c[:, None]) ** 2).sum(-1).mean(1))     # RMS radius in original px
        s = r / a.r0
        sx = sy = s
        tx = c[:, 0] - s * W / 2
        ty = c[:, 1] - s * H / 2
    else:
        s = a.mmpp / D.mm[idx]                        # original px per input px (isotropic)
        sx = sy = s
        tx = (w - W * s) / 2                          # centre the image on the canvas
        ty = (h - H * s) / 2
    M = torch.zeros(B, 2, 3, device="cuda")
    M[:, 0, 0], M[:, 1, 1], M[:, 0, 2], M[:, 1, 2] = sx, sy, tx, ty
    if aug:
        sc = torch.exp((torch.rand(B, device="cuda") * 2 - 1) * np.log(1 + a.aug_scale))
        ang = (torch.rand(B, device="cuda") * 2 - 1) * a.aug_rot * np.pi / 180
        dx = (torch.rand(B, device="cuda") * 2 - 1) * a.aug_shift * W
        dy = (torch.rand(B, device="cuda") * 2 - 1) * a.aug_shift * H
        cx, cy = W / 2, H / 2
        cos, sin = torch.cos(ang) * sc, torch.sin(ang) * sc
        A = torch.zeros(B, 3, 3, device="cuda")       # input px -> input px (about the centre)
        A[:, 0, 0], A[:, 0, 1], A[:, 0, 2] = cos, -sin, cx - cos * cx + sin * cy + dx
        A[:, 1, 0], A[:, 1, 1], A[:, 1, 2] = sin, cos, cy - sin * cx - cos * cy + dy
        A[:, 2, 2] = 1
        M = torch.bmm(M, A)
    return M


def render(D, idx, M, a):
    """Sample the cached images through M.  Returns (B, 1, H, W) float in [0, 1]."""
    B = len(idx)
    H, W = a.H, a.W
    ys, xs = torch.meshgrid(torch.arange(H, device="cuda") + 0.5, torch.arange(W, device="cuda") + 0.5, indexing="ij")
    P = torch.stack([xs, ys, torch.ones_like(xs)], -1).view(1, H * W, 3).expand(B, -1, -1)
    o = torch.bmm(P, M.transpose(1, 2))               # original px
    cs = (L / torch.maximum(D.w[idx], D.h[idx])).view(B, 1, 1)                    # original px -> cache px
    g = o * cs / L * 2 - 1                            # cache is L x L, image top-left anchored
    x = D.x[idx].float().unsqueeze(1) / 255.0
    return F.grid_sample(x, g.view(B, H, W, 2), mode="bilinear", padding_mode="zeros", align_corners=False)


def to_input(lm_orig, M):
    """Original px -> input px (inverse of M)."""
    A = M[:, :, :2]
    t = M[:, :, 2]
    return torch.einsum("bij,bkj->bki", torch.linalg.inv(A), lm_orig - t[:, None, :])


def to_orig(p_in, M):
    return torch.einsum("bij,bkj->bki", M[:, :, :2], p_in) + M[:, None, :, 2]


def photometric(x, a):
    if not a.aug_photo:
        return x
    B = x.shape[0]
    g = torch.exp((torch.rand(B, 1, 1, 1, device="cuda") * 2 - 1) * np.log(1.5))       # gamma
    c = 1 + (torch.rand(B, 1, 1, 1, device="cuda") * 2 - 1) * 0.25
    b = (torch.rand(B, 1, 1, 1, device="cuda") * 2 - 1) * 0.1
    x = x.clamp(1e-4, 1) ** g
    inv = torch.rand(B, 1, 1, 1, device="cuda") < a.aug_invert
    x = torch.where(inv, 1 - x, x)
    return ((x - 0.5) * c + 0.5 + b).clamp(0, 1)


@torch.no_grad()
def evaluate(model, D, a, bs=8, anchor=None):
    """anchor: (N, 19, 2) first-stage predictions in original px (canonical view only)."""
    model.eval()
    P = []
    for i in range(0, D.n, bs):
        idx = torch.arange(i, min(i + bs, D.n), device="cuda")
        anc = None if anchor is None else anchor[i:i + bs]
        for it in range(max(1, a.iters if a.view == "canon" else 1)):
            M = view_matrix(D, idx, a, False, anchor=anc)
            x = (render(D, idx, M, a) - 0.5) / 0.25
            with torch.autocast("cuda", dtype=torch.bfloat16, enabled=bool(a.amp)):
                hm = model(x)
            anc = to_orig(soft_argmax(hm, a.H, a.W), M)
        P.append(anc)
    P = torch.cat(P)
    mre, d = C.mre_mm(P.cpu().numpy(), D.lm.cpu().numpy(), D.mm.cpu().numpy())
    return mre, d, P.cpu().numpy()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", required=True)
    ap.add_argument("--train", default="aariz")       # aariz | isbi | joint | lodo:<device name>
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--view", default="squash", choices=["squash", "phys", "letter", "canon"])
    ap.add_argument("--r0", type=float, default=120.0)
    ap.add_argument("--stage1", default="")
    ap.add_argument("--iters", type=int, default=1)
    ap.add_argument("--H", type=int, default=512)
    ap.add_argument("--W", type=int, default=512)
    ap.add_argument("--mmpp", type=float, default=0.5)
    ap.add_argument("--epochs", type=int, default=60)
    ap.add_argument("--bs", type=int, default=6)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--amp", type=int, default=1)
    ap.add_argument("--pretrained", type=int, default=1)
    ap.add_argument("--aug_scale", type=float, default=0.0)
    ap.add_argument("--aug_rot", type=float, default=0.0)
    ap.add_argument("--aug_shift", type=float, default=0.0)
    ap.add_argument("--aug_photo", type=int, default=0)
    ap.add_argument("--aug_invert", type=float, default=0.0)
    ap.add_argument("--enc", default="resnet34", choices=["resnet34", "convnext_tiny"])
    # mean-teacher consistency on UNLABELLED images of another dataset (its training split; no landmarks are read)
    ap.add_argument("--unlab", default="", choices=["", "aariz", "isbi"])
    ap.add_argument("--u_start", type=int, default=10)
    ap.add_argument("--u_full", type=int, default=30)
    ap.add_argument("--u_weight", type=float, default=1.0)
    ap.add_argument("--ema", type=float, default=0.99)
    ap.add_argument("--steps_per_epoch", type=int, default=0)        # 0 = one pass over the training set
    a = ap.parse_args()
    out = f"{C.RUNS}/{a.tag}/{a.train.replace(':', '_').replace(' ', '_')}_s{a.seed}"
    if os.path.exists(f"{out}/metrics.json"):
        return print("done already")
    os.makedirs(out, exist_ok=True)
    torch.manual_seed(a.seed)
    np.random.seed(a.seed)
    if a.train.startswith("lodo:"):
        tr, d_in, d_out = lodo_split(a.train[5:], a.seed)
        ev = {"lodo_in": d_in, "lodo_out": d_out, "isbi_test1": Data("isbi", "test1")}
    else:
        isbi_tr = Data("isbi", "train")
        tr = {"aariz": lambda: Data("aariz", "train"), "isbi": lambda: isbi_tr,
              # the authors' balanced joint set: Aariz train (700) + ISBI train repeated four times (600)
              "joint": lambda: cat([Data("aariz", "train")] + [isbi_tr] * 4)}[a.train]()
        ev = {"aariz_valid": Data("aariz", "valid"), "isbi_test1": Data("isbi", "test1"), "aariz_test": Data("aariz", "test"), "isbi_test2": Data("isbi", "test2")}
    model = build(a)
    U = Data(a.unlab, "train") if a.unlab else None
    if U is not None:
        import copy
        teacher = copy.deepcopy(model)
        for q in teacher.parameters():
            q.requires_grad_(False)
    opt = torch.optim.AdamW(model.parameters(), lr=a.lr, weight_decay=1e-4)
    nb = a.steps_per_epoch or int(np.ceil(tr.n / a.bs))
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=a.lr, total_steps=a.epochs * nb, pct_start=0.1)
    anchors = {}
    if a.view == "canon":                               # first-stage predictions of the SAME training set and seed
        z1 = np.load(f"{C.RUNS}/{a.stage1}/{a.train}_s{a.seed}/preds.npz")
        anchors = {k: torch.tensor(z1[f"pred_{k}"], device="cuda") for k in ev}
    t0 = time.time()
    for ep in range(a.epochs):
        model.train()
        perm = torch.randperm(tr.n, device="cuda")
        tot = 0.0
        for b in range(nb):
            idx = perm[(b * a.bs) % tr.n:(b * a.bs) % tr.n + a.bs]
            M = view_matrix(tr, idx, a, True)
            x = ((photometric(render(tr, idx, M, a), a) - 0.5) / 0.25).contiguous(memory_format=torch.channels_last)
            y = to_input(tr.lm[idx], M)
            with torch.autocast("cuda", dtype=torch.bfloat16, enabled=bool(a.amp)):
                hm = model(x)
            loss = F.l1_loss(soft_argmax(hm, a.H, a.W), y)
            if U is not None and ep >= a.u_start:
                uw = a.u_weight * min(1.0, (ep - a.u_start + 1) / max(1, a.u_full - a.u_start))
                ui = torch.randint(0, U.n, (a.bs,), device="cuda")
                with torch.no_grad():                    # teacher: plain view, no augmentation
                    teacher.eval()
                    Mt = view_matrix(U, ui, a, False)
                    with torch.autocast("cuda", dtype=torch.bfloat16, enabled=bool(a.amp)):
                        ht = teacher(((render(U, ui, Mt, a) - 0.5) / 0.25).contiguous(memory_format=torch.channels_last))
                    pt = to_orig(soft_argmax(ht, a.H, a.W), Mt)
                Ms = view_matrix(U, ui, a, True)         # student: augmented view of the same images
                xs = ((photometric(render(U, ui, Ms, a), a) - 0.5) / 0.25).contiguous(memory_format=torch.channels_last)
                with torch.autocast("cuda", dtype=torch.bfloat16, enabled=bool(a.amp)):
                    hs = model(xs)
                loss = loss + uw * F.l1_loss(soft_argmax(hs, a.H, a.W), to_input(pt, Ms))
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            sched.step()
            if U is not None:
                with torch.no_grad():
                    for qt, qs in zip(teacher.parameters(), model.parameters()):
                        qt.lerp_(qs, 1 - a.ema)
                    for bt, bs_ in zip(teacher.buffers(), model.buffers()):
                        bt.copy_(bs_)
            tot += float(loss.detach())
        if (ep + 1) % 10 == 0 or ep == a.epochs - 1:
            msg = " ".join(f"{k} {evaluate(model, D, a, anchor=anchors.get(k))[0]:.2f}" for k, D in list(ev.items())[:2])
            print(f"ep {ep + 1:3d} loss {tot / nb:.2f} px | MRE mm: {msg} | {time.time() - t0:.0f}s", flush=True)
    res = dict(tag=a.tag, train=a.train, seed=a.seed, args=vars(a), train_sec=time.time() - t0, params=sum(p.numel() for p in model.parameters()))
    save = {}
    for k, D in ev.items():
        mre, d, P = evaluate(model, D, a, anchor=anchors.get(k))
        res[f"mre_{k}"] = mre
        res[f"mre_img_{k}"] = float(np.nanmean(np.nanmean(d, 1)))           # per-image mean, then mean (the authors' LODO pooling)
        for t in (2.0, 2.5, 3.0, 4.0):
            res[f"sdr{t:g}_{k}"] = C.sdr(d, t)
        save[f"ids_{k}"] = D.ids.astype(str)
        save[f"err_{k}"], save[f"pred_{k}"] = d.astype(np.float32), P.astype(np.float32)
    np.savez_compressed(f"{out}/preds.npz", **save)
    json.dump(res, open(f"{out}/metrics.json", "w"), indent=1)
    torch.save(model.state_dict(), f"{out}/model.pt")
    print({k: round(v, 3) for k, v in res.items() if k.startswith("mre_") or k.startswith("sdr2_")})


if __name__ == "__main__":
    main()
