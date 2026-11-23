"""Fine-tune a DINOv2 ViT for one task (crop or disease) with the v3 options. Saves log-probs for val/test splits.

Recommendations covered here:
  1  --attn       disease head = gated attention pooling over patch tokens (weakly supervised lesion finder)
  2  --adv W      gradient-reversal crop adversary on the disease feature (crop-blind disease encoder)
  4  --synth      add the cut-paste synthetic combinations (make_synthetic.py) to disease training
  6  --field      field-style augmentation + field oversampling (+ lab/field domain adversary for disease)

Model choice (epoch) uses val + val_unseen only: H of task accuracy on the two splits.
Test splits are scored once, with the chosen weights.

Example: python v3_disease/train.py --task disease --attn --adv 0.3 --synth --field --seed 0 --name dis_full_s0
"""
import argparse
import copy
import math
import os
import time

import pandas as pd
import timm
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler
from torchvision.transforms import v2

from common import DATA, EV, FIELD_SOURCES, H, HERE, RESULTS, SYNTH, Labels, device, save_logits, write_json

MEAN, STD = (0.485, 0.456, 0.406), (0.229, 0.224, 0.225)


def tf_eval(mean=MEAN, std=STD):
    return v2.Compose([v2.ToImage(), v2.Resize(224), v2.ToDtype(torch.float32, scale=True), v2.Normalize(mean, std)])


def tf_train(field: bool, mean=MEAN, std=STD):
    if not field:   # same strength as the team's dataset.build_transforms(True)
        return v2.Compose([
            v2.ToImage(), v2.RandomResizedCrop(224, scale=(0.6, 1.0), ratio=(0.8, 1.25)),
            v2.RandomHorizontalFlip(), v2.RandomVerticalFlip(), v2.RandomRotation(20),
            v2.ColorJitter(0.2, 0.2, 0.2, 0.02), v2.ToDtype(torch.float32, scale=True), v2.Normalize(mean, std),
            v2.RandomErasing(p=0.25, scale=(0.02, 0.15))])
    # field-style: harsher crops, light, blur, perspective, JPEG artefacts
    return v2.Compose([
        v2.ToImage(), v2.RandomResizedCrop(224, scale=(0.3, 1.0), ratio=(0.75, 1.33)),
        v2.RandomHorizontalFlip(), v2.RandomVerticalFlip(), v2.RandomRotation(30),
        v2.RandomApply([v2.ColorJitter(0.45, 0.45, 0.4, 0.04)], p=0.9),
        v2.RandomApply([v2.GaussianBlur(7, sigma=(0.3, 2.5))], p=0.3),
        v2.RandomApply([v2.RandomPerspective(0.25, p=1.0)], p=0.3),
        v2.RandomApply([v2.JPEG((25, 90))], p=0.5),
        v2.ToDtype(torch.float32, scale=True), v2.Normalize(mean, std),
        v2.RandomErasing(p=0.25, scale=(0.02, 0.15))])


class Frame(Dataset):
    """Rows with an absolute image path in column 'abs'. Returns (x, crop, disease, adv_mask, field)."""

    def __init__(self, df: pd.DataFrame, tf):
        self.p = df["abs"].tolist()
        self.c, self.d = torch.tensor(df.crop_id.values), torch.tensor(df.disease_id.values)
        self.m = torch.tensor(df.adv_mask.values, dtype=torch.bool)
        self.f = torch.tensor(df.source.isin(FIELD_SOURCES).values, dtype=torch.long)
        self.tf = tf

    def __len__(self):
        return len(self.p)

    def __getitem__(self, i):
        with Image.open(self.p[i]) as im:
            x = self.tf(im.convert("RGB"))
        return x, self.c[i], self.d[i], self.m[i], self.f[i]


class GradRev(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, lam):
        ctx.lam = lam
        return x.view_as(x)

    @staticmethod
    def backward(ctx, g):
        return -ctx.lam * g, None


class AttnPool(nn.Module):
    """Gated attention MIL pooling (Ilse et al. 2018) over patch tokens."""

    def __init__(self, d, h=256):
        super().__init__()
        self.v, self.u, self.w = nn.Linear(d, h), nn.Linear(d, h), nn.Linear(h, 1)

    def forward(self, t):
        a = self.w(torch.tanh(self.v(t)) * torch.sigmoid(self.u(t))).squeeze(-1).softmax(1)   # (B, N)
        return (a.unsqueeze(-1) * t).sum(1), a


class Net(nn.Module):
    def __init__(self, backbone, n_out, n_crop, attn, adv, domain):
        super().__init__()
        self.bb = timm.create_model(backbone, pretrained=True, num_classes=0, img_size=224, drop_path_rate=0.1)
        d = self.bb.num_features
        self.pool = AttnPool(d) if attn else None
        self.head = nn.Linear(d, n_out)
        self.adv = nn.Sequential(nn.Linear(d, 512), nn.GELU(), nn.Linear(512, n_crop)) if adv else None
        self.dom = nn.Sequential(nn.Linear(d, 256), nn.GELU(), nn.Linear(256, 2)) if domain else None

    def forward(self, x):
        t = self.bb.forward_features(x)
        if self.pool is None:
            z, a = t[:, 0], None
        else:
            z, a = self.pool(t[:, self.bb.num_prefix_tokens:])
        return self.head(z), z, a


def build_train_df(L: Labels, task: str, synth: bool) -> pd.DataFrame:
    tr = L.dfs["train"].copy()
    tr["abs"] = [str(DATA / p) for p in tr.path]
    n_crops = tr.groupby("disease_id").crop_id.nunique()
    tr["adv_mask"] = tr.disease_id.map(n_crops).ge(2)     # adversary only where crop varies within a disease
    if synth and task == "disease":
        sy = pd.read_csv(SYNTH / "index.csv")
        sy["abs"] = [str(HERE / p) for p in sy.path]
        sy["adv_mask"] = True
        tr = pd.concat([tr, sy], ignore_index=True)
    return tr


_LOADERS = {}


def eval_loader(split, df, tf, workers):
    """One eval loader per split, kept for the whole run. On Windows each worker process re-imports torch
    (about a minute per loader), which costs more than decoding the eval images in the main process."""
    if os.name == "nt":
        workers = 0
    if split not in _LOADERS:
        _LOADERS[split] = DataLoader(Frame(df, tf), batch_size=128, num_workers=workers,
                                     persistent_workers=workers > 0, pin_memory=torch.cuda.is_available())
    return _LOADERS[split]


@torch.no_grad()
def predict(model, L, split, task, dev, bs, workers, amp):
    df = L.dfs[split].copy()
    df["abs"], df["adv_mask"] = [str(DATA / p) for p in df.path], False
    dl = eval_loader(split, df, tf_eval(), workers)
    model.eval()
    out = []
    for x, *_ in dl:
        with torch.autocast("cuda", dtype=torch.bfloat16, enabled=amp):
            out.append(F.log_softmax(model(x.to(dev, non_blocking=True))[0].float(), 1).cpu())
    y = torch.tensor(df[f"{task}_id"].values)
    lp = torch.cat(out)
    return lp, 100 * (lp.argmax(1) == y).float().mean().item()


def attention_figure(model, L, path, dev, amp):
    """Attention overlays on 8 test_seen + 8 test_unseen images (where the disease head looks)."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    rows = pd.concat([L.dfs[s].sample(8, random_state=0) for s in ("test_seen", "test_unseen")])
    fig, ax = plt.subplots(2, 8, figsize=(20, 5.6))
    model.eval()
    for a, (_, r) in zip(ax.flat, rows.iterrows()):
        im = Image.open(DATA / r.path).convert("RGB").resize((224, 224))
        with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16, enabled=amp):
            _, _, att = model(tf_eval()(im).unsqueeze(0).to(dev))
        g = int(math.isqrt(att.shape[1]))
        m = att[0].float().view(g, g).cpu()
        m = F.interpolate(m[None, None], size=(224, 224), mode="bilinear")[0, 0]
        a.imshow(im); a.imshow(m / m.max(), cmap="jet", alpha=0.45); a.axis("off")
        a.set_title(f"{r.crop} / {r.disease}"[:34], fontsize=7)
    fig.suptitle("Disease-head attention. Top: test_seen. Bottom: test_unseen.")
    fig.tight_layout(); fig.savefig(path, dpi=100); plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--task", choices=["crop", "disease"], required=True)
    ap.add_argument("--name", required=True)
    ap.add_argument("--backbone", default="vit_large_patch14_dinov2.lvd142m")
    ap.add_argument("--epochs", type=int, default=15)
    ap.add_argument("--bs", type=int, default=48)
    ap.add_argument("--lr_bb", type=float, default=1e-5)
    ap.add_argument("--lr_head", type=float, default=1e-3)
    ap.add_argument("--attn", action="store_true")
    ap.add_argument("--adv", type=float, default=0.0, help="crop-adversary weight (disease task)")
    ap.add_argument("--synth", action="store_true")
    ap.add_argument("--field", action="store_true")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--max_batches", type=int, default=0, help="smoke test: stop each epoch after N batches")
    ap.add_argument("--save_weights", action="store_true")
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    dev = device()
    amp = dev == "cuda"
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    out = RESULTS / "runs" / args.name
    out.mkdir(parents=True, exist_ok=True)

    L = Labels()
    n_out = L.n_crop if args.task == "crop" else L.n_dis
    use_adv = args.task == "disease" and args.adv > 0
    use_dom = args.task == "disease" and args.field
    model = Net(args.backbone, n_out, L.n_crop, args.attn, use_adv, use_dom).to(dev)

    df = build_train_df(L, args.task, args.synth)
    sampler = None
    if args.field:   # oversample field photos to ~25% of each epoch
        fld = df.source.isin(FIELD_SOURCES).values
        w = torch.where(torch.tensor(fld), 0.25 / max(fld.mean(), 1e-6), 0.75 / max(1 - fld.mean(), 1e-6)).double()
        sampler = WeightedRandomSampler(w, num_samples=len(df), replacement=True,
                                        generator=torch.Generator().manual_seed(args.seed))
    dl = DataLoader(Frame(df, tf_train(args.field)), batch_size=args.bs, shuffle=sampler is None, sampler=sampler,
                    num_workers=args.workers, pin_memory=dev == "cuda", drop_last=True,
                    persistent_workers=args.workers > 0)

    head_params = [p for n, p in model.named_parameters() if not n.startswith("bb.")]
    opt = torch.optim.AdamW([{"params": model.bb.parameters(), "lr": args.lr_bb},
                             {"params": head_params, "lr": args.lr_head}], weight_decay=0.05)
    steps = args.epochs * (args.max_batches or len(dl))
    warm = max(1, steps // 15)
    sched = torch.optim.lr_scheduler.LambdaLR(
        opt, lambda s: min(1.0, (s + 1) / warm) * 0.5 * (1 + math.cos(math.pi * min(1.0, s / steps))))

    best, log, step = (-1.0, None, -1), [], 0
    for ep in range(args.epochs):
        model.train()
        t0, tot, n = time.time(), 0.0, 0
        for i, (x, c, d, m, f) in enumerate(dl):
            if args.max_batches and i >= args.max_batches:
                break
            x, c, d, m, f = (t.to(dev, non_blocking=True) for t in (x, c, d, m, f))
            y = c if args.task == "crop" else d
            lam = 2 / (1 + math.exp(-10 * step / steps)) - 1          # DANN ramp 0 -> 1
            with torch.autocast("cuda", dtype=torch.bfloat16, enabled=amp):
                logits, z, _ = model(x)
                loss = F.cross_entropy(logits.float(), y, label_smoothing=0.1)
                if use_adv and m.any():
                    loss = loss + args.adv * F.cross_entropy(model.adv(GradRev.apply(z[m], lam)).float(), c[m])
                if use_dom:
                    loss = loss + 0.1 * F.cross_entropy(model.dom(GradRev.apply(z, lam)).float(), f)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step(); sched.step(); step += 1
            tot, n = tot + loss.item(), n + 1
        _, va = predict(model, L, "val", args.task, dev, 128, args.workers, amp)
        _, vu = predict(model, L, "val_unseen", args.task, dev, 128, args.workers, amp)
        h = H(va, vu)
        log.append({"epoch": ep + 1, "loss": tot / max(n, 1), "val acc": va, "val_unseen acc": vu, "val H": h,
                    "sec": time.time() - t0})
        print(f"[{args.name}] ep {ep + 1}/{args.epochs} loss {tot / max(n, 1):.3f} val {va:.2f} "
              f"val_unseen {vu:.2f} H {h:.2f} ({time.time() - t0:.0f}s)", flush=True)
        if h > best[0]:
            best = (h, copy.deepcopy({k: v.detach().cpu() for k, v in model.state_dict().items()}), ep + 1)

    model.load_state_dict(best[1])
    lps, accs = {}, {}
    for s in EV:
        lps[s], accs[s] = predict(model, L, s, args.task, dev, 128, args.workers, amp)
    fld = {s: L.field(s) for s in ("test_seen", "test_unseen")}
    y = {s: torch.tensor(L.dfs[s][f"{args.task}_id"].values) for s in fld}
    ok = torch.cat([lps[s].argmax(1) == y[s] for s in fld]); fm = torch.cat(list(fld.values()))
    save_logits(out / "logits.npz", lps)
    pd.DataFrame(log).round(3).to_csv(out / "train_log.csv", index=False)
    write_json(out / "meta.json", vars(args) | {
        "best epoch": best[2], "val H": best[0], **{f"{s} acc": accs[s] for s in EV},
        "test field acc": 100 * ok[fm].float().mean().item(), "test lab acc": 100 * ok[~fm].float().mean().item()})
    if args.attn:
        attention_figure(model, L, out / "attention.png", dev, amp)
    if args.save_weights:
        (HERE / "weights").mkdir(exist_ok=True)
        torch.save(best[1], HERE / "weights" / f"{args.name}.pt")
    print(f"[{args.name}] done. best epoch {best[2]}, val H {best[0]:.2f}, "
          + ", ".join(f"{s} {accs[s]:.2f}" for s in EV), flush=True)


if __name__ == "__main__":
    main()
