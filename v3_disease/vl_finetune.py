"""Recommendation 3: symptom-text disease classifier with a vision-language model (BioCLIP-2).

Each disease is a frozen text embedding of crop-agnostic prompts plus a symptom description
(symptoms.json; disease name only when no description exists). No crop word appears in any prompt,
so the text side cannot carry crop identity. The image encoder's last K blocks are fine-tuned so image
embeddings match the right disease text (cross-entropy over the 175 disease texts).
Epoch 0 (no training) is also saved as the zero-shot reference run.

Run: python v3_disease/vl_finetune.py --seed 0 --name vl_s0
"""
import argparse
import copy
import json
import math
import time

import pandas as pd
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from common import DATA, EV, H, HERE, RESULTS, Labels, device, save_logits, write_json
from train import Frame, build_train_df, tf_eval, tf_train

CLIP_MEAN, CLIP_STD = (0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711)
FIX = {"segatoka": "sigatoka", "xamthomonas": "xanthomonas", "crinckle": "crinkle", "monillia laxa": "monilinia laxa",
       "aphis spp": "aphids", "eriosoma lanigerum": "woolly apple aphid", "verticulium wilt": "verticillium wilt"}
TEMPLATES = ["a photo of a plant leaf with {d}.", "a close-up photo of {d} on a leaf.", "a leaf infected with {d}."]


def prompts(name: str, desc: str | None):
    d = FIX.get(name.replace("_", " "), name.replace("_", " "))
    if name == "healthy":
        p = ["a photo of a healthy plant leaf.", "a healthy green leaf."]
    else:
        p = [t.format(d=d) for t in TEMPLATES]
    if desc:
        p.append(f"a plant leaf with {d}: {desc}.")
    return p


@torch.no_grad()
def text_bank(model, tok, L, dev):
    sym = json.loads((HERE / "symptoms.json").read_text(encoding="utf-8"))
    out = []
    for i in range(L.n_dis):
        n = L.maps["disease"][i]
        e = F.normalize(model.encode_text(tok(prompts(n, sym.get(n))).to(dev)).float(), dim=-1).mean(0)
        out.append(F.normalize(e, dim=0))
    return torch.stack(out)   # (175, D)


@torch.no_grad()
def predict(model, T, scale, L, split, dev, workers, amp):
    df = L.dfs[split].copy()
    df["abs"], df["adv_mask"] = [str(DATA / p) for p in df.path], False
    dl = DataLoader(Frame(df, tf_eval(CLIP_MEAN, CLIP_STD)), batch_size=128, num_workers=workers)
    model.eval()
    out = []
    for x, *_ in dl:
        with torch.autocast("cuda", dtype=torch.bfloat16, enabled=amp):
            z = F.normalize(model.encode_image(x.to(dev)).float(), dim=-1)
        out.append(F.log_softmax(scale * z @ T.T, 1).cpu())
    lp = torch.cat(out)
    return lp, 100 * (lp.argmax(1) == torch.tensor(df.disease_id.values)).float().mean().item()


def finish(name, lps, accs, meta):
    out = RESULTS / "runs" / name
    save_logits(out / "logits.npz", lps)
    write_json(out / "meta.json", meta | {f"{s} acc": accs[s] for s in EV})


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--name", required=True)
    ap.add_argument("--model", default="hf-hub:imageomics/bioclip-2")
    ap.add_argument("--epochs", type=int, default=10)
    ap.add_argument("--bs", type=int, default=48)
    ap.add_argument("--lr", type=float, default=1e-5)
    ap.add_argument("--blocks", type=int, default=6, help="fine-tune the last K vision blocks")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--max_batches", type=int, default=0)
    args = ap.parse_args()
    import open_clip

    torch.manual_seed(args.seed)
    dev = device(); amp = dev == "cuda"
    model, _, _ = open_clip.create_model_and_transforms(args.model)
    tok = open_clip.get_tokenizer(args.model)
    model.to(dev)
    L = Labels()
    T = text_bank(model, tok, L, dev)
    scale = model.logit_scale.exp().item()
    meta = vars(args) | {"task": "disease"}

    if args.seed == 0:   # zero-shot reference, once
        r = {s: predict(model, T, scale, L, s, dev, args.workers, amp) for s in EV}
        finish("vl_zeroshot", {s: r[s][0] for s in EV}, {s: r[s][1] for s in EV}, meta | {"epochs": 0})

    for p in model.parameters():
        p.requires_grad = False
    vis = model.visual
    train_mods = list(vis.transformer.resblocks[-args.blocks:]) + [vis.ln_post]
    params = [p for m in train_mods for p in m.parameters()] + ([vis.proj] if vis.proj is not None else [])
    for p in params:
        p.requires_grad = True

    df = build_train_df(L, "disease", synth=False)
    dl = DataLoader(Frame(df, tf_train(False, CLIP_MEAN, CLIP_STD)), batch_size=args.bs, shuffle=True,
                    num_workers=args.workers, pin_memory=amp, drop_last=True, persistent_workers=args.workers > 0)
    opt = torch.optim.AdamW(params, lr=args.lr, weight_decay=0.05)
    steps = args.epochs * (args.max_batches or len(dl))
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda s: 0.5 * (1 + math.cos(math.pi * min(1.0, s / steps))))
    Td = T.to(dev)
    best, log = (-1.0, None, 0), []
    for ep in range(args.epochs):
        model.train(); t0, tot, n = time.time(), 0.0, 0
        for i, (x, c, d, m, f) in enumerate(dl):
            if args.max_batches and i >= args.max_batches:
                break
            with torch.autocast("cuda", dtype=torch.bfloat16, enabled=amp):
                z = F.normalize(model.encode_image(x.to(dev)).float(), dim=-1)
                loss = F.cross_entropy(scale * z @ Td.T, d.to(dev), label_smoothing=0.1)
            opt.zero_grad(set_to_none=True); loss.backward(); opt.step(); sched.step()
            tot, n = tot + loss.item(), n + 1
        _, va = predict(model, T, scale, L, "val", dev, args.workers, amp)
        _, vu = predict(model, T, scale, L, "val_unseen", dev, args.workers, amp)
        h = H(va, vu)
        log.append({"epoch": ep + 1, "loss": tot / max(n, 1), "val acc": va, "val_unseen acc": vu, "val H": h,
                    "sec": time.time() - t0})
        print(f"[{args.name}] ep {ep + 1}/{args.epochs} loss {tot / max(n, 1):.3f} val {va:.2f} "
              f"val_unseen {vu:.2f} H {h:.2f} ({time.time() - t0:.0f}s)", flush=True)
        if h > best[0]:
            best = (h, copy.deepcopy({k: v.detach().cpu() for k, v in model.visual.state_dict().items()}), ep + 1)
    model.visual.load_state_dict(best[1])
    r = {s: predict(model, T, scale, L, s, dev, args.workers, amp) for s in EV}
    finish(args.name, {s: r[s][0] for s in EV}, {s: r[s][1] for s in EV}, meta | {"best epoch": best[2], "val H": best[0]})
    pd.DataFrame(log).round(3).to_csv(RESULTS / "runs" / args.name / "train_log.csv", index=False)
    print(f"[{args.name}] done. best epoch {best[2]}, " + ", ".join(f"{s} {r[s][1]:.2f}" for s in EV), flush=True)


if __name__ == "__main__":
    main()
