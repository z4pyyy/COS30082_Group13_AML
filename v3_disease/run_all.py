"""Run every v3 experiment in order, then evaluate. Safe to stop and re-run: finished runs are skipped.

    python v3_disease/run_all.py                 # full run (RTX 5090: about 4-5 h for 3 seeds)
    python v3_disease/run_all.py --seeds 0       # one seed only (about 1.5 h)
    python v3_disease/run_all.py --smoke         # 2-minute pipeline check with a tiny model; writes results_smoke/
"""
import argparse
import os
import subprocess
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent

# name -> train.py flags (disease ablation ladder: each row adds one recommendation)
DISEASE = {
    "dis_base": [],
    "dis_attn": ["--attn"],
    "dis_attn_adv": ["--attn", "--adv", "0.3"],
    "dis_attn_adv_syn": ["--attn", "--adv", "0.3", "--synth"],
    "dis_full": ["--attn", "--adv", "0.3", "--synth", "--field"],
}


def run(cmd, env):
    print("\n>>>", " ".join(cmd), flush=True)
    t = time.time()
    subprocess.run([sys.executable, *cmd], cwd=HERE, env=env, check=True)
    print(f"<<< {time.time() - t:.0f}s", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--synth_n", type=int, default=4000)
    ap.add_argument("--smoke", action="store_true")
    args = ap.parse_args()

    import torch
    if not args.smoke and not torch.cuda.is_available():
        sys.exit("CUDA GPU not found. Install the CUDA build of torch (see v3_disease/README.md), or use --smoke.")
    if torch.cuda.is_available():
        print("GPU:", torch.cuda.get_device_name(0), "| torch", torch.__version__, flush=True)
    env = dict(os.environ)
    extra, vl_extra = ["--workers", str(args.workers)], ["--workers", str(args.workers)]
    if args.smoke:
        env |= {"V3_OUT": "results_smoke", "V3_SYNTH": "synthetic_smoke"}
        args.seeds, args.synth_n = [0], 40
        extra += ["--backbone", "vit_small_patch14_dinov2.lvd142m", "--epochs", "1", "--bs", "8", "--max_batches", "3"]
        vl_extra += ["--model", "hf-hub:imageomics/bioclip", "--epochs", "1", "--bs", "8", "--max_batches", "3"]
    out = HERE / env.get("V3_OUT", "results") / "runs"
    done = lambda n: (out / n / "logits.npz").exists()

    if not (HERE / env.get("V3_SYNTH", "synthetic") / "index.csv").exists():
        run(["make_synthetic.py", "--n", str(args.synth_n)], env)
    for sd in args.seeds:
        s = ["--seed", str(sd)]
        if not done(f"crop_s{sd}"):   # crop model: same field-robust training for every disease variant
            run(["train.py", "--task", "crop", "--field", "--name", f"crop_s{sd}", *s, *extra], env)
        for name, flags in DISEASE.items():
            if not done(f"{name}_s{sd}"):
                run(["train.py", "--task", "disease", *flags, "--name", f"{name}_s{sd}", *s, *extra], env)
        if not done(f"vl_s{sd}") or (sd == 0 and not done("vl_zeroshot")):
            run(["vl_finetune.py", "--name", f"vl_s{sd}", *s, *vl_extra], env)
    run(["evaluate.py"], env)
    print("\nAll done. Commit the results:\n  git add v3_disease/results && git commit -m \"v3 results\" && git push")


if __name__ == "__main__":
    main()
