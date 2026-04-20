#!/usr/bin/env python3
"""
run_external_benchmark.py — Queue and run all backbone+DDPM benchmarks
on external resting-state datasets.

Runs: 10 backbones × N datasets × 2 modes = up to 60 runs
Each run saves to: external_results/{dataset}_results.json

Usage:
  python run_external_benchmark.py --datasets ds003768 ds005795 noddi
  python run_external_benchmark.py --datasets all
  python run_external_benchmark.py --mode intra --datasets ds003768
  python run_external_benchmark.py --resume --datasets ds005795 noddi
"""

import os, sys, json, subprocess, argparse
from pathlib import Path

BASE = Path("C:/Users/PC/Auto-claude-code-research-in-sleep/neurobolt_fm")
OUT_DIR = BASE / "external_results"
OUT_DIR.mkdir(exist_ok=True)

PYTHON = sys.executable

BACKBONES = [
    "sparc", "contrawr", "ffcl", "cnn_trans", "stt_trans",
    "biot", "labram", "beira", "li2024", "neurobolt",
]

DATASETS = {
    "ds003768": {"tr": 2.1},
    "ds005795": {"tr": 2.0},
    "noddi":    {"tr": 2.16},
    "ds006040": {"tr": 2.0},
}

NEUROBOLT_CKPT = str(BASE / "code" / "checkpoints" / "Cuneus.pth")

EPOCHS_BB   = 100
EPOCHS_DDPM = 300


def result_key_exists(ds_key: str, backbone: str, mode: str) -> bool:
    out_path = OUT_DIR / f"{ds_key}_results.json"
    if not out_path.exists():
        return False
    with open(out_path) as f:
        data = json.load(f)
    return f"{backbone}_{mode}" in data


def run_one(ds_key: str, backbone: str, mode: str) -> bool:
    """Run one backbone × dataset × mode. Returns True on success."""
    data_root = str(BASE / "external_processed" / ds_key)
    if not os.path.isdir(data_root):
        print(f"  SKIP: {data_root} not found")
        return False

    out_path = str(OUT_DIR / f"{ds_key}_results.json")
    cmd = [
        PYTHON, str(BASE / "train_comparison_ddpm_external.py"),
        "--backbone",     backbone,
        "--mode",         mode,
        "--dataset",      ds_key,
        "--data_root",    data_root,
        "--epochs_bb",    str(EPOCHS_BB),
        "--epochs_ddpm",  str(EPOCHS_DDPM),
        "--out",          out_path,
    ]
    if backbone == "neurobolt" and os.path.exists(NEUROBOLT_CKPT):
        cmd += ["--neurobolt_ckpt", NEUROBOLT_CKPT]

    env = os.environ.copy()
    env["PYTHONIOENCODING"] = "utf-8"

    print(f"\n{'='*60}")
    print(f"  {backbone:12s} | {mode:6s} | {ds_key}")
    print(f"{'='*60}")
    print(f"  CMD: {' '.join(cmd[-8:])}")

    result = subprocess.run(cmd, env=env)
    return result.returncode == 0


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--datasets", nargs="+", default=["ds003768", "ds005795", "noddi"],
                        help="Datasets to run. Use 'all' for all 4.")
    parser.add_argument("--mode", choices=["intra", "pooled", "both"], default="both")
    parser.add_argument("--backbones", nargs="+", default=BACKBONES)
    parser.add_argument("--resume", action="store_true",
                        help="Skip already-completed runs")
    parser.add_argument("--dry_run", action="store_true",
                        help="Print plan without running")
    args = parser.parse_args()

    datasets = list(DATASETS.keys()) if "all" in args.datasets else args.datasets
    modes    = ["intra", "pooled"] if args.mode == "both" else [args.mode]

    # Build run queue
    queue = []
    for ds in datasets:
        for mode in modes:
            for bb in args.backbones:
                if args.resume and result_key_exists(ds, bb, mode):
                    continue
                queue.append((ds, bb, mode))

    print(f"Plan: {len(queue)} runs | datasets={datasets} | modes={modes}")
    if args.dry_run:
        for ds, bb, mode in queue:
            print(f"  {bb:12s} {mode:6s} {ds}")
        return

    # Execute
    ok, fail = 0, 0
    for i, (ds, bb, mode) in enumerate(queue):
        print(f"\n[{i+1}/{len(queue)}] {bb} | {mode} | {ds}")
        success = run_one(ds, bb, mode)
        if success:
            ok += 1
        else:
            fail += 1
            print(f"  FAILED: {bb} {mode} {ds}")

    print(f"\n=== Done: {ok} ok, {fail} failed ===")


if __name__ == "__main__":
    main()
