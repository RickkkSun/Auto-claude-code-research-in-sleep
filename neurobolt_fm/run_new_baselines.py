"""
run_new_baselines.py — Run REVE and BrainOmni (NeurIPS 2025) on all 8 datasets.

Resting-state:  ds003768, ds005795, ds006040 + neurobolt (original)
Non-resting:    ds002336, ds002725, natview-monkey1_run-01, ds007216

Usage:
  python run_new_baselines.py
  python run_new_baselines.py --backbones brainomni --datasets ds003768 ds002336
  python run_new_baselines.py --backbones reve brainomni --modes intra
"""

import subprocess, sys, os, json, time, argparse
from pathlib import Path

BASE    = Path("C:/Users/PC/Auto-claude-code-research-in-sleep/neurobolt_fm")
PYTHON  = "C:/Users/PC/AppData/Local/Programs/Python/Python312/python.exe"
RES_DIR = BASE / "external_results"

# All 8 datasets
EXTERNAL_DATASETS = [
    "ds003768", "ds005795", "ds006040",          # resting-state (external)
    "ds002336", "ds002725",                       # non-resting (external)
    "natview-monkey1_run-01", "ds007216",         # non-resting (external)
]
ORIGINAL_DATASET = "neurobolt"   # original dataset, uses comparison_ddpm.py

NEW_BACKBONES = ["reve", "brainomni"]
MODES = ["intra", "pooled"]

os.environ["PYTHONIOENCODING"] = "utf-8"


def result_key_exists(ds: str, key: str) -> bool:
    if ds == ORIGINAL_DATASET:
        f = BASE / "comparison_ddpm_results.json"
    else:
        f = RES_DIR / f"{ds}_results.json"
    if not f.exists():
        return False
    try:
        data = json.loads(f.read_text(encoding="utf-8"))
        return key in data
    except Exception:
        return False


def run_cmd(cmd, label):
    print(f"\n  >>> {label}", flush=True)
    t0 = time.time()
    result = subprocess.run(cmd, cwd=str(BASE), capture_output=False,
                            text=True, encoding="utf-8", errors="replace")
    elapsed = time.time() - t0
    ok = result.returncode == 0
    print(f"  {'OK' if ok else 'FAILED'} ({elapsed:.0f}s)", flush=True)
    return ok


def run_external(ds, backbone, mode):
    key = f"{backbone}_{mode}"
    if result_key_exists(ds, key):
        print(f"  SKIP {backbone} {mode} on {ds} (already done)", flush=True)
        return True
    out_json = str(RES_DIR / f"{ds}_results.json")
    return run_cmd([PYTHON, "train_comparison_ddpm_external.py",
                    "--backbone", backbone, "--mode", mode,
                    "--dataset", ds, "--out", out_json],
                   f"{backbone} {mode} on {ds}")


def run_original(backbone, mode):
    key = f"{backbone}_{mode}"
    if result_key_exists(ORIGINAL_DATASET, key):
        print(f"  SKIP {backbone} {mode} on neurobolt (already done)", flush=True)
        return True
    out_json = str(BASE / "comparison_ddpm_results.json")
    return run_cmd([PYTHON, "train_comparison_ddpm.py",
                    "--backbone", backbone, "--mode", mode,
                    "--out", out_json],
                   f"{backbone} {mode} on neurobolt (original)")


def sync_neurobolt_results():
    """Copy comparison_ddpm_results.json + neuroflow_v4_results.json
    into external_results/neurobolt_results.json for unified table generation."""
    src1 = BASE / "comparison_ddpm_results.json"
    src2 = BASE / "comparison_experiment" / "results" / "neuroflow_v4_results.json"
    dst  = RES_DIR / "neurobolt_results.json"

    merged = {}
    if src1.exists():
        merged.update(json.loads(src1.read_text(encoding="utf-8")))
    if src2.exists():
        nfm = json.loads(src2.read_text(encoding="utf-8"))
        # Map neuroflow aliases to neurocfm_* keys
        alias_map = {
            "neuroflow_intra":       "neurocfm_intra",
            "neuroflow_v4_intra_tau":"neurocfm_intra",
            "neuroflow_pooled":      "neurocfm_pooled",
            "neuroflow_v4_pooled_tau":"neurocfm_pooled",
        }
        for k, v in nfm.items():
            target = alias_map.get(k, k)
            merged[target] = v

    RES_DIR.mkdir(exist_ok=True)
    dst.write_text(json.dumps(merged, indent=2), encoding="utf-8")
    print(f"  Synced neurobolt_results.json ({len(merged)} keys)", flush=True)


def generate_table(ds):
    print(f"\n  Generating table for {ds}...", flush=True)
    # Use neurobolt_results.json for original dataset
    if ds == ORIGINAL_DATASET:
        src_ds = "neurobolt"
    else:
        src_ds = ds
    cmd = [PYTHON, "generate_external_tables.py",
           "--datasets", src_ds,
           "--modes", "intra", "pooled",
           "--out_dir", str(RES_DIR / "tables")]
    result = subprocess.run(cmd, cwd=str(BASE), capture_output=False,
                            text=True, encoding="utf-8", errors="replace")
    if result.returncode == 0:
        print(f"  Table saved for {src_ds}", flush=True)
    else:
        print(f"  WARNING: table generation failed for {src_ds}", flush=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--backbones", nargs="+", default=NEW_BACKBONES)
    parser.add_argument("--modes",    nargs="+", default=MODES)
    parser.add_argument("--datasets", nargs="+",
                        default=EXTERNAL_DATASETS + [ORIGINAL_DATASET])
    args = parser.parse_args()

    t_start = time.time()

    for ds in args.datasets:
        print(f"\n{'='*60}")
        print(f"  DATASET: {ds}")
        print(f"{'='*60}", flush=True)

        for backbone in args.backbones:
            for mode in args.modes:
                if ds == ORIGINAL_DATASET:
                    ok = run_original(backbone, mode)
                else:
                    ok = run_external(ds, backbone, mode)
                if not ok:
                    print(f"  WARNING: {backbone} {mode} on {ds} FAILED", flush=True)

        # Sync & generate table after each dataset
        if ds == ORIGINAL_DATASET:
            sync_neurobolt_results()
        generate_table(ds)

    total = time.time() - t_start
    print(f"\n{'='*60}")
    print(f"  ALL DONE in {total/60:.1f} min")
    print(f"  Tables: {RES_DIR / 'tables'}")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
