"""
run_nonrest_full.py — Run all NeuroCFM + 10 baseline experiments on 4 non-resting-state
datasets, generating comparison tables after each dataset completes.

Datasets (in order): ds002336, ds002725, natview-monkey1_run-01, ds007216
Methods: neurocfm + sparc, contrawr, ffcl, biot, cnn_trans, stt_trans, beira, li2024, neurobolt, labram
Modes: intra, pooled

Generates tables to: external_results/tables/{ds}_table_{mode}.png
"""

import subprocess, sys, os, json, time
from pathlib import Path

BASE   = Path("C:/Users/PC/Auto-claude-code-research-in-sleep/neurobolt_fm")
PYTHON = "C:/Users/PC/AppData/Local/Programs/Python/Python312/python.exe"
RES_DIR = BASE / "external_results"

DATASETS  = ["ds002336", "ds002725", "natview-monkey1_run-01", "ds007216"]
BACKBONES = ["sparc", "contrawr", "ffcl", "biot", "cnn_trans",
             "stt_trans", "beira", "li2024", "neurobolt", "labram"]
MODES = ["intra", "pooled"]

os.environ["PYTHONIOENCODING"] = "utf-8"


def result_key_exists(ds: str, key: str) -> bool:
    f = RES_DIR / f"{ds}_results.json"
    if not f.exists():
        return False
    try:
        data = json.loads(f.read_text(encoding="utf-8"))
        # neurocfm may be stored as neuroflow_fm_*
        if key.startswith("neurocfm_"):
            mode = key.split("_", 1)[1]
            return (key in data or
                    f"neuroflow_fm_{mode}" in data or
                    f"neuroflow_{mode}" in data)
        return key in data
    except Exception:
        return False


def count_methods(ds: str, mode: str) -> int:
    f = RES_DIR / f"{ds}_results.json"
    if not f.exists():
        return 0
    try:
        data = json.loads(f.read_text(encoding="utf-8"))
        return sum(1 for k in data if k.endswith(f"_{mode}"))
    except Exception:
        return 0


def run_cmd(cmd: list, label: str) -> bool:
    print(f"\n  >>> {label}", flush=True)
    t0 = time.time()
    result = subprocess.run(cmd, cwd=str(BASE),
                            capture_output=False,
                            text=True, encoding="utf-8", errors="replace")
    elapsed = time.time() - t0
    ok = result.returncode == 0
    print(f"  {'OK' if ok else 'FAILED'} ({elapsed:.0f}s)", flush=True)
    return ok


def generate_table(ds: str):
    print(f"\n  >>> Generating table for {ds}...", flush=True)
    cmd = [PYTHON, "generate_external_tables.py",
           "--datasets", ds,
           "--modes", "intra", "pooled",
           "--out_dir", str(RES_DIR / "tables")]
    result = subprocess.run(cmd, cwd=str(BASE),
                            capture_output=False,
                            text=True, encoding="utf-8", errors="replace")
    if result.returncode == 0:
        print(f"  Table saved to external_results/tables/{ds}_table_*.png", flush=True)
    else:
        print(f"  WARNING: table generation failed for {ds}", flush=True)


def run_dataset(ds: str, modes: list = None):
    if modes is None:
        modes = MODES
    print(f"\n{'='*60}")
    print(f"  DATASET: {ds}")
    print(f"{'='*60}", flush=True)

    out_json = str(RES_DIR / f"{ds}_results.json")

    # 1. NeuroCFM (intra + pooled)
    for mode in modes:
        key = f"neurocfm_{mode}"
        if result_key_exists(ds, key):
            print(f"  SKIP neurocfm {mode} (already done)", flush=True)
            continue
        run_cmd([PYTHON, "train_neurocfm_external.py",
                 "--dataset", ds, "--mode", mode, "--out", out_json],
                f"NeuroCFM {mode} on {ds}")

    # 2. Baselines (all backbones × both modes)
    for mode in modes:
        for bb in BACKBONES:
            key = f"{bb}_{mode}"
            if result_key_exists(ds, key):
                print(f"  SKIP {bb} {mode} (already done)", flush=True)
                continue
            ok = run_cmd([PYTHON, "train_comparison_ddpm_external.py",
                          "--backbone", bb, "--mode", mode,
                          "--dataset", ds, "--out", out_json],
                         f"{bb} {mode} on {ds}")
            if not ok:
                print(f"  WARNING: {bb} {mode} failed, continuing...", flush=True)

    # 3. Generate table (both modes)
    intra_n  = count_methods(ds, "intra")
    pooled_n = count_methods(ds, "pooled")
    print(f"\n  Results: {intra_n} intra, {pooled_n} pooled methods", flush=True)

    if intra_n >= 2 or pooled_n >= 2:
        generate_table(ds)
    else:
        print(f"  Not enough results to generate table — skipping", flush=True)


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--datasets", nargs="+", default=DATASETS)
    parser.add_argument("--modes",    nargs="+", default=MODES)
    args = parser.parse_args()

    modes = args.modes
    t_start = time.time()
    for ds in args.datasets:
        run_dataset(ds, modes)

    total = time.time() - t_start
    print(f"\n{'='*60}")
    print(f"  ALL DONE in {total/60:.1f} min")
    print(f"  Tables: {RES_DIR / 'tables'}")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
