"""
Merge all comparison results into a single canonical JSON.
Run this after both briem2ym4 (pooled) and bi6h23ufc (intra) complete.

Sources:
  1. comparison_experiment/results/comparison_ddpm_results.json  -- 7 saved pooled keys
  2. comparison_ddpm_results.json                                 -- intra keys (10 after bi6h23ufc)
  3. labram_pooled injected from task output (see LABRAM_POOLED below)
  4. neuroflow_v4 results from comparison_experiment/results/neuroflow_v4_results.json

Missing after race condition (need re-runs):
  - sparc_pooled  -> python train_comparison_ddpm.py --backbone sparc  --mode pooled --out rerun_pooled.json
  - ffcl_pooled   -> python train_comparison_ddpm.py --backbone ffcl   --mode pooled --out rerun_pooled.json
"""

import json, os, sys

BASE = os.path.dirname(os.path.abspath(__file__))
RESULTS_DIR = os.path.join(BASE, "comparison_experiment", "results")

MAIN_JSON    = os.path.join(BASE, "comparison_ddpm_results.json")
BACKUP_JSON  = os.path.join(RESULTS_DIR, "comparison_ddpm_results.json")
NF_JSON      = os.path.join(RESULTS_DIR, "neuroflow_v4_results.json")
RERUN_JSON   = os.path.join(BASE, "rerun_pooled.json")
OUT_JSON     = MAIN_JSON   # write back to main file so build_paper_tables.py works unchanged
BACKUP_OUT   = os.path.join(RESULTS_DIR, "all_results_merged.json")  # also save copy

# ── labram_pooled placeholder (fill from briem2ym4 task output) ─────────────
# Will be populated once briem2ym4 finishes.
LABRAM_POOLED = {
    "backbone": "labram", "mode": "pooled",
    "avg_r": 0.3260579705238342,
    "roi_r": [0.3851, 0.3870, 0.2958, 0.3606, 0.1399, 0.2920, 0.4222],
    "crps": 0.28474,
    "fc_mae": 0.0645,
    "crps_ci": [0.2769, 0.2935],
    "r_ci": [0.2952, 0.3570]
}

def load(path):
    if not os.path.exists(path):
        return {}
    with open(path) as f:
        return json.load(f)

def main():
    merged = {}

    # 1. Pooled backbones from backup (neurobolt..li2024, 7 keys)
    backup = load(BACKUP_JSON)
    for k, v in backup.items():
        merged[k] = v
    print(f"[backup]  loaded {len(backup)} keys: {list(backup.keys())}")

    # 2. Intra backbones from main file
    main_data = load(MAIN_JSON)
    intra_keys = [k for k in main_data if k.endswith("_intra")]
    for k in intra_keys:
        merged[k] = main_data[k]
    print(f"[main]    loaded {len(intra_keys)} intra keys: {intra_keys}")

    # 3. labram_pooled from task output (manual inject)
    if LABRAM_POOLED is not None:
        merged["labram_pooled"] = LABRAM_POOLED
        print("[inject]  labram_pooled injected from task output")
    else:
        print("[inject]  labram_pooled NOT available yet — fill LABRAM_POOLED dict above")

    # 4. Re-run results (sparc_pooled, ffcl_pooled)
    if os.path.exists(RERUN_JSON):
        rerun = load(RERUN_JSON)
        for k, v in rerun.items():
            merged[k] = v
        print(f"[rerun]   loaded {len(rerun)} keys: {list(rerun.keys())}")
    else:
        print("[rerun]   rerun_pooled.json not found — sparc/ffcl pooled still needed")

    # 5. NeuroFlow results (pooled + intra)
    nf = load(NF_JSON)
    if "neuroflow_pooled" in nf:
        merged["neuroflow_fm_pooled"] = nf["neuroflow_pooled"]
    if "neuroflow_intra" in nf:
        merged["neuroflow_fm_intra"] = nf["neuroflow_intra"]
    print(f"[neuroflow] loaded NeuroFlow results")

    # Summary
    print("\n=== Merged keys ===")
    pooled_keys = sorted(k for k in merged if k.endswith("_pooled"))
    intra_keys2 = sorted(k for k in merged if k.endswith("_intra"))
    print(f"Pooled ({len(pooled_keys)}): {pooled_keys}")
    print(f"Intra  ({len(intra_keys2)}): {intra_keys2}")

    # Print comparison table
    print("\n=== R comparison (NeuroFlow FM vs baselines) ===")
    nf_pooled_r = merged.get("neuroflow_fm_pooled", {}).get("avg_r", None)
    nf_intra_r  = merged.get("neuroflow_fm_intra", {}).get("avg_r", None)
    print(f"{'Backbone':<20} {'Pooled R':>10} {'Intra R':>10} {'Pooled CRPS':>13} {'Intra CRPS':>12}")
    print("-" * 68)

    backbones = ["neurobolt", "sparc", "contrawr", "ffcl", "cnn_trans",
                 "stt_trans", "biot", "beira", "li2024", "labram"]
    for bb in backbones:
        pk = f"{bb}_pooled"
        ik = f"{bb}_intra"
        pr = merged.get(pk, {}).get("avg_r", float("nan"))
        ir = merged.get(ik, {}).get("avg_r", float("nan"))
        pc = merged.get(pk, {}).get("crps", float("nan"))
        ic = merged.get(ik, {}).get("crps", float("nan"))
        print(f"{bb:<20} {pr:>10.4f} {ir:>10.4f} {pc:>13.4f} {ic:>12.4f}")

    print("-" * 68)
    print(f"{'NeuroFlow FM':<20} {nf_pooled_r or float('nan'):>10.4f} "
          f"{nf_intra_r or float('nan'):>10.4f} "
          f"{merged.get('neuroflow_fm_pooled',{}).get('crps', float('nan')):>13.4f} "
          f"{merged.get('neuroflow_fm_intra',{}).get('crps', float('nan')):>12.4f}")

    with open(OUT_JSON, "w") as f:
        json.dump(merged, f, indent=2)
    with open(BACKUP_OUT, "w") as f:
        json.dump(merged, f, indent=2)
    print(f"\nSaved to {OUT_JSON}")
    print(f"Backup: {BACKUP_OUT}")

if __name__ == "__main__":
    main()
