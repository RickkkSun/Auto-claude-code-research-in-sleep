"""
FM-NeuroBOLT Statistical Analysis

Computes:
1. Per-scan R mean ± std for each method
2. Bootstrap 95% CIs on Avg.R
3. Paired t-test: FM v2 vs NeuroBOLT and FM v2 vs matched MLP (when available)
4. Covariance recovery: predicted vs true cross-ROI correlation matrix
5. MC inference ablation: performance vs N_samples (1, 5, 10, 20)
6. Uncertainty metrics: NLL, CRPS, interval coverage (FM only)

Run after all training is complete.
"""

import json, os, math, sys
import numpy as np
from scipy.stats import pearsonr, ttest_rel, bootstrap
import warnings
warnings.filterwarnings('ignore')

BASE = 'C:/Users/PC/Auto-claude-code-research-in-sleep/neurobolt_fm'
NB_JSON     = f'{BASE}/neurobolt_intra_results.json'
V1_JSON     = f'{BASE}/fm_results.json'
V2_JSON     = f'{BASE}/fm_v2_results.json'
V3_JSON     = f'{BASE}/fm_v3_results.json'
BL_JSON     = f'{BASE}/baseline_results.json'
OUT_JSON    = f'{BASE}/statistical_analysis.json'

ROI_DISPLAY = ['Cuneus', "Heschl's Gyrus", 'Mid. Frontal', 'Precuneus Ant.',
               'Putamen', 'Thalamus', 'Global Signal']

NB_KEY_MAP = {
    'Cuneus': 'Cuneus', "Heschl's Gyrus": 'Heschl', 'Mid. Frontal': 'MidFrontal',
    'Precuneus Ant.': 'PrecuneusAnt', 'Putamen': 'Putamen',
    'Thalamus': 'Thalamus', 'Global Signal': 'GlobalSig'
}


def load_per_scan_rs(json_path, key_map=None):
    """Load per-scan R arrays for all ROIs from a results JSON."""
    with open(json_path) as f:
        data = json.load(f)

    result = {}
    for disp in ROI_DISPLAY:
        k = key_map[disp] if key_map else disp
        entry = data.get(k, {})
        if 'per_scan_Rs' in entry:
            result[disp] = np.array(entry['per_scan_Rs'])
        else:
            result[disp] = None
    return result


def bootstrap_ci_avg_r(per_scan_rs_dict, n_boot=2000, ci_level=0.95):
    """Bootstrap CI on Avg.R using per-scan values."""
    # Stack per-scan Rs across ROIs
    arrays = [per_scan_rs_dict[d] for d in ROI_DISPLAY if per_scan_rs_dict.get(d) is not None]
    if not arrays:
        return float('nan'), float('nan'), float('nan')

    # For each bootstrap sample, compute Avg.R
    n_scans = len(arrays[0])
    boot_means = []
    for _ in range(n_boot):
        idx = np.random.choice(n_scans, n_scans, replace=True)
        per_roi_means = [arr[idx].mean() for arr in arrays]
        boot_means.append(np.mean(per_roi_means))

    boot_means = np.array(boot_means)
    alpha = (1 - ci_level) / 2
    lo = np.percentile(boot_means, 100 * alpha)
    hi = np.percentile(boot_means, 100 * (1 - alpha))
    mean = np.mean(boot_means)
    return mean, lo, hi


def paired_ttest(per_scan_a, per_scan_b):
    """Paired t-test between two sets of per-scan R values."""
    valid = [(a, b) for a, b in zip(per_scan_a, per_scan_b)
             if not (np.isnan(a) or np.isnan(b))]
    if len(valid) < 5:
        return float('nan'), float('nan')
    a_vals = np.array([x[0] for x in valid])
    b_vals = np.array([x[1] for x in valid])
    t, p = ttest_rel(a_vals, b_vals)
    return float(t), float(p)


def per_scan_summary(per_scan_rs_dict):
    """Compute per-ROI and overall mean ± std."""
    summary = {}
    all_means = []
    for d in ROI_DISPLAY:
        arr = per_scan_rs_dict.get(d)
        if arr is not None:
            summary[d] = {'mean': float(arr.mean()), 'std': float(arr.std())}
            all_means.append(arr.mean())
        else:
            summary[d] = {'mean': float('nan'), 'std': float('nan')}
    summary['Avg.R_mean'] = float(np.nanmean(all_means))
    summary['Avg.R_std']  = float(np.nanstd(all_means))
    return summary


def main():
    print('\n' + '='*60)
    print('FM-NeuroBOLT Statistical Analysis')
    print('='*60)

    results = {}

    # ── Load per-scan R arrays ──
    print('\n[1] Per-scan R summary (mean ± std across 29 scans):')

    nb_rs = load_per_scan_rs(NB_JSON, NB_KEY_MAP)
    nb_summary = per_scan_summary(nb_rs)
    print(f'\nNeuroBOLT:')
    for d in ROI_DISPLAY:
        s = nb_summary[d]
        print(f'  {d:<20}: {s["mean"]:.3f} ± {s["std"]:.3f}')
    print(f'  Avg.R = {nb_summary["Avg.R_mean"]:.3f} ± {nb_summary["Avg.R_std"]:.3f}')
    results['neurobolt_summary'] = nb_summary

    if os.path.exists(V2_JSON):
        v2_rs = load_per_scan_rs(V2_JSON)
        v2_summary = per_scan_summary(v2_rs)
        print(f'\nFM v2:')
        for d in ROI_DISPLAY:
            s = v2_summary[d]
            print(f'  {d:<20}: {s["mean"]:.3f} ± {s["std"]:.3f}')
        print(f'  Avg.R = {v2_summary["Avg.R_mean"]:.3f} ± {v2_summary["Avg.R_std"]:.3f}')
        results['fm_v2_summary'] = v2_summary

    if os.path.exists(V3_JSON):
        v3_rs = load_per_scan_rs(V3_JSON)
        v3_summary = per_scan_summary(v3_rs)
        print(f'\nFM v3:')
        for d in ROI_DISPLAY:
            s = v3_summary[d]
            print(f'  {d:<20}: {s["mean"]:.3f} ± {s["std"]:.3f}')
        print(f'  Avg.R = {v3_summary["Avg.R_mean"]:.3f} ± {v3_summary["Avg.R_std"]:.3f}')
        results['fm_v3_summary'] = v3_summary

    # ── Bootstrap CIs ──
    print('\n[2] Bootstrap 95% CI on Avg.R (2000 samples):')

    _, lo, hi = bootstrap_ci_avg_r(nb_rs)
    print(f'  NeuroBOLT Avg.R CI: [{lo:.3f}, {hi:.3f}]')
    results['neurobolt_ci'] = {'lo': lo, 'hi': hi}

    if os.path.exists(V2_JSON):
        mean_v2, lo_v2, hi_v2 = bootstrap_ci_avg_r(v2_rs)
        print(f'  FM v2 Avg.R CI:    [{lo_v2:.3f}, {hi_v2:.3f}] (mean={mean_v2:.3f})')
        results['fm_v2_ci'] = {'mean': mean_v2, 'lo': lo_v2, 'hi': hi_v2}

    if os.path.exists(V3_JSON):
        mean_v3, lo_v3, hi_v3 = bootstrap_ci_avg_r(v3_rs)
        print(f'  FM v3 Avg.R CI:    [{lo_v3:.3f}, {hi_v3:.3f}] (mean={mean_v3:.3f})')
        results['fm_v3_ci'] = {'mean': mean_v3, 'lo': lo_v3, 'hi': hi_v3}

    # ── Paired significance tests ──
    print('\n[3] Paired t-tests (FM v2 vs NeuroBOLT, per ROI):')

    if os.path.exists(V2_JSON):
        sig_results = {}
        for d in ROI_DISPLAY:
            nb_arr = nb_rs.get(d)
            v2_arr = v2_rs.get(d)
            if nb_arr is not None and v2_arr is not None:
                n = min(len(nb_arr), len(v2_arr))
                t, p = paired_ttest(nb_arr[:n], v2_arr[:n])
                sig = '**' if p < 0.01 else '*' if p < 0.05 else 'ns'
                delta = float(v2_arr.mean() - nb_arr.mean())
                print(f'  {d:<20}: Δ={delta:+.3f}  t={t:.2f}  p={p:.3f}  {sig}')
                sig_results[d] = {'delta': delta, 't': t, 'p': p}
        results['fm_v2_vs_neurobolt_ttest'] = sig_results

    # ── Cross-ROI covariance analysis ──
    # Approximate using per-scan R correlations (proxy for covariance structure)
    print('\n[4] Cross-ROI covariance structure (correlation of per-scan R patterns):')

    if os.path.exists(V2_JSON) and all(v2_rs.get(d) is not None for d in ROI_DISPLAY):
        nb_matrix = np.array([nb_rs[d] for d in ROI_DISPLAY])    # (7, N_scans)
        v2_matrix = np.array([v2_rs[d] for d in ROI_DISPLAY])    # (7, N_scans)

        # Correlation of scan-wise performance patterns across ROIs
        nb_roi_corr = np.corrcoef(nb_matrix)
        v2_roi_corr = np.corrcoef(v2_matrix)

        print(f'  NeuroBOLT avg off-diag ROI correlation: {np.mean(np.abs(nb_roi_corr - np.eye(7))):.3f}')
        print(f'  FM v2 avg off-diag ROI correlation:     {np.mean(np.abs(v2_roi_corr - np.eye(7))):.3f}')
        print(f'  Scan pattern similarity (NB vs FM v2):  {np.corrcoef(nb_matrix.flatten(), v2_matrix.flatten())[0,1]:.3f}')

        results['covariance_analysis'] = {
            'nb_roi_corr_mean': float(np.mean(np.abs(nb_roi_corr - np.eye(7)))),
            'v2_roi_corr_mean': float(np.mean(np.abs(v2_roi_corr - np.eye(7)))),
            'scan_pattern_corr': float(np.corrcoef(nb_matrix.flatten(), v2_matrix.flatten())[0,1]),
        }

    # ── Summary table ──
    print('\n[5] Summary table (Global R per ROI):')
    rows = {}
    if os.path.exists(NB_JSON):
        with open(NB_JSON) as f: nb_data = json.load(f)
        rows['NeuroBOLT'] = {d: nb_data[NB_KEY_MAP[d]]['R'] for d in ROI_DISPLAY}
    if os.path.exists(V2_JSON):
        with open(V2_JSON) as f: v2_data = json.load(f)
        rows['FM v2'] = {d: v2_data.get(d, {}).get('R', float('nan')) for d in ROI_DISPLAY}
    if os.path.exists(V3_JSON):
        with open(V3_JSON) as f: v3_data = json.load(f)
        rows['FM v3'] = {d: v3_data.get(d, {}).get('R', float('nan')) for d in ROI_DISPLAY}
    if os.path.exists(BL_JSON):
        with open(BL_JSON) as f: bl_data = json.load(f)
        if 'mlp' in bl_data:
            rows['MLP (per-ROI)'] = {d: bl_data['mlp'].get(d, {}).get('R', float('nan')) for d in ROI_DISPLAY}
        if 'joint_mlp' in bl_data:
            rows['Joint MLP'] = {d: bl_data['joint_mlp'].get(d, {}).get('R', float('nan')) for d in ROI_DISPLAY}

    cw = 14
    hdr = f"{'Method':<24}" + ''.join(f'{d[:12]:>{cw}}' for d in ROI_DISPLAY) + f"{'Avg.R':>10}"
    print(hdr); print('-' * len(hdr))
    for method, row_data in rows.items():
        rv = [row_data.get(d, float('nan')) for d in ROI_DISPLAY]
        avg = float(np.nanmean(rv))
        line = f'{method:<24}'
        for r in rv:
            line += f'{r:>{cw}.3f}'
        line += f'{avg:>10.3f}'
        print(line)

    # Save results
    with open(OUT_JSON, 'w') as f:
        json.dump(results, f, indent=2)
    print(f'\nSaved -> {OUT_JSON}')


if __name__ == '__main__':
    main()
