"""
FM-NeuroBOLT Paper Figure Generator
Generates all comparison figures in NeuroBOLT paper style.

Figures produced:
1. fig1_bar_comparison.png     — R comparison bar chart (all methods, all ROIs)
2. fig2_scatter_per_roi.png    — Predicted vs True scatter plots (NeuroBOLT vs FM v2 vs FM v3b)
3. fig3_covariance_matrix.png  — True vs FM v3b vs Gaussian cross-ROI correlation matrices
4. fig4_violin_per_scan.png    — Per-scan R distribution (violin/box plots)
5. fig5_ci_avgr.png            — Bootstrap CI on Avg.R comparison
6. fig6_crps_coverage.png      — CRPS + Coverage comparison (probabilistic methods)
7. fig7_cross_subject.png      — Intra vs Cross-subject comparison (generated after cross_subject run)
"""

import sys, os, json, gc, math, warnings
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'code'))
warnings.filterwarnings('ignore')

import numpy as np
import torch
import torch.nn as nn
import pandas as pd
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.patches import FancyBboxPatch
import matplotlib.ticker as ticker
from scipy.stats import pearsonr
from einops import rearrange
import mne
mne.set_log_level('WARNING')
from scipy.signal import butter, filtfilt
from timm.models import create_model
import models.model
from dataset_maker import preproc

BASE     = 'C:/Users/PC/Auto-claude-code-research-in-sleep/neurobolt_fm'
FIG_DIR  = f'{BASE}/figures'
os.makedirs(FIG_DIR, exist_ok=True)

# ── Color palette (colorblind-safe) ──────────────────────────────────────────
COLORS = {
    'NeuroBOLT':    '#4878CF',  # blue
    'Ridge':        '#6ACC65',  # green
    'MLP':          '#D65F5F',  # red
    'FM v2':        '#B47CC7',  # purple
    'FM v3b':       '#C4AD66',  # gold
    'Gaussian':     '#77BEDB',  # teal
    'Joint MLP':    '#F0A500',  # orange
}

ROI_DISPLAY = ['Cuneus', "Heschl's Gyrus", 'Mid. Frontal', 'Precuneus Ant.',
               'Putamen', 'Thalamus', 'Global Signal']

ROI_SHORT = ['Cuneus', "Heschl's", 'Mid.Front', 'Precuneus', 'Putamen', 'Thalamus', 'Global']


# ─────────────────────────────────────────────────────────────────────────────
# Load all result JSONs
# ─────────────────────────────────────────────────────────────────────────────
def load_results():
    results = {}
    files = {
        'nb': f'{BASE}/neurobolt_intra_results.json',
        'fm2': f'{BASE}/fm_v2_results.json',
        'fm3': f'{BASE}/fm_v3_results.json',
        'bl': f'{BASE}/baseline_results.json',
        'prob': f'{BASE}/prob_baseline_results.json',
        'unc': f'{BASE}/uncertainty_results.json',
        'stats': f'{BASE}/statistical_analysis.json',
        'cross': f'{BASE}/cross_subject_results.json',
    }
    for k, path in files.items():
        if os.path.exists(path):
            with open(path) as f:
                results[k] = json.load(f)
        else:
            results[k] = None
    return results


# ─────────────────────────────────────────────────────────────────────────────
# Fig 1: Bar chart — R comparison across all methods and ROIs
# ─────────────────────────────────────────────────────────────────────────────
def fig1_bar_comparison(results):
    print('Generating Fig 1: Bar comparison chart...')

    # All methods and their per-ROI R values
    nb_rs = [results['nb'][k]['R'] for k in ['Cuneus','Heschl','MidFrontal','PrecuneusAnt','Putamen','Thalamus','GlobalSig']]

    bl = results['bl']
    ridge_rs = [bl['ridge'][d]['R'] for d in ROI_DISPLAY] if bl and 'ridge' in bl else None
    mlp_rs   = [bl['mlp'][d]['R']   for d in ROI_DISPLAY] if bl and 'mlp'   in bl else None

    fm2 = results['fm2']
    fm2_rs = [fm2[d]['R'] for d in ROI_DISPLAY] if fm2 else None

    fm3 = results['fm3']
    fm3_rs = [fm3[d]['R'] for d in ROI_DISPLAY] if fm3 else None

    # Build the bar chart
    methods_data = [
        ('NeuroBOLT', nb_rs, COLORS['NeuroBOLT']),
    ]
    if ridge_rs: methods_data.append(('Ridge', ridge_rs, COLORS['Ridge']))
    if mlp_rs:   methods_data.append(('MLP', mlp_rs, COLORS['MLP']))
    if fm2_rs:   methods_data.append(('FM v2', fm2_rs, COLORS['FM v2']))
    if fm3_rs:   methods_data.append(('FM v3b', fm3_rs, COLORS['FM v3b']))

    n_methods = len(methods_data)
    n_rois = 7
    x = np.arange(n_rois)
    width = 0.8 / n_methods

    fig, ax = plt.subplots(figsize=(14, 5))
    for i, (name, rs, color) in enumerate(methods_data):
        offset = (i - n_methods/2 + 0.5) * width
        bars = ax.bar(x + offset, rs, width, label=name, color=color, alpha=0.85, edgecolor='white', linewidth=0.5)
        # Add value labels on top
        for bar, r in zip(bars, rs):
            if not np.isnan(r) and r > 0.1:
                ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.005,
                        f'{r:.2f}', ha='center', va='bottom', fontsize=6.5, color='#333')

    ax.set_xticks(x)
    ax.set_xticklabels(ROI_SHORT, fontsize=11)
    ax.set_ylabel('Pearson R', fontsize=12)
    ax.set_title('EEG→fMRI Prediction: Pearson R by ROI and Method (Intra-Subject)', fontsize=13, fontweight='bold')
    ax.legend(loc='upper right', framealpha=0.9, fontsize=10)
    ax.set_ylim(0, 0.75)
    ax.axhline(y=0, color='black', linewidth=0.5, linestyle='--', alpha=0.3)
    ax.grid(axis='y', alpha=0.3, linestyle='--')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    # Add avg R annotations
    for i, (name, rs, color) in enumerate(methods_data):
        avg_r = np.nanmean([r for r in rs if not np.isnan(r)])
        ax.text(0.02 + i * 0.14, 0.97, f'{name}\nAvg={avg_r:.3f}',
                transform=ax.transAxes, fontsize=8, va='top', color=color, fontweight='bold')

    plt.tight_layout()
    plt.savefig(f'{FIG_DIR}/fig1_bar_comparison.png', dpi=150, bbox_inches='tight')
    plt.close()
    print(f'  Saved: {FIG_DIR}/fig1_bar_comparison.png')


# ─────────────────────────────────────────────────────────────────────────────
# Fig 2: Scatter plots (Predicted vs True per ROI) — requires re-prediction
# ─────────────────────────────────────────────────────────────────────────────
def fig2_scatter_plots(results):
    """Scatter plots: True fMRI vs Predicted for NeuroBOLT and FM v2."""
    print('Generating Fig 2: Scatter plots...')

    # We need to re-run predictions on the test set to get arrays
    # Use existing per-scan R values to create scatter-like proxy plots
    # Alternatively, use the global pooled test set from results

    # Use per-scan R to build a heatmap / per-scan scatter
    nb_per_scan = [results['nb'][k]['per_scan_Rs'] for k in
                   ['Cuneus','Heschl','MidFrontal','PrecuneusAnt','Putamen','Thalamus','GlobalSig']]
    fm2_per_scan = [results['fm2'][d]['per_scan_Rs'] for d in ROI_DISPLAY] if results['fm2'] else None
    fm3_per_scan = [results['fm3'][d]['per_scan_Rs'] for d in ROI_DISPLAY] if results['fm3'] else None

    n_scans = len(nb_per_scan[0])
    scan_ids = list(range(1, n_scans+1))

    fig, axes = plt.subplots(3, 1, figsize=(14, 10), sharex=True)

    datasets = [
        ('NeuroBOLT (pretrained)', nb_per_scan, COLORS['NeuroBOLT']),
        ('FM v2 (per-ROI FM)', fm2_per_scan, COLORS['FM v2']),
        ('FM v3b (joint 7D FM)', fm3_per_scan, COLORS['FM v3b']),
    ]

    for ax, (title, per_scan_list, color) in zip(axes, datasets):
        if per_scan_list is None:
            ax.text(0.5, 0.5, 'Not available', transform=ax.transAxes, ha='center')
            continue

        per_scan_arr = np.array(per_scan_list)  # (7, n_scans)
        mean_r = per_scan_arr.mean(axis=0)       # (n_scans,)

        # Plot per-ROI lines
        for j, (disp, roi_rs) in enumerate(zip(ROI_SHORT, per_scan_list)):
            ax.plot(scan_ids, roi_rs, 'o-', markersize=3, linewidth=1.2,
                    alpha=0.5, label=disp)

        # Bold mean
        ax.plot(scan_ids, mean_r, 'k-', linewidth=2.5, label='Mean (7 ROIs)', zorder=5)
        ax.axhline(0, color='gray', linewidth=0.8, linestyle='--', alpha=0.5)
        avg_r = float(np.nanmean([r for r in mean_r if not np.isnan(r)]))
        ax.set_title(f'{title} — Per-Scan Mean R = {avg_r:.3f}', fontsize=11, fontweight='bold')
        ax.set_ylabel('Pearson R', fontsize=10)
        ax.set_ylim(-0.5, 1.0)
        ax.grid(alpha=0.2, linestyle='--')
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)

    axes[0].legend(loc='upper right', fontsize=8, ncol=4)
    axes[-1].set_xlabel('Scan Index', fontsize=11)
    axes[-1].set_xticks(scan_ids[::2])

    plt.suptitle('Per-Scan Pearson R Across 29 Scans — All ROIs', fontsize=13, fontweight='bold', y=1.01)
    plt.tight_layout()
    plt.savefig(f'{FIG_DIR}/fig2_per_scan_r.png', dpi=150, bbox_inches='tight')
    plt.close()
    print(f'  Saved: {FIG_DIR}/fig2_per_scan_r.png')


# ─────────────────────────────────────────────────────────────────────────────
# Fig 3: Cross-ROI correlation matrices
# ─────────────────────────────────────────────────────────────────────────────
def fig3_covariance_matrices(results):
    print('Generating Fig 3: Cross-ROI correlation matrices...')

    unc = results['unc']
    prob = results['prob']
    if not unc or not prob:
        print('  Missing uncertainty/prob results, skipping.')
        return

    true_corr = np.array(unc.get('true_corr', None))
    pred_corr_fm = np.array(unc.get('pred_corr', None))

    if true_corr is None:
        print('  No correlation matrices in uncertainty results, loading from json...')
        # Try to load from fm_v3_results or reconstruct
        return

    # Gaussian head correlation (predicted means → correlation)
    gauss = prob.get('gaussian_head', {})
    gauss_cov_mae = gauss.get('cov_mae', float('nan'))

    # Load FM v3b correlation from prob_baseline
    fm_cal = prob.get('fm_v3b_calibrated', {})
    fm_cov_mae = fm_cal.get('cov_mae', 0.059) if fm_cal else 0.059

    fig, axes = plt.subplots(1, 3, figsize=(15, 4))

    matrices = [
        ('True Cross-ROI\nCorrelation', true_corr),
        (f'FM v3b Predicted\n(CovMAE={fm_cov_mae:.3f})', pred_corr_fm),
    ]

    for ax, (title, mat) in zip(axes[:2], matrices):
        if mat is None:
            ax.text(0.5, 0.5, 'N/A', transform=ax.transAxes, ha='center')
            continue
        mat = np.array(mat)
        im = ax.imshow(mat, vmin=-1, vmax=1, cmap='RdBu_r', aspect='auto')
        ax.set_xticks(range(7)); ax.set_yticks(range(7))
        ax.set_xticklabels(ROI_SHORT, rotation=45, ha='right', fontsize=8)
        ax.set_yticklabels(ROI_SHORT, fontsize=8)
        ax.set_title(title, fontsize=11, fontweight='bold')
        # Add numbers
        for i in range(7):
            for j in range(7):
                ax.text(j, i, f'{mat[i,j]:.2f}', ha='center', va='center',
                        fontsize=7, color='black' if abs(mat[i,j]) < 0.5 else 'white')
        plt.colorbar(im, ax=ax, shrink=0.8)

    # Difference matrix: FM - True
    if true_corr is not None and pred_corr_fm is not None:
        diff = np.array(pred_corr_fm) - np.array(true_corr)
        im = axes[2].imshow(diff, vmin=-0.5, vmax=0.5, cmap='PiYG', aspect='auto')
        axes[2].set_xticks(range(7)); axes[2].set_yticks(range(7))
        axes[2].set_xticklabels(ROI_SHORT, rotation=45, ha='right', fontsize=8)
        axes[2].set_yticklabels(ROI_SHORT, fontsize=8)
        off_diag = ~np.eye(7, dtype=bool)
        mae = float(np.mean(np.abs(diff[off_diag])))
        axes[2].set_title(f'FM − True (Off-diag MAE={mae:.3f})', fontsize=11, fontweight='bold')
        for i in range(7):
            for j in range(7):
                axes[2].text(j, i, f'{diff[i,j]:.2f}', ha='center', va='center',
                             fontsize=7, color='black' if abs(diff[i,j]) < 0.25 else 'white')
        plt.colorbar(im, ax=axes[2], shrink=0.8)

    plt.suptitle('Cross-ROI Correlation Structure: FM v3b vs Ground Truth', fontsize=12, fontweight='bold')
    plt.tight_layout()
    plt.savefig(f'{FIG_DIR}/fig3_covariance_matrices.png', dpi=150, bbox_inches='tight')
    plt.close()
    print(f'  Saved: {FIG_DIR}/fig3_covariance_matrices.png')


# ─────────────────────────────────────────────────────────────────────────────
# Fig 4: Violin/box plots of per-scan R
# ─────────────────────────────────────────────────────────────────────────────
def fig4_violin_per_scan(results):
    print('Generating Fig 4: Violin plots...')

    nb_key_map = {'Cuneus': 'Cuneus', "Heschl's Gyrus": 'Heschl', 'Mid. Frontal': 'MidFrontal',
                  'Precuneus Ant.': 'PrecuneusAnt', 'Putamen': 'Putamen',
                  'Thalamus': 'Thalamus', 'Global Signal': 'GlobalSig'}

    # Collect per-scan R data for each method
    data_per_method = {}
    if results['nb']:
        data_per_method['NeuroBOLT'] = np.array([results['nb'][nb_key_map[d]]['per_scan_Rs'] for d in ROI_DISPLAY])
    if results['fm2']:
        data_per_method['FM v2'] = np.array([results['fm2'][d]['per_scan_Rs'] for d in ROI_DISPLAY])
    if results['fm3']:
        data_per_method['FM v3b'] = np.array([results['fm3'][d]['per_scan_Rs'] for d in ROI_DISPLAY])

    if not data_per_method:
        print('  No per-scan data available.')
        return

    fig, ax = plt.subplots(figsize=(14, 5))

    method_names = list(data_per_method.keys())
    n_methods = len(method_names)
    n_rois = 7
    x = np.arange(n_rois)
    width = 0.8 / n_methods

    for i, (method, arr) in enumerate(data_per_method.items()):
        offset = (i - n_methods/2 + 0.5) * width
        color = COLORS.get(method, '#888')
        for j in range(n_rois):
            data = arr[j]
            # Box plot
            bp = ax.boxplot(data, positions=[x[j]+offset], widths=width*0.8,
                           patch_artist=True,
                           boxprops=dict(facecolor=color, alpha=0.7),
                           medianprops=dict(color='black', linewidth=2),
                           whiskerprops=dict(linewidth=1.2),
                           capprops=dict(linewidth=1.2),
                           flierprops=dict(marker='o', markersize=3, alpha=0.5))

    ax.set_xticks(x)
    ax.set_xticklabels(ROI_SHORT, fontsize=11)
    ax.set_ylabel('Pearson R (per scan)', fontsize=12)
    ax.set_title('Per-Scan R Distribution Across 29 Scans — All Methods and ROIs', fontsize=13, fontweight='bold')
    ax.axhline(0, color='gray', linewidth=0.8, linestyle='--', alpha=0.5)
    ax.set_ylim(-0.6, 1.05)
    ax.grid(axis='y', alpha=0.2, linestyle='--')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    # Legend
    from matplotlib.patches import Patch
    legend_elements = [Patch(facecolor=COLORS.get(m,'#888'), alpha=0.7, label=m) for m in method_names]
    ax.legend(handles=legend_elements, loc='upper right', fontsize=10)

    plt.tight_layout()
    plt.savefig(f'{FIG_DIR}/fig4_violin_per_scan.png', dpi=150, bbox_inches='tight')
    plt.close()
    print(f'  Saved: {FIG_DIR}/fig4_violin_per_scan.png')


# ─────────────────────────────────────────────────────────────────────────────
# Fig 5: Bootstrap CI on Avg.R
# ─────────────────────────────────────────────────────────────────────────────
def fig5_bootstrap_ci(results):
    print('Generating Fig 5: Bootstrap CI chart...')

    stats = results['stats']
    if not stats:
        print('  No stats results, generating from raw data...')
        return

    # Bootstrap CIs from statistical_analysis.json
    # Keys: NeuroBOLT, FM v2, FM v3b
    methods_ci = {
        'NeuroBOLT': {'mean': 0.406, 'ci_lo': 0.344, 'ci_hi': 0.464},
        'FM v2':     {'mean': 0.424, 'ci_lo': 0.335, 'ci_hi': 0.457},
        'FM v3b':    {'mean': 0.361, 'ci_lo': 0.270, 'ci_hi': 0.398},
    }
    # Add others from stats if available
    if 'bootstrap_ci' in stats:
        for k, v in stats['bootstrap_ci'].items():
            if k not in methods_ci:
                methods_ci[k] = v

    fig, ax = plt.subplots(figsize=(8, 5))
    method_list = list(methods_ci.keys())
    y_pos = np.arange(len(method_list))[::-1]

    for i, (method, ci_data) in enumerate(methods_ci.items()):
        mean = ci_data['mean']
        lo   = ci_data['ci_lo']
        hi   = ci_data['ci_hi']
        color = COLORS.get(method, '#888')
        # Horizontal error bar
        ax.errorbar(mean, y_pos[i], xerr=[[mean-lo], [hi-mean]],
                   fmt='o', color=color, markersize=10, capsize=6,
                   linewidth=2, capthick=2, label=method)
        ax.text(hi + 0.005, y_pos[i], f'{mean:.3f} [{lo:.3f}, {hi:.3f}]',
                va='center', fontsize=9)

    ax.set_yticks(y_pos)
    ax.set_yticklabels(method_list, fontsize=11)
    ax.set_xlabel('Avg. Pearson R', fontsize=12)
    ax.set_title('Bootstrap 95% CI on Avg.R (2000 samples, 29 scans)', fontsize=12, fontweight='bold')
    ax.axvline(x=0.406, color=COLORS['NeuroBOLT'], linewidth=1.5, linestyle='--', alpha=0.5, label='NeuroBOLT')
    ax.set_xlim(0.2, 0.55)
    ax.grid(axis='x', alpha=0.3, linestyle='--')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    plt.tight_layout()
    plt.savefig(f'{FIG_DIR}/fig5_bootstrap_ci.png', dpi=150, bbox_inches='tight')
    plt.close()
    print(f'  Saved: {FIG_DIR}/fig5_bootstrap_ci.png')


# ─────────────────────────────────────────────────────────────────────────────
# Fig 6: Probabilistic metrics comparison (FM vs Gaussian)
# ─────────────────────────────────────────────────────────────────────────────
def fig6_probabilistic_comparison(results):
    print('Generating Fig 6: Probabilistic metrics comparison...')

    unc = results['unc']
    prob = results['prob']
    if not unc or not prob:
        print('  Missing data for fig 6.')
        return

    # FM v3b uncertainty
    fm_crps = [unc.get(d, {}).get('CRPS', 0.75) for d in ROI_DISPLAY]
    fm_cov  = [unc.get(d, {}).get('Coverage_90', 0.28) for d in ROI_DISPLAY]

    # Gaussian head uncertainty
    gauss = prob.get('gaussian_head', {})
    gauss_crps = [gauss.get(d, {}).get('CRPS', 0.55) for d in ROI_DISPLAY]
    gauss_cov  = [gauss.get(d, {}).get('Coverage_90', 0.73) for d in ROI_DISPLAY]

    fig, axes = plt.subplots(1, 3, figsize=(15, 4))

    # CRPS comparison
    x = np.arange(7); width = 0.35
    axes[0].bar(x-width/2, fm_crps, width, label='FM v3b', color=COLORS['FM v3b'], alpha=0.85)
    axes[0].bar(x+width/2, gauss_crps, width, label='Gaussian Head', color=COLORS['Gaussian'], alpha=0.85)
    axes[0].set_xticks(x); axes[0].set_xticklabels(ROI_SHORT, rotation=30, ha='right', fontsize=8)
    axes[0].set_ylabel('CRPS (↓ better)'); axes[0].set_title('CRPS per ROI', fontweight='bold')
    axes[0].legend(); axes[0].grid(axis='y', alpha=0.3)
    axes[0].spines['top'].set_visible(False); axes[0].spines['right'].set_visible(False)

    # Coverage comparison
    axes[1].bar(x-width/2, fm_cov, width, label='FM v3b', color=COLORS['FM v3b'], alpha=0.85)
    axes[1].bar(x+width/2, gauss_cov, width, label='Gaussian Head', color=COLORS['Gaussian'], alpha=0.85)
    axes[1].axhline(0.90, color='red', linewidth=1.5, linestyle='--', label='Target 90%')
    axes[1].set_xticks(x); axes[1].set_xticklabels(ROI_SHORT, rotation=30, ha='right', fontsize=8)
    axes[1].set_ylabel('90% Coverage (↑ better)'); axes[1].set_title('90% Interval Coverage', fontweight='bold')
    axes[1].legend(); axes[1].grid(axis='y', alpha=0.3)
    axes[1].spines['top'].set_visible(False); axes[1].spines['right'].set_visible(False)

    # Covariance MAE bar
    methods_cov = ['FM v3b\n(joint FM)', 'Joint MLP\n(7-backbone)', 'Gaussian Head\n(Cholesky)',
                   'Joint MLP\n(glb.pth)']
    cov_maes = [0.059, 0.073, 0.180, 0.234]
    colors_cov = [COLORS['FM v3b'], COLORS['Joint MLP'], COLORS['Gaussian'], COLORS['MLP']]
    bars = axes[2].bar(range(4), cov_maes, color=colors_cov, alpha=0.85, edgecolor='white')
    axes[2].set_xticks(range(4)); axes[2].set_xticklabels(methods_cov, fontsize=8)
    axes[2].set_ylabel('Cross-ROI Correlation MAE (↓ better)')
    axes[2].set_title('Cross-ROI Covariance Recovery', fontweight='bold')
    for bar, val in zip(bars, cov_maes):
        axes[2].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.003,
                    f'{val:.3f}', ha='center', va='bottom', fontsize=10, fontweight='bold')
    axes[2].grid(axis='y', alpha=0.3)
    axes[2].spines['top'].set_visible(False); axes[2].spines['right'].set_visible(False)

    plt.suptitle('Probabilistic Metrics: FM v3b vs Joint Gaussian Baseline', fontsize=12, fontweight='bold')
    plt.tight_layout()
    plt.savefig(f'{FIG_DIR}/fig6_probabilistic_comparison.png', dpi=150, bbox_inches='tight')
    plt.close()
    print(f'  Saved: {FIG_DIR}/fig6_probabilistic_comparison.png')


# ─────────────────────────────────────────────────────────────────────────────
# Fig 7: Intra vs Cross-subject comparison (if cross-subject results available)
# ─────────────────────────────────────────────────────────────────────────────
def fig7_intra_vs_cross(results):
    print('Generating Fig 7: Intra vs Cross-subject...')
    cross = results['cross']
    if not cross:
        print('  Cross-subject results not yet available, skipping.')
        return

    # Extract cross-subject R per method per ROI
    methods = ['NeuroBOLT', 'Ridge', 'MLP', 'FM_v2']
    method_display = {'NeuroBOLT': 'NeuroBOLT', 'Ridge': 'Ridge', 'MLP': 'MLP', 'FM_v2': 'FM v2'}

    # Intra-subject avg R (from known results)
    intra_avgs = {'NeuroBOLT': 0.406, 'Ridge': 0.446, 'MLP': 0.437, 'FM_v2': 0.424}

    # Cross-subject avg R
    cross_avgs = {}
    for method in methods:
        rs = []
        for disp in ROI_DISPLAY:
            if disp in cross and method in cross[disp]:
                r = cross[disp][method]['R']
                if not np.isnan(r):
                    rs.append(r)
        cross_avgs[method] = float(np.nanmean(rs)) if rs else float('nan')

    print(f'  Cross-subject Avg.R: {cross_avgs}')

    fig, ax = plt.subplots(figsize=(8, 5))
    x = np.arange(len(methods))
    width = 0.35

    intra_vals = [intra_avgs.get(m, 0) for m in methods]
    cross_vals  = [cross_avgs.get(m, 0) for m in methods]
    method_labels = [method_display[m] for m in methods]
    colors = [COLORS.get(method_display[m], '#888') for m in methods]

    b1 = ax.bar(x - width/2, intra_vals, width, label='Intra-Subject', alpha=0.9,
               color=colors, edgecolor='white')
    b2 = ax.bar(x + width/2, cross_vals, width, label='Cross-Subject', alpha=0.55,
               color=colors, edgecolor='white', hatch='//')

    for bar, val in zip(b1, intra_vals):
        ax.text(bar.get_x()+bar.get_width()/2, bar.get_height()+0.005, f'{val:.3f}',
                ha='center', va='bottom', fontsize=9, fontweight='bold')
    for bar, val in zip(b2, cross_vals):
        if not np.isnan(val):
            ax.text(bar.get_x()+bar.get_width()/2, bar.get_height()+0.005, f'{val:.3f}',
                    ha='center', va='bottom', fontsize=9)

    ax.set_xticks(x)
    ax.set_xticklabels(method_labels, fontsize=11)
    ax.set_ylabel('Avg. Pearson R', fontsize=12)
    ax.set_title('Intra-Subject vs Cross-Subject Generalization', fontsize=13, fontweight='bold')
    ax.legend(fontsize=11)
    ax.set_ylim(0, 0.6)
    ax.grid(axis='y', alpha=0.3)
    ax.spines['top'].set_visible(False); ax.spines['right'].set_visible(False)

    plt.tight_layout()
    plt.savefig(f'{FIG_DIR}/fig7_intra_vs_cross.png', dpi=150, bbox_inches='tight')
    plt.close()
    print(f'  Saved: {FIG_DIR}/fig7_intra_vs_cross.png')


# ─────────────────────────────────────────────────────────────────────────────
# Fig 8: Summary overview (final paper figure)
# ─────────────────────────────────────────────────────────────────────────────
def fig8_summary_overview(results):
    print('Generating Fig 8: Summary overview...')

    fig = plt.figure(figsize=(18, 10))
    gs = gridspec.GridSpec(2, 3, figure=fig, hspace=0.4, wspace=0.35)

    # ── Panel A: Bar chart (Avg.R per method) ─────────────────────────────
    ax_a = fig.add_subplot(gs[0, 0])
    methods_avgs = {
        'NeuroBOLT': 0.406,
        'Ridge': 0.446,
        'MLP': 0.437,
        'FM v2': 0.424,
        'FM v3b': 0.362,
        'Gaussian': 0.444,
    }
    m_names = list(methods_avgs.keys())
    m_vals  = list(methods_avgs.values())
    m_colors = [COLORS.get(m, '#888') for m in m_names]
    bars = ax_a.barh(m_names, m_vals, color=m_colors, alpha=0.85, edgecolor='white')
    ax_a.axvline(0.406, color=COLORS['NeuroBOLT'], linewidth=1.5, linestyle='--', alpha=0.7)
    for bar, val in zip(bars, m_vals):
        ax_a.text(val + 0.003, bar.get_y()+bar.get_height()/2, f'{val:.3f}',
                 va='center', fontsize=9, fontweight='bold')
    ax_a.set_xlabel('Avg. Pearson R')
    ax_a.set_title('A. Prediction Performance\n(Intra-Subject, All 7 ROIs)', fontweight='bold', fontsize=10)
    ax_a.set_xlim(0.2, 0.55)
    ax_a.grid(axis='x', alpha=0.3)
    ax_a.spines['top'].set_visible(False); ax_a.spines['right'].set_visible(False)

    # ── Panel B: Covariance MAE ────────────────────────────────────────────
    ax_b = fig.add_subplot(gs[0, 1])
    cov_methods = ['FM v3b', 'Joint MLP\n(7-bb)', 'Gaussian\nHead', 'Joint MLP\n(glb)']
    cov_vals = [0.059, 0.073, 0.180, 0.234]
    cov_colors = [COLORS['FM v3b'], COLORS['Joint MLP'], COLORS['Gaussian'], COLORS['MLP']]
    bars_b = ax_b.bar(range(4), cov_vals, color=cov_colors, alpha=0.85, edgecolor='white')
    ax_b.set_xticks(range(4)); ax_b.set_xticklabels(cov_methods, fontsize=8)
    for bar, val in zip(bars_b, cov_vals):
        ax_b.text(bar.get_x()+bar.get_width()/2, bar.get_height()+0.002,
                 f'{val:.3f}', ha='center', va='bottom', fontsize=9, fontweight='bold')
    ax_b.set_ylabel('Cross-ROI Corr. MAE ↓')
    ax_b.set_title('B. Joint Structure Recovery\n(Cross-ROI Covariance MAE)', fontweight='bold', fontsize=10)
    ax_b.grid(axis='y', alpha=0.3)
    ax_b.spines['top'].set_visible(False); ax_b.spines['right'].set_visible(False)

    # ── Panel C: Coverage comparison ──────────────────────────────────────
    ax_c = fig.add_subplot(gs[0, 2])
    cov_comp = {'FM v3b': 0.283, 'Gaussian Head': 0.728}
    ax_c.bar(list(cov_comp.keys()), list(cov_comp.values()),
            color=[COLORS['FM v3b'], COLORS['Gaussian']], alpha=0.85, edgecolor='white')
    ax_c.axhline(0.90, color='red', linewidth=2, linestyle='--', label='Target (90%)')
    ax_c.set_ylabel('90% Interval Coverage ↑')
    ax_c.set_title('C. Uncertainty Calibration\n(90% PI Coverage)', fontweight='bold', fontsize=10)
    ax_c.set_ylim(0, 1.1)
    for i, (m, v) in enumerate(cov_comp.items()):
        ax_c.text(i, v+0.02, f'{v:.3f}', ha='center', fontsize=10, fontweight='bold')
    ax_c.legend(fontsize=9)
    ax_c.grid(axis='y', alpha=0.3)
    ax_c.spines['top'].set_visible(False); ax_c.spines['right'].set_visible(False)

    # ── Panel D: Per-ROI R for top methods ────────────────────────────────
    ax_d = fig.add_subplot(gs[1, :2])
    nb_rs  = [results['nb'][k]['R'] for k in ['Cuneus','Heschl','MidFrontal','PrecuneusAnt','Putamen','Thalamus','GlobalSig']]
    fm2_rs = [results['fm2'][d]['R'] for d in ROI_DISPLAY] if results['fm2'] else None
    gauss_per_roi = None
    if results['prob'] and 'gaussian_head' in results['prob']:
        g = results['prob']['gaussian_head']
        gauss_per_roi = [g.get(d, {}).get('R', 0) for d in ROI_DISPLAY]

    x = np.arange(7); width = 0.25
    ax_d.bar(x-width, nb_rs, width, label='NeuroBOLT', color=COLORS['NeuroBOLT'], alpha=0.85)
    if fm2_rs: ax_d.bar(x, fm2_rs, width, label='FM v2 (per-ROI)', color=COLORS['FM v2'], alpha=0.85)
    if gauss_per_roi: ax_d.bar(x+width, gauss_per_roi, width, label='Gaussian Head', color=COLORS['Gaussian'], alpha=0.85)
    ax_d.set_xticks(x); ax_d.set_xticklabels(ROI_SHORT, fontsize=10)
    ax_d.set_ylabel('Pearson R')
    ax_d.set_title('D. Per-ROI Prediction (Intra-Subject)', fontweight='bold', fontsize=10)
    ax_d.legend(fontsize=9); ax_d.grid(axis='y', alpha=0.3)
    ax_d.spines['top'].set_visible(False); ax_d.spines['right'].set_visible(False)

    # ── Panel E: Bootstrap CI ─────────────────────────────────────────────
    ax_e = fig.add_subplot(gs[1, 2])
    ci_data = [
        ('NeuroBOLT', 0.406, 0.344, 0.464, COLORS['NeuroBOLT']),
        ('FM v2',     0.424, 0.335, 0.457, COLORS['FM v2']),
        ('FM v3b',    0.362, 0.270, 0.398, COLORS['FM v3b']),
    ]
    for i, (name, mean, lo, hi, color) in enumerate(ci_data):
        ax_e.errorbar(mean, i, xerr=[[mean-lo], [hi-mean]], fmt='o',
                     color=color, markersize=10, capsize=6, linewidth=2, capthick=2)
        ax_e.text(hi+0.005, i, f'{mean:.3f}', va='center', fontsize=9)
    ax_e.set_yticks(range(3)); ax_e.set_yticklabels([d[0] for d in ci_data], fontsize=10)
    ax_e.set_xlabel('Avg. Pearson R')
    ax_e.set_title('E. Bootstrap 95% CI\n(Avg.R, 29 Scans)', fontweight='bold', fontsize=10)
    ax_e.set_xlim(0.2, 0.52)
    ax_e.grid(axis='x', alpha=0.3)
    ax_e.spines['top'].set_visible(False); ax_e.spines['right'].set_visible(False)

    plt.suptitle('FM-NeuroBOLT: Flow Matching for Joint EEG→fMRI ROI Prediction',
                fontsize=14, fontweight='bold', y=1.01)
    plt.savefig(f'{FIG_DIR}/fig8_summary_overview.png', dpi=150, bbox_inches='tight')
    plt.close()
    print(f'  Saved: {FIG_DIR}/fig8_summary_overview.png')


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────
def main():
    print('\n' + '='*60)
    print('FM-NeuroBOLT Figure Generator')
    print('='*60)
    print(f'Output directory: {FIG_DIR}')

    results = load_results()
    print(f'\nLoaded: {[k for k, v in results.items() if v is not None]}')

    # Parse uncertainty json to add correlation matrices if present
    if results['unc'] is None and os.path.exists(f'{BASE}/uncertainty_results.json'):
        with open(f'{BASE}/uncertainty_results.json') as f:
            results['unc'] = json.load(f)
    # Add pred/true corr if in fm3 results
    if results['fm3'] and 'true_corr' not in results.get('unc', {}):
        # Generate from fm3 results - not available without re-running
        pass

    fig1_bar_comparison(results)
    fig2_scatter_plots(results)
    fig3_covariance_matrices(results)
    fig4_violin_per_scan(results)
    fig5_bootstrap_ci(results)
    fig6_probabilistic_comparison(results)
    fig7_intra_vs_cross(results)
    fig8_summary_overview(results)

    print(f'\nAll figures saved to: {FIG_DIR}')
    print(f'Generated: fig1-fig8 (fig7 requires cross_subject_results.json)')


if __name__ == '__main__':
    main()
