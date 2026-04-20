#!/usr/bin/env python3
"""
generate_academic_figures_v3.py
Updated figures for Round 2 of new loop.
Includes CNN head, Attention head, and joint coverage data.
"""

import json, os
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.ticker as mticker
from matplotlib.gridspec import GridSpec
from matplotlib.patches import FancyBboxPatch
import warnings
warnings.filterwarnings('ignore')

BASE    = 'C:/Users/PC/Auto-claude-code-research-in-sleep/neurobolt_fm'
CONF_JSON  = f'{BASE}/conformalized_comparison.json'
SCAN_JSON  = f'{BASE}/per_scan_results.json'
PHASEC_JSON= f'{BASE}/phase_c_results.json'
FIG_DIR = f'{BASE}/figures'
os.makedirs(FIG_DIR, exist_ok=True)

plt.rcParams.update({
    'font.family'       : 'DejaVu Sans',
    'font.size'         : 9,
    'axes.titlesize'    : 10,
    'axes.titleweight'  : 'bold',
    'axes.labelsize'    : 9,
    'xtick.labelsize'   : 8,
    'ytick.labelsize'   : 8,
    'legend.fontsize'   : 8,
    'legend.framealpha' : 0.9,
    'legend.edgecolor'  : '#CCCCCC',
    'figure.dpi'        : 180,
    'savefig.dpi'       : 300,
    'savefig.bbox'      : 'tight',
    'savefig.pad_inches': 0.05,
    'axes.spines.top'   : False,
    'axes.spines.right' : False,
    'axes.linewidth'    : 0.8,
    'xtick.major.size'  : 3,
    'ytick.major.size'  : 3,
    'xtick.major.width' : 0.8,
    'ytick.major.width' : 0.8,
    'grid.alpha'        : 0.3,
    'grid.linewidth'    : 0.5,
})

PALETTE = {
    'FM v3b (Ours)' : '#1565C0',
    'MDN (K=3)'     : '#00838F',
    'Ridge'         : '#E65100',
    'MLP'           : '#6A1B9A',
    'CNN Head'      : '#2E7D32',
    'Attn Head'     : '#AD1457',
    'Gaussian'      : '#B71C1C',
    'NeuroBOLT'     : '#37474F',
    'BIOT'          : '#795548',
    'LaBraM'        : '#0288D1',
    'BEIRA'         : '#F57F17',
    'Li et al.'     : '#558B2F',
    'FFCL'          : '#7B1FA2',
    'CNN-Trans.'    : '#1B5E20',
    'STT-Trans.'    : '#4E342E',
}

ROI_LABELS = ['Cuneus', "Heschl's", 'Mid.\nFront.', 'Precuneus', 'Putamen', 'Thalamus', 'Global']

PAPER_INTRA = {
    'BIOT'     : [0.531, 0.518, 0.490, 0.459, 0.410, 0.411, 0.493, 0.473],
    'LaBraM'   : [0.540, 0.519, 0.493, 0.490, 0.411, 0.449, 0.487, 0.484],
    'BEIRA'    : [0.357, 0.396, 0.294, 0.320, 0.234, 0.328, 0.456, 0.341],
    'Li et al.': [0.460, 0.515, 0.376, 0.457, 0.324, 0.398, 0.583, 0.445],
    'NeuroBOLT': [0.588, 0.566, 0.502, 0.559, 0.437, 0.480, 0.587, 0.531],
}
PAPER_INTER = {
    'FFCL'      : [0.326, 0.412, 0.327, 0.437, 0.243, 0.373, 0.512, 0.376],
    'CNN-Trans.' : [0.218, 0.412, 0.298, 0.316, 0.232, 0.180, 0.282, 0.273],
    'STT-Trans.' : [0.269, 0.188, 0.226, 0.280, 0.074, 0.142, 0.347, 0.218],
    'BIOT'      : [0.457, 0.512, 0.393, 0.445, 0.299, 0.413, 0.529, 0.435],
    'NeuroBOLT' : [0.482, 0.561, 0.423, 0.496, 0.335, 0.453, 0.564, 0.473],
}


def load_all_results():
    with open(CONF_JSON) as f:
        conf = json.load(f)
    with open(SCAN_JSON) as f:
        scan = json.load(f)

    phasec = {}
    if os.path.exists(PHASEC_JSON):
        with open(PHASEC_JSON) as f:
            phasec = json.load(f)

    # Merge all results
    methods = {
        'FM v3b (Ours)': {
            'per_roi_r': conf['fm_v3b_conformalized']['per_roi_r'],
            'avg_r'    : conf['fm_v3b_conformalized']['avg_r'],
            'avg_crps' : conf['fm_v3b_conformalized']['avg_crps'],
            'cov_mae'  : conf['fm_v3b_conformalized']['cov_mae'],
            'avg_cover90': conf['fm_v3b_conformalized']['avg_cover90'],
            'avg_width': conf['fm_v3b_conformalized']['avg_width'],
            'joint_cov': phasec.get('fm_v3b_joint_coverage', {}).get('joint_cov', None),
        },
        'Ridge': {
            'per_roi_r': conf['ridge_conformalized']['per_roi_r'],
            'avg_r'    : conf['ridge_conformalized']['avg_r'],
            'avg_crps' : conf['ridge_conformalized']['avg_crps'],
            'cov_mae'  : conf['ridge_conformalized']['cov_mae'],
            'avg_cover90': conf['ridge_conformalized']['avg_cover90'],
            'avg_width': conf['ridge_conformalized']['avg_width'],
            'joint_cov': phasec.get('ridge_joint_coverage', {}).get('joint_cov', None),
        },
        'MLP': {
            'per_roi_r': conf['mlp_conformalized']['per_roi_r'],
            'avg_r'    : conf['mlp_conformalized']['avg_r'],
            'avg_crps' : conf['mlp_conformalized']['avg_crps'],
            'cov_mae'  : conf['mlp_conformalized']['cov_mae'],
            'avg_cover90': conf['mlp_conformalized']['avg_cover90'],
            'avg_width': conf['mlp_conformalized']['avg_width'],
            'joint_cov': None,
        },
        'Gaussian': {
            'per_roi_r': conf['gaussian_conformalized']['per_roi_r'],
            'avg_r'    : conf['gaussian_conformalized']['avg_r'],
            'avg_crps' : conf['gaussian_conformalized']['avg_crps'],
            'cov_mae'  : conf['gaussian_conformalized']['cov_mae'],
            'avg_cover90': conf['gaussian_conformalized']['avg_cover90'],
            'avg_width': conf['gaussian_conformalized']['avg_width'],
            'joint_cov': None,
        },
        'CNN Head': {
            'per_roi_r': scan['intra_subject']['cnn']['per_roi_r'],
            'avg_r'    : scan['intra_subject']['cnn']['avg_r'],
            'avg_crps' : None, 'cov_mae': None, 'avg_cover90': None,
            'avg_width': None, 'joint_cov': None,
        },
        'Attn Head': {
            'per_roi_r': scan['intra_subject']['attn']['per_roi_r'],
            'avg_r'    : scan['intra_subject']['attn']['avg_r'],
            'avg_crps' : None, 'cov_mae': None, 'avg_cover90': None,
            'avg_width': None, 'joint_cov': None,
        },
    }
    if phasec.get('mdn_k3'):
        methods['MDN (K=3)'] = {
            'per_roi_r'  : phasec['mdn_k3']['per_roi_r'],
            'avg_r'      : phasec['mdn_k3']['avg_r'],
            'avg_crps'   : phasec['mdn_k3']['avg_crps'],
            'cov_mae'    : phasec['mdn_k3']['cov_mae'],
            'avg_cover90': phasec['mdn_k3']['avg_marginal_cov'],
            'avg_width'  : phasec['mdn_k3']['width'],
            'joint_cov'  : phasec['mdn_k3']['joint_cov'],
        }
    return methods, conf.get('bootstrap_ci', {}), phasec


# ─── Figure G: Full intra-subject per-ROI comparison (all 8 methods) ─────────
def fig_full_perROI(methods):
    # Only methods with per_roi_r
    point_order = ['Ridge', 'MLP', 'CNN Head', 'Attn Head']
    prob_order  = ['Gaussian', 'MDN (K=3)', 'FM v3b (Ours)'] if 'MDN (K=3)' in methods else \
                  ['Gaussian', 'FM v3b (Ours)']

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.0), gridspec_kw={'width_ratios': [4, 3]})

    n_roi = 7
    x = np.arange(n_roi)

    # Left panel: all methods per-ROI
    ax = axes[0]
    all_methods = point_order + prob_order
    n_m = len(all_methods)
    bw = 0.10
    offsets = np.linspace(-(n_m-1)/2 * bw * 1.1, (n_m-1)/2 * bw * 1.1, n_m)

    for i, m in enumerate(all_methods):
        if m not in methods: continue
        r_vals = methods[m]['per_roi_r']
        lw = 2.0 if m == 'FM v3b (Ours)' else 0.5
        ec = '#0D47A1' if m == 'FM v3b (Ours)' else 'white'
        ax.bar(x + offsets[i], r_vals, bw, color=PALETTE[m], label=m,
               alpha=0.88, edgecolor=ec, linewidth=lw, zorder=3)

    # NeuroBOLT paper reference
    nb_per_roi = PAPER_INTRA['NeuroBOLT'][:7]
    ax.plot(x, nb_per_roi, 's--', color='#37474F', linewidth=1.5,
            markersize=5, label='NeuroBOLT (cited)', zorder=5, alpha=0.7)

    ax.set_xticks(x)
    ax.set_xticklabels(ROI_LABELS, fontsize=8)
    ax.set_ylabel('Pearson R', fontsize=9)
    ax.set_ylim(0, 0.72)
    ax.set_title('(a) Intra-Subject Per-ROI Accuracy\n(NeuroBOLT as Feature Extractor)', fontsize=10, fontweight='bold')
    ax.legend(loc='upper right', ncol=2, fontsize=7, framealpha=0.9)
    ax.grid(axis='y', alpha=0.25)

    # Right panel: Avg.R bar chart (sorted)
    ax2 = axes[1]
    # Sort by Avg.R
    sorted_methods = sorted(all_methods, key=lambda m: methods[m]['avg_r'] if m in methods else 0)
    sorted_methods = [m for m in sorted_methods if m in methods]
    avg_rs = [methods[m]['avg_r'] for m in sorted_methods]
    colors = [PALETTE[m] for m in sorted_methods]
    bars = ax2.barh(range(len(sorted_methods)), avg_rs, color=colors, alpha=0.88,
                    edgecolor='white', linewidth=0.5, height=0.65)
    for bi, (bar, v) in enumerate(zip(bars, avg_rs)):
        ax2.text(v + 0.004, bi, f'{v:.3f}', va='center', fontsize=8,
                 fontweight='bold' if sorted_methods[bi] == 'FM v3b (Ours)' else 'normal')
    ax2.axvline(PAPER_INTRA['NeuroBOLT'][7], color='#37474F', linewidth=1.2,
                linestyle='--', alpha=0.6, label=f'NeuroBOLT cited ({PAPER_INTRA["NeuroBOLT"][7]:.3f})')
    ax2.set_yticks(range(len(sorted_methods)))
    ax2.set_yticklabels(sorted_methods, fontsize=8.5)
    ax2.set_xlabel('Avg. Pearson R', fontsize=9)
    ax2.set_title('(b) Avg. R Ranking', fontsize=10, fontweight='bold')
    ax2.set_xlim(0, 0.60)
    ax2.legend(fontsize=7, loc='lower right')
    ax2.grid(axis='x', alpha=0.25)

    fig.suptitle('EEG-to-fMRI Prediction: Intra-Subject Point Prediction Accuracy\n'
                 '(All methods use NeuroBOLT features; conformalized to ≈90% marginal coverage)',
                 fontsize=10.5, fontweight='bold', y=1.01)
    fig.tight_layout(pad=0.5)
    path = f'{FIG_DIR}/figG_full_perROI.png'
    fig.savefig(path, dpi=300)
    plt.close(fig)
    print(f'  Saved {path}')


# ─── Figure H: Probabilistic metrics — point + probabilistic methods ──────────
def fig_probabilistic_comparison(methods):
    prob_methods = ['Gaussian', 'MDN (K=3)', 'Ridge', 'MLP', 'FM v3b (Ours)']
    prob_methods = [m for m in prob_methods if m in methods and methods[m]['avg_crps'] is not None]

    n_m = len(prob_methods)
    metrics = [
        ('avg_r',    'Avg. R', '↑', (0.25, 0.54)),
        ('avg_crps', 'CRPS',   '↓', (0.20, 0.46)),
        ('cov_mae',  'CovMAE', '↓', (0.0, 0.28)),
        ('joint_cov','Joint Coverage', '≈ target', (0.0, 1.05)),
    ]

    fig, axes = plt.subplots(1, 4, figsize=(11, 3.6))
    bw = 0.6

    for ax, (key, label, arrow, (ylo, yhi)) in zip(axes, metrics):
        vals = [methods[m][key] for m in prob_methods]
        has_vals = [v is not None for v in vals]
        colors = [PALETTE[m] for m in prob_methods]

        bars = ax.bar(range(n_m), [v if v is not None else 0 for v in vals],
                      bw, color=colors, alpha=0.88, edgecolor='white', linewidth=0.5, zorder=3)

        # Hatching for FM
        for bi, (bar, m) in enumerate(zip(bars, prob_methods)):
            if m == 'FM v3b (Ours)':
                bar.set_edgecolor('#0D47A1')
                bar.set_linewidth(2.0)

        for bi, (bar, v, hv) in enumerate(zip(bars, vals, has_vals)):
            if hv and v is not None:
                ax.text(bar.get_x() + bar.get_width()/2,
                        v + (yhi - ylo) * 0.015 * (1 if arrow == '↑' else 1),
                        f'{v:.3f}', ha='center', va='bottom', fontsize=7,
                        fontweight='bold' if prob_methods[bi] == 'FM v3b (Ours)' else 'normal')
            elif not hv:
                ax.text(bar.get_x() + bar.get_width()/2, ylo + (yhi-ylo)*0.05,
                        'N/A', ha='center', va='bottom', fontsize=7, color='gray')

        # Joint coverage target line
        if key == 'joint_cov':
            ax.axhline(0.90**7, color='red', linewidth=1.0, linestyle=':',
                       label='Indep. bound (0.90^7)', alpha=0.7)
            ax.axhline(0.90, color='green', linewidth=1.0, linestyle='--',
                       label='Marginal target', alpha=0.7)
            ax.legend(fontsize=6.5, loc='upper left')

        ax.set_xticks(range(n_m))
        ax.set_xticklabels([m.replace(' (Ours)', '\n(Ours)').replace(' (K=3)', '\n(K=3)')
                            for m in prob_methods], fontsize=7.5)
        ax.set_ylabel(label, fontsize=9)
        ax.set_title(f'{label} {arrow}', fontsize=9.5, fontweight='bold')
        ax.set_ylim(ylo, yhi)
        ax.grid(axis='y', alpha=0.25, zorder=0)
        ax.yaxis.set_major_formatter(mticker.FormatStrFormatter('%.2f'))

    fig.suptitle('Probabilistic EEG-to-fMRI Forecasting: Multi-Metric Comparison\n'
                 '(Marginally Calibrated Joint Predictive Distributions)',
                 fontsize=10.5, fontweight='bold', y=1.01)
    fig.tight_layout(pad=0.4)
    path = f'{FIG_DIR}/figH_probabilistic.png'
    fig.savefig(path, dpi=300)
    plt.close(fig)
    print(f'  Saved {path}')


# ─── Figure I: Sample sensitivity for FM ─────────────────────────────────────
def fig_sample_sensitivity(phasec):
    if not phasec.get('fm_sensitivity'):
        print('  No FM sensitivity data, skipping figI.')
        return

    sens = phasec['fm_sensitivity']
    ns_vals = sorted([int(k) for k in sens.keys()])

    fig, axes = plt.subplots(1, 3, figsize=(9, 3.2))
    metrics = [
        ('avg_r',    'Avg. Pearson R', '↑ higher better'),
        ('avg_crps', 'CRPS',           '↓ lower better'),
        ('cov_mae',  'CovMAE',         '↓ lower better'),
    ]

    for ax, (key, label, note) in zip(axes, metrics):
        vals = [sens[str(n)][key] for n in ns_vals]
        ax.plot(ns_vals, vals, 'o-', color='#1565C0', linewidth=2,
                markersize=6, markeredgecolor='white', markeredgewidth=1.2)
        for n, v in zip(ns_vals, vals):
            ax.annotate(f'{v:.3f}', (n, v), textcoords='offset points',
                        xytext=(3, 4), fontsize=7.5)
        ax.set_xlabel('Number of FM Samples', fontsize=9)
        ax.set_ylabel(label, fontsize=9)
        ax.set_title(f'{label}\n{note}', fontsize=9.5, fontweight='bold')
        ax.set_xticks(ns_vals)
        ax.grid(alpha=0.25)
        # Shade stable region
        if len(vals) >= 3:
            ax.axvspan(50, max(ns_vals), alpha=0.08, color='#1565C0')
            ax.text(50, min(vals) + (max(vals)-min(vals))*0.1, 'Stable',
                    fontsize=8, color='#1565C0', alpha=0.7)

    fig.suptitle('FM Sample Count Sensitivity Analysis\n'
                 '(Convergence of key metrics with number of ODE samples)',
                 fontsize=10.5, fontweight='bold', y=1.02)
    fig.tight_layout(pad=0.5)
    path = f'{FIG_DIR}/figI_sample_sensitivity.png'
    fig.savefig(path, dpi=300)
    plt.close(fig)
    print(f'  Saved {path}')


# ─── Figure J: Joint vs marginal coverage comparison ──────────────────────────
def fig_joint_vs_marginal(methods, phasec):
    # Collect joint + marginal coverage data
    data = {}
    if phasec.get('fm_v3b_joint_coverage'):
        data['FM v3b'] = {
            'marginal': phasec['fm_v3b_joint_coverage']['avg_marginal_cov'],
            'joint': phasec['fm_v3b_joint_coverage']['joint_cov'],
            'marginal_per_roi': phasec['fm_v3b_joint_coverage']['marginal_cov_per_roi'],
        }
    if phasec.get('ridge_joint_coverage'):
        data['Ridge'] = {
            'marginal': phasec['ridge_joint_coverage']['avg_marginal_cov'],
            'joint': phasec['ridge_joint_coverage']['joint_cov'],
            'marginal_per_roi': phasec['ridge_joint_coverage']['marginal_cov_per_roi'],
        }
    if phasec.get('mdn_k3') and 'joint_cov' in phasec['mdn_k3']:
        data['MDN (K=3)'] = {
            'marginal': phasec['mdn_k3']['avg_marginal_cov'],
            'joint': phasec['mdn_k3']['joint_cov'],
            'marginal_per_roi': None,
        }

    if not data:
        print('  No joint coverage data, skipping figJ.')
        return

    n_m = len(data)
    method_names = list(data.keys())

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(9, 3.5))

    # Panel 1: marginal vs joint coverage bar comparison
    x = np.arange(n_m)
    bw = 0.3
    margs = [data[m]['marginal'] for m in method_names]
    joints = [data[m]['joint'] for m in method_names]
    colors = [PALETTE.get(m.split(' v')[0].strip(), '#607D8B') for m in method_names]

    ax1.bar(x - bw/2, margs, bw, label='Marginal (per-ROI)', alpha=0.85,
            color=colors, edgecolor='white')
    ax1.bar(x + bw/2, joints, bw, label='Joint (all 7 ROIs)', alpha=0.85,
            color=colors, edgecolor='white', hatch='//')

    ax1.axhline(0.90, color='gray', linestyle='--', linewidth=1.0, label='90% target')
    ax1.axhline(0.90**7, color='red', linestyle=':', linewidth=1.0,
                label=f'Indep. bound ({0.90**7:.2f})')
    ax1.set_xticks(x)
    ax1.set_xticklabels(method_names, fontsize=9)
    ax1.set_ylabel('Coverage', fontsize=9)
    ax1.set_title('(a) Marginal vs. Joint Coverage', fontsize=10, fontweight='bold')
    ax1.set_ylim(0, 1.05)
    ax1.legend(fontsize=7.5, framealpha=0.9)
    ax1.grid(axis='y', alpha=0.25)

    # Add value labels
    for xi, (m, vbar, vjoint) in enumerate(zip(method_names, margs, joints)):
        ax1.text(xi - bw/2, vbar + 0.01, f'{vbar:.2f}', ha='center', va='bottom', fontsize=7.5)
        ax1.text(xi + bw/2, vjoint + 0.01, f'{vjoint:.2f}', ha='center', va='bottom', fontsize=7.5)

    # Panel 2: per-ROI marginal coverage (FM v3b)
    if data.get('FM v3b', {}).get('marginal_per_roi'):
        roi_labels_short = ['Cun.', 'Hes.', 'MFG', 'Pre.', 'Put.', 'Tha.', 'Glo.']
        roi_covs = data['FM v3b']['marginal_per_roi']
        colors2 = ['#4CAF50' if c >= 0.90 else '#FF5722' for c in roi_covs]
        ax2.bar(range(7), roi_covs, color=colors2, alpha=0.88, edgecolor='white')
        ax2.axhline(0.90, color='gray', linestyle='--', linewidth=1.2,
                    label='90% target', zorder=5)
        ax2.set_xticks(range(7))
        ax2.set_xticklabels(roi_labels_short, fontsize=8.5)
        ax2.set_ylabel('Marginal Coverage', fontsize=9)
        ax2.set_title('(b) FM v3b Per-ROI Marginal Coverage', fontsize=10, fontweight='bold')
        ax2.set_ylim(0.80, 1.02)
        ax2.legend(fontsize=8)
        ax2.grid(axis='y', alpha=0.25)
        for xi, v in enumerate(roi_covs):
            ax2.text(xi, v + 0.003, f'{v:.3f}', ha='center', va='bottom', fontsize=7.5)

    fig.suptitle('Calibration Analysis: Marginal vs. Joint Predictive Coverage\n'
                 '(Split conformal calibration guarantees marginal, not joint, coverage)',
                 fontsize=10.5, fontweight='bold', y=1.01)
    fig.tight_layout(pad=0.5)
    path = f'{FIG_DIR}/figJ_joint_coverage.png'
    fig.savefig(path, dpi=300)
    plt.close(fig)
    print(f'  Saved {path}')


if __name__ == '__main__':
    print('Loading results...')
    methods, bootstrap_ci, phasec = load_all_results()

    print('Generating figures...')
    fig_full_perROI(methods)
    fig_probabilistic_comparison(methods)
    fig_sample_sensitivity(phasec)
    fig_joint_vs_marginal(methods, phasec)

    print('\nAll done. Figures in:', FIG_DIR)
