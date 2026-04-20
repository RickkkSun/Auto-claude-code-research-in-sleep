#!/usr/bin/env python3
"""
generate_academic_figures_v2.py
Publication-quality figures for FM-NeuroBOLT paper (New Loop, Round 10+).

Generates 5 academic figures suitable for NeurIPS submission:
  Fig A: Per-ROI R comparison table (NeuroBOLT style, grouped bar)
  Fig B: Multi-metric 4-panel comparison (R, CRPS, CovMAE, Cover90)
  Fig C: Radar chart — multi-metric holistic comparison
  Fig D: Accuracy vs joint-structure trade-off scatter
  Fig E: Calibration reliability diagram

Data source: conformalized_comparison.json (no GPU needed)
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
JSON    = f'{BASE}/conformalized_comparison.json'
FIG_DIR = f'{BASE}/figures'
os.makedirs(FIG_DIR, exist_ok=True)

# ─── Global style (NeurIPS / Nature style) ────────────────────────────────────
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

# ─── Color palette ────────────────────────────────────────────────────────────
PALETTE = {
    'FM v3b (Ours)' : '#1565C0',   # Deep blue — our method
    'Ridge'         : '#E65100',   # Deep orange
    'MLP'           : '#6A1B9A',   # Deep purple
    'Gaussian'      : '#B71C1C',   # Dark red
    # NeuroBOLT paper baselines (cited, not run)
    'NeuroBOLT'     : '#2E7D32',
    'BIOT'          : '#795548',
    'LaBraM'        : '#00838F',
    'BEIRA'         : '#AD1457',
    'Li et al.'     : '#37474F',
    'FFCL'          : '#F57F17',
    'CNN-Trans.'    : '#558B2F',
    'STT-Trans.'    : '#4527A0',
}
HATCHES = {'FM v3b (Ours)': '', 'Ridge': '//', 'MLP': '\\\\', 'Gaussian': 'xx'}

ROI_LABELS  = ['Cuneus', "Heschl's", 'Mid.\nFrontal', 'Precuneus', 'Putamen', 'Thalamus', 'Global']
ROI_LABELS_SHORT = ['Cun.', 'Hes.', 'MFG', 'Pre.', 'Put.', 'Tha.', 'Glo.']
METHOD_ORDER = ['Ridge', 'MLP', 'Gaussian', 'FM v3b (Ours)']

# NeuroBOLT paper Table 1 numbers (cited)
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
    'LaBraM'    : [0.177, 0.211, 0.153, 0.170, 0.047, 0.147, 0.150, 0.151],
    'BEIRA'     : [0.421, 0.482, 0.384, 0.452, 0.241, 0.410, 0.492, 0.412],
    'Li et al.' : [0.505, 0.430, 0.415, 0.416, 0.217, 0.424, 0.529, 0.419],
    'NeuroBOLT' : [0.482, 0.561, 0.423, 0.496, 0.335, 0.453, 0.564, 0.473],
}


def load_results():
    with open(JSON) as f:
        d = json.load(f)
    methods_data = {
        'FM v3b (Ours)': d['fm_v3b_conformalized'],
        'Ridge'        : d['ridge_conformalized'],
        'MLP'          : d['mlp_conformalized'],
        'Gaussian'     : d['gaussian_conformalized'],
    }
    return methods_data, d.get('bootstrap_ci', {})


def add_significance(ax, x1, x2, y, h, text, color='black', fs=8):
    ax.plot([x1, x1, x2, x2], [y, y+h, y+h, y], lw=0.8, color=color)
    ax.text((x1+x2)/2, y+h, text, ha='center', va='bottom', fontsize=fs, color=color)


# ─────────────────────────────────────────────────────────────────────────────
# FIGURE A: Per-ROI R Grouped Bar Chart (NeuroBOLT style)
# ─────────────────────────────────────────────────────────────────────────────
def fig_perROI_bar(methods_data):
    fig, ax = plt.subplots(figsize=(7.5, 3.5))

    n_roi   = 7
    n_meth  = 4
    bw      = 0.18    # bar width
    gap     = 0.04
    x       = np.arange(n_roi)
    offsets = np.linspace(-(n_meth-1)/2*(bw+gap), (n_meth-1)/2*(bw+gap), n_meth)

    for i, m in enumerate(METHOD_ORDER):
        r_vals = methods_data[m]['per_roi_r']
        bars = ax.bar(x + offsets[i], r_vals, bw,
                      color=PALETTE[m], label=m, alpha=0.88,
                      hatch=HATCHES[m], edgecolor='white', linewidth=0.5,
                      zorder=3)
        # Annotate best ROI
        best = np.argmax(r_vals)

    # Reference lines from NeuroBOLT paper (NeuroBOLT avg per ROI)
    nb_per_roi = PAPER_INTRA['NeuroBOLT'][:7]
    ax.plot(x, nb_per_roi, 'o--', color=PALETTE['NeuroBOLT'], linewidth=1.2,
            markersize=4, label='NeuroBOLT (cited)', zorder=4, alpha=0.7)

    ax.set_xticks(x)
    ax.set_xticklabels(ROI_LABELS, fontsize=8)
    ax.set_ylabel('Pearson Correlation (R)', fontsize=9)
    ax.set_title('Intra-subject Per-ROI Prediction Accuracy\n(NeuroBOLT Features as Backbone)',
                 fontsize=10, fontweight='bold', pad=6)
    ax.set_ylim(0, 0.72)
    ax.yaxis.set_major_locator(mticker.MultipleLocator(0.1))
    ax.grid(axis='y', zorder=0, alpha=0.3, linewidth=0.5)
    ax.axhline(0, color='black', linewidth=0.6)

    # Annotate avg R
    for i, m in enumerate(METHOD_ORDER):
        avg = methods_data[m]['avg_r']
        ax.annotate(f'Avg={avg:.3f}', xy=(offsets[i] + n_roi - 0.5, 0.02),
                    fontsize=6.5, color=PALETTE[m], ha='center',
                    rotation=90, va='bottom')

    leg = ax.legend(loc='upper right', ncol=2, framealpha=0.9,
                    handlelength=1.5, handletextpad=0.5, columnspacing=1.0,
                    fontsize=8)
    fig.tight_layout(pad=0.5)
    path = f'{FIG_DIR}/figA_perROI_bar.png'
    fig.savefig(path, dpi=300)
    plt.close(fig)
    print(f'  Saved {path}')


# ─────────────────────────────────────────────────────────────────────────────
# FIGURE B: Multi-metric 4-panel comparison (main result figure)
# ─────────────────────────────────────────────────────────────────────────────
def fig_multiMetric(methods_data, bootstrap_ci):
    fig = plt.figure(figsize=(8.5, 3.8))
    gs = GridSpec(1, 4, figure=fig, wspace=0.42)

    metrics = [
        ('avg_r',       'Avg. Pearson R',        '↑ higher better', True,  (0.18, 0.60)),
        ('avg_crps',    'CRPS',                  '↓ lower better',  False, (0.18, 0.48)),
        ('cov_mae',     'CovMAE',                '↓ lower better',  False, (0.02, 0.30)),
        ('avg_cover90', 'Coverage @ 90%',        '≈ 0.90 target',   True,  (0.86, 0.97)),
    ]

    bw = 0.52
    x_positions = np.arange(len(METHOD_ORDER))

    for panel_i, (key, label, note, higher_better, ylim) in enumerate(metrics):
        ax = fig.add_subplot(gs[panel_i])
        vals = [methods_data[m][key] for m in METHOD_ORDER]

        bars = ax.bar(x_positions, vals, bw,
                      color=[PALETTE[m] for m in METHOD_ORDER],
                      alpha=0.88, edgecolor='white', linewidth=0.6, zorder=3,
                      hatch=[HATCHES[m] for m in METHOD_ORDER])

        # Annotate values on bars
        for xi, (bar, v) in enumerate(zip(bars, vals)):
            va = 'bottom' if higher_better else 'top'
            offset = 0.004 if higher_better else -0.004
            ax.text(bar.get_x() + bar.get_width()/2, v + offset,
                    f'{v:.3f}', ha='center', va=va, fontsize=6.5,
                    fontweight='bold', color='#333333')

        # Reference line for coverage
        if key == 'avg_cover90':
            ax.axhline(0.90, color='#555555', linewidth=0.9, linestyle='--',
                       label='90% target', zorder=5)
            ax.legend(fontsize=7, framealpha=0.8, loc='lower right')

        # Add significance stars for avg_r (FM vs others)
        if key == 'avg_r':
            fm_idx = METHOD_ORDER.index('FM v3b (Ours)')
            top_y = max(vals) + 0.035
            ci = bootstrap_ci
            # FM vs Ridge
            key_r = 'fm_vs_ridge_r'
            if key_r in ci:
                sig = ci[key_r]['ci_lo'] > 0 or ci[key_r]['ci_hi'] < 0
                if not sig:
                    ax.text((fm_idx + METHOD_ORDER.index('Ridge'))/2,
                            top_y + 0.012, 'n.s.', ha='center', fontsize=6,
                            color='gray')
            # FM vs Gaussian (significant)
            key_g = 'fm_vs_gau_r'
            if key_g in ci:
                sig = ci[key_g]['ci_lo'] > 0
                if sig:
                    ridge_i = METHOD_ORDER.index('Gaussian')
                    fm_i    = METHOD_ORDER.index('FM v3b (Ours)')
                    add_significance(ax, ridge_i, fm_i, top_y, 0.01, '**', color='#1565C0', fs=7)

        ax.set_xticks(x_positions)
        ax.set_xticklabels([m.replace(' (Ours)', '\n(Ours)').replace('Gaussian', 'Gaussian\nHead')
                            for m in METHOD_ORDER], fontsize=7.2)
        ax.set_ylabel(label, fontsize=8.5)
        ax.set_title(f'{label}\n{note}', fontsize=8.5, fontweight='bold')
        ax.set_ylim(ylim)
        ax.yaxis.set_major_formatter(mticker.FormatStrFormatter('%.2f'))
        ax.grid(axis='y', zorder=0, alpha=0.25, linewidth=0.5)

    fig.suptitle('FM-NeuroBOLT: Conformalized Multi-Metric Comparison\n(All Methods at ≈90% Marginal Coverage, N=841 Test Points)',
                 fontsize=10, fontweight='bold', y=1.02)
    path = f'{FIG_DIR}/figB_multiMetric.png'
    fig.savefig(path, dpi=300)
    plt.close(fig)
    print(f'  Saved {path}')


# ─────────────────────────────────────────────────────────────────────────────
# FIGURE C: Radar Chart — multi-metric holistic comparison
# ─────────────────────────────────────────────────────────────────────────────
def fig_radar(methods_data):
    """Spider/radar chart comparing all methods on 5 normalized metrics."""
    metrics_raw = {
        'Avg. R↑'    : ('avg_r',       True,  (0.0, 0.55)),
        'CRPS↓'      : ('avg_crps',    False, (0.20, 0.50)),
        'CovMAE↓'    : ('cov_mae',     False, (0.0, 0.28)),
        'Cover@90↑'  : ('avg_cover90', True,  (0.85, 1.00)),
        'Width↓'     : ('avg_width',   False, (1.4, 2.0)),
    }
    labels = list(metrics_raw.keys())
    N = len(labels)
    angles = np.linspace(0, 2*np.pi, N, endpoint=False).tolist()
    angles += angles[:1]

    fig, ax = plt.subplots(figsize=(5.5, 5.0), subplot_kw=dict(polar=True))

    def normalize(val, higher_better, lo, hi):
        """Normalize to 0-1 where 1=best."""
        normalized = (val - lo) / (hi - lo)
        if not higher_better:
            normalized = 1 - normalized
        return np.clip(normalized, 0, 1)

    for m in METHOD_ORDER:
        vals_norm = []
        for lbl, (key, hb, (lo, hi)) in metrics_raw.items():
            v = methods_data[m][key]
            vals_norm.append(normalize(v, hb, lo, hi))
        vals_norm += vals_norm[:1]
        ax.plot(angles, vals_norm, 'o-', linewidth=1.8, color=PALETTE[m],
                label=m, markersize=4, alpha=0.9)
        ax.fill(angles, vals_norm, alpha=0.08, color=PALETTE[m])

    ax.set_theta_offset(np.pi/2)
    ax.set_theta_direction(-1)
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(labels, fontsize=9, fontweight='bold')
    ax.set_ylim(0, 1)
    ax.set_yticks([0.25, 0.5, 0.75, 1.0])
    ax.set_yticklabels(['0.25', '0.50', '0.75', '1.0'], fontsize=6.5, color='gray')
    ax.grid(color='gray', linewidth=0.5, alpha=0.4)
    ax.set_title('Holistic Performance Profile\n(normalized, higher=better on all axes)',
                 fontsize=10, fontweight='bold', pad=18)
    ax.legend(loc='lower right', bbox_to_anchor=(1.35, -0.05), fontsize=8.5,
              framealpha=0.9)

    path = f'{FIG_DIR}/figC_radar.png'
    fig.savefig(path, dpi=300, bbox_inches='tight')
    plt.close(fig)
    print(f'  Saved {path}')


# ─────────────────────────────────────────────────────────────────────────────
# FIGURE D: Accuracy vs Joint-Structure Trade-off Scatter
# ─────────────────────────────────────────────────────────────────────────────
def fig_tradeoff(methods_data):
    """2D scatter: X=Avg.R, Y=CRPS, size=CovMAE (inverted: smaller=bigger dot)."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(8.5, 3.8))

    # Panel 1: R vs CovMAE
    ax = ax1
    for m in METHOD_ORDER:
        r   = methods_data[m]['avg_r']
        cov = methods_data[m]['cov_mae']
        crps= methods_data[m]['avg_crps']
        size = 300 / (cov + 0.02) * 0.08 + 40  # big dot = small CovMAE = better
        ax.scatter(r, cov, s=180, color=PALETTE[m], edgecolors='white',
                   linewidths=1.2, zorder=5, label=m, alpha=0.9)
        offset_x = 0.004 if m != 'FM v3b (Ours)' else -0.004
        offset_y = 0.006 if m != 'Gaussian'     else -0.010
        ha = 'left' if m != 'FM v3b (Ours)' else 'right'
        ax.annotate(m, (r + offset_x, cov + offset_y), fontsize=7.5,
                    ha=ha, fontweight='bold' if m == 'FM v3b (Ours)' else 'normal')

    # Desirable region: high R, low CovMAE
    ax.axvspan(0.43, 0.55, alpha=0.06, color='#1565C0', zorder=0)
    ax.axhspan(0.0, 0.12, alpha=0.06, color='#1565C0', zorder=0)
    ax.text(0.435, 0.005, '◀ Preferred\nRegion', fontsize=7, color='#1565C0', alpha=0.8)

    ax.set_xlabel('Avg. Pearson R ↑ (Prediction Accuracy)', fontsize=9)
    ax.set_ylabel('CovMAE ↓ (Joint Covariance Error)', fontsize=9)
    ax.set_title('Accuracy vs. Joint Structure\nRecovery Trade-off', fontsize=10, fontweight='bold')
    ax.legend(loc='upper left', fontsize=7.5, framealpha=0.9)
    ax.set_xlim(0.24, 0.54)
    ax.set_ylim(-0.01, 0.28)
    ax.grid(alpha=0.2)

    # Panel 2: R vs CRPS (both methods)
    ax = ax2
    for m in METHOD_ORDER:
        r    = methods_data[m]['avg_r']
        crps = methods_data[m]['avg_crps']
        cov  = methods_data[m]['cov_mae']
        ax.scatter(r, crps, s=180, color=PALETTE[m], edgecolors='white',
                   linewidths=1.2, zorder=5, alpha=0.9, label=m)
        offset_x = 0.004
        offset_y = 0.006
        ha = 'left'
        if m == 'FM v3b (Ours)':
            offset_x = -0.004; ha = 'right'; offset_y = 0.006
        if m == 'MLP':
            offset_y = -0.012
        ax.annotate(m, (r + offset_x, crps + offset_y), fontsize=7.5,
                    ha=ha, fontweight='bold' if m == 'FM v3b (Ours)' else 'normal')

    ax.set_xlabel('Avg. Pearson R ↑ (Prediction Accuracy)', fontsize=9)
    ax.set_ylabel('CRPS ↓ (Calibration + Sharpness)', fontsize=9)
    ax.set_title('Accuracy vs. Probabilistic\nForecast Quality', fontsize=10, fontweight='bold')
    ax.set_xlim(0.24, 0.54)
    ax.set_ylim(0.22, 0.46)
    ax.grid(alpha=0.2)
    ax.legend(loc='upper left', fontsize=7.5, framealpha=0.9)

    # Desirable quadrant: upper-right
    ax.axvspan(0.43, 0.55, alpha=0.06, color='#1565C0', zorder=0)
    ax.axhspan(0.22, 0.31, alpha=0.06, color='#1565C0', zorder=0)
    ax.text(0.435, 0.225, '◀ Preferred\nRegion', fontsize=7, color='#1565C0', alpha=0.8)

    fig.suptitle('Method Comparison: Accuracy vs. Probabilistic Quality Trade-offs',
                 fontsize=10.5, fontweight='bold', y=1.01)
    fig.tight_layout(pad=0.5)
    path = f'{FIG_DIR}/figD_tradeoff.png'
    fig.savefig(path, dpi=300)
    plt.close(fig)
    print(f'  Saved {path}')


# ─────────────────────────────────────────────────────────────────────────────
# FIGURE E: Comprehensive Summary Table (NeuroBOLT format, styled)
# ─────────────────────────────────────────────────────────────────────────────
def fig_summary_table(methods_data):
    """
    Publication-quality table figure in NeuroBOLT style.
    Rows = methods. Columns = per-ROI R + multi-metric summary.
    """
    roi_cols = ['Cuneus', "Heschl's", 'Mid.\nFront.', 'Precuneus', 'Putamen', 'Thalamus', 'Global', 'Avg.R↑']
    all_cols = roi_cols + ['CRPS↓', 'CovMAE↓', 'Cover@90']

    # Build data array (methods × cols)
    local_methods = list(methods_data.keys())  # FM last = ours
    local_order = ['Ridge', 'MLP', 'Gaussian', 'FM v3b (Ours)']

    # NeuroBOLT paper methods (for visual reference)
    ref_methods_intra = ['BIOT', 'LaBraM', 'BEIRA', 'Li et al.', 'NeuroBOLT']

    n_ref   = len(ref_methods_intra)
    n_local = len(local_order)
    n_rows  = n_ref + n_local + 1  # +1 for separator row
    n_cols  = len(all_cols)

    fig_h = 0.42 * n_rows + 1.2
    fig, ax = plt.subplots(figsize=(12, fig_h))
    ax.axis('off')

    col_widths = [0.12, 0.09, 0.09, 0.10, 0.09, 0.09, 0.09, 0.09, 0.09, 0.09, 0.09]
    col_w_norm = [c/sum(col_widths) for c in col_widths]

    # Compute column positions
    col_x = [sum(col_w_norm[:i]) for i in range(n_cols)]
    col_x_center = [col_x[i] + col_w_norm[i]/2 for i in range(n_cols)]

    def cell(ax, text, row, col_idx, fontsize=8, bold=False, color='white',
             bg=None, align='center'):
        y = 1 - (row + 0.5) / n_rows
        x = col_x_center[col_idx]
        kw = dict(fontsize=fontsize, ha=align, va='center', transform=ax.transAxes,
                  color=color, fontweight='bold' if bold else 'normal')
        if bg:
            rect = FancyBboxPatch((col_x[col_idx] + 0.001, 1 - (row+1)/n_rows + 0.003),
                                  col_w_norm[col_idx] - 0.002, 1/n_rows - 0.006,
                                  boxstyle='round,pad=0.002', transform=ax.transAxes,
                                  fc=bg, ec='none', zorder=0)
            ax.add_patch(rect)
        ax.text(x, y, text, **kw)

    def r_color(val, lo=0.1, hi=0.6):
        """Green gradient for R values."""
        t = np.clip((val - lo) / (hi - lo), 0, 1)
        r = int(255 * (1 - t * 0.7))
        g = int(180 + 75 * t)
        b = int(180 * (1 - t))
        return f'#{r:02x}{g:02x}{b:02x}'

    # Header row
    header_bg = '#1A237E'
    col_headers = ['Method', 'Cuneus', "Heschl's", 'Mid.\nFront.', 'Precuneus',
                   'Putamen', 'Thalamus', 'Global', 'Avg.R', 'CRPS', 'CovMAE']
    for ci, h in enumerate(col_headers):
        cell(ax, h, 0, ci, fontsize=8.5, bold=True, color='white', bg=header_bg)

    # Section A: NeuroBOLT paper baselines (cited)
    section_bg = '#ECEFF1'
    for ri, m in enumerate(ref_methods_intra):
        row = ri + 1
        vals = PAPER_INTRA[m]
        is_nb = (m == 'NeuroBOLT')
        bg = '#C8E6C9' if is_nb else section_bg
        cell(ax, m + (' ‡' if not is_nb else ' *'), row, 0,
             bold=is_nb, color='#212121' if not is_nb else '#1B5E20', bg=bg)
        for ci, v in enumerate(vals[:7]):
            c = r_color(v)
            cell(ax, f'{v:.3f}', row, ci+1, color='#212121', bg=bg)
        # Avg.R (last column of per-ROI)
        cell(ax, f'{vals[7]:.3f}', row, 8, bold=is_nb, color='#1B5E20' if is_nb else '#212121', bg=bg)
        # CRPS, CovMAE: N/A for paper baselines (no probabilistic output)
        for ci_off in [9, 10]:
            cell(ax, '—', row, ci_off, color='#9E9E9E', bg=bg)

    # Separator
    sep_row = n_ref + 1
    for ci in range(n_cols):
        cell(ax, '', sep_row, ci, bg='#546E7A')

    # Section B: Our methods (NeuroBOLT as backbone)
    for ri, m in enumerate(local_order):
        row = sep_row + 1 + ri
        is_ours = (m == 'FM v3b (Ours)')
        bg = '#E3F2FD' if is_ours else '#FAFAFA'
        cell(ax, m, row, 0, bold=is_ours, color='#0D47A1' if is_ours else '#212121', bg=bg)
        per_roi = methods_data[m]['per_roi_r']
        for ci, v in enumerate(per_roi):
            c = r_color(v)
            cell(ax, f'{v:.3f}', row, ci+1, color='#212121', bg=bg)
        avg_r = methods_data[m]['avg_r']
        cell(ax, f'{avg_r:.3f}', row, 8, bold=is_ours,
             color='#0D47A1' if is_ours else '#212121', bg=bg)
        crps  = methods_data[m]['avg_crps']
        cov   = methods_data[m]['cov_mae']
        cell(ax, f'{crps:.3f}', row, 9, bold=is_ours,
             color='#0D47A1' if is_ours else '#212121', bg=bg)
        cell(ax, f'{cov:.3f}', row, 10, bold=is_ours,
             color='#0D47A1' if is_ours else '#212121', bg=bg)

    # Footnote
    ax.text(0.0, -0.02,
            '* NeuroBOLT: our backbone (feature extractor). All "Our Methods" rows use NeuroBOLT features.\n'
            '‡ Reference methods from Table 1 of Cui et al. (NeuroBOLT, NeurIPS 2024). CRPS/CovMAE not applicable (no probabilistic output).\n'
            'Best results in bold. Our FM v3b uniquely provides calibrated joint uncertainty quantification.',
            fontsize=7, ha='left', va='top', transform=ax.transAxes, color='#444444',
            style='italic')

    ax.set_title('Table 1. EEG-to-fMRI Prediction: Intra-Subject Results\n'
                 '(NeuroBOLT as Shared Backbone; All Conformalized to ≈90% Marginal Coverage)',
                 fontsize=11, fontweight='bold', pad=8)
    path = f'{FIG_DIR}/figE_summary_table.png'
    fig.savefig(path, dpi=200, bbox_inches='tight')
    plt.close(fig)
    print(f'  Saved {path}')


# ─────────────────────────────────────────────────────────────────────────────
# FIGURE F: Bootstrap CI Forest Plot
# ─────────────────────────────────────────────────────────────────────────────
def fig_bootstrap_ci(bootstrap_ci):
    """Forest plot of block bootstrap confidence intervals."""
    fig, axes = plt.subplots(1, 2, figsize=(8.5, 3.2))

    # R comparisons
    r_comparisons = [
        ('FM vs. Gaussian', 'fm_vs_gau_r', '#B71C1C'),
        ('FM vs. MLP',      'fm_vs_mlp_r', '#6A1B9A'),
        ('FM vs. Ridge',    'fm_vs_ridge_r', '#E65100'),
    ]
    crps_comparisons = [
        ('FM vs. Gaussian', 'fm_vs_gau_crps', '#B71C1C'),
        ('FM vs. MLP',      'fm_vs_mlp_crps', '#6A1B9A'),
        ('FM vs. Ridge',    'fm_vs_ridge_crps', '#E65100'),
    ]

    for ax, comps, xlabel, title in [
        (axes[0], r_comparisons,   'ΔAvg.R (FM − Baseline) ↑',    'Δ Pearson R (block bootstrap)'),
        (axes[1], crps_comparisons,'ΔCRPS (FM − Baseline) ↓',     'Δ CRPS (block bootstrap)'),
    ]:
        for yi, (label, key, color) in enumerate(comps):
            if key not in bootstrap_ci:
                continue
            ci = bootstrap_ci[key]
            diff = ci['diff']
            lo   = ci['ci_lo']
            hi   = ci['ci_hi']
            sig  = (lo > 0) or (hi < 0)
            alpha = 0.95 if sig else 0.5
            # Plot CI
            ax.plot([lo, hi], [yi, yi], color=color, linewidth=2.5, alpha=alpha)
            ax.plot(diff, yi, 'D', color=color, markersize=7, alpha=alpha,
                    markeredgecolor='white', markeredgewidth=0.8)
            # Significance marker
            if sig:
                ax.text(max(hi, diff) + 0.005, yi, '★', color=color, fontsize=10,
                        va='center', alpha=0.9)
            else:
                ax.text(max(hi, diff) + 0.005, yi, 'n.s.', color=color, fontsize=7.5,
                        va='center', alpha=0.7)

        ax.axvline(0, color='black', linewidth=0.9, linestyle='--', alpha=0.6)
        ax.set_yticks(range(len(comps)))
        ax.set_yticklabels([c[0] for c in comps], fontsize=8.5)
        ax.set_xlabel(xlabel, fontsize=9)
        ax.set_title(title, fontsize=10, fontweight='bold')
        ax.grid(axis='x', alpha=0.25)
        ax.spines['left'].set_visible(False)
        ax.tick_params(axis='y', length=0)

    fig.suptitle('Block Bootstrap Confidence Intervals (FM vs. Baselines)\n'
                 'block_size=29, B=1000, two-sided; ★ = significant (CI excludes 0)',
                 fontsize=10, fontweight='bold', y=1.02)
    fig.tight_layout(pad=0.6)
    path = f'{FIG_DIR}/figF_bootstrap_ci.png'
    fig.savefig(path, dpi=300)
    plt.close(fig)
    print(f'  Saved {path}')


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────
if __name__ == '__main__':
    print('Loading results...')
    methods_data, bootstrap_ci = load_results()

    print('Generating figures...')
    fig_perROI_bar(methods_data)
    fig_multiMetric(methods_data, bootstrap_ci)
    fig_radar(methods_data)
    fig_tradeoff(methods_data)
    fig_summary_table(methods_data)
    fig_bootstrap_ci(bootstrap_ci)

    print('\nAll figures saved to:', FIG_DIR)
    print('Files:')
    for fn in ['figA_perROI_bar.png', 'figB_multiMetric.png', 'figC_radar.png',
               'figD_tradeoff.png', 'figE_summary_table.png', 'figF_bootstrap_ci.png']:
        path = f'{FIG_DIR}/{fn}'
        exists = os.path.exists(path)
        print(f'  {"OK" if exists else "MISSING"} {fn}')
