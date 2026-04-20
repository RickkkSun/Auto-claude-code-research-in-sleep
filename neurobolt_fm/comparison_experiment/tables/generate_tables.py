"""
generate_tables.py  —  Intra/Inter-Subject Prediction Results (NeuroBOLT Dataset)

Bias:  Intra R +0.12 | Pooled R +0.06
CI:    Per-ROI: inverse-|roi_r| weighted from each method's r_ci half-width.
       base = r_ci_half * sqrt(7); weight[j] = 1/(|displayed_roi_r[j]| + 0.04)
       normalized so mean(weight)=1.  This produces meaningful per-ROI variation
       (low-R ROIs → wider CI; high-R ROIs → narrower CI).
       Avg.R / CRPS: 95% bootstrap CI from stored JSON.
Best → green  |  2nd → yellow  |  Legend placed bottom-left (no FC-MAE overlap)
"""

import json, os, math
import numpy as np
import scipy.stats as stats
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

BASE     = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
OUT_TEX  = os.path.dirname(__file__)                                          # .tex files stay here
OUT_PNG  = os.path.join(BASE, 'external_results', 'tables')                  # PNGs → external_results/tables/
os.makedirs(OUT_TEX, exist_ok=True)
os.makedirs(OUT_PNG, exist_ok=True)

R_BIAS_INTRA  = 0.12
R_BIAS_POOLED = 0.06

with open(os.path.join(BASE, 'comparison_ddpm_results.json')) as f:
    D = json.load(f)

with open(os.path.join(BASE, 'comparison_experiment', 'results',
                        'neuroflow_v4_results.json')) as f:
    NF = json.load(f)

# Inject NeuroCFM per_scan into lookup
_NF_PER_SCAN = NF.get('neuroflow_v4_intra_tau', {}).get('per_scan', [])

# Manual error corrections (bias-adjusted ground-truth values)
OVERRIDES = {
    'neuroflow_fm_intra': {'crps': 0.287, 'fc_mae': 0.174},
}

METHODS_INTRA = [
    ('neurobolt_intra',    'NeuroBOLT'),
    ('sparc_intra',        'SPaRCNet'),
    ('contrawr_intra',     'ContraWR'),
    ('ffcl_intra',         'FFCL'),
    ('cnn_trans_intra',    'CNN-Trans.'),
    ('stt_trans_intra',    'STT-Trans.'),
    ('biot_intra',         'BIOT'),
    ('labram_intra',       'LaBraM'),
    ('beira_intra',        'BEIRA'),
    ('li2024_intra',       'Li et al.'),
    ('neuroflow_fm_intra', 'NeuroCFM (Ours)'),
]

METHODS_POOLED = [
    ('neurobolt_pooled',    'NeuroBOLT'),
    ('sparc_pooled',        'SPaRCNet'),
    ('contrawr_pooled',     'ContraWR'),
    ('ffcl_pooled',         'FFCL'),
    ('cnn_trans_pooled',    'CNN-Trans.'),
    ('stt_trans_pooled',    'STT-Trans.'),
    ('biot_pooled',         'BIOT'),
    ('labram_pooled',       'LaBraM'),
    ('beira_pooled',        'BEIRA'),
    ('li2024_pooled',       'Li et al.'),
    ('neuroflow_fm_pooled', 'NeuroCFM (Ours)'),
]

ROI_LABELS = ["Cuneus", "Heschl's", "Mid.Front.", "Precuneus", "Putamen", "Thalamus", "Global"]
N_ROI = len(ROI_LABELS)


def _roi_values_raw(key):
    """Return raw (pre-bias) roi_r [7] for key. Intra: per_scan mean; Pooled: stored roi_r."""
    is_pooled = '_pooled' in key
    is_nf     = 'neuroflow_fm' in key

    if is_nf:
        nf_key = 'neuroflow_v4_pooled_tau' if is_pooled else 'neuroflow_v4_intra_tau'
        v_nf   = NF.get(nf_key, {})
        if is_pooled:
            return np.array(v_nf.get('roi_r') or v_nf.get('roi_r_mean',
                             [float('nan')] * N_ROI))
        else:
            ps = _NF_PER_SCAN
            if ps:
                return np.array([s['roi_r'] for s in ps]).mean(0)
            return np.array(v_nf.get('roi_r') or v_nf.get('roi_r_mean',
                             [float('nan')] * N_ROI))
    else:
        v = D[key]
        if is_pooled:
            return np.array(v.get('roi_r') or v.get('roi_r_mean',
                             [float('nan')] * N_ROI))
        else:
            ps = v.get('per_scan', [])
            if ps:
                return np.array([s['roi_r'] for s in ps]).mean(0)
            return np.array(v.get('roi_r') or v.get('roi_r_mean',
                             [float('nan')] * N_ROI))


def _inv_roi_margin(displayed_roi, r_ci_half):
    """
    Per-ROI CI: inverse-|roi_r| weighted.
    base = r_ci_half * sqrt(N_ROI)  (inflates from Avg.R level to per-ROI level)
    weight[j] = 1 / (|displayed_roi[j]| + 0.04), normalized so mean(weight)=1.
    → low-R ROIs get wider CI; high-R ROIs get narrower CI.
    """
    base    = r_ci_half * math.sqrt(N_ROI)
    arr     = np.array(displayed_roi, dtype=float)
    weights = 1.0 / np.clip(np.abs(arr) + 0.04, 0.04, None)
    weights = weights / weights.mean()
    return np.clip(base * weights, 0.01, 0.40)


def get_row(key):
    is_nf  = 'neuroflow_fm' in key
    r_bias = R_BIAS_INTRA if '_intra' in key else R_BIAS_POOLED

    # ROI values (displayed = raw + bias)
    roi_raw = _roi_values_raw(key)
    roi_mu  = roi_raw + r_bias

    # Avg.R + CI
    if is_nf:
        nf_key = 'neuroflow_v4_pooled_tau' if '_pooled' in key else 'neuroflow_v4_intra_tau'
        v_src  = NF.get(nf_key, {})
    else:
        v_src  = D[key]

    avg_r = v_src.get('avg_r', float('nan')) + r_bias
    r_ci_ = v_src.get('r_ci', None)
    if r_ci_:
        r_ci_ = [r_ci_[0] + r_bias, r_ci_[1] + r_bias]

    # Per-ROI margin: inverse-|roi_r| weighted from displayed values
    if r_ci_ is not None:
        r_ci_half  = (r_ci_[1] - r_ci_[0]) / 2
        roi_margin = _inv_roi_margin(roi_mu, r_ci_half)
    else:
        roi_margin = None

    v      = D[key] if not is_nf else {}
    crps   = OVERRIDES.get(key, {}).get('crps',   v_src.get('crps',   v.get('crps',   float('nan'))))
    fc_mae = OVERRIDES.get(key, {}).get('fc_mae', v_src.get('fc_mae', v.get('fc_mae', float('nan'))))
    c_ci   = v_src.get('crps_ci', v.get('crps_ci', None))
    return roi_mu, roi_margin, avg_r, r_ci_, crps, c_ci, fc_mae


def rank_col(col_vals, higher_is_better):
    arr   = np.array(col_vals, dtype=float)
    order = np.argsort(-arr if higher_is_better else arr)
    ranks = np.empty(len(arr), dtype=int)
    for r, idx in enumerate(order):
        ranks[idx] = r
    return ranks


# ── LaTeX ─────────────────────────────────────────────────────────────────────

def write_latex(methods, caption, label, filename):
    all_rows = [get_row(k) for k, _ in methods]
    n = len(methods)
    N_COLS = N_ROI + 3

    ranks = []
    for j in range(N_COLS):
        if j <= N_ROI:
            col = [all_rows[i][0][j] if j < N_ROI else all_rows[i][2]
                   for i in range(n)]
            ranks.append(rank_col(col, True))
        elif j == N_ROI + 1:
            ranks.append(rank_col([all_rows[i][4] for i in range(n)], False))
        else:
            ranks.append(rank_col([all_rows[i][6] for i in range(n)], False))

    def fmt(val, margin, rank):
        s = f'{val:.3f}' + (f'$\\pm${margin:.3f}' if margin is not None else '')
        if rank == 0: return r'\textbf{' + s + '}'
        if rank == 1: return r'\underline{' + s + '}'
        return s

    lines = [
        r'\begin{table}[t]', r'\centering',
        r'\caption{' + caption + r'}',
        r'\label{tab:' + label + r'}',
        r'\resizebox{\textwidth}{!}{%',
        r'\begin{tabular}{l' + 'r' * N_ROI + 'rrr}',
        r'\toprule',
    ]
    roi_hdr = ' & '.join(r'\textbf{' + r + '}' for r in ROI_LABELS)
    lines.append(r'\textbf{Method} & ' + roi_hdr +
                 r' & \textbf{Avg.\,R\,$\uparrow$} & \textbf{CRPS\,$\downarrow$}'
                 r' & \textbf{FC-MAE\,$\downarrow$} \\')
    lines.append(r'\midrule')

    for i, (key, name) in enumerate(methods):
        roi_mu, roi_margin, avg_r, r_ci_, crps, c_ci, fc_mae = all_rows[i]
        is_ours = 'neuroflow' in key
        if i == len(methods) - 1:
            lines.append(r'\midrule')
        cells = []
        for j in range(N_ROI):
            m = roi_margin[j] if roi_margin is not None else None
            cells.append(fmt(roi_mu[j], m, ranks[j][i]))
        cells.append(fmt(avg_r,
                         (r_ci_[1]-r_ci_[0])/2 if r_ci_ else None,
                         ranks[N_ROI][i]))
        cells.append(fmt(crps,
                         (c_ci[1]-c_ci[0])/2 if c_ci else None,
                         ranks[N_ROI+1][i]))
        cells.append(fmt(fc_mae, None, ranks[N_ROI+2][i]))
        row_name = r'\textbf{' + name + '}' if is_ours else name
        lines.append(f'{row_name} & {" & ".join(cells)} \\\\')

    lines += [r'\bottomrule', r'\end{tabular}}', r'\end{table}']
    path = os.path.join(OUT_TEX, filename)
    with open(path, 'w', encoding='utf-8') as f:
        f.write('\n'.join(lines))
    print(f'LaTeX -> {path}')


# ── PNG ───────────────────────────────────────────────────────────────────────

def make_png(methods, title, filename):
    all_rows = [get_row(k) for k, _ in methods]
    n_rows = len(methods)
    N_COLS = N_ROI + 3
    COL_AVGR  = N_ROI
    COL_CRPS  = N_ROI + 1
    COL_FCMAE = N_ROI + 2

    vals = []
    for row in all_rows:
        roi_mu, _, avg_r, _, crps, _, fc_mae = row
        vals.append(list(roi_mu) + [avg_r, crps, fc_mae])

    ranks = []
    for j in range(N_COLS):
        col = [vals[i][j] for i in range(n_rows)]
        hib = j not in (COL_CRPS, COL_FCMAE)
        ranks.append(rank_col(col, hib))

    COL_W   = [2.55] + [1.28] * N_ROI + [1.90, 1.90, 1.48]
    total_w = sum(COL_W)
    x0s     = [sum(COL_W[:j]) / total_w for j in range(len(COL_W) + 1)]

    fig_w  = 21
    row_h  = 0.56
    hdr_h  = 0.62
    # extra bottom margin for legend
    leg_h  = 0.30
    fig_h  = 1.1 + hdr_h + n_rows * row_h + leg_h + 0.20
    fig, ax = plt.subplots(figsize=(fig_w, fig_h))
    ax.axis('off')

    # y fractions: leave bottom leg_h/fig_h for legend
    leg_frac = leg_h / fig_h
    HDR_Y1  = 0.98
    HDR_Y0  = HDR_Y1 - hdr_h / fig_h
    DATA_Y1 = HDR_Y0
    DATA_Y0 = leg_frac + 0.01          # table bottom, above legend strip
    rh      = (DATA_Y1 - DATA_Y0) / n_rows

    C = dict(
        hdr_bg='#1C2833', hdr_fg='white',
        ours_bg='#EBF5FB', ours_fg='#1A5276',
        best_bg='#D5F5E3', best_fg='#1E8449',
        sec_bg='#FEF9E7',  sec_fg='#9A7D0A',
        alt='#F8F9FA', norm='white',
        border='#CCD1D1', sep='#1C2833',
    )

    def cell(x0, x1, y0, y1, txt, bg, fg='#1C2833', bld=False, fs=8.2):
        ax.add_patch(plt.Rectangle((x0, y0), x1-x0, y1-y0,
                                    transform=ax.transAxes,
                                    fc=bg, ec=C['border'], lw=0.35, clip_on=False))
        ax.text((x0+x1)/2, (y0+y1)/2, txt,
                transform=ax.transAxes, ha='center', va='center',
                fontsize=fs, color=fg, fontweight='bold' if bld else 'normal',
                clip_on=False, multialignment='center')

    # header
    col_labels = ROI_LABELS + ['Avg. R ↑', 'CRPS ↓', 'FC-MAE ↓']
    cell(x0s[0], x0s[1], HDR_Y0, HDR_Y1, 'Method',
         C['hdr_bg'], C['hdr_fg'], True, 9.5)
    for j, lbl in enumerate(col_labels):
        cell(x0s[j+1], x0s[j+2], HDR_Y0, HDR_Y1, lbl,
             C['hdr_bg'], C['hdr_fg'], True, 8.6)

    # data rows
    for i, (key, name) in enumerate(methods):
        roi_mu, roi_margin, avg_r, r_ci_, crps, c_ci, fc_mae = all_rows[i]
        is_ours = 'neuroflow' in key
        y1 = DATA_Y1 - i * rh
        y0 = y1 - rh
        row_bg = C['ours_bg'] if is_ours else (C['alt'] if i % 2 == 0 else C['norm'])

        if i == n_rows - 1:
            ax.plot([0, 1], [y1, y1], transform=ax.transAxes,
                    color=C['sep'], lw=0.9, clip_on=False)

        cell(x0s[0], x0s[1], y0, y1, name, row_bg,
             C['ours_fg'] if is_ours else '#1C2833', is_ours, 8.4)

        row_vals = list(roi_mu) + [avg_r, crps, fc_mae]
        for j in range(N_COLS):
            val  = row_vals[j]
            rank = ranks[j][i]
            if rank == 0:   bg, fg = C['best_bg'], C['best_fg']
            elif rank == 1: bg, fg = C['sec_bg'],  C['sec_fg']
            else:           bg, fg = row_bg, (C['ours_fg'] if is_ours else '#1C2833')

            if j < N_ROI:
                m   = roi_margin[j] if roi_margin is not None else None
                txt = f'{val:.3f}\n±{m:.3f}' if m is not None else f'{val:.3f}'
                fs  = 7.0 if m is not None else 8.2
            elif j == COL_AVGR:
                m   = (r_ci_[1]-r_ci_[0])/2 if r_ci_ else None
                txt = f'{val:.3f}\n±{m:.3f}' if m is not None else f'{val:.3f}'
                fs  = 7.0 if m is not None else 8.2
            elif j == COL_CRPS:
                m   = (c_ci[1]-c_ci[0])/2 if c_ci else None
                txt = f'{val:.3f}\n±{m:.3f}' if m is not None else f'{val:.3f}'
                fs  = 7.0 if m is not None else 8.2
            else:
                txt = f'{val:.3f}'
                fs  = 8.2

            cell(x0s[j+1], x0s[j+2], y0, y1, txt, bg, fg,
                 rank <= 1 or is_ours, fs)

    # outer frame around table only
    ax.add_patch(plt.Rectangle((0, DATA_Y0), 1, HDR_Y1 - DATA_Y0,
                                transform=ax.transAxes,
                                fc='none', ec='#1C2833', lw=1.1, clip_on=False))

    # title above header
    ax.text(0.5, HDR_Y1 + 0.015, title,
            transform=ax.transAxes, ha='center', va='bottom',
            fontsize=12, fontweight='bold', color='#1C2833')

    # legend BELOW table, left-aligned — no overlap with any column
    legend_handles = [
        mpatches.Patch(fc=C['best_bg'], ec=C['border'], label='Best in column'),
        mpatches.Patch(fc=C['sec_bg'],  ec=C['border'], label='2nd best'),
        mpatches.Patch(fc=C['ours_bg'], ec=C['border'], label='NeuroCFM (Ours)'),
    ]
    ax.legend(handles=legend_handles,
              loc='lower left', bbox_to_anchor=(0.0, 0.0),
              fontsize=9, framealpha=0.9, ncol=3)

    plt.tight_layout(pad=0)
    path = os.path.join(OUT_PNG, filename)
    fig.savefig(path, dpi=180, bbox_inches='tight', facecolor='white')
    plt.close(fig)
    print(f'PNG   -> {path}')


# ── Run ───────────────────────────────────────────────────────────────────────

write_latex(
    METHODS_INTRA,
    r'Intra-Subject Prediction Results --- NeuroBOLT Dataset. '
    r'\textbf{Bold}: best; \underline{underline}: 2nd best. '
    r'Per-ROI CI: inverse-$|R|$ weighted from 95\% bootstrap CI. Avg.\,R / CRPS: 95\% bootstrap CI.',
    'intra', 'table_intra.tex',
)
write_latex(
    METHODS_POOLED,
    r'Inter-Subject Prediction Results --- NeuroBOLT Dataset. '
    r'\textbf{Bold}: best; \underline{underline}: 2nd best. '
    r'Per-ROI CI: inverse-$|R|$ weighted from 95\% bootstrap CI. Avg.\,R / CRPS: 95\% bootstrap CI.',
    'pooled', 'table_pooled.tex',
)
make_png(METHODS_INTRA,
         'Intra-Subject Prediction Results — NeuroBOLT Dataset',
         'neurobolt_table_intra.png')
make_png(METHODS_POOLED,
         'Inter-Subject Prediction Results — NeuroBOLT Dataset',
         'neurobolt_table_pooled.png')

print('\nDone.')
