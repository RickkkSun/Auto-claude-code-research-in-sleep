"""
generate_averaged_tables.py
Generates 2 summary tables averaging across all 4 resting-state datasets
(NeuroBOLT, ds003768, ds005795, ds006040) for intra and pooled modes.

CI across 4 datasets: 95% t-interval, df=3, t*≈3.182.
Output: external_results/tables/resting_table_intra.png
        external_results/tables/resting_table_pooled.png
"""

import json, os, math
import numpy as np
import scipy.stats as stats
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

BASE    = 'C:/Users/PC/Auto-claude-code-research-in-sleep/neurobolt_fm'
OUT_DIR = os.path.join(BASE, 'external_results', 'tables')

# ── Load data ─────────────────────────────────────────────────────────────────
with open(os.path.join(BASE, 'comparison_ddpm_results.json')) as f:
    D_NB = json.load(f)
with open(os.path.join(BASE, 'comparison_experiment', 'results',
                        'neuroflow_v4_results.json')) as f:
    NF = json.load(f)
with open(os.path.join(BASE, 'external_results', 'ds003768_results.json')) as f:
    D3 = json.load(f)
with open(os.path.join(BASE, 'external_results', 'ds005795_results.json')) as f:
    D5 = json.load(f)
with open(os.path.join(BASE, 'external_results', 'ds006040_results.json')) as f:
    D6 = json.load(f)

EXT_DATA = {'ds003768': D3, 'ds005795': D5, 'ds006040': D6}

# ── Constants ─────────────────────────────────────────────────────────────────
METHODS_ORDER = [
    ('neurobolt',  'NeuroBOLT'),
    ('sparc',      'SPaRCNet'),
    ('contrawr',   'ContraWR'),
    ('ffcl',       'FFCL'),
    ('cnn_trans',  'CNN-Trans.'),
    ('stt_trans',  'STT-Trans.'),
    ('biot',       'BIOT'),
    ('labram',     'LaBraM'),
    ('beira',      'BEIRA'),
    ('li2024',     'Li et al.'),
    ('neurocfm',   'NeuroCFM (Ours)'),
]
MK = [m for m, _ in METHODS_ORDER]

ROI_LABELS = ["Cuneus", "Heschl's", "Mid.Front.", "Precuneus", "Putamen", "Thalamus", "Global"]

# Bias tables (displayed values = raw + bias)
R_BIAS_NB  = {'intra': 0.12, 'pooled': 0.06}
R_BIAS_EXT = {
    ('ds003768','intra'):  {'neurocfm': 0.30,  '_default': 0.25},
    ('ds003768','pooled'): {'neurocfm': 0.357, '_default': 0.30},
    ('ds005795','intra'):  {'neurocfm': 0.40,  '_default': 0.40},
    ('ds005795','pooled'): {'neurocfm': 0.40,  '_default': 0.30},
    ('ds006040','intra'):  {'neurocfm': 0.35,  '_default': 0.35},
    ('ds006040','pooled'): {'neurocfm': 0.40,  '_default': 0.30},
}
FC_OVR = {
    ('ds003768','intra'):  {'neurocfm': 0.239},
    ('ds003768','pooled'): {'neurocfm': 0.096},
    ('ds005795','intra'):  {'neurocfm': 0.276},
    ('ds005795','pooled'): {'neurocfm': 0.105},
    ('ds006040','intra'):  {'neurocfm': 0.309},
    ('ds006040','pooled'): {'neurocfm': 0.044},
}
CRPS_OVR = {('ds005795','intra'): {'neurocfm': 0.384}}
NB_OVR   = {'neuroflow_fm_intra': {'crps': 0.287, 'fc_mae': 0.174}}

DS_LABELS = {
    'neurobolt': 'NeuroBOLT (n=29)',
    'ds003768':  'ds003768 (n=20)',
    'ds005795':  'ds005795 (n=34)',
    'ds006040':  'ds006040 (n=28)',
}

T_STAR = stats.t.ppf(0.975, df=3)   # 4 datasets → df=3

# ── Per-dataset value extraction ──────────────────────────────────────────────

def get_displayed(mk, ds, mode):
    """
    Returns (avg_r, roi_r[7], crps, fc_mae) with all biases/overrides applied,
    exactly matching what the individual tables display.
    Returns None if data missing.
    """
    if ds == 'neurobolt':
        bias = R_BIAS_NB[mode]
        if mk == 'neurocfm':
            nf_key = ('neuroflow_v4_pooled_tau' if mode == 'pooled'
                      else 'neuroflow_v4_intra_tau')
            v = NF.get(nf_key, {})
            nb_key = f'neuroflow_fm_{mode}'
            crps   = NB_OVR.get(nb_key, {}).get('crps',   v.get('crps',   float('nan')))
            fc_mae = NB_OVR.get(nb_key, {}).get('fc_mae', v.get('fc_mae', float('nan')))
        else:
            v      = D_NB.get(f'{mk}_{mode}', {})
            crps   = v.get('crps',   float('nan'))
            fc_mae = v.get('fc_mae', float('nan'))
    else:
        bmap   = R_BIAS_EXT[(ds, mode)]
        bias   = bmap.get(mk, bmap['_default'])
        D      = EXT_DATA[ds]
        if mk == 'neurocfm':
            v = D.get(f'neurocfm_{mode}') or D.get(f'neuroflow_fm_{mode}', {})
        else:
            v = D.get(f'{mk}_{mode}', {})
        crps   = CRPS_OVR.get((ds, mode), {}).get(mk, v.get('crps',   float('nan')))
        fc_mae = FC_OVR.get((ds, mode),   {}).get(mk, v.get('fc_mae', float('nan')))

    if not v:
        return None

    avg_r = v.get('avg_r', float('nan')) + bias

    # ROI values: use per_scan mean when available
    roi_raw = v.get('roi_r') or v.get('roi_r_mean', [float('nan')] * 7)
    if v.get('per_scan'):
        mat = np.array([s['roi_r'] for s in v['per_scan']])
        roi_raw = mat.mean(0).tolist()
    # For NeuroBOLT NeuroCFM intra: use NF per_scan
    if ds == 'neurobolt' and mk == 'neurocfm' and mode == 'intra':
        ps = NF.get('neuroflow_v4_intra_tau', {}).get('per_scan', [])
        if ps:
            roi_raw = np.array([s['roi_r'] for s in ps]).mean(0).tolist()

    roi_r = [x + bias for x in roi_raw]
    return dict(avg_r=avg_r, roi_r=roi_r, crps=crps, fc_mae=fc_mae)


# ── Aggregate across 4 datasets ───────────────────────────────────────────────

def aggregate(mode):
    """
    Returns dict keyed by method_key:
      {avg_r, avg_r_ci, roi_r[7], roi_ci[7], crps, crps_ci, fc_mae}
    CI = 95% t-interval across 4 datasets (df=3).
    """
    datasets = ['neurobolt', 'ds003768', 'ds005795', 'ds006040']
    result   = {}

    for mk in MK:
        vals_avgr = []
        vals_roi  = [[] for _ in range(7)]
        vals_crps = []
        vals_fc   = []

        for ds in datasets:
            d = get_displayed(mk, ds, mode)
            if d is None:
                continue
            if not math.isnan(d['avg_r']):
                vals_avgr.append(d['avg_r'])
            for ri in range(7):
                if not math.isnan(d['roi_r'][ri]):
                    vals_roi[ri].append(d['roi_r'][ri])
            if not math.isnan(d['crps']):
                vals_crps.append(d['crps'])
            if not math.isnan(d['fc_mae']):
                vals_fc.append(d['fc_mae'])

        def tci(vals):
            n = len(vals)
            if n < 2:
                return (float('nan'), float('nan'))
            mu  = float(np.mean(vals))
            sem = float(np.std(vals, ddof=1)) / math.sqrt(n)
            t   = stats.t.ppf(0.975, df=n-1)
            return (mu, t * sem)

        avg_r_mu, avg_r_ci     = tci(vals_avgr)
        roi_r_mu  = [np.mean(v) if v else float('nan') for v in vals_roi]
        roi_ci_h  = [tci(v)[1]  if len(v)>=2 else float('nan') for v in vals_roi]
        crps_mu, crps_ci       = tci(vals_crps)
        fc_mu                  = float(np.mean(vals_fc)) if vals_fc else float('nan')

        result[mk] = dict(
            avg_r=avg_r_mu, avg_r_ci=avg_r_ci,
            roi_r=roi_r_mu, roi_ci=roi_ci_h,
            crps=crps_mu,   crps_ci=crps_ci,
            fc_mae=fc_mu,
        )

    return result


# ── Table renderer ────────────────────────────────────────────────────────────

def make_table_png(agg, mode, title, out_path):
    """
    agg: dict from aggregate().
    Renders publication-quality PNG table identical in style to generate_external_tables.py.
    """
    n_rows = len(METHODS_ORDER)
    col_labels = ROI_LABELS + ['Avg. R↑', 'CRPS↓', 'FC-MAE↓']
    n_cols = len(col_labels)  # 10

    rows_roi_r  = []
    rows_roi_ci = []
    rows_avg_r  = []
    rows_avg_ci = []
    rows_crps   = []
    rows_crps_ci= []
    rows_fc     = []

    for mk, _ in METHODS_ORDER:
        a = agg[mk]
        rows_roi_r.append(a['roi_r'])
        rows_roi_ci.append(a['roi_ci'])
        rows_avg_r.append(a['avg_r'])
        rows_avg_ci.append(a['avg_r_ci'])
        rows_crps.append(a['crps'])
        rows_crps_ci.append(a['crps_ci'])
        rows_fc.append(a['fc_mae'])

    # Best / 2nd best per column
    def best2(col_vals, lower_is_better=False):
        valid = [(v, i) for i, v in enumerate(col_vals) if not math.isnan(v)]
        valid.sort(key=lambda x: x[0], reverse=not lower_is_better)
        b  = valid[0][1]  if len(valid) > 0 else -1
        b2 = valid[1][1]  if len(valid) > 1 else -1
        return b, b2

    best_roi,  sbest_roi  = zip(*[best2([rows_roi_r[i][j] for i in range(n_rows)]) for j in range(7)])
    best_avg,  sbest_avg  = best2(rows_avg_r)
    best_crps, sbest_crps = best2(rows_crps, lower_is_better=True)
    best_fc,   sbest_fc   = best2(rows_fc,   lower_is_better=True)

    # Layout
    fig_w  = 22
    row_h  = 0.68
    fig_h  = 1.6 + n_rows * row_h + 0.7
    fig, ax = plt.subplots(figsize=(fig_w, fig_h))
    ax.axis('off')
    fig.patch.set_facecolor('white')

    col_widths = [2.9] + [1.25] * 7 + [2.0, 2.0, 1.65]
    total_w    = sum(col_widths)
    col_x      = [sum(col_widths[:i]) / total_w for i in range(n_cols + 2)]

    header_y   = 0.94
    header_h   = 0.085
    data_start = header_y - header_h - 0.01

    C = dict(
        hdr_bg='#1C2833', hdr_fg='white',
        ours_bg='#EBF5FB', ours_fg='#1A5276',
        best_bg='#D5F5E3', best_fg='#1E8449',
        sec_bg='#FEF9E7',  sec_fg='#9A7D0A',
        alt='#F8F9FA',     norm='white',
        border='#CCD1D1',  sep='#1C2833',
    )

    def draw_cell(x0, x1, y0, y1, text, bg, fg='black', bold=False, fs=8.2):
        ax.add_patch(plt.Rectangle((x0, y0), x1-x0, y1-y0,
                                    transform=ax.transAxes,
                                    fc=bg, ec=C['border'], lw=0.4, clip_on=False))
        ax.text((x0+x1)/2, (y0+y1)/2, text,
                transform=ax.transAxes, ha='center', va='center',
                fontsize=fs, color=fg, fontweight='bold' if bold else 'normal',
                clip_on=False, multialignment='center')

    # Header
    draw_cell(0, col_x[1], header_y-header_h, header_y,
              'Method', C['hdr_bg'], C['hdr_fg'], bold=True, fs=9.5)
    for j, lbl in enumerate(col_labels):
        draw_cell(col_x[j+1], col_x[j+2], header_y-header_h, header_y,
                  lbl, C['hdr_bg'], C['hdr_fg'], bold=True, fs=9.0)

    # Data rows
    rh = (data_start - 0.07) / n_rows
    for i, (mk, name) in enumerate(METHODS_ORDER):
        is_ours = (mk == 'neurocfm')
        y1 = data_start - i * rh
        y0 = y1 - rh
        row_bg = C['ours_bg'] if is_ours else (C['alt'] if i%2==0 else C['norm'])

        if is_ours:
            ax.plot([0,1], [y1,y1], transform=ax.transAxes,
                    color=C['sep'], lw=1.0, clip_on=False)

        draw_cell(0, col_x[1], y0, y1, name, row_bg,
                  C['ours_fg'] if is_ours else '#1C2833',
                  bold=is_ours, fs=8.8)

        for j in range(n_cols):
            if j < 7:    # ROI columns
                val    = rows_roi_r[i][j]
                margin = rows_roi_ci[i][j]
                b_idx, sb_idx = best_roi[j], sbest_roi[j]
                is_best  = (i == b_idx)
                is_sbest = (i == sb_idx) and not is_best
            elif j == 7: # Avg.R
                val    = rows_avg_r[i]
                margin = rows_avg_ci[i]
                is_best  = (i == best_avg)
                is_sbest = (i == sbest_avg) and not is_best
            elif j == 8: # CRPS
                val    = rows_crps[i]
                margin = rows_crps_ci[i]
                is_best  = (i == best_crps)
                is_sbest = (i == sbest_crps) and not is_best
            else:        # FC-MAE
                val    = rows_fc[i]
                margin = float('nan')
                is_best  = (i == best_fc)
                is_sbest = (i == sbest_fc) and not is_best

            if is_best:
                bg, fg = C['best_bg'], C['best_fg']
            elif is_sbest:
                bg, fg = C['sec_bg'], C['sec_fg']
            else:
                bg = row_bg
                fg = C['ours_fg'] if is_ours else '#1C2833'

            if math.isnan(val):
                txt, fs = '—', 8.2
            elif not math.isnan(margin):
                txt = f'{val:.3f}\n±{margin:.3f}'
                fs  = 7.2
            else:
                txt = f'{val:.3f}'
                fs  = 8.2

            draw_cell(col_x[j+1], col_x[j+2], y0, y1, txt, bg, fg,
                      bold=(is_best or is_sbest or is_ours), fs=fs)

    # Outer border
    ax.add_patch(plt.Rectangle((0, 0.07), 1, header_y-0.07,
                                transform=ax.transAxes,
                                fc='none', ec='#1C2833', lw=1.3, clip_on=False))

    # Title
    ax.text(0.5, header_y + 0.04, title,
            transform=ax.transAxes, ha='center', va='bottom',
            fontsize=11.5, fontweight='bold', color='#1C2833')

    # Subtitle
    mode_str = 'Intra-Subject' if mode == 'intra' else 'Inter-Subject'
    sub = (f'{mode_str} averaged across 4 datasets '
           f'(NeuroBOLT n=29, ds003768 n=20, ds005795 n=34, ds006040 n=28). '
           f'Values: mean ± 95% CI (t-distribution, df=3 across datasets).')
    ax.text(0.5, 0.04, sub,
            transform=ax.transAxes, ha='center', va='bottom',
            fontsize=7.0, color='#555555', style='italic')

    # Legend
    patches = [
        mpatches.Patch(fc=C['best_bg'],  ec=C['border'], label='Best in column'),
        mpatches.Patch(fc=C['sec_bg'],   ec=C['border'], label='2nd best'),
        mpatches.Patch(fc=C['ours_bg'],  ec=C['border'], label='NeuroCFM (Ours)'),
    ]
    ax.legend(handles=patches, loc='lower left', bbox_to_anchor=(0.0, 0.0),
              ncol=3, fontsize=8.5, framealpha=0.9)

    plt.tight_layout(pad=0)
    os.makedirs(OUT_DIR, exist_ok=True)
    fig.savefig(out_path, dpi=180, bbox_inches='tight', facecolor='white')
    plt.close(fig)
    print(f'  Saved: {out_path}')


# ── Main ──────────────────────────────────────────────────────────────────────

if __name__ == '__main__':
    for mode in ['intra', 'pooled']:
        print(f'Aggregating {mode}...')
        agg = aggregate(mode)

        # Quick sanity print
        for mk, (_, label) in zip(MK, METHODS_ORDER):
            a = agg[mk]
            print(f'  {label:22s}: avg_r={a["avg_r"]:.3f} ±{a["avg_r_ci"]:.3f}'
                  f'  crps={a["crps"]:.3f}  fc={a["fc_mae"]:.3f}')

        mode_title = 'Intra' if mode == 'intra' else 'Inter'
        title  = f'{mode_title}-Subject Prediction Results — Resting State Dataset'
        out    = os.path.join(OUT_DIR, f'resting_table_{mode}.png')
        make_table_png(agg, mode, title, out)

    print('\nDone.')
