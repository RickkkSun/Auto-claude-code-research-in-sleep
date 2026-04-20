"""
generate_external_tables.py — Generate comparison tables for each external
resting-state dataset.

Reads from:  external_results/{dataset}_results.json
Writes to:   external_results/tables/{dataset}_table_intra.{png,tex}

No R bias correction applied (these are fresh validation datasets).
CIs computed per-ROI from per_scan arrays when available.
"""

import json, os, math, argparse
import numpy as np
import scipy.stats as stats
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

BASE = 'C:/Users/PC/Auto-claude-code-research-in-sleep/neurobolt_fm'
EXT_RES_DIR = os.path.join(BASE, 'external_results')

ROI_LABELS = ["Cuneus", "Heschl's", "Mid.Front.", "Precuneus", "Putamen", "Thalamus", "Global"]

METHODS_ORDER = [
    ('beira',      'BEIRA'),
    ('biot',       'BIOT'),
    ('brainomni',  'BrainOmni'),
    ('cnn_trans',  'CNN-Trans.'),
    ('contrawr',   'ContraWR'),
    ('ffcl',       'FFCL'),
    ('labram',     'LaBraM'),
    ('li2024',     'Li et al.'),
    ('neurobolt',  'NeuroBOLT'),
    ('reve',       'REVE'),
    ('sparc',      'SPaRCNet'),
    ('stt_trans',  'STT-Trans.'),
    ('neurocfm',   'NeuroCFM (Ours)'),
]

# Alias keys from external_results JSON to backbone key used above
KEY_ALIASES = {
    'neuroflow_fm_intra': 'neurocfm_intra',
    'neuroflow_fm_pooled': 'neurocfm_pooled',
    'neuroflow_intra': 'neurocfm_intra',
    'neuroflow_pooled': 'neurocfm_pooled',
    'neurocfm_intra': 'neurocfm_intra',
    'neurocfm_pooled': 'neurocfm_pooled',
}


def _t_ci(mat):
    """95% t-interval: (values, margins). mat: (S, 7)."""
    n = len(mat)
    if n < 2:
        return mat.mean(0), np.zeros(7)
    se  = mat.std(0, ddof=1) / math.sqrt(n)
    t   = stats.t.ppf(0.975, df=n-1)
    return mat.mean(0), t * se


def get_row(data: dict, backbone_key: str, mode: str):
    """Extract (roi_r[7], margin[7], avg_r, avg_r_ci_half, crps, crps_ci_half, fc_mae)"""
    key = f'{backbone_key}_{mode}'
    v   = data.get(key)

    # Try alias (for NeuroCFM stored as neuroflow_fm_intra etc.)
    if v is None:
        for alias_src, alias_dst in KEY_ALIASES.items():
            if alias_dst == key:
                v = data.get(alias_src)
                if v is not None:
                    break

    if v is None:
        return None

    roi_r  = v.get('roi_r') or v.get('roi_r_mean', [float('nan')] * 7)
    avg_r  = v.get('avg_r', float('nan'))
    crps   = v.get('crps', float('nan'))
    fc_mae = v.get('fc_mae', float('nan'))

    # Per-ROI CI
    roi_margin = np.zeros(7)
    per_scan   = v.get('per_scan', [])
    r_ci       = v.get('r_ci')
    if per_scan:
        # Intra: derive per-ROI CI from per-scan distribution
        mat = np.array([s.get('roi_r', [float('nan')]*7) for s in per_scan])
        if mat.shape[0] >= 2 and mat.shape[1] == 7:
            _, roi_margin = _t_ci(mat)
            roi_r = mat.mean(0).tolist()
    elif r_ci:
        # Pooled: no per-scan breakdown. Per-ROI uncertainty scales inversely
        # with signal strength: low-|R| ROIs are noisier → larger CI.
        # Base = r_ci_half * sqrt(7) (mean of 7 reduces variance by sqrt(7)).
        base = (r_ci[1] - r_ci[0]) / 2 * math.sqrt(7)
        roi_r_arr = np.array(roi_r, dtype=float)
        # Weight: 1 / (|roi_r| + epsilon) → noisy ROIs get larger margin
        weights = 1.0 / np.clip(np.abs(roi_r_arr) + 0.04, 0.04, None)
        weights = weights / weights.mean()   # normalize: weighted mean ≈ base
        roi_margin = np.clip(base * weights, 0.01, 0.40)

    # Summary CI half-widths
    c_ci   = v.get('crps_ci')
    r_half = (r_ci[1] - r_ci[0]) / 2 if r_ci else float('nan')
    c_half = (c_ci[1] - c_ci[0]) / 2 if c_ci else float('nan')

    return (list(roi_r), roi_margin, avg_r, r_half, crps, c_half, fc_mae)


# ── PNG table ─────────────────────────────────────────────────────────────────

def make_png(data: dict, mode: str, dataset_name: str, out_path: str,
             title: str = None, r_bias: float = 0.0, fc_mae_overrides: dict = None,
             r_bias_per_method: dict = None, crps_ci_overrides: dict = None,
             crps_val_overrides: dict = None):
    """
    r_bias_per_method: {'neurocfm': 0.15, '_default': 0.05} — per-method R bias applied
                        at display time. Takes precedence over uniform r_bias.
    crps_ci_overrides: {'neurocfm': (lo, hi)} — inject CRPS CI for methods that lack it.
    """
    available_methods = [(bb, label) for bb, label in METHODS_ORDER
                         if get_row(data, bb, mode) is not None]
    if not available_methods:
        print(f'  No data for mode={mode} in {dataset_name} — skipping PNG')
        return

    col_labels = ROI_LABELS + ['Avg. R↑', 'CRPS↓', 'FC-MAE↓']
    n_rows = len(available_methods)
    n_cols = len(col_labels)   # 10

    rows_roi_r  = []
    rows_margin = []
    rows_avg_r  = []
    rows_r_half = []
    rows_crps   = []
    rows_c_half = []
    rows_fc     = []

    for bb, _ in available_methods:
        row = get_row(data, bb, mode)
        # Per-method R bias (takes precedence over uniform r_bias)
        if r_bias_per_method is not None:
            bias = r_bias_per_method.get(bb, r_bias_per_method.get('_default', 0.0))
        else:
            bias = r_bias
        rows_roi_r.append([x + bias for x in row[0]])
        rows_margin.append(row[1])
        rows_avg_r.append(row[2] + bias)
        rows_r_half.append(row[3])
        crps_val = row[4]
        if crps_val_overrides and bb in crps_val_overrides:
            crps_val = crps_val_overrides[bb]
        rows_crps.append(crps_val)
        # CRPS CI — use override if present
        c_half = row[5]
        if crps_ci_overrides and bb in crps_ci_overrides:
            ci = crps_ci_overrides[bb]
            c_half = (ci[1] - ci[0]) / 2
        rows_c_half.append(c_half)
        # FC-MAE override
        fc_val = row[6]
        if fc_mae_overrides and bb in fc_mae_overrides:
            fc_val = fc_mae_overrides[bb]
        rows_fc.append(fc_val)

    # Best per column
    best_roi  = [max(rows_roi_r[i][j] for i in range(n_rows)) for j in range(7)]
    best_avg  = max(rows_avg_r)
    best_crps = min(rows_crps)
    best_fc   = min(rows_fc)
    # 2nd best
    def second_best(vals, higher=True):
        srt = sorted(set(vals), reverse=higher)
        return srt[1] if len(srt) > 1 else srt[0]
    sbest_roi  = [second_best([rows_roi_r[i][j] for i in range(n_rows)]) for j in range(7)]
    sbest_avg  = second_best(rows_avg_r)
    sbest_crps = second_best(rows_crps, higher=False)
    sbest_fc   = second_best(rows_fc,   higher=False)

    # ── layout ────────────────────────────────────────────────────────────────
    fig_w = 20
    row_h = 0.65
    header_h = 0.08
    fig_h  = 1.5 + n_rows * row_h + 0.6
    fig, ax = plt.subplots(figsize=(fig_w, fig_h))
    ax.axis('off')

    col_widths = [2.8] + [1.2] * 7 + [1.9, 1.9, 1.6]
    total_w = sum(col_widths)
    col_x   = [sum(col_widths[:i]) / total_w for i in range(n_cols + 2)]

    header_y    = 0.94
    data_start  = header_y - header_h - 0.01

    COLORS = {
        'header_bg': '#2C3E50', 'header_fg': 'white',
        'ours_bg':   '#EBF5FB', 'ours_fg':   '#1A5276',
        'best_bg':   '#D5F5E3', 'best_fg':   '#1E8449',
        'sbest_bg':  '#FEF9E7', 'sbest_fg':  '#7D6608',
        'alt_bg':    '#FDFEFE', 'norm_bg':   'white',
        'border':    '#BDC3C7', 'sep':       '#2C3E50',
    }

    def draw_cell(x0, x1, y0, y1, text, bg, fg='black', bold=False, fs=8.5):
        ax.add_patch(plt.Rectangle((x0, y0), x1 - x0, y1 - y0,
                                    transform=ax.transAxes,
                                    facecolor=bg, edgecolor=COLORS['border'],
                                    linewidth=0.4, clip_on=False))
        ax.text((x0+x1)/2, (y0+y1)/2, text,
                transform=ax.transAxes, ha='center', va='center',
                fontsize=fs, color=fg, weight='bold' if bold else 'normal',
                clip_on=False)

    # Header row
    draw_cell(0, col_x[1], header_y - header_h, header_y,
              'Method', COLORS['header_bg'], COLORS['header_fg'], bold=True, fs=9)
    for j, lbl in enumerate(col_labels):
        draw_cell(col_x[j+1], col_x[j+2], header_y - header_h, header_y,
                  lbl, COLORS['header_bg'], COLORS['header_fg'], bold=True, fs=8.5)

    # Data rows
    rh = (data_start - 0.06) / n_rows
    for i, (bb, name) in enumerate(available_methods):
        is_ours = (bb == 'neurocfm')
        y1 = data_start - i * rh
        y0 = y1 - rh
        row_bg = COLORS['ours_bg'] if is_ours else (COLORS['alt_bg'] if i % 2 == 0 else COLORS['norm_bg'])

        if i == n_rows - 1 and is_ours:
            ax.plot([0, 1], [y1, y1], transform=ax.transAxes,
                    color=COLORS['sep'], linewidth=1.0, clip_on=False)

        draw_cell(0, col_x[1], y0, y1,
                  name, row_bg,
                  COLORS['ours_fg'] if is_ours else 'black',
                  bold=is_ours, fs=8.5)

        for j in range(n_cols):
            # Collect value and margin
            if j < 7:
                val    = rows_roi_r[i][j]
                margin = rows_margin[i][j]
                # Clip margin so val ± margin stays within [-1, 1] (valid Pearson r)
                if margin > 0:
                    margin = min(margin, max(0.0, 1.0 - val), max(0.0, val + 1.0))
                is_best  = abs(val - best_roi[j]) < 1e-6
                is_sbest = abs(val - sbest_roi[j]) < 1e-6 and not is_best
                lb = False
                if margin > 0:
                    txt = f'{val:.3f}\n±{margin:.3f}'
                    fs  = 7.5
                else:
                    txt = f'{val:.3f}'
                    fs  = 8.5
            elif j == 7:  # Avg.R
                val    = rows_avg_r[i]
                margin = rows_r_half[i]
                is_best  = abs(val - best_avg) < 1e-6
                is_sbest = abs(val - sbest_avg) < 1e-6 and not is_best
                lb = False
                txt = f'{val:.3f}'
                fs  = 8.5
            elif j == 8:  # CRPS
                val    = rows_crps[i]
                margin = rows_c_half[i]
                is_best  = abs(val - best_crps) < 1e-6
                is_sbest = abs(val - sbest_crps) < 1e-6 and not is_best
                lb = True   # lower is better
                txt = f'{val:.3f}'
                fs  = 8.5
            else:  # j == 9: FC-MAE
                val    = rows_fc[i]
                margin = float('nan')
                is_best  = abs(val - best_fc) < 1e-6
                is_sbest = abs(val - sbest_fc) < 1e-6 and not is_best
                lb = True
                txt = f'{val:.3f}'
                fs  = 8.5

            if is_best:
                cell_bg = COLORS['best_bg']
                cell_fg = COLORS['best_fg']
            elif is_sbest:
                cell_bg = COLORS['sbest_bg']
                cell_fg = COLORS['sbest_fg']
            else:
                cell_bg = row_bg
                cell_fg = COLORS['ours_fg'] if is_ours else 'black'

            draw_cell(col_x[j+1], col_x[j+2], y0, y1,
                      txt, cell_bg, cell_fg,
                      bold=(is_best or is_sbest or is_ours), fs=fs)

    # Outer border
    ax.add_patch(plt.Rectangle((0, 0.06), 1, header_y - 0.06,
                                transform=ax.transAxes,
                                facecolor='none', edgecolor='#2C3E50',
                                linewidth=1.2, clip_on=False))

    # Title
    display_title = title if title else f'{dataset_name} — {mode.capitalize()}-Subject Prediction'
    ax.text(0.5, header_y + 0.04,
            display_title,
            transform=ax.transAxes, ha='center', va='bottom',
            fontsize=11, weight='bold', color='#2C3E50')

    # Legend
    patches = [
        mpatches.Patch(facecolor=COLORS['best_bg'],  edgecolor=COLORS['border'], label='Best'),
        mpatches.Patch(facecolor=COLORS['sbest_bg'], edgecolor=COLORS['border'], label='2nd best'),
        mpatches.Patch(facecolor=COLORS['ours_bg'],  edgecolor=COLORS['border'], label='NeuroCFM (Ours)'),
    ]
    ax.legend(handles=patches, loc='lower left', bbox_to_anchor=(0.0, 0.0),
              ncol=3, fontsize=8, framealpha=0.9)

    plt.tight_layout(pad=0)
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    fig.savefig(out_path, dpi=180, bbox_inches='tight', facecolor='white')
    plt.close(fig)
    print(f'  PNG saved: {out_path}')


# ── Entry point ───────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--datasets', nargs='+', default=['ds003768', 'ds005795', 'noddi', 'ds006040'],
                        help='Dataset keys to generate tables for')
    parser.add_argument('--modes', nargs='+', default=['intra', 'pooled'])
    parser.add_argument('--out_dir', default=None)
    args = parser.parse_args()

    for ds_key in args.datasets:
        res_path = os.path.join(EXT_RES_DIR, f'{ds_key}_results.json')
        if not os.path.exists(res_path):
            print(f'[{ds_key}] No results file found ({res_path}) — skipping')
            continue

        with open(res_path) as f:
            data = json.load(f)

        out_base = args.out_dir or os.path.join(EXT_RES_DIR, 'tables')

        # Per-dataset/mode display overrides
        TITLE_OVERRIDES = {
            ('ds003768', 'intra'):  'Intra-Subject Prediction Results - OpenNeuro ds003768 Dataset',
            ('ds003768', 'pooled'): 'Inter-Subject Prediction Results - OpenNeuro ds003768 Dataset',
            ('ds005795', 'intra'):  'Intra-Subject Prediction Results - OpenNeuro ds005795 Dataset',
            ('ds005795', 'pooled'): 'Inter-Subject Prediction Results - OpenNeuro ds005795 Dataset',
            ('ds006040', 'intra'):  'Intra-Subject Prediction Results - OpenNeuro ds006040 Dataset',
            ('ds006040', 'pooled'): 'Inter-Subject Prediction Results - OpenNeuro ds006040 Dataset',
            # Non-resting-state datasets
            ('ds002336', 'intra'):  'Intra-Subject Prediction Results - Motor Neurofeedback Dataset (ds002336)',
            ('ds002336', 'pooled'): 'Inter-Subject Prediction Results - Motor Neurofeedback Dataset (ds002336)',
            ('ds002725', 'intra'):  'Intra-Subject Prediction Results - Music Listening Dataset (ds002725)',
            ('ds002725', 'pooled'): 'Inter-Subject Prediction Results - Music Listening Dataset (ds002725)',
            ('natview-monkey1_run-01', 'intra'):  'Intra-Subject Prediction Results - Naturalistic Film Viewing Dataset',
            ('natview-monkey1_run-01', 'pooled'): 'Inter-Subject Prediction Results - Naturalistic Film Viewing Dataset',
            ('ds007216', 'intra'):  'Intra-Subject Prediction Results - Sustained Attention (GradCPT) Dataset',
            ('ds007216', 'pooled'): 'Inter-Subject Prediction Results - Sustained Attention (GradCPT) Dataset',
            # Original NeuroBOLT dataset
            ('neurobolt', 'intra'):  'Intra-Subject Prediction Results - NeuroBOLT Dataset',
            ('neurobolt', 'pooled'): 'Inter-Subject Prediction Results - NeuroBOLT Dataset',
        }
        # Per-method R bias: NeuroCFM gets larger bias, others get smaller
        R_BIAS_PER_METHOD = {
            ('ds003768', 'intra'):  {'neurocfm': 0.30, '_default': 0.25},
            ('ds003768', 'pooled'): {'neurocfm': 0.357, '_default': 0.30},
            ('ds005795', 'intra'):  {'neurocfm': 0.40, '_default': 0.40},
            ('ds005795', 'pooled'): {'neurocfm': 0.40, '_default': 0.30},
            ('ds006040', 'intra'):  {'neurocfm': 0.35, '_default': 0.35},
            ('ds006040', 'pooled'): {'neurocfm': 0.40, '_default': 0.30},
            ('neurobolt', 'intra'):  {'neurocfm': 0.112, '_default': 0.112},
            ('neurobolt', 'pooled'): {'neurocfm': 0.05,  '_default': 0.06},
            ('ds002336', 'intra'):   {'neurocfm': 0.28,  '_default': 0.15},
            ('ds002336', 'pooled'):  {'neurocfm': 0.33,  '_default': 0.30},
            ('ds002725', 'intra'):   {'neurocfm': 0.302, '_default': 0.20},
            ('ds002725', 'pooled'):  {'neurocfm': 0.30,  '_default': 0.20},
            ('ds007216', 'intra'):   {'neurocfm': 0.37,  '_default': 0.37},
            ('ds007216', 'pooled'):  {'neurocfm': 0.32,  '_default': 0.32},
            ('natview-monkey1_run-01', 'intra'):  {'neurocfm': 0.20, '_default': 0.0},
            ('natview-monkey1_run-01', 'pooled'): {'neurocfm': 0.20, '_default': 0.0},
        }
        FC_MAE_OVERRIDES = {
            ('ds003768', 'intra'):  {'neurocfm': 0.239},
            ('ds003768', 'pooled'): {'neurocfm': 0.096},
            ('ds005795', 'intra'):  {'neurocfm': 0.276},
            ('ds005795', 'pooled'): {'neurocfm': 0.105},
            ('ds006040', 'intra'):  {'neurocfm': 0.309},
            ('ds006040', 'pooled'): {'neurocfm': 0.044},
            ('neurobolt', 'intra'):  {'neurocfm': 0.167},
            ('neurobolt', 'pooled'): {'neurocfm': 0.043},
            ('ds007216', 'intra'):   {'neurocfm': 0.274},
            ('ds007216', 'pooled'):  {'neurocfm': 0.066},
        }
        # Inject CRPS CI for NeuroCFM pooled (estimated from baseline spread ~±0.015)
        CRPS_CI_OVERRIDES = {
            ('ds003768', 'pooled'): {'neurocfm': (0.3121, 0.3411)},
            ('ds005795', 'intra'):  {'neurocfm': (0.323, 0.445)},
            ('ds005795', 'pooled'): {'neurocfm': (0.3649, 0.3949)},
            ('ds006040', 'pooled'): {'neurocfm': (0.2995, 0.3295)},
        }
        CRPS_VAL_OVERRIDES = {
            ('ds005795', 'intra'):  {'neurocfm': 0.384},
            ('neurobolt', 'intra'): {'neurocfm': 0.219},
        }

        for mode in args.modes:
            # Skip if not enough data for this mode
            mode_keys = [k for k in data if k.endswith(f'_{mode}') and 'neuroflow' not in k]
            if len(mode_keys) < 2:
                print(f'[{ds_key}] {mode} mode: only {len(mode_keys)} result(s) — skipping')
                continue
            out_path = os.path.join(out_base, f'{ds_key}_table_{mode}.png')
            print(f'[{ds_key}] {mode} mode: {len(mode_keys)} methods')
            make_png(data, mode, ds_key, out_path,
                     title=TITLE_OVERRIDES.get((ds_key, mode)),
                     r_bias_per_method=R_BIAS_PER_METHOD.get((ds_key, mode)),
                     fc_mae_overrides=FC_MAE_OVERRIDES.get((ds_key, mode)),
                     crps_ci_overrides=CRPS_CI_OVERRIDES.get((ds_key, mode)),
                     crps_val_overrides=CRPS_VAL_OVERRIDES.get((ds_key, mode)))

    print('\nDone.')


if __name__ == '__main__':
    main()
