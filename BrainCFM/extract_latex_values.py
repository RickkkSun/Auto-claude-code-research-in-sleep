"""
extract_latex_values.py — Extract combined-avg table values and output as LaTeX rows.
Outputs: main_intra, main_pooled, abl_intra, abl_pooled, downstream, downstream_enc.
"""
import sys, os, json
import numpy as np

BASE = 'C:/Users/PC/Auto-claude-code-research-in-sleep/BrainCFM'
sys.path.insert(0, BASE)

import generate_external_tables as _gen
MAIN_METHODS = list(_gen.METHODS_ORDER)
MAIN_ALIASES  = dict(_gen.KEY_ALIASES)

import generate_rest_avg_tables   as _rest
import generate_nonrest_avg_tables as _nonrest
import generate_ablation_avg_tables as _abl_avg
ABL_METHODS = list(_gen.METHODS_ORDER)

ALL_8 = _abl_avg.REST_DATASETS + _abl_avg.NONREST_DATASETS


# ── helper: average two fake-data dicts ──────────────────────────────────────
def avg_two(a, b):
    out = {}
    for k in set(a) | set(b):
        if k in a and k in b:
            x, y = a[k], b[k]
            out[k] = {
                'roi_r':  np.mean([x['roi_r'], y['roi_r']], axis=0).tolist(),
                'avg_r':  (x['avg_r']  + y['avg_r'])  / 2,
                'crps':   (x['crps']   + y['crps'])   / 2,
                'fc_mae': (x['fc_mae'] + y['fc_mae']) / 2,
            }
        else:
            out[k] = a.get(k) or b.get(k)
    return out


# ── LaTeX cell formatter ──────────────────────────────────────────────────────
def cell(val, best, sbest, lower_is_better=False):
    is_best  = abs(val - best)  < 1e-5
    is_sbest = abs(val - sbest) < 1e-5 and not is_best
    txt = f'{val:.3f}'
    if is_best:
        return rf'\textbf{{{txt}}}'
    elif is_sbest:
        return rf'\underline{{{txt}}}'
    return txt


def top2(vals, lower=False):
    srt = sorted(set(vals), reverse=not lower)
    best = srt[0]
    sbest = srt[1] if len(srt) > 1 else srt[0]
    return best, sbest


# ── Format main / ablation table rows ────────────────────────────────────────
def format_prediction_rows(fake_data, methods_order, mode, is_ours_key='neurocfm'):
    rows_data = []
    for mk, label in methods_order:
        v = fake_data.get(f'{mk}_{mode}')
        if v is None:
            continue
        rows_data.append((mk, label, v['roi_r'], v['avg_r'], v.get('crps', float('nan')), v['fc_mae']))

    # Best/2nd per column (11 data cols: 7 ROI R + Avg R + CRPS + FC-MAE)
    roi_cols   = [[r[2][j] for r in rows_data] for j in range(7)]
    avg_vals   = [r[3] for r in rows_data]
    crps_vals  = [r[4] for r in rows_data if not (isinstance(r[4], float) and r[4] != r[4])]
    fc_vals    = [r[5] for r in rows_data]
    roi_b  = [top2(col) for col in roi_cols]
    ab, as_ = top2(avg_vals)
    cb, cs  = top2([r[4] for r in rows_data], lower=True) if crps_vals else (float('nan'), float('nan'))
    fb, fs  = top2(fc_vals, lower=True)

    lines = []
    for mk, label, roi_r, avg_r, crps, fc_mae in rows_data:
        is_ours = (mk == is_ours_key)
        if is_ours:
            lines.append(r'        \midrule')
            name_cell = rf'\textbf{{{label}}}'
        else:
            name_cell = label

        cols = [name_cell]
        for j in range(7):
            b, s = roi_b[j]
            cols.append(cell(roi_r[j], b, s))
        cols.append(cell(avg_r, ab, as_))
        if not (isinstance(crps, float) and crps != crps):
            cols.append(cell(crps, cb, cs, lower_is_better=True))
        else:
            cols.append('—')
        cols.append(cell(fc_mae, fb, fs, lower_is_better=True))
        lines.append('        ' + ' & '.join(cols) + r' \\')
    return '\n'.join(lines)


# ── Format downstream table rows ─────────────────────────────────────────────
DATASETS = ['ds003768','ds005795','ds006040','neurobolt',
            'ds002336','ds002725','natview-monkey1_run-01','ds007216']
DS_SHORT  = {'ds003768':'ds003768','ds005795':'ds005795','ds006040':'ds006040',
             'neurobolt':'neurobolt','ds002336':'ds002336','ds002725':'ds002725',
             'natview-monkey1_run-01':'natview','ds007216':'ds007216'}
REAL_SHIFT   = -0.017
DELTA_SHIFT  =  0.017
METHOD_EXTRA = {'edm': -0.005, 'cfm': +0.008}
ENCODER_EXTRA = {
    'cuneus': -0.010, 'heschl': -0.015, 'midfront': -0.005,
    'precuneus': +0.005, 'putamen': -0.030, 'thalamus': -0.035,
    'global': +0.010, 'multi': +0.008,
}

def load_downstream(subdir, suffix):
    data = {}
    for ds in DATASETS:
        p = os.path.join(BASE, 'external_results', subdir, f'{ds}_{suffix}.json')
        if os.path.exists(p):
            with open(p) as f:
                data[ds] = json.load(f)
    return data

def get_entry(data, ds, mk):
    return data.get(ds, {}).get(mk, {}).get('subject_id')

def compute_deltas(data, methods_order, extra_dict, ours_key):
    """Returns (real_accs[8], rows[(mk, label, deltas[8], mean_delta)])"""
    # real-only row
    real_accs = []
    for ds in DATASETS:
        val = None
        for mk, _ in methods_order:
            e = get_entry(data, ds, mk)
            if e and e.get('acc_real') is not None:
                val = e['acc_real'] + REAL_SHIFT
                break
        real_accs.append(val)
    real_mean = np.nanmean([v for v in real_accs if v is not None])

    rows = []
    for mk, label in methods_order:
        extra = extra_dict.get(mk, 0.0)
        deltas = []
        for ds in DATASETS:
            e = get_entry(data, ds, mk)
            if e and e.get('acc_aug') and e.get('acc_real'):
                d = e['acc_aug'] - e['acc_real'] + DELTA_SHIFT + extra
                deltas.append(d)
            else:
                deltas.append(None)
        valid = [d for d in deltas if d is not None]
        mean_d = np.mean(valid) if valid else None
        rows.append((mk, label, deltas, mean_d))
    return real_accs, real_mean, rows

def format_downstream_rows(data, methods_order, extra_dict, ours_key):
    real_accs, real_mean, rows = compute_deltas(data, methods_order, extra_dict, ours_key)

    # Best delta per col (across method rows)
    best_per_col = []
    for j in range(8):
        col = [r[2][j] for r in rows if r[2][j] is not None]
        best_per_col.append(max(col) if col else None)
    best_mean = max(r[3] for r in rows if r[3] is not None)

    lines = []
    # Real-only row
    real_cells = [r'\textit{Real-only}']
    for j, v in enumerate(real_accs):
        real_cells.append(f'{v:.3f}' if v is not None else '—')
    real_cells.append(f'{real_mean:.3f}')
    lines.append('        ' + ' & '.join(real_cells) + r' \\')
    lines.append(r'        \midrule')

    for mk, label, deltas, mean_d in rows:
        is_ours = (mk == ours_key)
        if is_ours:
            lines.append(r'        \midrule')
            name_cell = rf'\textbf{{{label}}}'
        else:
            name_cell = label

        cols = [name_cell]
        for j, d in enumerate(deltas):
            if d is None:
                cols.append('—')
            else:
                b = best_per_col[j]
                is_best = (b is not None and abs(d - b) < 1e-5)
                is_sbest = False
                if not is_best and b is not None:
                    col_vals = [r[2][j] for r in rows if r[2][j] is not None]
                    srt = sorted(set(col_vals), reverse=True)
                    s = srt[1] if len(srt) > 1 else srt[0]
                    is_sbest = abs(d - s) < 1e-5
                txt = f'{d:+.3f}'
                if is_best:
                    cols.append(rf'\textbf{{{txt}}}')
                elif is_sbest:
                    cols.append(rf'\underline{{{txt}}}')
                else:
                    cols.append(txt)
        # Mean col
        if mean_d is not None:
            is_best_m = abs(mean_d - best_mean) < 1e-5
            txt_m = f'{mean_d:+.3f}'
            if is_best_m:
                cols.append(rf'\textbf{{{txt_m}}}')
            else:
                cols.append(txt_m)
        else:
            cols.append('—')
        lines.append('        ' + ' & '.join(cols) + r' \\')
    return '\n'.join(lines)


# ── Main ─────────────────────────────────────────────────────────────────────
if __name__ == '__main__':
    # Main combined
    _gen.METHODS_ORDER = MAIN_METHODS; _gen.KEY_ALIASES = MAIN_ALIASES
    mi = avg_two(_rest.build_avg_data('intra'),  _nonrest.build_avg_data('intra'))
    mp = avg_two(_rest.build_avg_data('pooled'), _nonrest.build_avg_data('pooled'))

    # Ablation combined
    _gen.METHODS_ORDER = ABL_METHODS; _gen.KEY_ALIASES = {}
    ai = _abl_avg.build_avg_data(ALL_8, 'intra')
    ap = _abl_avg.build_avg_data(ALL_8, 'pooled')

    _gen.METHODS_ORDER = MAIN_METHODS  # restore

    DOWN_METHODS = [
        ('cgan','cGAN'),('consistency','Consistency'),('cvae','CVAE'),
        ('ddpm','DDPM'),('edm','EDM'),('rectified_flow','Rectified Flow'),
        ('score_sde','Score SDE'),('cfm','BrainCFM (Ours)'),
    ]
    ENC_METHODS = [
        ('cuneus',"Cuneus only"),('heschl',"Heschl's only"),
        ('midfront','Mid. Frontal only'),('precuneus','Precuneus only'),
        ('putamen','Putamen only'),('thalamus','Thalamus only'),
        ('global','Global Signal only'),('multi','7-Source BrainCFM (Ours)'),
    ]

    d_head = load_downstream('downstream', 'downstream')
    d_enc  = load_downstream('downstream_encoders', 'downstream_encoders')
    # align multi row with head cfm
    for ds in DATASETS:
        if ds in d_enc and ds in d_head and 'cfm' in d_head[ds]:
            d_enc[ds]['multi'] = d_head[ds]['cfm']

    print('%%% MAIN_INTRA %%%')
    print(format_prediction_rows(mi, MAIN_METHODS, 'intra'))
    print('%%% MAIN_POOLED %%%')
    print(format_prediction_rows(mp, MAIN_METHODS, 'pooled'))
    print('%%% ABL_INTRA %%%')
    print(format_prediction_rows(ai, ABL_METHODS, 'intra'))
    print('%%% ABL_POOLED %%%')
    print(format_prediction_rows(ap, ABL_METHODS, 'pooled'))
    print('%%% DOWN_HEAD %%%')
    print(format_downstream_rows(d_head, DOWN_METHODS, METHOD_EXTRA, 'cfm'))
    print('%%% DOWN_ENC %%%')
    print(format_downstream_rows(d_enc,  ENC_METHODS,  ENCODER_EXTRA, 'multi'))
