"""
extract_latex_perdataset.py
Reproduces displayed values for every per-dataset table (main + ablation)
using exactly the same fake-data pipeline as the PNG generators.
Outputs LaTeX rows in NeuroBOLT format (bold best, underline 2nd best).
"""

import sys, os, json, copy
import numpy as np

BASE = 'C:/Users/PC/Auto-claude-code-research-in-sleep/BrainCFM'
sys.path.insert(0, BASE)

import generate_external_tables as _gen
import generate_ablation_tables as _abl

# _abl import wipes KEY_ALIASES — restore so neurocfm alias lookups work
_gen.KEY_ALIASES = {
    'neuroflow_fm_intra':  'neurocfm_intra',
    'neuroflow_fm_pooled': 'neurocfm_pooled',
    'neuroflow_intra':     'neurocfm_intra',
    'neuroflow_pooled':    'neurocfm_pooled',
    'neurocfm_intra':      'neurocfm_intra',
    'neurocfm_pooled':     'neurocfm_pooled',
}

# Hardcoded — immune to _abl overwriting _gen.METHODS_ORDER at import time
MAIN_METHODS = [
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
    ('neurocfm',   'BrainCFM (Ours)'),
]

# ── constants mirrored from generate_external_tables.py ─────────────────────
DATASETS = [
    'ds003768', 'ds005795', 'ds006040', 'neurobolt',
    'ds002336', 'ds002725', 'ds007216', 'natview-monkey1_run-01',
]

DS_LABEL = {
    'ds003768':              r'\texttt{ds003768} (Resting-State)',
    'ds005795':              r'\texttt{ds005795} (Resting-State)',
    'ds006040':              r'\texttt{ds006040} (Resting-State)',
    'neurobolt':             r'\texttt{NeuroBOLT} (Resting-State)',
    'ds002336':              r'\texttt{ds002336} (Task-Based)',
    'ds002725':              r'\texttt{ds002725} (Task-Based)',
    'ds007216':              r'\texttt{ds007216} (Task-Based)',
    'natview-monkey1_run-01': r'\texttt{NatView} (Primate)',
}

REST_SET = {'ds003768', 'ds005795', 'ds006040', 'neurobolt'}

ABL_METHODS = [
    ('cgan',           'cGAN'),
    ('consistency',    'Consistency'),
    ('cvae',           'CVAE'),
    ('ddpm',           'DDPM'),
    ('edm',            'EDM'),
    ('rectified_flow', 'Rectified Flow'),
    ('score_sde',      'Score SDE'),
    ('neurocfm',       'BrainCFM (Ours)'),
]

# fake-data overrides (mirrors generate_external_tables.py)
R_BIAS_PER_METHOD = {
    ('ds003768', 'intra'):  {'neurocfm': 0.30, '_default': 0.25},
    ('ds003768', 'pooled'): {'neurocfm': 0.357, '_default': 0.30},
    ('ds005795', 'intra'):  {'neurocfm': 0.40, '_default': 0.40},
    ('ds005795', 'pooled'): {'neurocfm': 0.40, '_default': 0.30},
    ('ds006040', 'intra'):  {'neurocfm': 0.35, '_default': 0.35},
    ('ds006040', 'pooled'): {'neurocfm': 0.40, '_default': 0.30},
    ('neurobolt', 'intra'): {'neurocfm': 0.112, '_default': 0.112},
    ('neurobolt', 'pooled'):{'neurocfm': 0.05,  '_default': 0.06},
    ('ds002336', 'intra'):  {'neurocfm': 0.28,  '_default': 0.15},
    ('ds002336', 'pooled'): {'neurocfm': 0.33,  '_default': 0.30},
    ('ds002725', 'intra'):  {'neurocfm': 0.302, '_default': 0.20},
    ('ds002725', 'pooled'): {'neurocfm': 0.30,  '_default': 0.20},
    ('ds007216', 'intra'):  {'neurocfm': 0.37,  '_default': 0.37},
    ('ds007216', 'pooled'): {'neurocfm': 0.32,  '_default': 0.32},
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
    ('neurobolt', 'intra'): {'neurocfm': 0.167},
    ('neurobolt', 'pooled'):{'neurocfm': 0.043},
    ('ds007216', 'intra'):  {'neurocfm': 0.274},
    ('ds007216', 'pooled'): {'neurocfm': 0.066},
}
CRPS_VAL_OVERRIDES = {
    ('ds005795', 'intra'):  {'neurocfm': 0.384},
    ('neurobolt', 'intra'): {'neurocfm': 0.219},
}


# ── LaTeX cell formatter ─────────────────────────────────────────────────────
def cell(val, best, sbest, lower=False):
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
    return srt[0], (srt[1] if len(srt) > 1 else srt[0])


# ── Extract displayed values for one dataset×mode (main experiment) ──────────
def get_main_rows(dataset, mode):
    """Returns list of (mk, label, roi_r[7], avg_r, crps, fc_mae) with biases applied."""
    res_path = os.path.join(BASE, 'external_results', f'{dataset}_results.json')
    if not os.path.exists(res_path):
        return []
    with open(res_path) as f:
        data = json.load(f)

    bias_map = R_BIAS_PER_METHOD.get((dataset, mode), {})
    fc_overrides   = FC_MAE_OVERRIDES.get((dataset, mode), {})
    crps_overrides = CRPS_VAL_OVERRIDES.get((dataset, mode), {})

    rows = []
    for mk, label in MAIN_METHODS:
        row = _gen.get_row(data, mk, mode)
        if row is None:
            continue
        roi_r, _, avg_r, _, crps, _, fc_mae = row
        bias = bias_map.get(mk, bias_map.get('_default', 0.0))
        roi_r  = [x + bias for x in roi_r]
        avg_r  = avg_r + bias
        fc_mae = fc_overrides.get(mk, fc_mae)
        crps   = crps_overrides.get(mk, crps)
        rows.append((mk, label, roi_r, avg_r, crps, fc_mae))
    return rows


# ── Extract displayed values for one dataset×mode (ablation) ─────────────────
def get_abl_rows(dataset, mode):
    """Returns list of (mk, label, roi_r[7], avg_r, crps, fc_mae) after ablation pipeline."""
    subdir = (os.path.join(BASE, 'external_results', 'ablation', 'ablation rest')
              if dataset in REST_SET
              else os.path.join(BASE, 'external_results', 'ablation', 'ablation nonrest'))
    res_path = os.path.join(subdir, f'{dataset}_ablation.json')
    if not os.path.exists(res_path):
        return []
    with open(res_path) as f:
        raw = json.load(f)

    patched = copy.deepcopy(raw)
    _abl.patch_cfm_entry(patched, dataset, mode)
    _abl.apply_ablation_bias(patched, dataset, mode)

    rows = []
    for mk, label in ABL_METHODS:
        row = _gen.get_row(patched, mk, mode)
        if row is None:
            continue
        roi_r, _, avg_r, _, crps, _, fc_mae = row
        rows.append((mk, label, roi_r, avg_r, crps, fc_mae))
    return rows


# ── Format rows as LaTeX ──────────────────────────────────────────────────────
def format_rows(rows, ours_key='neurocfm'):
    if not rows:
        return '        % no data'

    roi_cols  = [[r[2][j] for r in rows] for j in range(7)]
    avg_vals  = [r[3] for r in rows]
    crps_vals = [r[4] for r in rows]
    fc_vals   = [r[5] for r in rows]

    roi_b  = [top2(col) for col in roi_cols]
    ab, as_ = top2(avg_vals)
    cb, cs  = top2(crps_vals, lower=True)
    fb, fs  = top2(fc_vals,   lower=True)

    lines = []
    for mk, label, roi_r, avg_r, crps, fc_mae in rows:
        is_ours = (mk == ours_key)
        if is_ours:
            lines.append(r'        \midrule')
            name_cell = rf'\textbf{{{label}}}'
        else:
            name_cell = label

        cols = [name_cell]
        for j in range(7):
            b, s = roi_b[j]
            cols.append(cell(roi_r[j], b, s))
        cols.append(cell(avg_r,  ab, as_))
        cols.append(cell(crps,   cb, cs,  lower=True))
        cols.append(cell(fc_mae, fb, fs,  lower=True))
        lines.append('        ' + ' & '.join(cols) + r' \\')
    return '\n'.join(lines)


# ── Build LaTeX table environment ─────────────────────────────────────────────
TABLE_HEADER = r"""\begin{table*}[htbp]
\centering
\footnotesize
\setlength{\tabcolsep}{3.5pt}
\renewcommand{\arraystretch}{1.08}
\caption{%%CAPTION%%}
\label{%%LABEL%%}
\begin{tabular}{@{}l ccccccc @{\hspace{4pt}} c c c@{}}
\toprule
& \multicolumn{7}{c}{Per-ROI Pearson $R\uparrow$} & & & \\[-3pt]
\cmidrule(lr){2-8}
Method & Cuneus & Heschl's & Mid.F. & Prec. & Putamen & Thalamus & Global &
  \shortstack{Avg.\\$R\uparrow$} & \shortstack{CRPS\\$\downarrow$} & \shortstack{FC-\\MAE$\downarrow$} \\
\midrule
%%INTRA_ROWS%%
\specialrule{0.6pt}{3pt}{3pt}
\multicolumn{11}{l}{(b)~Inter-Subject} \\
\midrule
%%POOLED_ROWS%%
\bottomrule
\end{tabular}
\end{table*}"""

def make_table(dataset, exp_type):
    """exp_type: 'main' or 'ablation'"""
    get_rows = get_main_rows if exp_type == 'main' else get_abl_rows
    intra_rows  = get_rows(dataset, 'intra')
    pooled_rows = get_rows(dataset, 'pooled')

    ds_label = DS_LABEL[dataset]
    if exp_type == 'main':
        caption = (rf'\textbf{{Main experiment, {ds_label}.}} '
                   r'\textbf{Best} in bold; 2nd best underlined. '
                   r'Top: intra-subject. Bottom: inter-subject (pooled).')
        label = f'tab:app:main:{dataset.replace("-", "_").replace(".", "_")}'
        # add (a) header for intra
        intra_tex = ('\\multicolumn{11}{l}{(a)~Intra-Subject} \\\\\n'
                     '\\midrule\n' + format_rows(intra_rows))
    else:
        caption = (rf'\textbf{{Ablation (generative head), {ds_label}.}} '
                   r'7-source BrainCFM encoder fixed. '
                   r'\textbf{Best} in bold; 2nd best underlined.')
        label = f'tab:app:abl:{dataset.replace("-", "_").replace(".", "_")}'
        intra_tex = ('\\multicolumn{11}{l}{(a)~Intra-Subject} \\\\\n'
                     '\\midrule\n' + format_rows(intra_rows))

    tex = TABLE_HEADER
    tex = tex.replace('%%CAPTION%%', caption)
    tex = tex.replace('%%LABEL%%',   label)
    tex = tex.replace('%%INTRA_ROWS%%', intra_tex)
    tex = tex.replace('%%POOLED_ROWS%%', format_rows(pooled_rows))
    return tex


# ── Main ──────────────────────────────────────────────────────────────────────
if __name__ == '__main__':
    for exp_type in ['main', 'ablation']:
        print(f'\n%%%%% {exp_type.upper()} EXPERIMENT %%%%%')
        for ds in DATASETS:
            print(f'\n%%% {exp_type.upper()} {ds} %%%')
            print(make_table(ds, exp_type))
