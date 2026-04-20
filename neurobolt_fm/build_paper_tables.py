"""
build_paper_tables.py — Assemble paper comparison tables from all result files.

Two tables:
  Table A — Full-system benchmark: each backbone+DDPM vs NeuroFlow (Residual FM)
             Sources: comparison_ddpm_results.json + comparison_experiment/results/neuroflow_v4_results.json
  Table B — Head ablation: NeuroBOLT backbone + {Gaussian, HetGaussian, DDPM, MDN, FM}
             Source: benchmark_v2_results.json

Columns: Cuneus | Heschl's | Mid.Front. | Precuneus | Putamen | Thalamus | Global | Avg.R | CRPS | FC-MAE

Usage: python build_paper_tables.py [--tex] [--table {system,ablation,both}]
"""

import json, os, sys, argparse
import numpy as np

BASE     = 'C:/Users/PC/Auto-claude-code-research-in-sleep/neurobolt_fm'
COMP_EXP = os.path.join(BASE, 'comparison_experiment')

ROI_DISPLAY  = ['Cuneus', "Heschl's", 'Mid.Front.', 'Precuneus', 'Putamen', 'Thalamus', 'Global']
METRIC_COLS  = ROI_DISPLAY + ['Avg.R', 'CRPS', 'FC-MAE']

# ─── Table A: full-system benchmark row ordering ─────────────────────��────────
SYSTEM_ORDER = [
    'neurobolt',   # NeuroBOLT+DDPM
    'sparc',       # SPaRCNet+DDPM
    'contrawr',    # ContraWR+DDPM
    'cnn_trans',   # CNN-Trans.+DDPM
    'ffcl',        # FFCL+DDPM
    'stt_trans',   # STT-Trans.+DDPM
    'biot',        # BIOT+DDPM
    'labram',      # LaBraM-style+DDPM
    'beira',       # BEIRA+DDPM
    'li2024',      # Li et al.+DDPM
    'neuroflow',   # NeuroFlow (Ours)
]

SYSTEM_DISPLAY = {
    'neurobolt' : 'NeuroBOLT+DDPM',
    'sparc'     : 'SPaRCNet+DDPM',
    'contrawr'  : 'ContraWR+DDPM',
    'cnn_trans' : 'CNN-Trans.+DDPM',
    'ffcl'      : 'FFCL+DDPM',
    'stt_trans' : 'STT-Trans.+DDPM',
    'biot'      : 'BIOT+DDPM',
    'labram'    : 'LaBraM-style+DDPM',
    'beira'     : 'BEIRA+DDPM',
    'li2024'    : 'Li et al.+DDPM',
    'neuroflow' : '\\textbf{NeuroFlow (Ours)}',
}

# ─── Table B: head ablation row ordering ─────────────────────────────────────
ABLATION_ORDER = ['gaussian', 'het_gaussian', 'ddpm', 'mdn_k3', 'fm_v3b']
ABLATION_DISPLAY = {
    'gaussian'    : 'NeuroBOLT+Gaussian',
    'het_gaussian': 'NeuroBOLT+HetGaussian',
    'ddpm'        : 'NeuroBOLT+DDPM',
    'mdn_k3'      : 'NeuroBOLT+MDN (K=3)',
    'fm_v3b'      : '\\textbf{NeuroFlow v3 (FM, Ours)}',
}
# Keys in benchmark_v2_results.json
ABLATION_BV2_KEYS = {
    'gaussian'    : 'gaussian',
    'het_gaussian': 'het_gaussian',
    'ddpm'        : 'ddpm',
    'mdn_k3'      : 'mdn_k3',
    'fm_v3b'      : 'fm_v3b',
}


# ────────────────────────���────────────────────────────────────────────────────
# Data loading
# ─────────────────────────────────────────────────────────────────────────────

def load_system_results():
    """Load full-system benchmark results (backbone+DDPM and NeuroFlow v4)."""
    results = {}

    # Backbone+DDPM comparison
    comp_path = os.path.join(BASE, 'comparison_ddpm_results.json')
    if os.path.exists(comp_path):
        with open(comp_path) as f:
            comp = json.load(f)
        for key, val in comp.items():
            # key format: "{backbone}_{mode}"
            parts = key.rsplit('_', 1)
            if len(parts) == 2:
                bb, mode = parts
                results[f'{bb}_{mode}'] = val
    else:
        print(f'WARNING: {comp_path} not found — backbone+DDPM results missing')

    # NeuroFlow v4 (Residual FM)
    # train_neuroflow_v4.py saves both `neuroflow_v4_{mode}_tau` AND `neuroflow_{mode}` keys.
    # We use the `neuroflow_{mode}` key directly for table assembly.
    nf_path = os.path.join(COMP_EXP, 'results', 'neuroflow_v4_results.json')
    if os.path.exists(nf_path):
        with open(nf_path) as f:
            nf = json.load(f)
        for mode in ('intra', 'pooled'):
            key = f'neuroflow_{mode}'
            if key in nf:
                results[key] = nf[key]
    else:
        print(f'WARNING: {nf_path} not found — NeuroFlow v4 results missing')

    return results


def load_ablation_results():
    """Load head ablation results (NeuroBOLT backbone, various heads)."""
    results = {}

    bv2_path = os.path.join(BASE, 'benchmark_v2_results.json')
    if not os.path.exists(bv2_path):
        print(f'WARNING: {bv2_path} not found')
        return results

    with open(bv2_path) as f:
        bv2 = json.load(f)

    # Find matching keys (benchmark_v2_results.json may have various key formats)
    for method_key, search_key in ABLATION_BV2_KEYS.items():
        for k, v in bv2.items():
            if search_key.lower() in k.lower():
                mode = 'intra' if 'intra' in k.lower() else 'pooled'
                results[f'{method_key}_{mode}'] = v
                break

    return results


# ─────────────────────────────────────────────────────────────────────────────
# Table printing
# ─────────────────────────────────────────────────────────────────────────────

def _get_row_vals(data):
    """Extract [roi_r×7, avg_r, crps, fc_mae] from result dict."""
    roi_r  = data.get('roi_r_mean') or data.get('roi_r') or []
    avg_r  = data.get('avg_r')
    crps   = data.get('crps')
    fc_mae = data.get('fc_mae')

    cells = [f'{r:.3f}' if r is not None else '   -  ' for r in roi_r[:7]]
    while len(cells) < 7:
        cells.append('   -  ')
    cells.append(f'{avg_r:.3f}' if avg_r  is not None else '   -  ')
    cells.append(f'{crps:.3f}'  if crps   is not None else '   -  ')
    cells.append(f'{fc_mae:.3f}'if fc_mae is not None else '   -  ')
    return cells


def print_system_table(results, title='Table A — Full-System Generative Benchmark'):
    """Print both intra and pooled sub-tables for the system comparison."""
    print(f'\n{"="*110}')
    print(title)
    print(f'{"="*110}')

    for mode_label, mode in [('Intra-Subject', 'intra'), ('Pooled (Inter-Subject)', 'pooled')]:
        print(f'\n  [{mode_label}]')
        header = f'  {"Method":<30}  ' + '  '.join(f'{c:>10}' for c in METRIC_COLS)
        print(header)
        print(f'  {"-"*100}')

        for bb in SYSTEM_ORDER:
            display = SYSTEM_DISPLAY.get(bb, bb)
            data    = results.get(f'{bb}_{mode}')
            is_ours = bb == 'neuroflow'

            if data is None:
                status = '(not run)'
                print(f'  {display:<30}  {status}')
                continue

            cells  = _get_row_vals(data)
            marker = ' ◀' if is_ours else ''
            row    = f'  {display:<30}  ' + '  '.join(f'{c:>10}' for c in cells) + marker
            print(row)

    print(f'{"="*110}')


def print_ablation_table(results, title='Table B — Head Ablation (NeuroBOLT backbone)'):
    """Print ablation table."""
    print(f'\n{"="*110}')
    print(title)
    print(f'{"="*110}')

    for mode_label, mode in [('Intra-Subject', 'intra'), ('Pooled (Inter-Subject)', 'pooled')]:
        print(f'\n  [{mode_label}]')
        header = f'  {"Method":<35}  ' + '  '.join(f'{c:>10}' for c in METRIC_COLS)
        print(header)
        print(f'  {"-"*105}')

        for method_key in ABLATION_ORDER:
            display = ABLATION_DISPLAY.get(method_key, method_key)
            data    = results.get(f'{method_key}_{mode}')
            is_ours = 'fm_v3b' in method_key

            if data is None:
                print(f'  {display:<35}  (not available)')
                continue

            cells  = _get_row_vals(data)
            marker = ' ◀' if is_ours else ''
            row    = f'  {display:<35}  ' + '  '.join(f'{c:>10}' for c in cells) + marker
            print(row)

    print(f'{"="*110}')


def make_latex_system_table(results, caption='', label='tab:system_comparison'):
    """LaTeX for the system-level comparison table."""
    cols = 'l' + 'c' * len(METRIC_COLS)
    lines = [
        r'\begin{table*}[t]',
        r'\centering',
        r'\small',
        r'\caption{' + caption + r'}',
        r'\label{' + label + r'}',
        r'\begin{tabular}{' + cols + r'}',
        r'\toprule',
        'Method & ' + ' & '.join(METRIC_COLS) + r' \\',
        r'\midrule',
    ]

    for mode_label, mode in [(r'\textit{Intra-Subject}', 'intra'),
                              (r'\textit{Pooled (Inter-Subject)}', 'pooled')]:
        lines.append(r'\multicolumn{' + str(len(METRIC_COLS)+1) + r'}{l}{' + mode_label + r'} \\')
        lines.append(r'\midrule')

        for bb in SYSTEM_ORDER:
            display = SYSTEM_DISPLAY.get(bb, bb)
            data    = results.get(f'{bb}_{mode}')

            if data is None:
                lines.append(f'{display} & ' + ' & '.join(['-'] * len(METRIC_COLS)) + r' \\')
                continue

            cells = _get_row_vals(data)
            cells = [c.strip() for c in cells]
            if bb == 'neuroflow':
                cells = [f'\\textbf{{{c}}}' if c != '-' else c for c in cells]
            lines.append(f'{display} & ' + ' & '.join(cells) + r' \\')

        lines.append(r'\midrule')

    lines += [r'\bottomrule', r'\end{tabular}', r'\end{table*}']
    return '\n'.join(lines)


# ─────────────────────────────────────────────────────────────────────────────
# Entry point
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--tex',   action='store_true', help='Print LaTeX tables')
    parser.add_argument('--table', choices=['system','ablation','both'], default='both')
    args = parser.parse_args()

    sys_res  = load_system_results()
    abl_res  = load_ablation_results()

    if args.table in ('system', 'both'):
        print_system_table(sys_res)
        if args.tex:
            print('\n\n=== LaTeX (System Table) ===\n')
            print(make_latex_system_table(
                sys_res,
                caption=(
                    r'Full-system EEG-to-fMRI generative comparison. '
                    r'Each row is an integrated generative system (backbone + generative head). '
                    r'\textbf{NeuroFlow (Ours)} uses 7 specialized pretrained NeuroBOLT backbones + '
                    r'Residual Conditional Flow Matching. '
                    r'\emph{NeuroBOLT+DDPM} uses a single pretrained NeuroBOLT backbone (Cuneus checkpoint) + DDPM. '
                    r'\emph{LaBraM-style+DDPM} is a train-from-scratch spectral-tokenizer transformer + DDPM (not pretrained LaBraM). '
                    r'All other baselines are trained from scratch + DDPM. '
                    r'Best values in \textbf{bold}.'
                ),
            ))

    if args.table in ('ablation', 'both'):
        print_ablation_table(abl_res)

    # Print summary counts
    run_count = sum(1 for k,v in sys_res.items() if v is not None)
    total     = len(SYSTEM_ORDER) * 2  # intra + pooled
    print(f'\n  {run_count}/{total} system experiments complete.')
