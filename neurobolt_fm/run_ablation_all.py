"""
run_ablation_all.py — Master runner for generative head ablation study.

Runs 8 heads × 8 datasets × 2 modes (intra + pooled).
After each dataset completes, generates comparison table.

Usage:
  python run_ablation_all.py
  python run_ablation_all.py --heads edm cfm --datasets neurobolt ds003768
  python run_ablation_all.py --modes intra
"""

import subprocess, sys, os, json, time, argparse

BASE = 'C:/Users/PC/Auto-claude-code-research-in-sleep/neurobolt_fm'
ABL_DIR = os.path.join(BASE, 'external_results', 'ablation')

ALL_DATASETS = [
    # Resting-state
    'ds003768', 'ds005795', 'ds006040', 'neurobolt',
    # Task
    'ds002336', 'ds002725', 'natview-monkey1_run-01', 'ds007216',
]

ALL_HEADS = ['ddpm', 'edm', 'score_sde', 'consistency', 'cvae', 'cgan',
             'rectified_flow', 'cfm']

ALL_MODES = ['intra', 'pooled']


def run_one(head, dataset, mode, epochs=300):
    """Run one ablation experiment via subprocess."""
    cmd = [
        sys.executable, os.path.join(BASE, 'train_ablation.py'),
        '--head', head,
        '--mode', mode,
        '--dataset', dataset,
        '--epochs', str(epochs),
    ]
    print(f'\n  >>> {head} {mode} on {dataset}')
    t0 = time.time()
    result = subprocess.run(cmd, capture_output=False)
    elapsed = time.time() - t0
    status = 'OK' if result.returncode == 0 else f'FAIL (rc={result.returncode})'
    print(f'  {status} ({elapsed:.0f}s)')
    return result.returncode == 0


def generate_table(dataset):
    """Generate ablation table for one dataset."""
    cmd = [
        sys.executable, os.path.join(BASE, 'generate_ablation_tables.py'),
        '--datasets', dataset,
        '--modes', 'intra', 'pooled',
    ]
    subprocess.run(cmd, capture_output=False)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--heads', nargs='+', default=ALL_HEADS)
    parser.add_argument('--datasets', nargs='+', default=ALL_DATASETS)
    parser.add_argument('--modes', nargs='+', default=ALL_MODES)
    parser.add_argument('--epochs', type=int, default=300)
    args = parser.parse_args()

    os.makedirs(ABL_DIR, exist_ok=True)
    total = len(args.heads) * len(args.datasets) * len(args.modes)
    done, failed = 0, 0
    t_start = time.time()

    print('='*60)
    print(f'  ABLATION: {len(args.heads)} heads × {len(args.datasets)} datasets × {len(args.modes)} modes = {total} runs')
    print('='*60)

    for ds in args.datasets:
        print(f'\n{"="*60}')
        print(f'  DATASET: {ds}')
        print(f'{"="*60}')

        for head in args.heads:
            for mode in args.modes:
                # Check if already done
                res_path = os.path.join(ABL_DIR, f'{ds}_ablation.json')
                key = f'{head}_{mode}'
                if os.path.exists(res_path):
                    with open(res_path) as f:
                        existing = json.load(f)
                    if key in existing:
                        print(f'  >>> {head} {mode} on {ds} — SKIP (already done)')
                        done += 1
                        continue

                ok = run_one(head, ds, mode, args.epochs)
                done += 1
                if not ok:
                    failed += 1

        # Generate tables after each dataset
        print(f'\n  Generating tables for {ds}...')
        generate_table(ds)

    elapsed = (time.time() - t_start) / 60
    print(f'\n{"="*60}')
    print(f'  ALL DONE in {elapsed:.1f} min  ({done} runs, {failed} failed)')
    print(f'  Tables: {ABL_DIR}')
    print(f'{"="*60}')


if __name__ == '__main__':
    main()
