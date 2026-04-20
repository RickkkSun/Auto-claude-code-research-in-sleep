"""
run_downstream_all.py — Master runner for downstream augmentation experiments.

Runs 8 heads × 8 datasets, evaluating:
  - Subject identification (all datasets)
  - Brain state classification (task datasets only)

Metrics: TSTR, Augmentation Lift, Data Efficiency Ratio

Usage:
  python run_downstream_all.py
  python run_downstream_all.py --heads cfm ddpm --datasets neurobolt ds002725
"""

import subprocess, sys, os, json, time, argparse

BASE = 'C:/Users/PC/Auto-claude-code-research-in-sleep/neurobolt_fm'
DOWN_DIR = os.path.join(BASE, 'external_results', 'downstream')

ALL_DATASETS = [
    'ds003768', 'ds005795', 'ds006040', 'neurobolt',         # rest
    'ds002336', 'ds002725', 'natview-monkey1_run-01', 'ds007216',  # task
]

ALL_HEADS = ['ddpm', 'edm', 'score_sde', 'consistency',
             'cvae', 'cgan', 'rectified_flow', 'cfm']


def run_one(head, dataset, epochs=300):
    cmd = [
        sys.executable, os.path.join(BASE, 'train_downstream.py'),
        '--head', head,
        '--dataset', dataset,
        '--epochs', str(epochs),
    ]
    print(f'\n  >>> {head} on {dataset}')
    t0 = time.time()
    result = subprocess.run(cmd, capture_output=False)
    elapsed = time.time() - t0
    status = 'OK' if result.returncode == 0 else f'FAIL (rc={result.returncode})'
    print(f'  {status} ({elapsed:.0f}s)')
    return result.returncode == 0


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--heads', nargs='+', default=ALL_HEADS)
    parser.add_argument('--datasets', nargs='+', default=ALL_DATASETS)
    parser.add_argument('--epochs', type=int, default=300)
    args = parser.parse_args()

    os.makedirs(DOWN_DIR, exist_ok=True)
    total = len(args.heads) * len(args.datasets)
    done, failed = 0, 0
    t_start = time.time()

    print('='*60)
    print(f'  DOWNSTREAM: {len(args.heads)} heads × {len(args.datasets)} datasets = {total} runs')
    print('='*60)

    for ds in args.datasets:
        print(f'\n{"="*60}')
        print(f'  DATASET: {ds}')
        print(f'{"="*60}')

        for head in args.heads:
            # Skip if already done
            res_path = os.path.join(DOWN_DIR, f'{ds}_downstream.json')
            if os.path.exists(res_path):
                with open(res_path) as f:
                    existing = json.load(f)
                if head in existing:
                    print(f'  >>> {head} on {ds} — SKIP (already done)')
                    done += 1
                    continue

            ok = run_one(head, ds, args.epochs)
            done += 1
            if not ok:
                failed += 1

    elapsed = (time.time() - t_start) / 60
    print(f'\n{"="*60}')
    print(f'  ALL DONE in {elapsed:.1f} min  ({done} runs, {failed} failed)')
    print(f'  Results: {DOWN_DIR}')
    print(f'{"="*60}')


if __name__ == '__main__':
    main()
