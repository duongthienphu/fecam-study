"""Validate E3 paired runs and report per-seed plus mean±sample-std results."""

import argparse
import csv
import json
import statistics
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


ROOT = Path(__file__).resolve().parents[1]
METHODS = {
    ('ncm', 'none'): 'Euclidean NCM',
    ('fecam', 'shared'): 'FeCAM shared covariance',
    ('fecam', 'classwise'): 'FeCAM classwise covariance',
}
SEEDS = (1, 2, 3)
FIELDS = (
    'row_type', 'seed', 'method', 'classifier', 'covariance_mode',
    'checkpoint_sha256', 'avg_acc_percent', 'final_acc_percent',
    'avg_acc_std_percent', 'final_acc_std_percent',
    'avg_delta_vs_ncm_pp', 'final_delta_vs_ncm_pp',
    'avg_delta_std_pp', 'final_delta_std_pp',
)


def load_summary(path):
    summary = json.loads(Path(path).read_text(encoding='utf-8'))
    if summary.get('status') != 'completed' or summary.get('action') != 'evaluate':
        raise ValueError(f'Not a completed evaluation summary: {path}')
    if summary.get('config', {}).get('experiment') not in ('e1-table1', 'e3-seed-stability'):
        raise ValueError(f'Not an E1/E3 summary: {path}')
    if not summary.get('checkpoint_sha256') or not summary.get('class_order_sha256'):
        raise ValueError(f'Missing checkpoint or class-order hash: {path}')
    return summary


def key_for(summary):
    classifier = summary['evaluation']['classifier']
    key = (classifier['name'], classifier.get('covariance_mode', 'none'))
    if key not in METHODS:
        raise ValueError(f'Unexpected E3 method: {key}')
    return key


def validate_summaries(summaries):
    if len(summaries) != 9:
        raise ValueError(f'Expected 9 summaries (3 methods × 3 seeds), got {len(summaries)}')
    by_seed = {seed: {} for seed in SEEDS}
    reference_config = None
    classifier_configs = {}
    class_orders = set()
    checkpoints_by_seed = {}
    for summary in summaries:
        config = summary['config']
        seed = config['seed']
        if seed not in by_seed:
            raise ValueError(f'Unexpected E3 training seed: {seed}')
        experiment = config['experiment']
        if (seed == 1 and experiment != 'e1-table1') or (seed != 1 and experiment != 'e3-seed-stability'):
            raise ValueError(f'Seed {seed} summary has unexpected experiment type {experiment}')
        key = key_for(summary)
        if key in by_seed[seed]:
            raise ValueError(f'Duplicate method {key} for seed {seed}')
        by_seed[seed][key] = summary
        if key in classifier_configs and classifier_configs[key] != config['classifier']:
            raise ValueError(f'Classifier settings for {key} changed between seeds')
        classifier_configs[key] = config['classifier']
        if config['split']['class_order_seed'] != 1993:
            raise ValueError('E3 must keep class_order_seed=1993 for every training seed')
        if config['training']['weight_decay'] != 0.0001:
            raise ValueError('E3 configs must match the checkpoint weight_decay=0.0001')
        class_orders.add(summary['class_order_sha256'])
        checkpoints_by_seed.setdefault(seed, set()).add(summary['checkpoint_sha256'])
        comparable_config = config.copy()
        for field in ('classifier', 'experiment', 'seed', 'paper_reference', 'require_checkpoint'):
            comparable_config.pop(field, None)
        if reference_config is None:
            reference_config = comparable_config
        elif comparable_config != reference_config:
            raise ValueError('E3 runs differ in protocol/training settings beyond classifier and seed')
        sessions = summary['evaluation']['sessions']
        if [row['task'] for row in sessions] != list(range(6)):
            raise ValueError(f'Seed {seed} must contain session 0 and incremental sessions 1–5')
        if [row['classes_seen'] for row in sessions] != [50, 60, 70, 80, 90, 100]:
            raise ValueError(f'Seed {seed} session class counts do not match B=50/increment=10')

    expected_methods = set(METHODS)
    for seed, methods in by_seed.items():
        if set(methods) != expected_methods:
            raise ValueError(f'Seed {seed} does not contain exactly the three E1 methods')
        if len(checkpoints_by_seed[seed]) != 1:
            raise ValueError(f'The three methods for seed {seed} do not share one checkpoint')
        hashes = {summary['checkpoint_sha256'] for summary in methods.values()}
        if len(hashes) != 1:
            raise ValueError(f'Checkpoint mismatch for seed {seed}')
    if len(class_orders) != 1:
        raise ValueError('E3 runs do not use the same class order')
    seed_hashes = {seed: next(iter(values)) for seed, values in checkpoints_by_seed.items()}
    if len(set(seed_hashes.values())) != 3:
        raise ValueError('Expected a distinct trained checkpoint for each of seeds 1, 2, and 3')
    return by_seed


def metric_value(summary, metric):
    key = 'avg_acc' if metric == 'avg' else 'last_acc'
    return float(summary['evaluation']['metrics'][key])


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('summaries', nargs=9, help='Nine summaries: three methods for each of seeds 1, 2, 3')
    parser.add_argument('--output-dir', default='results/e3')
    args = parser.parse_args(argv)
    summaries = [load_summary(path) for path in args.summaries]
    by_seed = validate_summaries(summaries)

    output_dir = Path(args.output_dir)
    if not output_dir.is_absolute():
        output_dir = ROOT / output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    rows = []
    deltas = {key: {'avg': [], 'final': []} for key in METHODS}
    scores = {key: {'avg': [], 'final': []} for key in METHODS}

    for seed in SEEDS:
        ncm = by_seed[seed][('ncm', 'none')]
        for key, summary in by_seed[seed].items():
            avg = metric_value(summary, 'avg')
            final = metric_value(summary, 'final')
            delta_avg = avg - metric_value(ncm, 'avg')
            delta_final = final - metric_value(ncm, 'final')
            scores[key]['avg'].append(avg)
            scores[key]['final'].append(final)
            deltas[key]['avg'].append(delta_avg)
            deltas[key]['final'].append(delta_final)
            rows.append({
                'row_type': 'seed', 'seed': seed, 'method': METHODS[key],
                'classifier': key[0], 'covariance_mode': key[1],
                'checkpoint_sha256': summary['checkpoint_sha256'],
                'avg_acc_percent': avg, 'final_acc_percent': final,
                'avg_acc_std_percent': '', 'final_acc_std_percent': '',
                'avg_delta_vs_ncm_pp': delta_avg,
                'final_delta_vs_ncm_pp': delta_final,
                'avg_delta_std_pp': '', 'final_delta_std_pp': '',
            })

    aggregate = {}
    for key in METHODS:
        avg_mean = statistics.mean(scores[key]['avg'])
        final_mean = statistics.mean(scores[key]['final'])
        avg_std = statistics.stdev(scores[key]['avg'])
        final_std = statistics.stdev(scores[key]['final'])
        avg_delta_mean = statistics.mean(deltas[key]['avg'])
        final_delta_mean = statistics.mean(deltas[key]['final'])
        avg_delta_std = statistics.stdev(deltas[key]['avg'])
        final_delta_std = statistics.stdev(deltas[key]['final'])
        aggregate[key] = {
            'avg_mean': avg_mean, 'avg_std': avg_std,
            'final_mean': final_mean, 'final_std': final_std,
            'avg_delta_mean': avg_delta_mean, 'avg_delta_std': avg_delta_std,
            'final_delta_mean': final_delta_mean, 'final_delta_std': final_delta_std,
        }
        rows.append({
            'row_type': 'mean±sample_std', 'seed': '1–3', 'method': METHODS[key],
            'classifier': key[0], 'covariance_mode': key[1],
            'checkpoint_sha256': '',
            'avg_acc_percent': avg_mean, 'final_acc_percent': final_mean,
            'avg_acc_std_percent': avg_std, 'final_acc_std_percent': final_std,
            'avg_delta_vs_ncm_pp': avg_delta_mean,
            'final_delta_vs_ncm_pp': final_delta_mean,
            'avg_delta_std_pp': avg_delta_std,
            'final_delta_std_pp': final_delta_std,
        })

    csv_path = output_dir / 'e3_seed_stability.csv'
    with csv_path.open('w', newline='', encoding='utf-8-sig') as handle:
        writer = csv.DictWriter(handle, fieldnames=FIELDS)
        writer.writeheader()
        writer.writerows(rows)

    fig, axes = plt.subplots(1, 2, figsize=(12, 5), constrained_layout=True)
    for axis, metric, title in zip(axes, ('avg', 'final'), ('Average accuracy', 'Final accuracy')):
        x = range(len(METHODS))
        keys = list(METHODS)
        means = [aggregate[key][f'{metric}_mean'] for key in keys]
        stds = [aggregate[key][f'{metric}_std'] for key in keys]
        axis.errorbar(x, means, yerr=stds, fmt='o', capsize=5, linewidth=1.8,
                      color='#174a7e', label='Mean ± sample SD')
        for index, key in enumerate(keys):
            axis.scatter([index] * len(SEEDS), scores[key][metric],
                         color='#f28e2b', alpha=0.8, zorder=3)
        axis.set_xticks(list(x), [METHODS[key] for key in keys], rotation=18, ha='right')
        axis.set_ylabel('Accuracy (%)')
        axis.set_title(title)
        axis.grid(True, axis='y', alpha=0.3)
    axes[0].legend()
    plot_path = output_dir / 'e3_seed_stability.png'
    fig.savefig(plot_path, dpi=180)
    plt.close(fig)
    print(f'Wrote {csv_path}\nWrote {plot_path}')
    for key in METHODS:
        result = aggregate[key]
        print(
            f"{METHODS[key]}: Avg {result['avg_mean']:.3f} ± {result['avg_std']:.3f}; "
            f"Final {result['final_mean']:.3f} ± {result['final_std']:.3f}"
        )


if __name__ == '__main__':
    main()
