"""Validate paired E1 summaries and create CSV tables plus a task-accuracy plot."""

import argparse
import csv
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


ROOT = Path(__file__).resolve().parents[1]
PAPER_AVG = {
    ('ncm', 'none'): 64.8,
    ('fecam', 'shared'): 68.8,
    ('fecam', 'classwise'): 70.9,
}
DISPLAY = {
    ('ncm', 'none'): 'Euclidean NCM',
    ('fecam', 'shared'): 'FeCAM shared covariance',
    ('fecam', 'classwise'): 'FeCAM classwise covariance',
}


def load_summary(path):
    summary = json.loads(Path(path).read_text(encoding='utf-8'))
    if summary.get('status') != 'completed' or summary.get('action') != 'evaluate':
        raise ValueError(f'Not a completed evaluation summary: {path}')
    if summary.get('config', {}).get('experiment') != 'e1-table1':
        raise ValueError(f'Not an E1 summary: {path}')
    if not summary.get('checkpoint_sha256'):
        raise ValueError(f'Missing checkpoint SHA-256: {path}')
    return summary


def method_key(summary):
    classifier = summary['evaluation']['classifier']
    name = classifier['name']
    mode = classifier.get('covariance_mode', 'none' if name == 'ncm' else None)
    key = (name, mode)
    if key not in PAPER_AVG:
        raise ValueError(f'Unexpected E1 classifier/mode: {key}')
    return key


def check_pairing(summaries):
    keys = [method_key(summary) for summary in summaries]
    if set(keys) != set(PAPER_AVG) or len(set(keys)) != 3:
        raise ValueError(f'Expected exactly the three E1 methods, got {keys}')
    checkpoint_hashes = {summary['checkpoint_sha256'] for summary in summaries}
    class_order_hashes = {summary.get('class_order_sha256') for summary in summaries}
    if len(checkpoint_hashes) != 1:
        raise ValueError('E1 runs do not use the same checkpoint SHA-256')
    if len(class_order_hashes) != 1 or None in class_order_hashes:
        raise ValueError('E1 runs do not have the same recorded class-order hash')
    reference = summaries[0]['config'].copy()
    reference.pop('classifier', None)
    reference.pop('paper_reference', None)
    for summary in summaries[1:]:
        config = summary['config'].copy()
        config.pop('classifier', None)
        config.pop('paper_reference', None)
        if config != reference:
            raise ValueError('E1 configs differ beyond classifier settings')
        if summary['class_order'] != summaries[0]['class_order']:
            raise ValueError('E1 class orders differ')
    for summary in summaries:
        sessions = summary['evaluation']['sessions']
        if [session['task'] for session in sessions] != list(range(6)):
            raise ValueError('Expected six evaluations: base task 0 and incremental tasks 1–5')
        if [session['classes_seen'] for session in sessions] != [50, 60, 70, 80, 90, 100]:
            raise ValueError('E1 session class counts do not match B=50, increment=10')
        key = method_key(summary)
        paper_value = summary['config'].get('paper_reference', {}).get('avg_acc_percent')
        if paper_value != PAPER_AVG[key]:
            raise ValueError(f'Paper reference mismatch for {key}: {paper_value}')


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('summaries', nargs=3, help='Three E1 outputs/summary.json files')
    parser.add_argument('--output-dir', default='results/e1')
    args = parser.parse_args(argv)

    summaries = [load_summary(path) for path in args.summaries]
    check_pairing(summaries)
    output_dir = Path(args.output_dir)
    if not output_dir.is_absolute():
        output_dir = ROOT / output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    summary_rows = []
    task_rows = []
    for summary in summaries:
        key = method_key(summary)
        metrics = summary['evaluation']['metrics']
        summary_rows.append({
            'method': DISPLAY[key],
            'classifier': key[0],
            'covariance_mode': key[1],
            'seed': summary['config']['seed'],
            'checkpoint_sha256': summary['checkpoint_sha256'],
            'class_order_sha256': summary['class_order_sha256'],
            'group_weight_decay': summary['config']['training']['weight_decay'],
            'paper_weight_decay': 0.0001,
            'weight_decay_matches_paper': summary['config']['training']['weight_decay'] == 0.0001,
            'avg_acc_percent': metrics['avg_acc'],
            'final_acc_percent': metrics['last_acc'],
            'paper_table1_avg_acc_percent': PAPER_AVG[key],
            'delta_vs_paper_avg_pp': metrics['avg_acc'] - PAPER_AVG[key],
        })
        for session in summary['evaluation']['sessions']:
            task_rows.append({
                'method': DISPLAY[key],
                'classifier': key[0],
                'covariance_mode': key[1],
                'task': session['task'],
                'classes_seen': session['classes_seen'],
                'seen_acc_percent': session['seen_accuracy'],
            })

    summary_path = output_dir / 'e1_table1_comparison.csv'
    task_path = output_dir / 'e1_seen_accuracy_by_task.csv'
    for path, rows in ((summary_path, summary_rows), (task_path, task_rows)):
        with path.open('w', newline='', encoding='utf-8-sig') as handle:
            writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)

    figure, axis = plt.subplots(figsize=(8, 5), constrained_layout=True)
    for key in PAPER_AVG:
        rows = [row for row in task_rows if (row['classifier'], row['covariance_mode']) == key]
        axis.plot(
            [row['task'] for row in rows],
            [row['seen_acc_percent'] for row in rows],
            marker='o', linewidth=2, label=DISPLAY[key],
        )
    axis.set(title='E1: CIFAR-100 seen-class accuracy by session',
             xlabel='Session (0 = base, 1–5 = incremental)', ylabel='Seen accuracy (%)')
    axis.set_xticks(range(6))
    axis.grid(True, alpha=0.3)
    axis.legend()
    plot_path = output_dir / 'e1_seen_accuracy.png'
    figure.savefig(plot_path, dpi=180)
    plt.close(figure)
    print(f'Paired runs: checkpoint={summaries[0]["checkpoint_sha256"]}')
    print(f'Wrote {summary_path}\nWrote {task_path}\nWrote {plot_path}')


if __name__ == '__main__':
    main()
