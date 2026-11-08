"""Compare per-class and base/incremental accuracy for paired FeCAM runs."""

import argparse
import csv
import json
from pathlib import Path


def read_summary(path):
    with Path(path).open(encoding='utf-8') as handle:
        return json.load(handle)


def compatible_configs(full, identity):
    left = json.loads(json.dumps(full['config']))
    right = json.loads(json.dumps(identity['config']))
    if left['classifier'].get('mode', 'full') != 'full':
        raise ValueError('First input must be FeCAM mode=full')
    if right['classifier'].get('mode') != 'identity':
        raise ValueError('Second input must be FeCAM mode=identity')
    left['classifier'].pop('mode', None)
    right['classifier'].pop('mode', None)
    if left != right:
        raise ValueError('Run configs differ beyond classifier.mode; paired comparison is invalid')
    if full.get('class_order') != identity.get('class_order'):
        raise ValueError('Class orders differ')
    if (not full.get('checkpoint_sha256')
            or full.get('checkpoint_sha256') != identity.get('checkpoint_sha256')):
        raise ValueError('Runs did not use the exact same backbone checkpoint')


def flatten_classes(summary):
    result = {}
    for session in summary['evaluation']['sessions']:
        task = session['task']
        for row in session['per_class_accuracy']:
            result[(task, row['class_id'])] = row
    return result


def compare(full_path, identity_path, output_dir):
    full = read_summary(full_path)
    identity = read_summary(identity_path)
    compatible_configs(full, identity)
    full_rows = flatten_classes(full)
    identity_rows = flatten_classes(identity)
    if full_rows.keys() != identity_rows.keys():
        raise ValueError('The runs do not contain the same task/class evaluations')

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    per_class_path = output_dir / 'fecam_mode_per_class_comparison.csv'
    per_class_rows = []
    for key in sorted(full_rows):
        a, b = full_rows[key], identity_rows[key]
        if a['group'] != b['group'] or a['total'] != b['total']:
            raise ValueError(f'Mismatched class evaluation for task/class {key}')
        per_class_rows.append({
            'task': key[0], 'class_id': key[1], 'original_class_id': a['original_class_id'],
            'class_name': a['class_name'], 'group': a['group'], 'introduced_task': a['introduced_task'],
            'total': a['total'], 'full_correct': a['correct'], 'identity_correct': b['correct'],
            'full_accuracy_percent': a['accuracy_percent'],
            'identity_accuracy_percent': b['accuracy_percent'],
            'delta_identity_minus_full_pp': b['accuracy_percent'] - a['accuracy_percent'],
        })
    with per_class_path.open('w', newline='', encoding='utf-8') as handle:
        fields = list(per_class_rows[0]) if per_class_rows else [
            'task', 'class_id', 'original_class_id', 'class_name', 'group', 'introduced_task',
            'total', 'full_correct', 'identity_correct', 'full_accuracy_percent',
            'identity_accuracy_percent', 'delta_identity_minus_full_pp']
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(per_class_rows)

    group_path = output_dir / 'fecam_mode_group_comparison.csv'
    group_rows = []
    for task in sorted({key[0] for key in full_rows}):
        task_rows = [row for row in per_class_rows if row['task'] == task]
        for group in ('base', 'incremental', 'all_seen'):
            selected = task_rows if group == 'all_seen' else [row for row in task_rows if row['group'] == group]
            total = sum(row['total'] for row in selected)
            if not total:
                continue
            full_correct = sum(row['full_correct'] for row in selected)
            identity_correct = sum(row['identity_correct'] for row in selected)
            full_acc = 100.0 * full_correct / total
            identity_acc = 100.0 * identity_correct / total
            group_rows.append({
                'task': task, 'group': group, 'total': total,
                'full_accuracy_percent': full_acc, 'identity_accuracy_percent': identity_acc,
                'delta_identity_minus_full_pp': identity_acc - full_acc,
            })
    with group_path.open('w', newline='', encoding='utf-8') as handle:
        writer = csv.DictWriter(handle, fieldnames=(
            'task', 'group', 'total', 'full_accuracy_percent', 'identity_accuracy_percent',
            'delta_identity_minus_full_pp'))
        writer.writeheader()
        writer.writerows(group_rows)
    return per_class_path, group_path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--full-summary', required=True)
    parser.add_argument('--identity-summary', required=True)
    parser.add_argument('--output-dir', required=True)
    args = parser.parse_args()
    for path in compare(args.full_summary, args.identity_summary, args.output_dir):
        print(path)


if __name__ == '__main__':
    main()
