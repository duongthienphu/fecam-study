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

    full_memory = full.get('memory') or {}
    identity_memory = identity.get('memory') or {}

    def memory_value(summary_memory, key):
        return summary_memory.get(key)

    def memory_delta(key):
        left, right = memory_value(full_memory, key), memory_value(identity_memory, key)
        return None if left is None or right is None else right - left

    def final_matrix_storage(summary, field):
        rows = summary.get('evaluation', {}).get('matrix_storage_by_task', [])
        if not rows:
            return None
        state = rows[-1].get('matrix_state', {})
        return state.get(field)

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    per_class_rows = []
    per_class_rows = []
    for key in sorted(full_rows):
        a, b = full_rows[key], identity_rows[key]
        if a['group'] != b['group'] or a['class_role'] != b['class_role'] or a['total'] != b['total']:
            raise ValueError(f'Mismatched class evaluation for task/class {key}')
        per_class_rows.append({
            'task': key[0], 'class_name': a['class_name'],
            'original_class_id': a['original_class_id'], 'group': a['group'], 'total': a['total'],
            'full_accuracy_percent': a['accuracy_percent'],
            'identity_accuracy_percent': b['accuracy_percent'],
            'delta_identity_minus_full_pp': b['accuracy_percent'] - a['accuracy_percent'],
        })
    per_class_path = output_dir / 'fecam_mode_per_class_comparison.csv'
    per_class_fields = [
        'task', 'class_name', 'original_class_id', 'group', 'total', 'full_accuracy_percent',
        'identity_accuracy_percent', 'delta_identity_minus_full_pp']
    with per_class_path.open('w', newline='', encoding='utf-8') as handle:
        writer = csv.DictWriter(handle, fieldnames=per_class_fields)
        writer.writeheader()
        writer.writerows(per_class_rows)

    full_metrics = full['evaluation']['metrics']
    identity_metrics = identity['evaluation']['metrics']
    summary_row = {
        'protocol': full['config']['protocol'],
        'seed': full['config']['seed'],
        'checkpoint_sha256': full['checkpoint_sha256'],
        'full_avg_accuracy_percent': full_metrics['avg_acc'],
        'identity_avg_accuracy_percent': identity_metrics['avg_acc'],
        'delta_avg_accuracy_pp': identity_metrics['avg_acc'] - full_metrics['avg_acc'],
        'full_final_accuracy_percent': full_metrics['last_acc'],
        'identity_final_accuracy_percent': identity_metrics['last_acc'],
        'delta_final_accuracy_pp': identity_metrics['last_acc'] - full_metrics['last_acc'],
        'full_bwt_percent': full_metrics['bwt'],
        'identity_bwt_percent': identity_metrics['bwt'],
        'delta_bwt_pp': identity_metrics['bwt'] - full_metrics['bwt'],
        'full_peak_ram_rss_mb': memory_value(full_memory, 'peak_ram_rss_mb'),
        'identity_peak_ram_rss_mb': memory_value(identity_memory, 'peak_ram_rss_mb'),
        'delta_peak_ram_rss_mb': memory_delta('peak_ram_rss_mb'),
        'full_peak_vram_allocated_mb': memory_value(full_memory, 'peak_vram_allocated_mb'),
        'identity_peak_vram_allocated_mb': memory_value(identity_memory, 'peak_vram_allocated_mb'),
        'delta_peak_vram_allocated_mb': memory_delta('peak_vram_allocated_mb'),
        'full_peak_vram_reserved_mb': memory_value(full_memory, 'peak_vram_reserved_mb'),
        'identity_peak_vram_reserved_mb': memory_value(identity_memory, 'peak_vram_reserved_mb'),
        'delta_peak_vram_reserved_mb': memory_delta('peak_vram_reserved_mb'),
        # Actual backing-storage bytes for the persistent FeCAM matrix state,
        # measured from classifier.factors after the final task.
        'full_matrix_factor_ram_mib': final_matrix_storage(full, 'storage_ram_mib'),
        'identity_matrix_factor_ram_mib': final_matrix_storage(identity, 'storage_ram_mib'),
        'delta_matrix_factor_ram_mib': (
            final_matrix_storage(identity, 'storage_ram_mib')
            - final_matrix_storage(full, 'storage_ram_mib')
            if final_matrix_storage(full, 'storage_ram_mib') is not None
            and final_matrix_storage(identity, 'storage_ram_mib') is not None else None
        ),
        'full_matrix_factor_vram_mib': final_matrix_storage(full, 'storage_vram_mib'),
        'identity_matrix_factor_vram_mib': final_matrix_storage(identity, 'storage_vram_mib'),
        'delta_matrix_factor_vram_mib': (
            final_matrix_storage(identity, 'storage_vram_mib')
            - final_matrix_storage(full, 'storage_vram_mib')
            if final_matrix_storage(full, 'storage_vram_mib') is not None
            and final_matrix_storage(identity, 'storage_vram_mib') is not None else None
        ),
    }
    summary_path = output_dir / 'fecam_mode_summary_comparison.csv'
    summary_fields = list(summary_row)
    with summary_path.open('w', newline='', encoding='utf-8') as handle:
        writer = csv.DictWriter(handle, fieldnames=summary_fields)
        writer.writeheader()
        writer.writerow(summary_row)
    return per_class_path, summary_path


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
