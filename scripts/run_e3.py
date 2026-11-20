"""Run the six new E3 evaluations for seeds 2 and 3, reusing E1 seed-1 runs.

Seed-2/3 checkpoints are loaded when present. With --train-missing-checkpoints,
the script trains a missing base checkpoint and resumes an interrupted base
training run from its atomic per-epoch checkpoint.
"""

import argparse
import hashlib
import json
import subprocess
import sys
from pathlib import Path

import yaml

from report_e1 import check_pairing as check_e1_pairing
from report_e1 import load_summary as load_e1_summary


ROOT = Path(__file__).resolve().parents[1]
METHODS = (
    ('ncm', 'configs/experiments/e1_ncm_euclidean.yaml'),
    ('fecam_shared', 'configs/experiments/e1_fecam_shared.yaml'),
    ('fecam_classwise', 'configs/experiments/e1_fecam_classwise.yaml'),
)


def read_yaml(path):
    return yaml.safe_load(Path(path).read_text(encoding='utf-8'))


def write_yaml(path, config):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(yaml.safe_dump(config, sort_keys=False), encoding='utf-8')
    return path


def get_checkpoint(template, seed):
    if '{seed}' not in str(template):
        raise ValueError('--checkpoint-template must contain the literal {seed}')
    return Path(str(template).format(seed=seed)).expanduser().resolve()


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def prepare_checkpoint(seed, checkpoint, train_missing, config_dir, device):
    if checkpoint.is_file():
        return
    if not train_missing:
        raise FileNotFoundError(
            f'Missing seed-{seed} checkpoint: {checkpoint}. Supply it or use '
            '--train-missing-checkpoints to train/resume the backbone.'
        )

    base_config = read_yaml(ROOT / METHODS[0][1])
    base_config['experiment'] = 'e3-seed-stability'
    base_config['seed'] = seed
    base_config['require_checkpoint'] = False
    config_path = write_yaml(config_dir / f'e3_train_base_seed{seed}.yaml', base_config)
    output_dir = checkpoint.parent
    training_checkpoint = output_dir / 'base_training_checkpoint.pt'
    summary_path = output_dir / 'summary.json'
    if output_dir.exists():
        if summary_path.is_file():
            raise FileNotFoundError(
                f'{summary_path} exists but final base.pt is missing; inspect that '
                'checkpoint directory before restarting training.'
            )
        if not training_checkpoint.is_file():
            raise FileExistsError(
                f'Checkpoint output directory already exists without a resumable '
                f'training checkpoint: {output_dir}'
            )
        resume = True
    else:
        resume = False
    command = [
        sys.executable, 'scripts/run_experiment.py', 'train-base',
        '--config', str(config_path), '--output', str(output_dir), '--device', device,
    ]
    if resume:
        command.append('--resume')
        print(f'Resuming seed-{seed} backbone from its saved epoch checkpoint.', flush=True)
    else:
        print(f'Training seed-{seed} backbone (200 epochs).', flush=True)
    subprocess.run(command, cwd=ROOT, check=True)
    if not checkpoint.is_file():
        raise FileNotFoundError(f'Training finished without producing {checkpoint}')


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--seed1-summaries', nargs=3, required=True,
                        help='E1 seed-1 summaries: NCM, shared, classwise')
    parser.add_argument('--checkpoint-template', required=True,
                        help='Checkpoint path template containing {seed}, e.g. outputs/e3/seed{seed}/base.pt')
    parser.add_argument('--output-dir', default='outputs/e3')
    parser.add_argument('--device', default='auto')
    parser.add_argument('--train-missing-checkpoints', action='store_true',
                        help='Train/resume missing seed-2/3 backbones using the E1 training config')
    args = parser.parse_args(argv)

    seed1_summaries = [load_e1_summary(path) for path in args.seed1_summaries]
    check_e1_pairing(seed1_summaries)
    if any(summary['config']['seed'] != 1 for summary in seed1_summaries):
        raise ValueError('E1 seed-1 summaries must all have training seed 1')

    output_root = Path(args.output_dir)
    if not output_root.is_absolute():
        output_root = ROOT / output_root
    config_dir = output_root / 'generated_configs'
    e3_summaries = list(args.seed1_summaries)

    for seed in (2, 3):
        checkpoint = get_checkpoint(args.checkpoint_template, seed)
        prepare_checkpoint(
            seed, checkpoint, args.train_missing_checkpoints, config_dir, args.device
        )
        checkpoint_hash = sha256(checkpoint)
        for method, template_path in METHODS:
            config = read_yaml(ROOT / template_path)
            config['experiment'] = 'e3-seed-stability'
            config['seed'] = seed
            config['require_checkpoint'] = True
            config_path = write_yaml(config_dir / f'e3_{method}_seed{seed}.yaml', config)
            output = output_root / 'evaluations' / f'seed{seed}' / method
            summary_path = output / 'summary.json'
            if summary_path.is_file():
                existing = json.loads(summary_path.read_text(encoding='utf-8'))
                if (existing.get('status') != 'completed'
                        or existing.get('checkpoint_sha256') != checkpoint_hash
                        or existing.get('config', {}).get('seed') != seed
                        or existing.get('evaluation', {}).get('classifier') != config['classifier']):
                    raise ValueError(f'Existing E3 output is incomplete or mismatched: {summary_path}')
                print(f'Reusing completed seed-{seed} {method} evaluation.', flush=True)
            else:
                if output.exists():
                    raise FileExistsError(
                        f'Incomplete output directory: {output}. Preserve or move it before rerunning.'
                    )
                command = [
                    sys.executable, 'scripts/run_experiment.py', 'evaluate',
                    '--config', str(config_path), '--checkpoint', str(checkpoint),
                    '--device', args.device, '--output', str(output),
                ]
                print(f'Evaluating seed-{seed} {method}.', flush=True)
                subprocess.run(command, cwd=ROOT, check=True)
            e3_summaries.append(str(summary_path))

    report_dir = output_root / 'report'
    report_command = [
        sys.executable, 'scripts/report_e3.py', *e3_summaries,
        '--output-dir', str(report_dir),
    ]
    subprocess.run(report_command, cwd=ROOT, check=True)
    print(f'E3 completed. Results: {report_dir}', flush=True)


if __name__ == '__main__':
    main()
