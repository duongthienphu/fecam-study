"""Upload all nine paired E3 runs to W&B, grouped by backbone training seed."""

import argparse
import os
from pathlib import Path

from report_e3 import METHODS, key_for, load_summary, metric_value, validate_summaries


ROOT = Path(__file__).resolve().parents[1]


def load_api_key():
    env_path = ROOT / '.env'
    if env_path.is_file():
        for line in env_path.read_text(encoding='utf-8').splitlines():
            key, separator, value = line.partition('=')
            if separator and key.strip() == 'WANDB_API_KEY':
                os.environ.setdefault('WANDB_API_KEY', value.strip().strip('"\''))
                break
    if not os.environ.get('WANDB_API_KEY'):
        raise RuntimeError('Set WANDB_API_KEY in the environment or local ignored .env file')


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('summaries', nargs=9, help='Nine E1/E3 summaries for seeds 1, 2, and 3')
    parser.add_argument('--project', default='fecam-study')
    parser.add_argument('--entity', help='W&B entity; defaults to the logged-in account')
    parser.add_argument('--report', help='Optional E3 CSV to attach to each run')
    parser.add_argument('--plot', help='Optional E3 plot to attach to each run')
    args = parser.parse_args(argv)

    summaries = [load_summary(path) for path in args.summaries]
    by_seed = validate_summaries(summaries)
    load_api_key()
    import wandb

    for summary, summary_arg in zip(summaries, args.summaries):
        config = summary['config']
        seed = config['seed']
        key = key_for(summary)
        ncm = by_seed[seed][('ncm', 'none')]
        metrics = summary['evaluation']['metrics']
        run = wandb.init(
            project=args.project,
            entity=args.entity,
            group=f'e3-seed-{seed}',
            job_type='e3-seed-stability',
            name=f'e3-{key[0]}-{key[1]}-seed-{seed}',
            tags=['e3-seed-stability', f'seed-{seed}', key[0], key[1], config['protocol']],
            config=config,
            notes=(
                'Paired E3 run: all three methods for this training seed share one '
                'backbone checkpoint and class order. Weight decay is 0.0001 '
                'for ResNet-18, matching the paper and checkpoint metadata.'
            ),
        )
        try:
            for session in summary['evaluation']['sessions']:
                run.log({
                    'task': session['task'],
                    'classes_seen': session['classes_seen'],
                    'seen_acc_percent': session['seen_accuracy'],
                    **{f'task_{index}_acc_percent': value
                       for index, value in enumerate(session['task_accuracies'])},
                }, step=session['task'])
            run.summary.update({
                'avg_acc_percent': metrics['avg_acc'],
                'final_acc_percent': metrics['last_acc'],
                'bwt_percent': metrics['bwt'],
                'avg_delta_vs_ncm_pp': metric_value(summary, 'avg') - metric_value(ncm, 'avg'),
                'final_delta_vs_ncm_pp': metric_value(summary, 'final') - metric_value(ncm, 'final'),
                'checkpoint_sha256': summary['checkpoint_sha256'],
                'class_order_sha256': summary['class_order_sha256'],
                'group_weight_decay': config['training']['weight_decay'],
                'paper_weight_decay': 0.0001,
                'summary_path': str(Path(summary_arg).resolve()),
            })
            artifact = wandb.Artifact(
                name=f'{run.id}-e3-result', type='e3-result',
                metadata={'seed': seed, 'checkpoint_sha256': summary['checkpoint_sha256']},
            )
            artifact.add_file(str(Path(summary_arg).resolve()))
            for optional_path in (args.report, args.plot):
                if optional_path:
                    artifact.add_file(str(Path(optional_path).resolve()))
            run.log_artifact(artifact)
        finally:
            run.finish()
        print(f"seed={seed} {METHODS[key]}: {run.url}", flush=True)


if __name__ == '__main__':
    main()
