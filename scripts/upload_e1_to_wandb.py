"""Upload the three paired E1 summaries to W&B as a grouped set of runs."""

import argparse
import os
from pathlib import Path

from report_e1 import DISPLAY, check_pairing, load_summary, method_key


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
    parser.add_argument('summaries', nargs=3, help='Three E1 outputs/summary.json files')
    parser.add_argument('--project', default='fecam-study')
    parser.add_argument('--entity', help='W&B entity; defaults to the logged-in account')
    parser.add_argument('--plot', help='Optional e1_seen_accuracy.png to attach to each run')
    args = parser.parse_args(argv)

    summaries = [load_summary(path) for path in args.summaries]
    check_pairing(summaries)
    load_api_key()
    import wandb

    for summary, summary_arg in zip(summaries, args.summaries):
        key = method_key(summary)
        config = summary['config']
        metrics = summary['evaluation']['metrics']
        run = wandb.init(
            project=args.project,
            entity=args.entity,
            group=f"e1-table1-seed-{config['seed']}",
            job_type='e1-table1',
            name=f"e1-{key[0]}-{key[1]}-seed-{config['seed']}",
            tags=['e1-table1', key[0], key[1], config['protocol']],
            config=config,
            notes=(
                'Paired E1 evaluation; all three classifiers use one checkpoint and '
                'the same CIFAR-100 class order. See paper-average reference and delta '
                'in the run summary.'
            ),
        )
        try:
            for session in summary['evaluation']['sessions']:
                run.log({
                    'task': session['task'],
                    'classes_seen': session['classes_seen'],
                    'seen_acc_percent': session['seen_accuracy'],
                    'plasticity_percent': session['task_accuracies'][-1],
                    **{f'task_{index}_acc_percent': value
                       for index, value in enumerate(session['task_accuracies'])},
                }, step=session['task'])
            run.summary.update({
                'avg_acc_percent': metrics['avg_acc'],
                'final_acc_percent': metrics['last_acc'],
                'bwt_percent': metrics['bwt'],
                'paper_table1_avg_acc_percent': config['paper_reference']['avg_acc_percent'],
                'delta_vs_paper_avg_pp': metrics['avg_acc'] - config['paper_reference']['avg_acc_percent'],
                'checkpoint_sha256': summary['checkpoint_sha256'],
                'class_order_sha256': summary['class_order_sha256'],
                'group_weight_decay': config['training']['weight_decay'],
                'paper_weight_decay': 0.0001,
                'weight_decay_matches_paper': config['training']['weight_decay'] == 0.0001,
                'summary_path': str(Path(summary_arg).resolve()),
            })
            artifact = wandb.Artifact(
                name=f"{run.id}-summary", type='e1-summary',
                metadata={'checkpoint_sha256': summary['checkpoint_sha256']},
            )
            artifact.add_file(str(Path(summary_arg).resolve()))
            if args.plot:
                artifact.add_file(str(Path(args.plot).resolve()))
            run.log_artifact(artifact)
        finally:
            run.finish()
        print(f"{DISPLAY[key]}: {run.url}", flush=True)


if __name__ == '__main__':
    main()
