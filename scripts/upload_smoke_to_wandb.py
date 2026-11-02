"""Upload completed smoke summaries to the W&B project fecam-study."""

import argparse
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def load_local_api_key():
    """Read WANDB_API_KEY from the ignored .env without printing its value."""
    env_path = ROOT / '.env'
    if env_path.is_file():
        for line in env_path.read_text(encoding='utf-8').splitlines():
            key, separator, value = line.partition('=')
            if separator and key.strip() == 'WANDB_API_KEY':
                os.environ.setdefault('WANDB_API_KEY', value.strip().strip('"\''))
                break
    if not os.environ.get('WANDB_API_KEY'):
        raise RuntimeError('WANDB_API_KEY is not configured in environment or local .env')


def get_project(api, entity, project_name):
    """Load an existing project; create it if the account does not have one yet."""
    project = api.project(project_name, entity)
    project._load()
    if not project._attrs:
        api.create_project(project_name, entity)
        api.flush()
        project = api.project(project_name, entity)
        project._load()
    query = '''
    query CheckProjectAccess($name: String!, $entity: String!) {
      project(name: $name, entityName: $entity) { name access }
    }
    '''
    info = api._service_api.execute_graphql(
        query, variables={'name': project_name, 'entity': entity}
    )['project']
    return project, info['access']


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--project', default='fecam-study')
    parser.add_argument('--entity', help='Defaults to the account linked to WANDB_API_KEY')
    parser.add_argument('--check-only', action='store_true', help='Check project access without creating runs')
    parser.add_argument('--summaries', nargs='+', default=[
        'outputs/smoke-ncm-20261102-v3/summary.json',
        'outputs/smoke-fecam-20261102-v3/summary.json',
    ])
    args = parser.parse_args(argv)

    load_local_api_key()
    import json
    import wandb

    api = wandb.Api()
    entity = args.entity or api.default_entity
    if not entity:
        raise RuntimeError('Could not determine W&B entity for this API key')
    project, access = get_project(api, entity, args.project)
    print(f'W&B project: {project.url} (access={access})', flush=True)
    if args.check_only:
        return

    for summary_path in args.summaries:
        path = Path(summary_path)
        if not path.is_absolute():
            path = ROOT / path
        if not path.is_file():
            raise FileNotFoundError(path)
        summary = json.loads(path.read_text(encoding='utf-8'))
        if summary.get('status') != 'completed' or summary.get('action') != 'evaluate':
            raise ValueError(f'Not a completed evaluation summary: {path}')
        evaluation = summary['evaluation']
        classifier = evaluation['classifier']['name']
        run = wandb.init(
            project=args.project,
            entity=entity,
            name=f"smoke-{classifier}-cpu-{path.parent.name}",
            job_type='smoke-test',
            tags=['smoke-test', classifier, 'cpu', summary['config']['protocol']],
            config=summary['config'],
            notes=(
                'Smoke test only: one epoch, 10 classes and reduced samples per class. '
                f"Dataset source mode: {summary.get('dataset_source_mode', 'unknown')}."
            ),
        )
        try:
            for session in evaluation['sessions']:
                row = {
                    'task': session['task'],
                    'classes_seen': session['classes_seen'],
                    'acc_all': session['seen_accuracy'],
                    'plasticity': session['task_accuracies'][-1],
                }
                row.update({f"acc_task_{i}": value for i, value in enumerate(session['task_accuracies'])})
                run.log(row, step=session['task'])
            run.summary.update({f'final/{key}': value for key, value in evaluation['metrics'].items()
                                if value is not None and not isinstance(value, list)})
            run.summary['dataset_source_mode'] = summary.get('dataset_source_mode', 'unknown')
            run.summary['summary_file'] = path.name
        finally:
            run.finish()
        print(f'Uploaded {classifier}: {run.url}', flush=True)


if __name__ == '__main__':
    main()
