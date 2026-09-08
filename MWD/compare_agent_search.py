"""Replay matched-budget search orders over one frozen complete candidate table."""
import argparse
import hashlib
import json
from pathlib import Path

import pandas as pd

from src.agent_loop import BASELINE, observation_digest
from src.search_comparison import compare_search_orders


def load_session(session):
    paths = sorted(session.glob('state-*.json'))
    if not paths:
        raise ValueError('Session has no state snapshots')
    blobs = [path.read_bytes() for path in paths]
    states = [json.loads(blob) for blob in blobs]
    for index, state in enumerate(states):
        if state['revision'] != index:
            raise ValueError('Session revisions must be contiguous')
        if index == 0 and (state['actions'] or len(state['evaluations']) != 1
                           or state['evaluations'][0]['candidate'] != BASELINE):
            raise ValueError('Initial state must contain only the baseline and no actions')
        if index:
            previous = states[index - 1]
            if previous['status'] != 'active':
                raise ValueError('No actions are allowed after a stopped state')
            if state['source_sha256'] != previous['source_sha256'] or state['policy'] != previous['policy']:
                raise ValueError('Source or policy changed during session')
            if state['actions'][:-1] != previous['actions']:
                raise ValueError('Session action history was rewritten')
            if state['actions'][-1]['observation_sha256'] != observation_digest(previous):
                raise ValueError('Action does not reference preceding observation')
            if state['evaluations'][:len(previous['evaluations'])] != previous['evaluations']:
                raise ValueError('Previous candidate evaluations were rewritten')
            action = state['actions'][-1]
            if action['action'] == 'run_candidate':
                if (len(state['evaluations']) != len(previous['evaluations']) + 1
                        or state['evaluations'][-1]['candidate'] != action['candidate']):
                    raise ValueError('Action and candidate evaluation do not match')
            elif action['action'] != 'stop' or state['evaluations'] != previous['evaluations']:
                raise ValueError('Unexpected action or evaluations after stop')
    return states[-1], {path.name: hashlib.sha256(blob).hexdigest() for path, blob in zip(paths, blobs)}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--session', type=Path, default=Path('reports/agent_loop/2026-09-08'))
    parser.add_argument('--output-dir', type=Path, required=True)
    args = parser.parse_args()
    state, hashes = load_session(args.session)
    result = compare_search_orders(state)
    baseline = state['evaluations'][0]
    result = {**result, 'fixed_model_reference': {
        'candidate': baseline['candidate'], 'training_attempts': len(baseline['attempted_seeds']),
        'selection_score': baseline['summary']['selection_score_mean'],
        'macro_f1': baseline['summary']['macro_f1_mean'],
        'transition_macro_f1': baseline['summary']['transition_macro_f1_mean'],
        'replayed_evaluation_seconds': baseline['elapsed_seconds'],
        'note': 'Fixed model performs no additional search; its cost is not expanded to the search budget.'},
        'provenance': {'session': str(args.session), 'state_sha256': hashes,
        'source_sha256': state['source_sha256'],
        'comparison_code_sha256': {name: hashlib.sha256((Path(__file__).parent / name).read_bytes()).hexdigest()
                                  for name in ('compare_agent_search.py', 'src/search_comparison.py', 'src/agent_loop.py')},
        'new_model_fits': 0, 'new_llm_trials': 0,
        'limitations': 'Retrospective search-order replay on one frozen table. Random permutations are not independent model or LLM trials. Excludes host inference cost.'}}
    args.output_dir.mkdir(parents=True, exist_ok=False)
    (args.output_dir / 'comparison.json').write_text(json.dumps(result, indent=2, allow_nan=False), encoding='utf-8')
    pd.json_normalize(result['budget_summary']).to_csv(args.output_dir / 'budget_summary.csv', index=False)
    rows = []
    for policy in ('host_agent', 'fixed_order', 'random_search'):
        trajectories = result['trajectories'][policy]
        if policy != 'random_search':
            trajectories = [trajectories]
        for index, trajectory in enumerate(trajectories):
            rows.extend({'policy': policy, 'order_index': index, **step} for step in trajectory['steps'])
    pd.DataFrame(rows).to_csv(args.output_dir / 'trajectories.csv', index=False)
    print(json.dumps({'oracle': result['oracle'], 'budgets': result['budget_summary']}, indent=2, allow_nan=False))


if __name__ == '__main__':
    main()
