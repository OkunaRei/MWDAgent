"""Replay three frozen validation evaluations for a host stop decision; never train."""
import argparse
from copy import deepcopy
import hashlib
import json
from pathlib import Path

import pandas as pd

from src.agent_loop import MIDPOINT_POLICY, initialize, apply_action, observe, observation_digest
from src.benchmark import SEED_LIST
from src.class_impact import compare_class_impact


def build_replay(baseline, weighted, midpoint, source_sha256):
    tables = {}
    for strategy, runs in zip(('paper_smote', 'class_weight', 'midpoint_smote'), (baseline, weighted, midpoint)):
        if [row['seed'] for row in runs] != list(SEED_LIST):
            raise ValueError('Replay requires all fixed seeds exactly once in order')
        for row in runs:
            if (row['training_strategy'] != strategy or row['model'] != 'lightgbm'
                    or row['feature_group'] != 'all_48' or row['validation_size'] != .2):
                raise ValueError('Replay candidate or validation protocol mismatch')
        compare_class_impact(baseline, runs)
        tables[strategy] = {row['seed']: row for row in runs}

    def cached(frame, **kwargs):
        return deepcopy(tables[kwargs['training_strategy']][kwargs['seed']])

    state = initialize(pd.DataFrame(), source_sha256=source_sha256, budget=4,
                       policy=MIDPOINT_POLICY, evaluator=cached)
    states = [state]
    for strategy in ('class_weight', 'midpoint_smote'):
        action = {'action': 'run_candidate', 'candidate': f'all_48__lightgbm__{strategy}',
            'reason': 'Import previously measured evidence; this is not a new host proposal or training run.',
            'hypothesis': 'Recompute promotion with frozen v3 class guard.',
            'physical_risk': 'Retrospective development evidence only.',
            'observation_sha256': observation_digest(state)}
        state = apply_action(state, pd.DataFrame(), action, evaluator=cached)
        states.append(state)
    return states


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--impact-dir', type=Path, default=Path('reports/strategy_impact/2026-09-09'))
    parser.add_argument('--midpoint-state', type=Path, default=Path('reports/agent_loop/midpoint-v3/state-0001.json'))
    parser.add_argument('--output-dir', type=Path, required=True)
    parser.add_argument('--stop-action', type=Path)
    args = parser.parse_args()
    paths = [args.impact_dir / name for name in ('protocol.json', 'paper_smote.json', 'class_weight.json')]
    paths.append(args.midpoint_state)
    blobs = [path.read_bytes() for path in paths]
    protocol, baseline, weighted, old_state = [json.loads(blob) for blob in blobs]
    if protocol['source_sha256'] != old_state['source_sha256']:
        raise ValueError('Evidence source hashes differ')
    if old_state['policy'] != MIDPOINT_POLICY:
        raise ValueError('Midpoint policy differs from frozen v3')
    if old_state['evaluations'][0]['runs'] != baseline:
        raise ValueError('Historical baselines are not identical')
    item = next(row for row in old_state['evaluations'] if row['candidate'] == 'all_48__lightgbm__midpoint_smote')
    if item['status'] != 'completed':
        raise ValueError('Midpoint evidence is incomplete')
    states = build_replay(baseline, weighted, item['runs'], protocol['source_sha256'])
    if args.stop_action:
        action = json.loads(args.stop_action.read_text())
        if action.get('action') != 'stop':
            raise ValueError('This replay endpoint only accepts a host stop action')
        def forbidden(*args, **kwargs):
            raise RuntimeError('Replay stop must never train')
        states.append(apply_action(states[-1], pd.DataFrame(), action, evaluator=forbidden))
    metadata = {'mode': 'cached_evidence_replay', 'new_model_fits': 0, 'historical_model_fits': 15,
        'historical_code_hashes': {'impact': protocol.get('code_sha256'),
                                   'midpoint': old_state.get('provenance', {}).get('code_sha256')},
        'inputs': {str(path): hashlib.sha256(blob).hexdigest() for path, blob in zip(paths, blobs)},
        'note': 'Replay timings measure cache access, not training; unused budget is not measured compute savings.'}
    args.output_dir.mkdir(parents=True, exist_ok=False)
    (args.output_dir/'provenance.json').write_text(json.dumps(metadata, indent=2))
    for state in states:
        (args.output_dir/f"state-{state['revision']:04d}.json").write_text(json.dumps(state, indent=2, allow_nan=False))
    print(json.dumps({'observation': observe(states[-1]), 'observation_sha256': observation_digest(states[-1])}, indent=2))


if __name__ == '__main__':
    main()
