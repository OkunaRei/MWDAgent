"""Prepare frozen evidence, expose bounded observations, then compare cached orders."""
import argparse
from contextlib import redirect_stdout
import csv
import io
import json
from pathlib import Path

from run_mwd_agent import ROOT, TRAIN_SOURCE, sha256, dependencies, session_lock, latest_state
from src.agent_loop import MIDPOINT_POLICY, observe
from src.benchmark import SEED_LIST, run_validation_once
from src.contract import load_dataset
from src.decision_benchmark import start, execute, compact_observation, compare, validate_table

CODE = ('run_decision_benchmark.py', 'run_mwd_agent.py', 'src/decision_benchmark.py',
        'src/agent_loop.py', 'src/benchmark.py', 'src/contract.py', 'src/evaluate.py',
        'src/sampling.py', 'src/class_impact.py', 'docs/research-plan/decision-host-prompt.md')


def write_new(path, value):
    payload = json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False)
    with path.open('x', encoding='utf-8') as handle:
        handle.write(payload + '\n')


def fingerprints():
    return {'source_sha256': sha256(TRAIN_SOURCE), 'code_sha256': {name: sha256(ROOT/name) for name in CODE},
            'packages': dependencies()}


def prepare(directory):
    directory.mkdir(parents=True, exist_ok=False)
    before = fingerprints()
    protocol = {**before, 'policy': MIDPOINT_POLICY, 'budget_including_baseline': 3,
                'preparation_fits': 40, 'selection_budget_fits': 15,
                'random_comparator': 'all 42 ordered pairs without replacement',
                'fixed_order': MIDPOINT_POLICY['catalog'][:3],
                'disclosure': 'Known dataset and historically explored actions; fresh bounded host context only'}
    write_new(directory/'protocol.json', protocol)
    frame = load_dataset(TRAIN_SOURCE)
    files = {}
    for index, candidate in enumerate(MIDPOINT_POLICY['catalog']):
        feature, model, strategy = candidate.split('__')
        runs = []
        for seed in SEED_LIST:
            with redirect_stdout(io.StringIO()):
                run = run_validation_once(frame, model_name=model, feature_group=feature, seed=seed,
                    validation_size=.2, selection_weights=(.2, .5, .3), training_strategy=strategy,
                    include_diagnostics=True)
            runs.append(run)
            print(f'Prepared candidate {index+1}/8 seed {seed}', flush=True)
        name = f'candidate-{index:02d}.json'
        write_new(directory/name, {'candidate': candidate, 'runs': runs})
        files[candidate] = {'path': name, 'sha256': sha256(directory/name)}
    if fingerprints() != before:
        raise ValueError('Source, code or packages changed during preparation')
    write_new(directory/'manifest.json', {'protocol_sha256': sha256(directory/'protocol.json'), 'files': files})
    load_table(directory)


def load_table(directory):
    manifest = json.loads((directory/'manifest.json').read_text())
    if sha256(directory/'protocol.json') != manifest['protocol_sha256']:
        raise ValueError('Protocol checksum mismatch')
    protocol = json.loads((directory/'protocol.json').read_text())
    current = fingerprints()
    if any(protocol[key] != value for key, value in current.items()):
        raise ValueError('Frozen input fingerprint mismatch')
    candidates = {}
    if set(manifest['files']) != set(MIDPOINT_POLICY['catalog']):
        raise ValueError('Incomplete catalogue manifest')
    for index, candidate in enumerate(MIDPOINT_POLICY['catalog']):
        record = manifest['files'][candidate]
        if record['path'] != f'candidate-{index:02d}.json':
            raise ValueError('Unexpected candidate filename')
        path = directory/record['path']
        if sha256(path) != record['sha256']:
            raise ValueError('Candidate checksum mismatch')
        entry = json.loads(path.read_text())
        if entry['candidate'] != candidate:
            raise ValueError('Candidate identity mismatch')
        candidates[candidate] = entry['runs']
    table = {'source_sha256': protocol['source_sha256'], 'policy': protocol['policy'], 'candidates': candidates}
    validate_table(table)
    return table


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('command', choices=('prepare', 'observe', 'act', 'compare'))
    parser.add_argument('--dir', type=Path, required=True)
    parser.add_argument('--action', help='Exact JSON action, not a file path')
    args = parser.parse_args()
    if (args.command == 'act') != (args.action is not None):
        parser.error('--action is required only for act')
    if args.command == 'prepare':
        prepare(args.dir)
        return
    with session_lock(args.dir):
        table = load_table(args.dir)
        paths = list(args.dir.glob('state-*.json'))
        if not paths and args.command != 'observe':
            raise ValueError('Initialize with observe first')
        state = latest_state(args.dir) if paths else start(table)
        if not paths:
            write_new(args.dir/'state-0000.json', state)
        if args.command == 'act':
            updated = execute(state, table, json.loads(args.action))
            write_new(args.dir/f"state-{updated['revision']:04d}.json", updated)
            state = updated
        if args.command == 'compare':
            result = compare(state, table)
            write_new(args.dir/'comparison.json', result)
            with (args.dir/'comparison.csv').open('x', newline='', encoding='utf-8') as handle:
                writer = csv.DictWriter(handle, fieldnames=list(result['rows'][0]))
                writer.writeheader()
                writer.writerows({**row, 'candidate_order': json.dumps(row['candidate_order'])} for row in result['rows'])
            print(json.dumps({key: value for key, value in result.items() if key != 'rows'}, indent=2))
        else:
            observation_path = args.dir/f"observation-{state['revision']:04d}.json"
            if not observation_path.exists():
                write_new(observation_path, observe(state))
            print(json.dumps(compact_observation(state), indent=2, ensure_ascii=False))


if __name__ == '__main__':
    main()
