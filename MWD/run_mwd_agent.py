"""Validation-only experiment tool for a language-model host: init, observe, act."""
from __future__ import annotations

import argparse
from contextlib import contextmanager
import hashlib
from importlib.metadata import version
import json
from pathlib import Path
import platform
import tempfile

from src.agent_loop import MIDPOINT_POLICY, CLASS_GUARD_POLICY, STRATEGY_POLICY, initialize, observe, observation_digest, apply_action
from src.contract import load_dataset


ROOT = Path(__file__).resolve().parent
TRAIN_SOURCE = ROOT / 'data/mwd_rocktype_10358374/mwd_rocktype_blastholes_model_ready_train.csv'
CODE_FILES = ('run_mwd_agent.py', 'src/agent_loop.py', 'src/benchmark.py',
              'src/contract.py', 'src/evaluate.py', 'src/sampling.py', 'src/class_impact.py')


def sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def code_hashes():
    return {name: sha256(ROOT / name) for name in CODE_FILES}


def dependencies():
    return {name: version(name) for name in ('numpy', 'pandas', 'scikit-learn', 'lightgbm', 'imbalanced-learn')}


@contextmanager
def session_lock(session):
    lock = session / '.lock'
    try:
        lock.mkdir()
    except FileExistsError:
        raise ValueError('Session is locked; check for an active or interrupted experiment') from None
    try:
        yield
    finally:
        lock.rmdir()


def latest_state(session):
    paths = sorted(session.glob('state-*.json'))
    if not paths:
        raise ValueError('No initialized session state')
    return json.loads(paths[-1].read_text(encoding='utf-8'))


def write_state(session, state):
    path = session / f"state-{state['revision']:04d}.json"
    if path.exists():
        raise ValueError('Revision already exists')
    payload = json.dumps(state, indent=2, ensure_ascii=False, allow_nan=False)
    with tempfile.NamedTemporaryFile(mode='w', encoding='utf-8', dir=session, suffix='.tmp', delete=False) as handle:
        temporary = Path(handle.name)
        handle.write(payload)
    temporary.replace(path)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('command', choices=('init', 'observe', 'act'))
    parser.add_argument('--session', type=Path, required=True)
    parser.add_argument('--budget', type=int, default=4)
    parser.add_argument('--action', type=Path)
    parser.add_argument('--strategy-actions', action='store_true', help='Initialize versioned six-candidate training-strategy policy')
    parser.add_argument('--class-guard', action='store_true', help='Use v2 class-recall feedback and guard')
    parser.add_argument('--midpoint-actions', action='store_true', help='Use v3 midpoint catalogue with class guard')
    args = parser.parse_args()
    if args.midpoint_actions and (args.command != 'init' or not args.strategy_actions or not args.class_guard):
        parser.error('--midpoint-actions requires init --strategy-actions --class-guard')
    if args.class_guard and (args.command != 'init' or not args.strategy_actions):
        parser.error('--class-guard requires init --strategy-actions')
    if args.strategy_actions and args.command != 'init':
        parser.error('--strategy-actions is only valid with init')
    if args.command == 'init':
        if args.action:
            parser.error('init does not accept --action')
        # Source is pinned in code: the host cannot substitute a test path in an action.
        digest = sha256(TRAIN_SOURCE)
        frame = load_dataset(TRAIN_SOURCE)
        fingerprints = code_hashes()
        packages = dependencies()
        args.session.mkdir(parents=True, exist_ok=False)
        with session_lock(args.session):
            options = {'policy': STRATEGY_POLICY} if args.strategy_actions else {}
            if args.class_guard:
                options = {'policy': CLASS_GUARD_POLICY}
            if args.midpoint_actions:
                options = {'policy': MIDPOINT_POLICY}
            state = initialize(frame, source_sha256=digest, budget=args.budget, **options)
            if code_hashes() != fingerprints or dependencies() != packages or sha256(TRAIN_SOURCE) != digest:
                raise ValueError('Inputs changed during initialization')
            state = {**state, 'provenance': {'source': str(TRAIN_SOURCE),
                'source_sha256': digest, 'code_sha256': fingerprints,
                'python': platform.python_version(), 'packages': packages,
                'proposer': 'external_language_model_host',
                'limitations': 'Repeated random validation splits; no external generalization claim; no test data read by this tool.'}}
            write_state(args.session, state)
    elif args.command == 'act':
        if not args.action:
            parser.error('act requires --action JSON')
        with session_lock(args.session):
            state = latest_state(args.session)
            previous = state['provenance']
            if sha256(TRAIN_SOURCE) != previous['source_sha256'] or code_hashes() != previous['code_sha256']:
                raise ValueError('Source or executor code changed since session initialization')
            if dependencies() != previous['packages'] or platform.python_version() != previous['python']:
                raise ValueError('Runtime changed since session initialization')
            action = json.loads(args.action.read_text(encoding='utf-8'))
            state = apply_action(state, load_dataset(TRAIN_SOURCE), action)
            write_state(args.session, state)
    else:
        state = latest_state(args.session)
    print(json.dumps({'observation': observe(state), 'observation_sha256': observation_digest(state)},
                     indent=2, ensure_ascii=False, allow_nan=False))


if __name__ == '__main__':
    main()
