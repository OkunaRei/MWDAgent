"""Two feedback conditions over the same frozen candidate table and executor."""
import argparse
import json
from pathlib import Path

from run_decision_benchmark import load_table, write_new
from run_mwd_agent import sha256, latest_state, session_lock
from src.decision_benchmark import start, execute, compact_observation
from src.feedback_ablation import feedback_view


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('command', choices=('init', 'act', 'observe'))
    parser.add_argument('--table', type=Path, default=Path('reports/decision_benchmark/fresh-host-01'))
    parser.add_argument('--dir', type=Path, required=True)
    parser.add_argument('--condition', choices=('aggregate', 'class_feedback'))
    parser.add_argument('--action', help='JSON literal')
    args = parser.parse_args()
    if args.command == 'init' and not args.condition:
        parser.error('init requires --condition')
    if (args.command == 'act') != (args.action is not None):
        parser.error('--action is required only for act')
    table = load_table(args.table)
    code = {name: sha256(Path(__file__).parent/name) for name in ('run_feedback_ablation.py','src/feedback_ablation.py')}
    if args.command == 'init':
        args.dir.mkdir(parents=True, exist_ok=False)
        write_new(args.dir/'protocol.json', {'condition': args.condition, 'budget': 3,
            'table_manifest_sha256': sha256(args.table/'manifest.json'), 'code_sha256': code,
            'new_training_fits': 0, 'paired_trajectory_count': 1})
    with session_lock(args.dir):
        protocol = json.loads((args.dir/'protocol.json').read_text())
        if (protocol['table_manifest_sha256'] != sha256(args.table/'manifest.json')
                or protocol['code_sha256'] != code):
            raise ValueError('Frozen ablation inputs changed')
        state = start(table) if args.command == 'init' else latest_state(args.dir)
        if args.command == 'act':
            state = execute(state, table, json.loads(args.action))
        if args.command != 'observe':
            write_new(args.dir/f"state-{state['revision']:04d}.json", state)
        view = feedback_view(compact_observation(state), protocol['condition'])
        path = args.dir/f"delivered-{state['revision']:04d}.json"
        if not path.exists():
            write_new(path, view)
        print(json.dumps(view, ensure_ascii=False))


if __name__ == '__main__':
    main()
