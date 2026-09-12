"""Check review annotations and write a new audit report without changing source files."""
import argparse
import csv
import hashlib
import io
import json
from pathlib import Path

from src.review_annotations import validate_annotations


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--snapshot', type=Path, required=True)
    parser.add_argument('--annotations', type=Path, required=True)
    parser.add_argument('--output-dir', type=Path, required=True)
    args = parser.parse_args()
    snapshot_bytes = args.snapshot.read_bytes()
    annotation_bytes = args.annotations.read_bytes()
    snapshot = json.loads(snapshot_bytes)
    reader = csv.DictReader(io.StringIO(annotation_bytes.decode('utf-8-sig'), newline=''))
    if not reader.fieldnames or len(reader.fieldnames) != len(set(reader.fieldnames)):
        raise ValueError('CSV header must contain unique column names')
    rows = list(reader)
    result = validate_annotations(snapshot, rows)
    report = {**result, 'provenance': {
        'snapshot': str(args.snapshot), 'snapshot_sha256': hashlib.sha256(snapshot_bytes).hexdigest(),
        'annotations': str(args.annotations), 'annotations_sha256': hashlib.sha256(annotation_bytes).hexdigest(),
        'source_sha256': snapshot['metadata']['source_sha256'],
        'experiment_sha256': snapshot['metadata']['experiment_sha256'],
    }, 'limitations': 'Structural checks only; evidence references are not independently verified. No training labels are generated.'}
    args.output_dir.mkdir(parents=True, exist_ok=False)
    (args.output_dir / 'validation.json').write_text(json.dumps(report, indent=2, allow_nan=False), encoding='utf-8')
    print(json.dumps({key: result[key] for key in ('valid', 'row_count', 'status_counts', 'evidence_backed_count')}, indent=2))
    if not result['valid']:
        raise SystemExit(1)


if __name__ == '__main__':
    main()
