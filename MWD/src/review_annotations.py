"""Validate human annotations against an immutable review snapshot."""
from datetime import date
from decimal import Decimal, InvalidOperation
import json


EDITABLE = frozenset({'review_status', 'quality_label', 'reviewer', 'reviewed_at',
                      'evidence_reference', 'notes'})
STATUSES = ('pending', 'in_review', 'completed')
LABELS = ('measurement_issue', 'geological_variation', 'mixed', 'insufficient_evidence')


def _unchanged(value, original):
    try:
        if isinstance(original, (dict, list)):
            # Canonical JSON retains type distinctions, including booleans vs integers.
            return json.dumps(json.loads(value), sort_keys=True) == json.dumps(original, sort_keys=True)
        if isinstance(original, bool):
            return value == str(original)
        if isinstance(original, (int, float)):
            parsed = Decimal(value)
            return parsed.is_finite() and parsed == Decimal(str(original))
        return value == str(original)
    except (ValueError, TypeError, InvalidOperation):
        return False


def _review_errors(row):
    errors = []
    if row['review_status'] not in STATUSES:
        errors.append('Unknown review_status')
    if row['quality_label'] and row['quality_label'] not in LABELS:
        errors.append('Unknown quality_label')
    if row['reviewed_at']:
        try:
            parsed = date.fromisoformat(row['reviewed_at'])
            if parsed.isoformat() != row['reviewed_at']:
                raise ValueError('Noncanonical date')
        except ValueError:
            errors.append('reviewed_at must be a valid YYYY-MM-DD date')
    if row['review_status'] == 'completed':
        for field in ('quality_label', 'reviewer', 'reviewed_at', 'notes'):
            if not row[field]:
                errors.append(f'Completed review requires {field}')
        if row['quality_label'] != 'insufficient_evidence' and not row['evidence_reference']:
            errors.append('Substantive quality conclusion requires evidence_reference')
    return errors


def validate_annotations(snapshot, rows):
    if not isinstance(snapshot, dict) or not isinstance(snapshot.get('items'), list) or not snapshot['items']:
        raise ValueError('Snapshot must contain a nonempty items list')
    schema = None
    for item in snapshot['items']:
        if not isinstance(item, dict) or not (EDITABLE | {'row_index'}).issubset(item):
            raise ValueError('Snapshot item is missing required fields')
        if type(item['row_index']) is not int or item['row_index'] < 0:
            raise ValueError('Snapshot row_index must be a nonnegative integer')
        if schema is not None and set(item) != schema:
            raise ValueError('Snapshot items must share one schema')
        schema = set(item)
    expected = {str(row['row_index']): row for row in snapshot['items']}
    if len(expected) != len(snapshot['items']):
        raise ValueError('Snapshot has duplicate row indices')
    errors, completed = [], []
    seen = set()
    counts = {status: 0 for status in STATUSES}
    for line, row in enumerate(rows, start=2):
        index = row.get('row_index', '')
        if index not in expected or index in seen:
            errors.append({'line': line, 'row_index': index, 'message': 'Unknown or duplicate row_index'})
            continue
        seen.add(index)
        original = expected[index]
        if set(row) != set(original) or not all(isinstance(value, str) for value in row.values()):
            errors.append({'line': line, 'row_index': index, 'message': 'CSV schema differs from snapshot'})
            continue
        changed = [key for key in original if key not in EDITABLE and not _unchanged(row[key], original[key])]
        annotation = {key: row[key].strip() for key in EDITABLE}
        problems = [f'Immutable field changed: {key}' for key in changed] + _review_errors(annotation)
        errors.extend({'line': line, 'row_index': index, 'message': problem} for problem in problems)
        status = annotation['review_status']
        if status in counts:
            counts[status] += 1
        if not problems and status == 'completed':
            completed.append({'row_index': original['row_index'], **annotation})
    if seen != set(expected):
        errors.append({'line': None, 'row_index': None, 'message': f'Missing {len(set(expected) - seen)} snapshot rows'})
    return {'valid': not errors, 'row_count': len(rows), 'status_counts': counts,
            'errors': errors, 'completed_reviews': completed if not errors else [],
            'evidence_backed_count': sum(row['quality_label'] != 'insufficient_evidence'
                                         for row in completed) if not errors else 0}
