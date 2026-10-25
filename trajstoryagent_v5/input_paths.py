"""Strict selection of the projected CSVs used by InsightBench and v5."""
from pathlib import Path

# This table contains only computed burst summaries; the input projection has
# no retained columns. Every other declared CSV must have an input_ sibling.
OMITTED_INPUT_CSVS = {
    ('trajstory_examples', 'insight', 'LaDe',
     '15_delivery_acceptance_bursts', 'acceptance_bursts.csv'),
}


def resolve_input_csv(path: str | Path | None) -> Path | None:
    if path is None:
        return None
    source = Path(path).expanduser().resolve()
    if source.suffix.lower() != '.csv':
        return source
    original_name = source.name.removeprefix('input_')
    original_parts = (*source.parts[:-1], original_name)
    if tuple(original_parts[-5:]) in OMITTED_INPUT_CSVS:
        return None
    target = source if source.name.startswith('input_') else source.with_name('input_' + source.name)
    if not target.is_file():
        raise FileNotFoundError(f'Projected input CSV is required; no original CSV fallback: {target}')
    return target
