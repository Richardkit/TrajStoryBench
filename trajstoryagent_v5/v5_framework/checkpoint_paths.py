"""Resolve frozen Windows checkpoint paths against the current checkout.

Original artifacts and hashes are never rewritten. Only decoded path values are
relocated at runtime; all hash checks still inspect the original file bytes.
"""
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
ORIGINAL_ROOT = "B:/TrajStory"


def relocate_path(value, root=None):
    if not isinstance(value, str):
        return value
    normalized = value.replace(chr(92), "/")
    if normalized.lower() == ORIGINAL_ROOT.lower():
        if root is None and Path(value).exists():
            return value
        return str(ROOT if root is None else Path(root))
    prefix = ORIGINAL_ROOT + "/"
    if normalized.lower().startswith(prefix.lower()):
        relative = normalized[len(prefix):]
        if ".." in relative.split("/"):
            raise ValueError("Checkpoint path escapes repository")
        # A batch executes from a frozen source copy. Existing input/checkpoint
        # paths still belong to the caller, not that source-only directory.
        if root is None and Path(value).exists():
            return value
        return str((ROOT if root is None else Path(root)) / relative)
    return value


def relocate_decoded(value):
    if isinstance(value, dict):
        return {relocate_path(k): relocate_decoded(v) for k, v in value.items()}
    if isinstance(value, list):
        return [relocate_decoded(v) for v in value]
    return relocate_path(value)
