"""Formal v5 terminology implementation."""
import re

INPUT_POLICY = '''INPUT POLICY:
Only the named projected input tables and visible columns are available.
Previously supplied derived columns may be absent by design. Compute needed
derived quantities from visible fields where possible; do not recover them
from original CSVs, unlisted files, reference insights or prior experiment outputs.
If visible inputs cannot support a quantity or conclusion, state that limitation.
'''


class InsightClient:
    def __init__(self, client):
        self.client = client

    def __getattr__(self, name):
        return getattr(self.client, name)

    def complete(self, *, system, user, **kwargs):
        # Shared role code can retain its Python compatibility accessors, but
        # the actual prompt and provider audit consistently describe insights.
        def normalize(text):
            text = re.sub(r'\bclaim(s)?\b', lambda m: 'insights' if m[1] else 'insight', text)
            return re.sub(r'\ba insight\b', 'an insight', text)
        return self.client.complete(system=normalize(system),
                                    user=INPUT_POLICY + '\n' + normalize(user), **kwargs)
