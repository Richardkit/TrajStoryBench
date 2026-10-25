"""Offline tests must never issue provider/network requests."""
import urllib.request
import pytest

@pytest.fixture(autouse=True)
def forbid_network(monkeypatch):
    def blocked(*args, **kwargs):
        raise AssertionError('Network calls are forbidden in offline tests')
    monkeypatch.setattr(urllib.request, 'urlopen', blocked)
