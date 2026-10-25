"""Compatibility entry; implementation: trajstoryagent_v5.evaluation.evaluate_batch."""
from importlib import import_module
import sys
_module = import_module("trajstoryagent_v5.evaluation.evaluate_batch")
if __name__ == "__main__":
    _module.main()
else:
    sys.modules[__name__] = _module
