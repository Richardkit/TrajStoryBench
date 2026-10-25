"""Compatibility entry; implementation: trajstoryagent_v5.v5_framework.cli."""
from importlib import import_module
import sys
_module = import_module("trajstoryagent_v5.v5_framework.cli")
if __name__ == "__main__":
    _module.main()
else:
    sys.modules[__name__] = _module
