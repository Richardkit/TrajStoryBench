"""
Optimizer package.
"""

import os

from .types import Optimizer

__all__ = ["Optimizer"]
if os.getenv("TRAJSTORY_MINIMAL_IMPORTS") != "1":
    from .textgrad_optimizer import TextGradOptimizer, optimize_agent_with_textgrad
    from .reflection_optimizer import ReflectionOptimizer
    from .grpo_optimizer import GrpoOptimizer
    from .reinforce_plus_plus_optimizer import ReinforcePlusPlusOptimizer
    __all__ += [
        "TextGradOptimizer", "optimize_agent_with_textgrad",
        "ReflectionOptimizer", "GrpoOptimizer", "ReinforcePlusPlusOptimizer",
    ]
