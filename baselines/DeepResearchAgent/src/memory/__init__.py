"""Memory module for managing agent execution history."""

import os

from .server import MemoryManager, memory_manager
from .types import ChatEvent, EventType, Memory, MemoryConfig
from .context import MemoryContextManager

__all__ = [
    "MemoryManager",
    "memory_manager",
    "Memory",
    "MemoryConfig",
    "MemoryContextManager",
    "ChatEvent",
    "EventType",
]

if os.getenv("TRAJSTORY_MINIMAL_IMPORTS") != "1":
    from .general_memory_system import GeneralMemorySystem
    from .online_trading_memory_system import OnlineTradingMemorySystem
    from .offline_trading_memory_system import OfflineTradingMemorySystem
    from .optimizer_memory_system import OptimizerMemorySystem
    __all__ += [
        "GeneralMemorySystem", "OnlineTradingMemorySystem",
        "OfflineTradingMemorySystem", "OptimizerMemorySystem",
    ]
