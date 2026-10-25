"""Agents module for multi-agent system."""

import os

from .tool_calling_agent import ToolCallingAgent
from .planning_agent import PlanningAgent
from .orchestra_roles import (
    OrchestraAnalystAgent, OrchestraInterpreterAgent,
    OrchestraVerifierAgent, OrchestraSynthesizerAgent,
)
from .trajstory_agents import (
    TrajstoryTemporalSpatialAgent,
    TrajstoryBehaviorComparisonAgent,
    TrajstoryAnomalyAgent,
    TrajstoryEvidenceVerifierAgent,
    TrajstorySynthesisAgent,
)
from .server import acp


__all__ = [
    "ToolCallingAgent",
    "PlanningAgent",
    "TrajstoryTemporalSpatialAgent",
    "TrajstoryBehaviorComparisonAgent",
    "TrajstoryAnomalyAgent",
    "TrajstoryEvidenceVerifierAgent",
    "TrajstorySynthesisAgent",
    "acp",
]

if os.getenv("TRAJSTORY_MINIMAL_IMPORTS") != "1":
    from .interday_trading_agent import InterdayTradingAgent
    from .intraday_trading_agent import IntradayTradingAgent
    from .simple_chat_agent import SimpleChatAgent
    from .debate_manager import DebateManagerAgent
    from .operator_browser_agent import OperatorBrowserAgent
    from .mobile_agent import MobileAgent
    from .anthropic_mobile_agent import AnthropicMobileAgent
    from .online_trading_agent import OnlineTradingAgent
    from .offline_trading_agent import OfflineTradingAgent
    from .trading_strategy_agent import TradingStrategyAgent
    from .esg_agent import ESGAgent
    __all__ += [
        "InterdayTradingAgent", "IntradayTradingAgent", "SimpleChatAgent",
        "DebateManagerAgent", "OperatorBrowserAgent", "MobileAgent",
        "AnthropicMobileAgent", "OnlineTradingAgent", "OfflineTradingAgent",
        "TradingStrategyAgent", "ESGAgent",
    ]
