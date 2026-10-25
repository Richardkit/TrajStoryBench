"""Minimal TrajStory data adapter for the native Planner/AgentBus workflow."""
from mmengine.config import read_base

with read_base():
    from .agents.planning import planning_agent
    from .agents.orchestra_roles import (orchestra_analyst_agent,
        orchestra_interpreter_agent, orchestra_verifier_agent, orchestra_synthesizer_agent)
    from .tools.trajstory_analysis import trajstory_analysis_tool

tag = "trajstory_minimal"
workdir = f"workdir/{tag}"
log_path = "trajstory.log"
use_local_proxy = False
version = "0.1.0"
model_name = "openai/gpt-6-luna"
env_names = []
memory_names = []
skill_names = []
tool_names = ["trajstory_analysis", "done"]
agent_names = ["planning", "orchestra_analyst", "orchestra_interpreter",
               "orchestra_verifier", "orchestra_synthesizer"]
trajstory_analysis_tool.update(use_trajstory_helpers=False)

# Reuse upstream agents, prompts and their step/tool/review limits.
# External memory and unrelated application services are not needed for cases.
planning_agent.update(workdir=f"{workdir}/agent/planning", model_name=model_name,
                      memory_name=None, require_grad=False,
                      model_input_char_limit=24000,model_output_token_limit=2500)
