"""Tool calling agent implementation with manual agent logic."""

import asyncio
import os
from typing import List, Optional, Dict, Any
from langchain_core.messages import BaseMessage
from datetime import datetime
from pydantic import Field, ConfigDict

from src.agent.types import Agent, AgentResponse, AgentExtra, ThinkOutput
from src.config import config
from src.logger import logger
from src.utils import dedent, parse_tool_args
from src.tool.server import tcp
from src.tool.types import ToolResponse
from src.skill.server import scp
from src.environment.server import ecp
from src.memory import memory_manager, EventType
from src.tracer import Tracer, Record
from src.model import model_manager
from src.registry import AGENT
from src.session import SessionContext
from src.agent.trajstory_exploration_budget import TrajstoryExplorationBudget
from src.agent.orchestra_handoffs import model_limits

@AGENT.register_module(force=True)
class ToolCallingAgent(Agent):
    """Tool calling agent implementation with manual agent logic."""
    model_config = ConfigDict(arbitrary_types_allowed=True, extra="allow")
    
    name: str = Field(default="tool_calling", description="The name of the tool calling agent.")
    description: str = Field(default="A tool calling agent that can call tools to complete tasks.", description="The description of the tool calling agent.")
    metadata: Dict[str, Any] = Field(default={}, description="The metadata of the tool calling agent.")
    require_grad: bool = Field(default=False, description="Whether the agent requires gradients")
    
    def __init__(
        self,
        workdir: str,
        name: Optional[str] = None,
        description: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
        model_name: Optional[str] = None,
        prompt_name: Optional[str] = None,
        memory_name: Optional[str] = None,
        max_tools: int = 10,
        max_steps: int = 20,
        review_steps: int = 5,
        require_grad: bool = False,
        **kwargs
    ):
        # Set default prompt name for tool calling
        if not prompt_name:
            prompt_name = "tool_calling"
        
        super().__init__(
            workdir=workdir,
            name=name,
            description=description,
            metadata=metadata,
            model_name=model_name,
            prompt_name=prompt_name,
            memory_name=memory_name,
            max_tools=max_tools,
            max_steps=max_steps,
            review_steps=review_steps,
            require_grad=require_grad,
            **kwargs)
    
    async def initialize(self):
        """Initialize the agent."""
        self.tracer_save_path = os.path.join(self.workdir, "tracer.json")
        await super().initialize()
    
    async def _get_tracer_and_record(self) -> tuple[Tracer, Record]:
        """Get tracer and record for current call (coroutine-safe)."""
        tracer = Tracer()
        record = Record()
        
        if os.path.exists(self.tracer_save_path):
            await tracer.load_from_json(self.tracer_save_path)
            last_record = await tracer.get_last_record()
            if last_record:
                record = last_record
        
        return tracer, record
    
    async def _get_environment_context(self, ctx: SessionContext, record: Record = None, **kwargs) -> Dict[str, Any]:
        """Get the environment state."""
        
        environment_context = "<environment_context>"
        record_observation = {}
        
        # Only iterate over environments specified in config, not all registered environments
        for env_name in config.env_names:
            env_info = await ecp.get_info(env_name)
            rule_string = env_info.rules
            rule_string = dedent(f"""
                <rules>
                {rule_string}
                </rules>
            """)
            
            env_state = await ecp.get_state(env_name, ctx=ctx)
            state_string = "<state>"
            state_string += env_state["state"]
            extra = env_state["extra"]
            record_observation[env_name] = extra
            
            if "screenshots" in extra:
                for screenshot in extra["screenshots"]:
                    state_string += f"\n<img src={screenshot.screenshot_path} alt={screenshot.screenshot_description}/>"
            state_string += "</state>"
            
            environment_context += dedent(f"""
                <{env_name}>
                {rule_string}
                {state_string}
                </{env_name}>
            """)
        
        if record is not None:
            record.observation = record_observation
        
        environment_context += "</environment_context>"
        return {
            "environment_context": environment_context,
        }
        
    async def _get_tool_context(self, ctx: SessionContext, record: Record = None, **kwargs) -> Dict[str, Any]:
        """Get the tool context."""
        
        tool_context = "<tool_context>"

        tool_context += dedent(f"""
            <available_tools>
            {await tcp.get_contract()}
            </available_tools>
        """)

        tool_context += "</tool_context>"
        return {
            "tool_context": tool_context,
        }
        
    async def _think_and_tool(self, 
                              messages: List[BaseMessage], 
                              task_id: str,
                              step_number: int,
                              record: Record = None, 
                              ctx: SessionContext = None, 
                              **kwargs)->Dict[str, Any]:
        """Think and tool calls for one step."""
        
        done = False
        result = None
        reasoning = None
        analysis_budget: TrajstoryExplorationBudget | None = kwargs.get("analysis_budget")
        analysis_invoked_this_step = False
        
        record_data = {
            "thinking": None,
            "evaluation_previous_goal": None,
            "memory": None,
            "next_goal": None,
            "actions": [],
        }
        
        call_limits = model_limits(self, messages)
        try:
            think_output = await model_manager(
                model=self.model_name,
                messages=messages,
                response_format=ThinkOutput,
                **call_limits,
            )
            think_output = think_output.extra.parsed_model
            
            thinking = think_output.thinking
            evaluation_previous_goal = think_output.evaluation_previous_goal
            memory = think_output.memory
            next_goal = think_output.next_goal
            actions = think_output.actions
            
            record_data["thinking"] = thinking
            record_data["evaluation_previous_goal"] = evaluation_previous_goal
            record_data["memory"] = memory
            record_data["next_goal"] = next_goal
            
            logger.info(f"| 💭 Thinking: {thinking}")
            logger.info(f"| 🎯 Next Goal: {next_goal}")
            logger.info(f"| 🔧 Actions to execute: {actions}")
            
            # Execute actions sequentially, routing by type
            action_results = []
            
            for i, action in enumerate(actions):
                action_type = action.type
                action_name = action.name
                action_args_str = action.args
                action_args = parse_tool_args(action_args_str) if action_args_str else {}

                if analysis_budget is not None:
                    rejection = None
                    if action_name not in ("trajstory_analysis", "done") or action_type != "tool":
                        rejection = "Exploration agents may only call trajstory_analysis or done."
                    elif action_name == "trajstory_analysis":
                        if analysis_invoked_this_step:
                            rejection = "Only one analysis call is permitted per reasoning step; inspect its output first."
                        else:
                            allowed, reason = analysis_budget.authorize(action_args)
                            if not allowed:
                                rejection = reason
                    elif analysis_invoked_this_step or (
                        analysis_budget.successful_analyses < 1
                        and analysis_budget.failed_attempts_in_phase <= analysis_budget.max_repairs_per_phase
                    ):
                        rejection = "Inspect at least one successful analysis in a later step before calling done."
                    if rejection is not None:
                        rejected = action.model_dump()
                        rejected["output"] = f"TrajStory analysis error: exploration budget rejection: {rejection}"
                        record_data["actions"].append(rejected)
                        action_results.append(rejected)
                        continue
                    if action_name == "trajstory_analysis":
                        analysis_invoked_this_step = True

                logger.info(f"| 📝 Action {i+1}/{len(actions)}: [{action_type}] {action_name}")
                logger.info(f"| 📝 Args: {action_args}")

                if action_type == "skill":
                    # Route to SCP
                    response = await scp(
                        name=action_name,
                        input=action_args,
                        ctx=ctx,
                    )
                    action_result = response.message
                    action_extra = response.extra if hasattr(response, 'extra') else None

                    logger.info(f"| ✅ Skill '{action_name}' completed (success={response.success})")
                    logger.info(f"| 📄 Result: {str(action_result)[:500]}")

                    action_dict = action.model_dump()
                    action_dict["output"] = action_result
                    action_results.append(action_dict)

                    record_extra = {}
                    record_extra.update(action_dict)
                    if action_extra is not None:
                        record_extra['extra'] = action_extra.model_dump()
                    record_data["actions"].append(record_extra)

                else:
                    # Route to TCP (default: type == "tool")
                    if (action_name == "trajstory_analysis"
                            and (not isinstance(action_args, dict)
                                 or not isinstance(action_args.get("spec_path"), str)
                                 or not isinstance(action_args.get("code"), str)
                                 or not action_args["spec_path"].strip()
                                 or not action_args["code"].strip())):
                        tool_response = ToolResponse(
                            success=False,
                            message=(
                                "TrajStory analysis error: malformed or missing arguments. "
                                "Call trajstory_analysis with JSON args containing "
                                "spec_path (the assigned absolute case_input.json path) "
                                "and code (nonempty Python source)."
                            ),
                        )
                    elif action_name == "done" and (
                        not isinstance(action_args, dict)
                        or not isinstance(action_args.get("result"), str)
                        or not action_args["result"].strip()
                    ):
                        tool_response = ToolResponse(
                            success=False,
                            message=(
                                'TrajStory done error: missing result. Call done with '
                                'JSON args {"result": "supported findings", '
                                '"reasoning": "brief explanation"}.'
                            ),
                        )
                    else:
                        if action_name == "done" and "reasoning" not in action_args:
                            action_args["reasoning"] = ""
                        try:
                            tool_response = await tcp(
                                name=action_name,
                                input=action_args,
                                ctx=ctx,
                            )
                        except Exception as exc:
                            if action_name not in ("trajstory_analysis", "done"):
                                raise
                            tool_response = ToolResponse(
                                success=False,
                                message=f"TrajStory {action_name} error: {type(exc).__name__}: {exc}",
                            )
                    action_result = tool_response.message
                    if analysis_budget is not None and action_name == "trajstory_analysis":
                        analysis_budget.record_result(
                            bool(tool_response.success)
                            and not str(action_result).startswith("TrajStory analysis error:")
                        )
                    action_extra = tool_response.extra if hasattr(tool_response, 'extra') else None

                    logger.info(f"| ✅ Tool '{action_name}' completed")
                    logger.info(f"| 📄 Result: {str(action_result)}")

                    action_dict = action.model_dump()
                    action_dict["output"] = action_result
                    action_results.append(action_dict)

                    record_extra = {}
                    record_extra.update(action_dict)
                    if action_extra is not None:
                        record_extra['extra'] = action_extra.model_dump()
                    record_data["actions"].append(record_extra)

                    if action_name == "done" and tool_response.success:
                        done = True
                        result = action_result
                        reasoning = action_extra.data.get('reasoning', None) if action_extra and action_extra.data else None
                        break
            
            event_data = {
                "thinking": thinking,
                "evaluation_previous_goal": evaluation_previous_goal,
                "memory": memory,
                "next_goal": next_goal,
                "actions": action_results
            }
            
            if record is not None:
                if analysis_budget is not None:
                    record_data["exploration_budget"] = analysis_budget.snapshot()
                record.tool = record_data
            
            # Get memory system name
            memory_name = self.memory_name
            
            # Add event to memory if use_memory is enabled
            if self.use_memory and memory_name:
                await memory_manager.add_event(
                    memory_name=memory_name,
                    step_number=step_number,
                    event_type=EventType.TOOL_STEP,
                    data=event_data,
                    agent_name=self.name,
                    task_id=task_id,
                    ctx=ctx
                )
            
        except Exception as e:
            logger.error(f"| Error in thinking and tool step: {e}")
        
        response_dict = {
            "done": done,
            "result": result,
            "reasoning": reasoning
        }
        return response_dict
        
    async def __call__(self, 
                  task: str, 
                  files: Optional[List[str]] = None,
                  **kwargs
                  ) -> AgentResponse:
        """
        Main entry point for tool calling agent through acp.
        
        Args:
            task (str): The task to complete.
            files (Optional[List[str]]): The files to attach to the task.
            
        Returns:
            AgentResponse: The response of the agent.
        """
        logger.info(f"| 🚀 Starting ToolCallingAgent: {task}")
        
        ctx = kwargs.get("ctx", None)
        if ctx is None:
            ctx = SessionContext()
        
        # Create tracer and record as local variables (coroutine-safe)
        tracer, record = await self._get_tracer_and_record()
        
        if files:
            logger.info(f"| 📂 Attached files: {files}")
            files = await asyncio.gather(*[self._extract_file_content(file) for file in files])
            enhanced_task = await self._generate_enhanced_task(task, files)
        else:
            enhanced_task = task
        
        # Get memory system name
        memory_name = self.memory_name

        task_id = "task_" + datetime.now().strftime("%Y%m%d-%H%M%S")
        
        logger.info(f"| 📝 Context ID: {ctx.id}, Task ID: {task_id}")
        
        # Memory session management (only if use_memory is enabled)
        if self.use_memory and memory_name:
            await memory_manager.start_session(memory_name=memory_name, ctx=ctx)
            
            # Add task start event
            await memory_manager.add_event(
                memory_name=memory_name,
                step_number=0,
                event_type=EventType.TASK_START,
                data=dict(task=enhanced_task),
                agent_name=self.name,
                task_id=task_id,
                ctx=ctx
            )
        else:
            logger.info(f"| ⏭️ Memory disabled (use_memory={self.use_memory}), skipping session management")
        
        # Initialize messages
        messages = await self._get_messages(enhanced_task, ctx=ctx, step_number=0)
        local_tool_results: List[str] = []
        feedback_mode = kwargs.get("tool_feedback_mode", "history")
        premature_repeat_limit = kwargs.get("premature_done_repeat_limit", 0)
        prior_premature_done = None
        premature_done_repeats = 0
        analysis_calls = 0
        min_analysis_calls = max(0, int(kwargs.get("min_analysis_calls", 0)))
        analysis_budget: TrajstoryExplorationBudget | None = kwargs.get("analysis_budget")
        
        # Main loop
        step_number = 0
        
        while step_number < self.max_steps:
            logger.info(f"| 🔄 Step {step_number+1}/{self.max_steps}")
            
            # Execute one step
            prior_tool_record = record.tool
            response = await self._think_and_tool(
                messages, task_id, step_number, ctx=ctx, record=record,
                analysis_budget=analysis_budget,
            )
            step_number += 1

            # Memory may be disabled, but the next reasoning step still needs
            # the output of tools it just ran. Keep this feedback local to the
            # current task; it is not persistent cross-round memory.
            if record.tool is not prior_tool_record and record.tool:
                if feedback_mode == "latest":
                    local_tool_results.clear()
                for action in record.tool.get("actions", []):
                    output = str(action.get("output", ""))
                    if (action.get("name") == "trajstory_analysis"
                            and not output.startswith("TrajStory analysis error:")):
                        analysis_calls += 1
                    local_tool_results.append(
                        f"{action.get('name', 'tool')} output:\n"
                        f"{output[:12_000]}"
                    )

            budget_exhausted_without_success = (
                analysis_budget is not None
                and analysis_budget.failed_attempts_in_phase > analysis_budget.max_repairs_per_phase
            )
            if response["done"] and analysis_calls < min_analysis_calls and not budget_exhausted_without_success:
                attempted_result = str(response.get('result') or '').strip()
                premature_done_repeats = premature_done_repeats + 1 if attempted_result == prior_premature_done else 1
                prior_premature_done = attempted_result
                if premature_repeat_limit and premature_done_repeats >= premature_repeat_limit:
                    response = dict(done=False,result='Repeated termination without executed evidence; task stopped.',reasoning='Input or computation unavailable')
                    await tracer.add_record(observation=record.observation,tool=record.tool,task_id=task_id,ctx=ctx)
                    await tracer.save_to_json(self.tracer_save_path)
                    break
                local_tool_results.append(
                    "The premature done action was rejected. Continue with "
                    f"trajstory_analysis; completed analysis calls: {analysis_calls}, "
                    f"minimum required: {min_analysis_calls}. A schema-only preview "
                    "does not establish a numerical finding."
                )
                response = {"done": False, "result": None, "reasoning": None}

            if analysis_budget is not None and analysis_budget.failed_attempts_in_phase > analysis_budget.max_repairs_per_phase:
                local_tool_results.append(
                    "The exploration code-repair budget is exhausted. Do not run further analysis; "
                    "return only already supported findings with done."
                )
            
            # Update tracer and save to json
            await tracer.add_record(observation=record.observation, 
                                        tool=record.tool,
                                        task_id=task_id,
                                        ctx=ctx)
            await tracer.save_to_json(self.tracer_save_path)
            
            # Memory is automatically saved in add_event()
            next_task = enhanced_task
            if local_tool_results:
                feedback = "\n\n".join(local_tool_results)[-32_000:]
                next_task += f"\n\n<executed_tool_results>\n{feedback}\n</executed_tool_results>"
            messages = await self._get_messages(
                next_task, ctx=ctx, step_number=step_number
            )
            
            if response["done"]:
                break
        
        # Handle max steps reached
        if step_number >= self.max_steps and not response["done"]:
            logger.warning(f"| 🛑 Reached max steps ({self.max_steps}), stopping...")
            response = {
                "done": False,
                "result": "The task has not been completed.",
                "reasoning": "Reached the maximum number of steps."
            }
        
        # Get memory system name
        memory_name = self.memory_name
        
        # Add task end event and end session (only if use_memory is enabled)
        if self.use_memory and memory_name:
            await memory_manager.add_event(
                memory_name=memory_name,
                step_number=step_number,
                event_type=EventType.TASK_END,
                data=response,
                agent_name=self.name,
                task_id=task_id,
                ctx=ctx
            )
            
            # End session (automatically saves memory to JSON)
            await memory_manager.end_session(memory_name=memory_name, ctx=ctx)
        
        # Save tracer to json
        await tracer.save_to_json(self.tracer_save_path)
        
        logger.info(f"| ✅ Agent completed after {step_number}/{self.max_steps} steps")
        
        return AgentResponse(
            success=response["done"],
            message=response["result"],
            extra=AgentExtra(
                data=response
            )
        )
