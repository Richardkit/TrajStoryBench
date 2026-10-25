import json
import traceback
from typing import Any, Callable, Dict, List, Optional
from prompts import IMPORT_PMT
from openai import OpenAI
from tools import jupyter_code_interpreter_schema  # 确保 tools.py 里有这个 schema
from utils import *
import random
import string
from utils import _sanitize_code_parentheses


class DataScienceAgent:

    def __init__(
            self,
            client: OpenAI,
            model: str,
            system_prompt: str = "",
            max_iterations: int = 20,
            is_multimodal: bool = False,
            generation_kwargs: Optional[Dict[str, Any]] = None,
    ):
        self.client = client
        self.model = model
        self.system_prompt = system_prompt
        self.max_iterations = max_iterations
        self.is_multimodal = is_multimodal
        self.generation_kwargs = generation_kwargs or {}

        self.tools_registry: Dict[str, Callable] = {}
        self.notebook_cells = []
        self.messages: List[Dict[str, Any]] = [
            {"role": "system", "content": self.system_prompt}
        ]

    def register_tool(self, name: str, func: Callable[[Dict[str, Any]], Any]):
        self.tools_registry[name] = func

    def _build_tools_schema(self):
        return [jupyter_code_interpreter_schema()]

    def _assistant_message_to_dict(self, msg) -> Dict[str, Any]:
        content = msg.content or "..."

        base: Dict[str, Any] = {
            "role": msg.role,
            "content": content,
        }
        if getattr(msg, "tool_calls", None):
            base["tool_calls"] = [
                {
                    "id": tc.id,
                    "type": tc.type,
                    "function": {
                        "name": tc.function.name,
                        "arguments": tc.function.arguments,
                    },
                }
                for tc in msg.tool_calls
            ]
        return base

    def _answer_only_messages(self) -> List[Dict[str, Any]]:
        """Flatten executed observations so the last turn cannot continue a tool chain."""
        messages = []
        for message in self.messages:
            role = message["role"]
            if role == "tool":
                messages.append({
                    "role": "user",
                    "content": f"Executed Jupyter output:\n{message.get('content', '')}",
                })
            elif role == "assistant" and message.get("tool_calls"):
                continue
            elif role in ("system", "user", "assistant"):
                messages.append({"role": role, "content": message.get("content", "")})
        messages.append({
            "role": "user",
            "content": "The tool phase is over. Answer the current request using the executed evidence. "
                       "Return the required JSON only; do not write or request more code.",
        })
        return messages

    def _add_markdown_cell(self, text):
        if not text: return
        self.notebook_cells.append({
            "cell_type": "markdown",
            "metadata": {},
            "source": [line + "\n" for line in text.split("\n")]
        })

    def _add_code_cell(self, code, text_output, images_b64, execution_error=False):
        outputs = []
        if text_output:
            outputs.append({
                "name": "stdout",
                "output_type": "stream",
                "text": [line + "\n" for line in text_output.split("\n")]
            })
        if images_b64:
            for img in images_b64:
                outputs.append({
                    "data": {
                        "image/png": img,
                        "text/plain": "<IPython.core.display.Image object>"
                    },
                    "metadata": {},
                    "output_type": "display_data"
                })

        self.notebook_cells.append({
            "cell_type": "code",
            "execution_count": None,
            "metadata": {"execution_error": bool(execution_error)},
            "outputs": outputs,
            "source": [line + "\n" for line in code.split("\n")]
        })

    def reset_notebook_buffer(self):
        self.notebook_cells = []

    def save_notebook(self, filepath):
        notebook_json = {
            "cells": self.notebook_cells,
            "metadata": {
                "kernelspec": {
                    "display_name": "Python 3",
                    "language": "python",
                    "name": "python3"
                },
                "language_info": {"name": "python", "version": "3.10"}
            },
            "nbformat": 4,
            "nbformat_minor": 4
        }
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(notebook_json, f, indent=1, ensure_ascii=False)
        print(f"📘 Notebook saved to: {filepath}")

    def run(
        self,
        user_input: str,
        working_path: str,
        jupyter_sandbox,
        verbose: bool = True,
        max_iterations: Optional[int] = None,
        final_answer_on_last_step: bool = False,
        locked_question: Optional[str] = None,
    ):

        reasoning_logs = []
        final_report = ""
        usage_stats = {
            "prompt_tokens": 0,
            "completion_tokens": 0,
            "total_tokens": 0,
            "llm_calls": 0,
        }
        iteration_limit = self.max_iterations if max_iterations is None else max_iterations
        if iteration_limit < 1:
            raise ValueError("max_iterations must be positive")

        self._add_markdown_cell(f"# Task Request\n**User**: {user_input}")
        self.messages.append({"role": "user", "content": user_input})

        tools_schema = self._build_tools_schema()
        question_answered = False
        execution_attempts = 0
        blocked_tool_calls = 0
        if locked_question is not None:
            if not locked_question.strip():
                raise ValueError("locked_question must not be empty")
            # The adapter registers the question before tools are available.
            # This checks identity, not the semantic meaning of arbitrary Python.
            schema = tools_schema[0]["function"]["parameters"]
            schema["properties"]["question"] = {
                "type": "string", "enum": [locked_question],
                "description": "Repeat the locked question exactly; do not introduce another question.",
            }
            schema["required"] = [*schema["required"], "question"]
            tools_schema[0]["function"]["description"] += (
                " Answer only the locked question in one self-contained calculation. "
                "No exploratory screening or unrelated subquestions. After successful execution "
                "no more code is permitted; after failure only repair the same calculation."
            )

        def execute_jupyter(args: dict) -> dict:
            code = args.get("code", "")
            if 'google' in self.model:
                code = _sanitize_code_parentheses(code)
            # if 'matplotlib' or 'plt' in code and "%matplotlib inline" not in code:
            #     code = "%matplotlib inline\n" + code
            text_out, images = jupyter_sandbox.execute_code(code)
            compressed_text = compress_text_output(text_out)
            return {
                "text": compressed_text,
                "raw_text": text_out,
                "images": images
            }

        # imp_result = execute_jupyter({"code":IMPORT_PMT})
        if getattr(self, "_plotting_sandbox", None) is not jupyter_sandbox:
            execute_jupyter({"code":"%matplotlib inline"})
            self._plotting_sandbox = jupyter_sandbox

        self.tools_registry["jupyter_code_interpreter"] = execute_jupyter

        for step in range(iteration_limit):

            step_header = f"[Agent] step {step}: call LLM"
            reasoning_logs.append(step_header)
            if verbose: print("\n" + step_header)

            try:
                answer_only = question_answered or (final_answer_on_last_step and step == iteration_limit - 1)
                request = {
                    "model": self.model,
                    "messages": self._answer_only_messages() if answer_only else self.messages,
                    **self.generation_kwargs,
                }
                # Some compatible APIs still return tool calls for tool_choice="none".
                # Omitting tools on the final turn makes the answer-only budget reliable.
                if not answer_only:
                    request.update(tools=tools_schema, tool_choice="auto")
                resp = self.client.chat.completions.create(**request)
                usage_stats["llm_calls"] += 1

                if hasattr(resp, 'usage') and resp.usage:
                    usage_stats["prompt_tokens"] += resp.usage.prompt_tokens
                    usage_stats["completion_tokens"] += resp.usage.completion_tokens
                    usage_stats["total_tokens"] += resp.usage.total_tokens

            except Exception as e:
                error_log = f"API Error: {e}"
                reasoning_logs.append(error_log)
                print(error_log)
                break

            choice = resp.choices[0].message


            self.messages.append(self._assistant_message_to_dict(choice))
            tool_calls = getattr(choice, "tool_calls", None)

            if answer_only and tool_calls:
                reasoning_logs.append("[Agent] provider returned a tool call on an answer-only turn")
                break

            if not tool_calls:
                if locked_question is not None and not question_answered:
                    reasoning_logs.append("[Agent] rejected answer without execution for locked question")
                    self.messages.append({
                        "role": "user",
                        "content": "This question has not been calculated in this exploration. "
                                   "Do not reuse an earlier finding. Execute exactly one "
                                   "jupyter_code_interpreter call for the locked question first.",
                    })
                    continue
                final_report = choice.content or ""
                reasoning_logs.append("[Agent] final answer:")
                reasoning_logs.append(final_report)
                if verbose:
                    print("[Agent] final answer:")
                    print(final_report)

                gen_files = get_files_from_path(working_path)

                finished_log = {
                    "reasoning": "\n".join(reasoning_logs),
                    "final_report": final_report,
                    "finished": True,
                    "coding": len(self.notebook_cells) > 1,  # 简单判断有没有 coding
                    "generated_files": gen_files,
                    "token_usage": usage_stats
                }
                finished_log["question_execution"] = {
                    "locked_question": locked_question,
                    "execution_attempts": execution_attempts,
                    "successful_executions": int(question_answered),
                    "blocked_tool_calls": blocked_tool_calls,
                }
                return finished_log, final_report

            for tool_call in tool_calls:
                tool_id = tool_call.id
                multimodal_message = None

                tool_name = tool_call.function.name
                tool_args_str = tool_call.function.arguments or "{}"

                log_tool = f"[Agent] tool_call -> {tool_name}({tool_args_str})"
                reasoning_logs.append(log_tool)
                if verbose: print(log_tool)

                tool_content_for_llm = "No outputs found in kernel."

                if locked_question is not None and len(tool_calls) != 1:
                    blocked_tool_calls += 1
                    tool_content_for_llm = "Rejected: only one tool call for the locked question is permitted per response. No code was executed."
                elif tool_name not in self.tools_registry:
                    tool_content_for_llm = f"Unknown tool: {tool_name}"
                else:
                    try:
                        parsed_args = json.loads(tool_args_str)
                    except json.JSONDecodeError as e:
                        tool_content_for_llm = f"JSON Error: {e}"
                    else:
                        if locked_question is not None and (
                            not isinstance(parsed_args, dict)
                            or parsed_args.get("question") != locked_question
                            or not isinstance(parsed_args.get("code"), str)
                            or not parsed_args["code"].strip()
                        ):
                            blocked_tool_calls += 1
                            self.messages.append({
                                "role": "tool", "tool_call_id": tool_id, "name": tool_name,
                                "content": "Rejected: supply nonempty code and the exact locked question. No code was executed.",
                            })
                            reasoning_logs.append("[Agent] rejected mismatched question or empty code")
                            continue
                        try:
                            code_to_run = parsed_args.get("code", "")
                            execution_attempts += 1
                            tool_result = self.tools_registry[tool_name](parsed_args)
                            execution_error = getattr(jupyter_sandbox, "last_execution_error", False)
                            if locked_question is not None and not execution_error:
                                question_answered = True
                            text_output = tool_result["text"]
                            text_output_full = tool_result.get("raw_text", text_output)
                            images_b64 = tool_result["images"]
                            print(f"[Tool] tool result: {text_output_full}, \n{len(images_b64)}images.")

                            tool_content_for_llm = text_output if text_output and text_output != '' else "No execution information found in kernel."

                            self._add_code_cell(
                                code_to_run,
                                text_output_full,
                                images_b64,
                                execution_error=execution_error,
                            )

                            tool_output_text = text_output
                            if images_b64:
                                tool_output_text += f"\n[System] {len(images_b64)} image(s) generated."

                            if self.is_multimodal and images_b64:
                                content_parts = [{
                                    "type": "text",
                                    "text": "Here are the plots generated by the code:"
                                }]
                                for b64_str in images_b64:
                                    content_parts.append({
                                        "type": "image_url",
                                        "image_url": {"url": f"data:image/png;base64,{b64_str}"}
                                    })
                                multimodal_message = {
                                    "role": "user",
                                    "content": content_parts
                                }

                        except Exception as e:
                            reasoning_logs.append(f"Execution Exception: {repr(e)}")
                            tool_content_for_llm = f"Execution Exception: {repr(e)}\nYou MUST have a faster solution."
                            print(f"[Error] in execution {e}.")

                reasoning_logs.append(f"[Tool Output] {str(tool_content_for_llm)}")

                if "google" in self.model:
                    
                    self.messages.append({
                        "role": "user",
                        "tool_call_id": tool_id,
                        "name": tool_name,
                        "content": tool_content_for_llm
                    })

                else:
                    self.messages.append({
                        "role": "tool",
                        "tool_call_id": tool_id,
                        "name": tool_name,
                        "content": tool_content_for_llm
                    })

                if multimodal_message:
                    self.messages.append(multimodal_message)
                    reasoning_logs.append("[System] Sent Vision Observation to VLM.")

        final_report = "Returned with no final report."
        log_json = {
            "reasoning": "\n".join(reasoning_logs),
            "final_report": final_report,
            "finished": False,
            "token_usage": usage_stats
        }
        log_json["question_execution"] = {
            "locked_question": locked_question,
            "execution_attempts": execution_attempts,
            "successful_executions": int(question_answered),
            "blocked_tool_calls": blocked_tool_calls,
        }
        return log_json, final_report
