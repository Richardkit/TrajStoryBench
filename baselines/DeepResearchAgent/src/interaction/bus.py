"""AgentBus — async message bus that drives the planner loop.

Architecture
------------
::

    submit(Task)
       │
       ▼
    session queue  (one per session_id)
       │
       ▼
    session worker  (one asyncio.Task per session_id)
       │
       ├─── Round 1: call planner → get PlanDecision
       │       │
       │       ├── dispatches: [agent_a, agent_b]   (concurrent via gather)
       │       └── collect results
       │
       ├─── Round 2: call planner again (with results) → get PlanDecision
       │       │
       │       └── dispatches: [agent_c]             (unicast)
       │
       └─── Round N: planner returns is_done=True → resolve caller Future

Concurrency model
~~~~~~~~~~~~~~~~~
Cross-session
    Each ``session_id`` owns a dedicated ``asyncio.Queue`` + worker.
    Different sessions run in complete isolation.

Intra-round
    All agents listed in one ``PlanDecision.dispatches`` are dispatched
    concurrently via ``asyncio.gather``.

Correlation
    ``submit()`` registers an ``asyncio.Future`` keyed by ``correlation_id``.
    The worker resolves it when the planning loop finishes (or errors).

Usage
~~~~~
::

    from src.interaction import bus
    from src.task import Task
    from src.session import SessionContext

    await bus.initialize()                # sync agents from ACP
    ctx = SessionContext()
    task = Task(content="...", session_id=ctx.id)
    response = await bus.submit(task, session_ctx=ctx)
"""

from __future__ import annotations

import asyncio
import json
from typing import Any, Dict, List, Optional

from src.agent.server import acp
from src.interaction.types import BusEvent, BusMessage, BusMessageType, DeliveryMode
from src.logger import logger
from src.session import SessionContext
from src.task import Task
from src.model.request_budget import (
    ModelRequestBudget, ModelRequestBudgetExceeded, SampleNetworkFailureLimitExceeded, current_request_budget,
    request_budget_scope, request_role,
)

_DEFAULT_MAX_ROUNDS = 10


class AgentBus:
    """Session-isolated message bus that orchestrates the planner loop."""

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def __init__(self, planner_name: str = "planning") -> None:
        self.planner_name = planner_name

        self._session_queues: Dict[str, asyncio.Queue] = {}
        self._session_workers: Dict[str, asyncio.Task] = {}
        self._session_contexts: Dict[str, SessionContext] = {}

        self._pending_responses: Dict[str, asyncio.Future] = {}

        self._known_agents: Dict[str, str] = {}  # name → description
        self._event_log: List[BusEvent] = []
        self._lock: Optional[asyncio.Lock] = None

    def _get_lock(self) -> asyncio.Lock:
        if self._lock is None:
            self._lock = asyncio.Lock()
        return self._lock

    # ------------------------------------------------------------------
    # Agent registry  (synced from ACP)
    # ------------------------------------------------------------------

    async def initialize(self) -> None:
        """Sync agent names from ACP into the bus registry."""
        try:
            agent_names = await acp.list()
            for name in agent_names:
                info = await acp.get_info(name)
                if info and name not in self._known_agents:
                    self._known_agents[name] = info.description or ""
            logger.info(
                f"| Bus: synced {len(self._known_agents)} agents from ACP: "
                f"{list(self._known_agents.keys())}"
            )
        except Exception as exc:
            logger.warning(f"| Bus: failed to sync agents from ACP: {exc}")

    def register_agent(self, name: str, description: str = "") -> None:
        self._known_agents[name] = description

    def unregister_agent(self, name: str) -> None:
        self._known_agents.pop(name, None)

    def list_agents(self) -> List[str]:
        return list(self._known_agents.keys())

    def is_agent(self, name: str) -> bool:
        return name in self._known_agents

    def get_agent_description(self, name: str) -> Optional[str]:
        return self._known_agents.get(name)

    def _resolve_agent_name(self, raw_name: str) -> Optional[str]:
        """Try to fuzzy-match *raw_name* to a known agent.

        Handles common LLM mistakes like appending "_agent", using hyphens
        instead of underscores, or different casing.
        """
        known = self._known_agents
        # exact
        if raw_name in known:
            return raw_name
        lower = raw_name.lower().replace("-", "_")
        # strip trailing "_agent" / " agent"
        for suffix in ("_agent", " agent"):
            if lower.endswith(suffix):
                candidate = lower[: -len(suffix)]
                if candidate in known:
                    return candidate
        # case-insensitive
        for k in known:
            if k.lower() == lower:
                return k
        # substring match (e.g. "tool_calling" in "tool_calling_agent")
        for k in known:
            if k in lower or lower in k:
                return k
        return None

    # ------------------------------------------------------------------
    # Public API — submit
    # ------------------------------------------------------------------

    async def submit(
        self,
        task: Task,
        session_ctx: Optional[SessionContext] = None,
        max_rounds: int = _DEFAULT_MAX_ROUNDS,
        history_result_max_chars: int = 300,
        include_prior_results_in_subtasks: bool = False,
        subtask_context_max_chars: int = 48_000,
        dispatch_schedule: Optional[List[List[str]]] = None,
        planner_context_max_chars: Optional[int] = None,
        max_model_requests: Optional[int] = None,
    ) -> BusMessage:
        """Submit a task; the bus runs the full planner loop and returns the final result.

        Args:
            task:        Top-level work unit.
            session_ctx: Session context.  Auto-created if ``None``.
            max_rounds:  Maximum number of planner iterations.
            history_result_max_chars: Maximum characters retained from each
                sub-agent result in planner history.  The legacy default is
                intentionally small; evidence-heavy pipelines can opt in to a
                larger value.
            include_prior_results_in_subtasks: Append prior-round results to
                later sub-agent tasks.  This is useful for verifier/synthesis
                stages without coupling agents through shared mutable memory.
            subtask_context_max_chars: Maximum appended prior context per
                dispatched sub-agent.
            dispatch_schedule: Optional exact agent names for each planner
                round. An empty final round requires completion with no
                dispatches; a nonempty round rejects early completion and
                any missing, duplicate, or extra dispatches.
            planner_context_max_chars: Opt in to an indexed, deduplicated
                planner state bounded to this many characters. Full history
                remains in the local execution logs.
            max_model_requests: Optional whole-task provider invocation limit,
                shared by the planner and all concurrent/internal role calls.

        Returns:
            Terminal ``BusMessage`` (RESPONSE or ERROR).
        """
        if session_ctx is None:
            session_ctx = SessionContext()
        if planner_context_max_chars is not None and planner_context_max_chars < 1024:
            raise ValueError('planner_context_max_chars must be at least 1024')
        if max_model_requests is not None:
            ModelRequestBudget(max_model_requests)  # Validate before queuing.
        if dispatch_schedule is not None:
            if not dispatch_schedule or len(dispatch_schedule) != max_rounds:
                raise ValueError("dispatch_schedule must have exactly max_rounds entries")
            if dispatch_schedule[-1]:
                raise ValueError("dispatch_schedule must end with an empty completion round")
        if task.session_id is None:
            task.session_id = session_ctx.id

        await self._ensure_session(task.session_id, session_ctx)

        msg = BusMessage.task_message(
            session_id=task.session_id,
            task_id=task.id,
            content=task.content,
            files=task.files,
            recipients=[self.planner_name],
            delivery_mode=DeliveryMode.UNICAST,
        )
        msg.payload["max_rounds"] = max_rounds
        msg.payload["history_result_max_chars"] = max(1, history_result_max_chars)
        msg.payload["include_prior_results_in_subtasks"] = include_prior_results_in_subtasks
        msg.payload["subtask_context_max_chars"] = max(1, subtask_context_max_chars)
        msg.payload['planner_context_max_chars'] = planner_context_max_chars
        msg.payload['max_model_requests'] = max_model_requests
        if dispatch_schedule is not None:
            msg.payload["dispatch_schedule"] = [list(names) for names in dispatch_schedule]

        future: asyncio.Future = asyncio.get_event_loop().create_future()
        self._pending_responses[msg.correlation_id] = future

        task.mark_running()
        await self._session_queues[task.session_id].put(msg)
        self._log_event(msg, "message_enqueued")
        logger.info(f"| Bus: task '{task.id}' enqueued (session='{task.session_id}')")

        try:
            response: BusMessage = await future
            if response.type == BusMessageType.ERROR:
                task.mark_failed()
            else:
                task.mark_done()
            return response
        except asyncio.CancelledError:
            task.mark_cancelled()
            raise
        except Exception:
            task.mark_failed()
            raise

    # ------------------------------------------------------------------
    # Public API — direct dispatch (used ONLY for non-planner agents)
    # ------------------------------------------------------------------

    async def dispatch(
        self,
        agent_name: str,
        message: BusMessage,
        session_ctx: Optional[SessionContext] = None,
    ) -> BusMessage:
        """Dispatch a single message to a single agent via ACP.

        This is a **direct call**, not queued.  The bus uses it internally
        to call sub-agents inside the planner loop, and it can also be used
        externally for one-off agent calls outside the planning flow.
        """
        ctx = session_ctx or self._session_contexts.get(message.session_id)
        return await self._call_agent(agent_name, message, ctx=ctx)

    # ------------------------------------------------------------------
    # Shutdown
    # ------------------------------------------------------------------

    async def shutdown(self) -> None:
        """Cancel all workers and unblock all pending Futures."""
        logger.info("| Bus: shutting down")
        for sid, worker in list(self._session_workers.items()):
            worker.cancel()
            try:
                await worker
            except asyncio.CancelledError:
                pass
        for cid, future in list(self._pending_responses.items()):
            if not future.done():
                future.set_exception(RuntimeError("AgentBus shut down"))
        self._session_workers.clear()
        self._session_queues.clear()
        self._session_contexts.clear()
        self._pending_responses.clear()
        logger.info("| Bus: shutdown complete")

    # ------------------------------------------------------------------
    # Observability
    # ------------------------------------------------------------------

    def get_event_log(self, session_id: Optional[str] = None) -> List[BusEvent]:
        if session_id:
            return [e for e in self._event_log if e.session_id == session_id]
        return list(self._event_log)

    def list_sessions(self) -> List[str]:
        return list(self._session_queues.keys())

    # ------------------------------------------------------------------
    # Internal — session management
    # ------------------------------------------------------------------

    async def _ensure_session(
        self, session_id: str, session_ctx: SessionContext
    ) -> None:
        async with self._get_lock():
            if session_id not in self._session_queues:
                self._session_queues[session_id] = asyncio.Queue()
                self._session_contexts[session_id] = session_ctx
                worker = asyncio.create_task(
                    self._session_worker(session_id),
                    name=f"bus-worker-{session_id}",
                )
                self._session_workers[session_id] = worker
                logger.info(f"| Bus: session worker created for '{session_id}'")

    # ------------------------------------------------------------------
    # Internal — session worker (drains the queue)
    # ------------------------------------------------------------------

    async def _session_worker(self, session_id: str) -> None:
        """Long-lived coroutine: processes tasks sequentially within a session."""
        queue = self._session_queues[session_id]
        logger.info(f"| Bus: worker started (session='{session_id}')")

        while True:
            try:
                message: BusMessage = await queue.get()
                try:
                    await self._run_planner_loop(message)
                except Exception as exc:
                    logger.error(
                        f"| Bus: error in planner loop for '{message.id}': {exc}",
                        exc_info=True,
                    )
                    self._resolve_future(
                        message.correlation_id,
                        BusMessage.error_message(
                            session_id=message.session_id,
                            task_id=message.task_id,
                            correlation_id=message.correlation_id,
                            error=str(exc),
                            parent_id=message.id,
                        ),
                    )
                finally:
                    queue.task_done()
            except asyncio.CancelledError:
                logger.info(f"| Bus: worker cancelled (session='{session_id}')")
                break

    # ------------------------------------------------------------------
    # Internal — THE planner loop  (the core of the bus)
    # ------------------------------------------------------------------

    async def _run_planner_loop(self, message: BusMessage) -> None:
        # Bind in the queue worker, not the submitter: reused session workers
        # must receive a fresh case counter and parallel children inherit it.
        limit = message.payload.get("max_model_requests")
        budget = ModelRequestBudget(limit) if limit is not None else None
        with request_budget_scope(budget):
            try:
                await self._drive_planner_loop(message)
            except Exception as exc:
                if budget is None:
                    raise
                logger.error(f"| Bus: budgeted planner loop failed: {exc}")
                self._resolve_future(message.correlation_id, BusMessage.error_message(
                    session_id=message.session_id, task_id=message.task_id,
                    correlation_id=message.correlation_id, parent_id=message.id,
                    error=str(exc)))

    async def _drive_planner_loop(self, message: BusMessage) -> None:
        """Drive the full planner→dispatch→collect→repeat cycle for one task.

        This method is the heart of the bus.  It:

        1. Calls the planner agent via ACP to get a ``PlanDecision``.
        2. If ``is_done`` → resolve the caller's Future and return.
        3. Otherwise, dispatch all listed sub-agents concurrently.
        4. Feed results back to the planner in the next round.
        5. Repeat until ``is_done`` or ``max_rounds`` is exhausted.
        """
        # TTL check
        if message.is_expired():
            self._log_event(message, "ttl_expired")
            self._resolve_future(
                message.correlation_id,
                BusMessage.error_message(
                    session_id=message.session_id,
                    task_id=message.task_id,
                    correlation_id=message.correlation_id,
                    error=f"Message expired (TTL={message.ttl}s)",
                    parent_id=message.id,
                ),
            )
            return

        session_id = message.session_id
        task_id = message.task_id
        task_content = message.payload.get("content", "")
        task_files = message.payload.get("files", [])
        max_rounds = message.payload.get("max_rounds", _DEFAULT_MAX_ROUNDS)
        history_result_max_chars = message.payload.get("history_result_max_chars", 300)
        include_prior_results = message.payload.get(
            "include_prior_results_in_subtasks", False
        )
        subtask_context_max_chars = message.payload.get(
            "subtask_context_max_chars", 48_000
        )
        dispatch_schedule = message.payload.get("dispatch_schedule")
        planner_context_max_chars = message.payload.get('planner_context_max_chars')
        planner_state = None
        if planner_context_max_chars is not None:
            from src.agent.orchestra_handoffs import (PlannerEvidenceState, PLANNER_INDEXED_GUIDANCE,
                                                      dispatch_key, reusable_dispatch)
            planner_state = PlannerEvidenceState()
        ctx = self._session_contexts.get(session_id)

        # Build agent contract from bus-known agents only (exclude planner)
        sub_agent_lines = []
        for name, desc in self._known_agents.items():
            if name == self.planner_name:
                continue
            sub_agent_lines.append(f"- **{name}**: {desc or 'No description'}")
        if sub_agent_lines:
            agent_contract = (
                "Available agents (use these EXACT names in dispatches):\n"
                + "\n".join(sub_agent_lines)
            )
        else:
            agent_contract = "(no sub-agents available)"
        if planner_state is not None:
            agent_contract += PLANNER_INDEXED_GUIDANCE

        round_results: Dict[str, Any] = {}  # results from previous round
        execution_history = ""               # cumulative text log
        planner_history = ""
        dispatch_cache = {}
        reused_only_rounds = 0

        for round_num in range(1, max_rounds + 1):
            logger.info(
                f"| Bus: planner round {round_num}/{max_rounds} "
                f"(session='{session_id}')"
            )
            self._log_event(message, "planner_round_start", detail=str(round_num))

            # ---- 1. Call the planner ----
            try:
                planner_history = (planner_state.render(planner_context_max_chars)
                                   if planner_state is not None else execution_history)
            except ValueError as exc:
                self._resolve_future(message.correlation_id, BusMessage.error_message(
                    session_id=session_id, task_id=task_id,
                    correlation_id=message.correlation_id, error=str(exc), parent_id=message.id))
                return
            decision_dict = await self._call_planner_raw(
                task_content=task_content,
                task_files=task_files,
                ctx=ctx,
                task_id=task_id,
                round_number=round_num,
                max_rounds=max_rounds,
                agent_contract=agent_contract,
                execution_history=planner_history,
                round_results=round_results,
            )

            if decision_dict is None:
                logger.error("| Bus: planner returned None — aborting")
                self._resolve_future(
                    message.correlation_id,
                    BusMessage.error_message(
                        session_id=session_id,
                        task_id=task_id,
                        correlation_id=message.correlation_id,
                        error="Planner returned no decision",
                        parent_id=message.id,
                    ),
                )
                return

            is_done = decision_dict.get("is_done", False)
            final_result = decision_dict.get("final_result")
            dispatches = decision_dict.get("dispatches", [])
            plan_update = decision_dict.get("plan_update", "")

            # Optional strict schedule: a planner prompt alone cannot enforce
            # a fixed exploration budget or protect against early completion.
            if dispatch_schedule is not None:
                expected = dispatch_schedule[round_num - 1]
                actual = []
                for dispatch in dispatches:
                    raw_name = dispatch.get("agent_name", "")
                    actual.append(
                        raw_name if raw_name in self._known_agents
                        else self._resolve_agent_name(raw_name) or raw_name
                    )
                schedule_ok = (
                    is_done and not actual and not expected
                    if not expected
                    else not is_done and sorted(actual) == sorted(expected)
                )
                if not schedule_ok:
                    error = (
                        f"Planner violated dispatch schedule in round {round_num}: "
                        f"expected {expected or ['completion']}, got "
                        f"{actual or ['completion' if is_done else 'no dispatches']}"
                    )
                    logger.error(f"| Bus: {error}")
                    self._resolve_future(
                        message.correlation_id,
                        BusMessage.error_message(
                            session_id=session_id,
                            task_id=task_id,
                            correlation_id=message.correlation_id,
                            error=error,
                            parent_id=message.id,
                        ),
                    )
                    return

            # ---- 2. Done? ----
            if is_done:
                if planner_state is not None and planner_state.synthesis_stale:
                    planner_state.observe('planner', '', False, json.dumps(dict(
                        detail='New evidence or unresolved handoff after synthesis: '
                               'interpret, verify and synthesize the updated findings before finishing.')))
                    self._log_event(message, 'stale_synthesis_rejected', detail=str(round_num))
                    continue
                logger.info(f"| Bus: planner signalled done (round {round_num})")
                self._resolve_future(
                    message.correlation_id,
                    BusMessage.response_message(
                        session_id=session_id,
                        task_id=task_id,
                        correlation_id=message.correlation_id,
                        sender=self.planner_name,
                        success=True,
                        result=final_result or plan_update,
                    ),
                )
                return

            # ---- 3. No dispatches and not done → stall ----
            if not dispatches:
                logger.warning(
                    f"| Bus: planner returned no dispatches in round {round_num} — aborting"
                )
                self._resolve_future(
                    message.correlation_id,
                    BusMessage.error_message(
                        session_id=session_id,
                        task_id=task_id,
                        correlation_id=message.correlation_id,
                        error="Planner stalled (no dispatches and not done)",
                        parent_id=message.id,
                    ),
                )
                return

            # ---- 4. Validate & normalise agent names ----
            known = set(self._known_agents.keys())
            valid_dispatches = []
            for d in dispatches:
                raw_name = d["agent_name"]
                if raw_name in known:
                    valid_dispatches.append(d)
                else:
                    resolved = self._resolve_agent_name(raw_name)
                    if resolved:
                        logger.warning(
                            f"| Bus: LLM returned '{raw_name}', resolved to '{resolved}'"
                        )
                        d["agent_name"] = resolved
                        valid_dispatches.append(d)
                    else:
                        logger.error(
                            f"| Bus: unknown agent '{raw_name}' — skipping "
                            f"(known: {list(known)})"
                        )
            dispatches = valid_dispatches

            if not dispatches:
                logger.warning(
                    "| Bus: all dispatched agent names were invalid — aborting round"
                )
                execution_history += (
                    f"=== Round {round_num} ===\n"
                    f"ERROR: All agent names were invalid.\n\n"
                )
                continue

            agent_names = [d["agent_name"] for d in dispatches]
            delivery = "BROADCAST" if len(agent_names) > 1 else "UNICAST"
            logger.info(f"| Bus: {delivery} → {agent_names}")

            def subtask_content(dispatch: Dict[str, Any]) -> str:
                content = dispatch["task"]
                if include_prior_results and execution_history:
                    prior_context = execution_history[-subtask_context_max_chars:]
                    content += (
                        "\n\n<prior_agent_results>\n"
                        + prior_context
                        + "\n</prior_agent_results>"
                    )
                return content

            sub_messages = [
                BusMessage(
                    type=BusMessageType.PLAN,
                    session_id=session_id,
                    task_id=task_id,
                    sender="bus",
                    recipients=[d["agent_name"]],
                    delivery_mode=DeliveryMode.UNICAST,
                    payload={
                        "content": subtask_content(d),
                        "files": d.get("files", []),
                    },
                )
                for d in dispatches
            ]

            fresh_dispatches = 0
            pending_dispatches = {}
            reused_messages = set()
            round_evidence_version = tuple(planner_state.artifacts) if planner_state is not None else ()

            async def dispatch_once(sub_message):
                nonlocal fresh_dispatches
                if planner_state is None:
                    return await self._call_agent(sub_message.recipients[0], sub_message, ctx=ctx)
                key = dispatch_key(sub_message.recipients[0], sub_message.payload['content'],
                                   sub_message.payload.get('files'))
                evidence_version = tuple(planner_state.artifacts)
                cached = dispatch_cache.get(key)
                if cached and not cached[0].payload.get('success') and cached[1] != evidence_version:
                    del dispatch_cache[key]
                if key in dispatch_cache or key in pending_dispatches:
                    planner_state.reused_dispatches += 1
                    reused_messages.add(sub_message.id)
                    self._log_event(sub_message, 'dispatch_reused', agent_name=sub_message.recipients[0])
                    if key in dispatch_cache:
                        return dispatch_cache[key][0]
                    return await pending_dispatches[key]
                fresh_dispatches += 1
                pending_dispatches[key] = asyncio.create_task(
                    self._call_agent(sub_message.recipients[0], sub_message, ctx=ctx))
                response = await pending_dispatches[key]
                if reusable_dispatch(response.payload):
                    dispatch_cache[key] = (response, evidence_version)
                return response

            raw_responses = await asyncio.gather(
                *(dispatch_once(m) for m in sub_messages), return_exceptions=True)
            budget = current_request_budget()
            if budget is not None:
                budget.check_network_stop()
            if budget is not None and budget.snapshot()['blocked_requests']:
                raise ModelRequestBudgetExceeded(
                    f"Case model request budget exhausted ({budget.limit}/{budget.limit}); "
                    "one or more dispatches could not finish within the shared limit.")

            # ---- 5. Collect results ----
            round_results = {}
            history_parts: List[str] = [
                f"=== Round {round_num} ===",
                f"Plan: {plan_update}",
                f"Dispatched ({delivery}): {', '.join(agent_names)}",
            ]

            for d, raw, sub_message in zip(dispatches, raw_responses, sub_messages):
                name = d["agent_name"]
                subtask = d["task"]
                if planner_state is not None and sub_message.id not in reused_messages:
                    failed = isinstance(raw, BaseException)
                    planner_state.observe(
                        name, subtask, False if failed else bool(raw.payload.get('success', False)),
                        str(raw) if failed else str(raw.payload.get('result') or raw.payload.get('error') or ''),
                        synthesis_basis=round_evidence_version)
                history_parts.append(f"  {name} subtask: {subtask[:200]}")

                if isinstance(raw, BaseException):
                    logger.error(f"| Bus: agent '{name}' error: {raw}")
                    round_results[name] = {"success": False, "error": str(raw)}
                    history_parts.append(f"  FAIL {name}: {str(raw)[:300]}")
                else:
                    ok = raw.payload.get("success", False)
                    result_text = str(
                        raw.payload.get("result") or raw.payload.get("error") or ""
                    )
                    round_results[name] = {
                        "success": ok,
                        "result": result_text,
                        "error": raw.payload.get("error"),
                    }
                    tag = "OK" if ok else "FAIL"
                    history_parts.append(
                        f"  {tag} {name}: {result_text[:history_result_max_chars]}"
                    )
                    logger.info(f"| Bus: agent '{name}' → {tag}")

            history_parts.append("")
            execution_history += "\n".join(history_parts) + "\n"

            self._log_event(message, "round_complete", detail=str(round_num))
            if planner_state is not None:
                reused_only_rounds = reused_only_rounds + 1 if fresh_dispatches == 0 else 0
                if reused_only_rounds >= 2:
                    self._resolve_future(message.correlation_id, BusMessage.error_message(
                        session_id=session_id, task_id=task_id,
                        correlation_id=message.correlation_id, parent_id=message.id,
                        error='Planner repeated unchanged dispatches for two rounds without new evidence; '
                              'saved results were reused and no agents were rerun.'))
                    return

        # ---- Max rounds exhausted ----
        logger.warning(f"| Bus: max rounds ({max_rounds}) reached")
        self._resolve_future(
            message.correlation_id,
            BusMessage.error_message(
                session_id=session_id,
                task_id=task_id,
                correlation_id=message.correlation_id,
                error=f"Planner did not finish within {max_rounds} rounds",
                parent_id=message.id,
            ),
        )

        # Tell the planner to finalize plan.md as failed
        # Budgeted tasks cannot issue cleanup calls after their result and
        # usage have already been delivered to the submitter.
        if current_request_budget() is not None:
            return
        await self._call_planner_raw(
            task_content=task_content,
            task_files=task_files,
            ctx=ctx,
            task_id=task_id,
            round_number=max_rounds + 1,
            max_rounds=max_rounds,
            agent_contract=agent_contract,
            execution_history=(execution_history if planner_state is None else planner_history),
            round_results=round_results,
        )

    # ------------------------------------------------------------------
    # Internal — call planner via ACP and extract PlanDecision dict
    # ------------------------------------------------------------------

    async def _call_planner_raw(
        self,
        task_content: str,
        task_files: List[str],
        ctx: Optional[SessionContext],
        **kwargs,
    ) -> Optional[Dict[str, Any]]:
        """Call the planner via ACP and return the raw PlanDecision dict."""
        budget = current_request_budget()
        if budget is not None:
            budget.check_available()
            usage = budget.snapshot()
            kwargs['agent_contract'] = kwargs.get('agent_contract', '') + (
                f"\nShared case model request budget: {usage['requests_used']}/{budget.limit} "
                f"used, {usage['requests_remaining']} remaining before this planner call. "
                "Every planner call and every internal role call consumes one request; network failures are excluded. "
                "Allow room for interpretation, verification, synthesis and final planner completion."
            )
        try:
            with request_role(self.planner_name):
                agent_response = await acp(
                    name=self.planner_name,
                    input={
                        "task": task_content,
                        "files": task_files if task_files else None,
                    },
                    ctx=ctx,
                    **kwargs,
                )
            if hasattr(agent_response, "extra") and agent_response.extra:
                data = agent_response.extra.data or {}
                return data.get("decision")
        except (ModelRequestBudgetExceeded, SampleNetworkFailureLimitExceeded):
            raise
        except Exception as exc:
            logger.error(f"| Bus: planner call failed: {exc}", exc_info=True)
        return None

    # ------------------------------------------------------------------
    # Internal — call any agent via ACP → BusMessage
    # ------------------------------------------------------------------

    async def _call_agent(
        self,
        agent_name: str,
        message: BusMessage,
        ctx: Optional[SessionContext] = None,
        extra_kwargs: Optional[Dict[str, Any]] = None,
    ) -> BusMessage:
        """Call one agent via ACP, wrapping the result in a BusMessage.

        Never raises — errors are returned as ERROR-type BusMessages.
        """
        self._log_event(message, "agent_dispatched", agent_name=agent_name)
        logger.info(f"| Bus: dispatching → '{agent_name}'")

        try:
            call_kwargs: Dict[str, Any] = {}
            if extra_kwargs:
                call_kwargs.update(extra_kwargs)

            with request_role(agent_name):
                agent_result = await acp(
                    name=agent_name,
                    input={
                        "task": message.payload.get("content", ""),
                        "files": message.payload.get("files", []),
                    },
                    ctx=ctx,
                    **call_kwargs,
                )

            success = getattr(agent_result, "success", bool(agent_result))
            result_data = getattr(agent_result, "message", str(agent_result))
            error_str = None if success else result_data

            return BusMessage.response_message(
                session_id=message.session_id,
                task_id=message.task_id,
                correlation_id=message.correlation_id,
                sender=agent_name,
                success=success,
                result=result_data,
                error=error_str,
                parent_id=message.id,
            )

        except (ModelRequestBudgetExceeded, SampleNetworkFailureLimitExceeded):
            raise
        except Exception as exc:
            logger.error(f"| Bus: agent '{agent_name}' raised: {exc}", exc_info=True)
            return BusMessage.error_message(
                session_id=message.session_id,
                task_id=message.task_id,
                correlation_id=message.correlation_id,
                error=str(exc),
                parent_id=message.id,
            )

    # ------------------------------------------------------------------
    # Internal — helpers
    # ------------------------------------------------------------------

    def _resolve_future(self, correlation_id: str, response: BusMessage) -> None:
        budget = current_request_budget()
        if budget is not None:
            usage = budget.snapshot()
            if usage['network_failure_stop']:
                response = BusMessage.error_message(
                    session_id=response.session_id, task_id=response.task_id,
                    correlation_id=correlation_id, parent_id=response.parent_id,
                    error='Sample stopped after 5 consecutive network request failures; '
                          'network failures do not consume the model request budget.')
                response.payload['error_type'] = 'SampleNetworkFailureLimitExceeded'
            elif usage['blocked_requests']:
                response = BusMessage.error_message(
                    session_id=response.session_id, task_id=response.task_id,
                    correlation_id=correlation_id, parent_id=response.parent_id,
                    error=f"Case model request budget exhausted ({usage['requests_used']}/{budget.limit}); "
                          "workflow did not complete within the request limit.")
                response.payload['error_type'] = 'ModelRequestBudgetExceeded'
            response.payload['request_budget'] = usage
        future = self._pending_responses.pop(correlation_id, None)
        if future is not None and not future.done():
            future.set_result(response)

    def _log_event(
        self,
        message: BusMessage,
        event_type: str,
        agent_name: Optional[str] = None,
        detail: Optional[str] = None,
    ) -> None:
        self._event_log.append(
            BusEvent(
                session_id=message.session_id,
                task_id=message.task_id,
                message_id=message.id,
                event_type=event_type,
                agent_name=agent_name,
                detail=detail,
            )
        )


# ---------------------------------------------------------------------------
# Global singleton
# ---------------------------------------------------------------------------

bus = AgentBus()
