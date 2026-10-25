"""Enforce a shared 48-round budget while retaining the official Pandas loop."""
from langchain.agents import AgentExecutor
from langchain_core.agents import AgentFinish

MAX_ROUNDS = 48
ROUND_POLICY = (
    'You have a HARD LIMIT of 48 Agent rounds, including the round that returns '
    'the final answer. Each round is one planning/model decision; Python tool '
    'calls are not separate rounds. The program supplies the current round '
    'counter on every turn. Return SIX insight strings and Summary no later '
    'than round 48. On round 48 return the final JSON instead of requesting '
    'more tools. If you do not finish within 48 rounds, this case is FAILED. '
    'The limit is a ceiling, not a target.\n\n'
)

class RoundCountedPandasExecutor(AgentExecutor):
    rounds_used: int = 0
    completed_normally: bool = False

    def _iter_next_step(self, name_to_tool_map, color_mapping, inputs,
                        intermediate_steps, run_manager=None):
        if self.rounds_used >= MAX_ROUNDS:
            raise RuntimeError('RoundLimitExceeded: no 49th model decision is allowed')
        self.rounds_used += 1
        counted_inputs = dict(inputs)
        counter = (
            f'CURRENT AGENT ROUND: {self.rounds_used}/48. '
            f'Remaining rounds after this one: {MAX_ROUNDS - self.rounds_used}. '
            'Final answer must be submitted by round 48; otherwise this case fails.\n\n'
        )
        if self.rounds_used == MAX_ROUNDS:
            counter += 'LAST ROUND: return the final JSON now; no further analysis round is available.\n\n'
        counted_inputs['input'] = counter + inputs['input']
        for step in super()._iter_next_step(name_to_tool_map, color_mapping,
                                           counted_inputs, intermediate_steps, run_manager):
            if isinstance(step, AgentFinish):
                self.completed_normally = True
            yield step

def apply_round_budget(executor):
    return RoundCountedPandasExecutor(
        agent=executor.agent, tools=executor.tools,
        return_intermediate_steps=True, max_iterations=MAX_ROUNDS,
        max_execution_time=executor.max_execution_time,
        early_stopping_method='force', handle_parsing_errors=executor.handle_parsing_errors,
    )
