def _role(name, description, *, computation=False):
    return dict(name=name, description=description, workdir=f'workdir/trajstory/agent/{name}',
        model_name='openai/gpt-6-luna', prompt_name='tool_calling' if computation else None,
        memory_name=None, use_memory=False, use_todo=False, max_steps=50,
        max_tools=10, review_steps=5, require_grad=False)

orchestra_analyst_agent = _role('orchestra_analyst',
    'Compute one focused question with Python; return brief status only. Code/stdout are logged and returned by evidence ID.', computation=True)
orchestra_interpreter_agent = _role('orchestra_interpreter',
    'Interpret E_ execution IDs via short JSON tasks with instruction and evidence_ids; exact citations, no pasted logs or tools.')
orchestra_verifier_agent = _role('orchestra_verifier',
    'Review interpreter R_ IDs; execution evidence is resolved once. Approve/revise/reject with exact citations, no pasted logs or tools.')
orchestra_synthesizer_agent = _role('orchestra_synthesizer',
    'Synthesize verifier R_ IDs into compact insights plus Summary; no execution logs, tools or new findings.')
