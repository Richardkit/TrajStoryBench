"""Agent configs for the TrajStory multi-agent pipeline."""


def _specialist(name: str, description: str, *, max_steps: int = 15) -> dict:
    return dict(
        workdir=f"workdir/trajstory/agent/{name}",
        name=name,
        type="Agent",
        description=description,
        model_name="openai/gpt-5.6-luna",
        prompt_name="tool_calling",
        memory_name=None,
        max_tools=1,
        max_steps=max_steps,
        review_steps=3,
        log_max_length=4000,
        require_grad=False,
        use_memory=False,
        use_todo=False,
    )


trajstory_temporal_spatial_agent = _specialist(
    "trajstory_temporal_spatial",
    "Analyses temporal, spatial, route, movement, and time-space patterns.",
    max_steps=9,
)
trajstory_behavior_comparison_agent = _specialist(
    "trajstory_behavior_comparison",
    "Analyses entity behaviour, group contrasts, routines, and cross-table relationships.",
    max_steps=9,
)
trajstory_anomaly_agent = _specialist(
    "trajstory_anomaly",
    "Analyses robust anomalies, counter-patterns, rare regimes, and data-quality confounds.",
    max_steps=9,
)
trajstory_evidence_verifier_agent = _specialist(
    "trajstory_evidence_verifier",
    "Reverse-checks all candidate claims against the input tables and reports coverage gaps.",
)
trajstory_synthesis_agent = _specialist(
    "trajstory_synthesis",
    "Produces broad, non-redundant final insights and an integrated summary from verified evidence.",
)
