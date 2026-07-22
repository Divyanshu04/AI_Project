from app.graph.workflow import (
    build_clinical_graph,
)


clinical_graph = (
    build_clinical_graph()
)


def run_clinical_agent(
        clinical_case: str,
) -> dict:

    initial_state = {
        "clinical_case": clinical_case,
        "workflow_status": (
            "Clinical analysis started"
        ),
    }

    final_state = (
        clinical_graph.invoke(
            initial_state
        )
    )

    return {
        "response": final_state.get(
            "response",
            "",
        ),
        "safety": final_state.get(
            "safety",
            {},
        ),
        "evidence": final_state.get(
            "evidence",
            [],
        ),
        "sources": final_state.get(
            "sources",
            [],
        ),
        "intent": final_state.get(
            "intent",
            "",
        ),
        "clinical_domain": final_state.get(
            "clinical_domain",
            "",
        ),
        "output_safe": final_state.get(
            "output_safe",
            False,
        ),
        "safety_message": final_state.get(
            "safety_message",
            "",
        ),
        "workflow_status": final_state.get(
            "workflow_status",
            "",
        ),
    }