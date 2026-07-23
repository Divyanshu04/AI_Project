from app.graph.workflow import (
    build_clinical_graph,
)


# =========================================
# BUILD CLINICAL AGENT GRAPH
# =========================================

clinical_graph = (
    build_clinical_graph()
)


# =========================================
# RUN CLINICAL AGENT
# =========================================

def run_clinical_agent(
        clinical_case: str,
) -> dict:

    # =========================================
    # INITIAL STATE
    # =========================================

    initial_state = {

        "clinical_case": (
            clinical_case
        ),

        "workflow_status": (
            "Clinical analysis started"
        ),
    }


    # =========================================
    # EXECUTE LANGGRAPH WORKFLOW
    # =========================================

    final_state = (
        clinical_graph.invoke(
            initial_state
        )
    )


    # =========================================
    # RETURN FINAL AGENT RESULT
    # =========================================

    return {

        # =========================================
        # CLINICAL RESPONSE
        # =========================================

        "response": final_state.get(
            "response",
            "",
        ),


        # =========================================
        # SAFETY
        # =========================================

        "safety": final_state.get(
            "safety",
            {},
        ),


        # =========================================
        # ROUTING
        # =========================================

        "intent": final_state.get(
            "intent",
            "",
        ),

        "clinical_domain": final_state.get(
            "clinical_domain",
            "GENERAL_CLINICAL",
        ),


        # =========================================
        # RAG EVIDENCE
        # =========================================

        "evidence": final_state.get(
            "evidence",
            [],
        ),

        "evidence_details": final_state.get(
            "evidence_details",
            [],
        ),

        "sources": final_state.get(
            "sources",
            [],
        ),


        # =========================================
        # RAG CONFIDENCE
        # =========================================

        "rag_confidence": final_state.get(
            "rag_confidence",
            0.0,
        ),

        "rag_confidence_level": final_state.get(
            "rag_confidence_level",
            "LOW",
        ),


        # =========================================
        # EVIDENCE QUALITY
        # =========================================

        "evidence_quality": final_state.get(
            "evidence_quality",
            "UNKNOWN",
        ),

        "evidence_quality_message": final_state.get(
            "evidence_quality_message",
            "",
        ),

        "proceed_with_llm": final_state.get(
            "proceed_with_llm",
            False,
        ),


        # =========================================
        # OUTPUT SAFETY
        # =========================================

        "output_safe": final_state.get(
            "output_safe",
            False,
        ),

        "safety_message": final_state.get(
            "safety_message",
            "",
        ),


        # =========================================
        # WORKFLOW
        # =========================================

        "workflow_status": final_state.get(
            "workflow_status",
            "",
        ),
    }