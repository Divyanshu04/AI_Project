from langgraph.graph import (
    StateGraph,
    START,
    END,
)

from app.graph.state import (
    ClinicalAgentState,
)

from app.graph.nodes import (
    safety_node,
    intent_router_node,
    clinical_domain_router_node,
    high_risk_response_node,
    rag_node,
    prompt_node,
    llm_node,
    output_guardrail_node,
)


# =========================================
# CONDITIONAL ROUTER
# =========================================

def route_after_intent(
        state: ClinicalAgentState,
) -> str:

    intent = state.get(
        "intent",
        "CLINICAL_QUESTION",
    )

    if intent == "HIGH_RISK":

        return "high_risk"

    return "clinical"


# =========================================
# BUILD CLINICAL AGENT GRAPH
# =========================================

def build_clinical_graph():

    graph = StateGraph(
        ClinicalAgentState
    )


    # =========================================
    # ADD NODES
    # =========================================

    graph.add_node(
        "safety",
        safety_node,
    )

    graph.add_node(
        "intent_router",
        intent_router_node,
    )

    graph.add_node(
        "clinical_domain_router",
        clinical_domain_router_node,
    )

    graph.add_node(
        "high_risk_response",
        high_risk_response_node,
    )

    graph.add_node(
        "rag",
        rag_node,
    )

    graph.add_node(
        "prompt",
        prompt_node,
    )

    graph.add_node(
        "llm",
        llm_node,
    )

    graph.add_node(
        "output_guardrail",
        output_guardrail_node,
    )


    # =========================================
    # START → SAFETY
    # =========================================

    graph.add_edge(
        START,
        "safety",
    )


    # =========================================
    # SAFETY → INTENT ROUTER
    # =========================================

    graph.add_edge(
        "safety",
        "intent_router",
    )


    # =========================================
    # INTENT CONDITIONAL ROUTING
    # =========================================

    graph.add_conditional_edges(
        "intent_router",
        route_after_intent,
        {
            "high_risk": (
                "high_risk_response"
            ),
            "clinical": (
                "clinical_domain_router"
            ),
        },
    )


    # =========================================
    # HIGH-RISK → END
    # =========================================

    graph.add_edge(
        "high_risk_response",
        END,
    )


    # =========================================
    # DOMAIN ROUTER → MEDICAL RAG
    # =========================================

    graph.add_edge(
        "clinical_domain_router",
        "rag",
    )


    # =========================================
    # RAG → PROMPT BUILDER
    # =========================================

    graph.add_edge(
        "rag",
        "prompt",
    )


    # =========================================
    # PROMPT → GEMINI
    # =========================================

    graph.add_edge(
        "prompt",
        "llm",
    )


    # =========================================
    # GEMINI → OUTPUT GUARDRAIL
    # =========================================

    graph.add_edge(
        "llm",
        "output_guardrail",
    )


    # =========================================
    # OUTPUT GUARDRAIL → END
    # =========================================

    graph.add_edge(
        "output_guardrail",
        END,
    )


    # =========================================
    # COMPILE GRAPH
    # =========================================

    return graph.compile()