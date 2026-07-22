from app.prompts.router_prompt import (
    build_router_prompt,
)

from app.guardrails.safety import (
    run_safety_check,
)

from app.rag.retriever import (
    retrieve_clinical_evidence,
)

from app.prompts.clinical_prompt import (
    build_clinical_prompt,
)

from app.llm.gemini_client import (
    ask_gemini,
)


# =========================================
# NODE 1: SAFETY SCREENING
# =========================================

def safety_node(
        state: dict,
) -> dict:

    clinical_case = state[
        "clinical_case"
    ]

    safety_result = run_safety_check(
        clinical_case
    )

    return {
        "safety": safety_result,
        "workflow_status": (
            "Safety screening completed"
        ),
    }


# =========================================
# NODE 2: INTENT ROUTER
# =========================================

def intent_router_node(
        state: dict,
) -> dict:

    safety_result = state[
        "safety"
    ]

    if safety_result[
        "requires_urgent_attention"
    ]:

        intent = "HIGH_RISK"

    else:

        intent = "CLINICAL_QUESTION"

    return {
        "intent": intent,
        "workflow_status": (
            f"Intent classified as: {intent}"
        ),
    }


# =========================================
# NODE 3: CLINICAL DOMAIN ROUTER
# =========================================

def clinical_domain_router_node(
        state: dict,
) -> dict:

    clinical_case = state[
        "clinical_case"
    ]

    # -----------------------------------------
    # Build domain classification prompt
    # -----------------------------------------

    prompt = build_router_prompt(
        clinical_case
    )

    # -----------------------------------------
    # Ask Gemini for domain classification
    # -----------------------------------------

    raw_domain = ask_gemini(
        prompt
    )

    # -----------------------------------------
    # Normalize domain
    # -----------------------------------------

    domain = (
        raw_domain
        .strip()
        .upper()
        .replace(
            " ",
            "_",
        )
    )

    # -----------------------------------------
    # Allowed clinical domains
    # -----------------------------------------

    valid_domains = {
        "CARDIOVASCULAR",
        "HYPERTENSION",
        "DIABETES",
        "RESPIRATORY",
        "MEDICATION_SAFETY",
        "GENERAL_CLINICAL",
    }

    # -----------------------------------------
    # Fallback for unexpected LLM output
    # -----------------------------------------

    if domain not in valid_domains:

        domain = (
            "GENERAL_CLINICAL"
        )

    return {
        "clinical_domain": domain,
        "workflow_status": (
            f"Clinical domain classified as: "
            f"{domain}"
        ),
    }


# =========================================
# NODE 4: HIGH-RISK SAFETY RESPONSE
# =========================================

def high_risk_response_node(
        state: dict,
) -> dict:

    safety_result = state[
        "safety"
    ]

    response = f"""
## 🚨 High-Risk Clinical Alert

The initial safety screening identified potentially
serious symptoms or clinical concerns.

### Safety Message

{ safety_result.get(
        "message",
        "Immediate professional medical evaluation "
        "may be required."
    ) }

### Recommended Action

The patient should be evaluated promptly by a
qualified healthcare professional.

If symptoms are severe, rapidly worsening, or
potentially life-threatening, emergency medical
services should be contacted immediately.

### Important

ClinicaSense AI does not diagnose medical conditions
or replace professional clinical judgment.
"""

    return {
        "response": response,
        "workflow_status": (
            "High-risk safety response generated"
        ),
        "output_safe": True,
        "safety_message": (
            "High-risk case routed directly "
            "to safety response."
        ),
    }


# =========================================
# NODE 5: MEDICAL RAG
# =========================================

def rag_node(
        state: dict,
) -> dict:

    clinical_case = state[
        "clinical_case"
    ]

    # -----------------------------------------
    # Get clinical domain from router
    # -----------------------------------------

    clinical_domain = state.get(
        "clinical_domain",
        "GENERAL_CLINICAL",
    )

    # -----------------------------------------
    # Domain-aware Medical RAG
    # -----------------------------------------

    rag_result = (
        retrieve_clinical_evidence(
            query=clinical_case,
            n_results=3,
            clinical_domain=clinical_domain,
        )
    )

    # -----------------------------------------
    # Extract Sources
    # -----------------------------------------

    sources = [
        source["source"]
        for source in rag_result[
            "sources"
        ]
    ]

    # -----------------------------------------
    # Return RAG Results
    # -----------------------------------------

    return {
        "evidence": rag_result[
            "documents"
        ],
        "sources": sources,
        "workflow_status": (
            f"Clinical evidence retrieved "
            f"for domain: {clinical_domain}"
        ),
    }


# =========================================
# NODE 6: PROMPT BUILDER
# =========================================

def prompt_node(
        state: dict,
) -> dict:

    clinical_case = state[
        "clinical_case"
    ]

    safety_result = state[
        "safety"
    ]

    clinical_domain = state.get(
        "clinical_domain",
        "GENERAL_CLINICAL",
    )

    evidence = "\n\n".join(
        state.get(
            "evidence",
            [],
        )
    )

    prompt = build_clinical_prompt(
        clinical_case=clinical_case,
        safety_result=safety_result,
        evidence=evidence,
    )

    return {
        "prompt": prompt,
        "workflow_status": (
            f"Evidence-grounded prompt created "
            f"for domain: {clinical_domain}"
        ),
    }


# =========================================
# NODE 7: GEMINI LLM
# =========================================

def llm_node(
        state: dict,
) -> dict:

    prompt = state[
        "prompt"
    ]

    response = ask_gemini(
        prompt
    )

    return {
        "response": response,
        "workflow_status": (
            "Clinical response generated"
        ),
    }


# =========================================
# NODE 8: OUTPUT GUARDRAIL
# =========================================

def output_guardrail_node(
        state: dict,
) -> dict:

    response = state.get(
        "response",
        "",
    )

    unsafe_phrases = [
        "you definitely have",
        "you are diagnosed with",
        "take this medication",
        "stop your medication",
    ]

    response_lower = response.lower()

    detected_unsafe_phrases = [
        phrase
        for phrase in unsafe_phrases
        if phrase in response_lower
    ]

    # -----------------------------------------
    # Unsafe Output Detected
    # -----------------------------------------

    if detected_unsafe_phrases:

        return {
            "output_safe": False,
            "safety_message": (
                "The generated response contained "
                "potentially unsafe clinical language "
                "and requires professional review."
            ),
            "workflow_status": (
                "Output safety check failed"
            ),
        }

    # -----------------------------------------
    # Output Passed Guardrail
    # -----------------------------------------

    return {
        "output_safe": True,
        "safety_message": (
            "Output safety validation passed."
        ),
        "workflow_status": (
            "Output safety check completed"
        ),
    }