from typing import TypedDict


class ClinicalAgentState(
    TypedDict,
    total=False,
):

    # =========================================
    # USER INPUT
    # =========================================

    clinical_case: str


    # =========================================
    # ROUTING
    # =========================================

    intent: str

    clinical_domain: str


    # =========================================
    # SAFETY
    # =========================================

    safety: dict


    # =========================================
    # RAG
    # =========================================

    # Retrieved evidence text

    evidence: list[str]


    # Detailed evidence objects
    #
    # Example:
    #
    # {
    #     "text": "...",
    #     "source": "hypertension.txt",
    #     "domain": "HYPERTENSION",
    #     "document_id": "hypertension",
    #     "evidence_level": "DEMO",
    #     "publication_year": "N/A",
    #     "distance": 0.25,
    #     "confidence": 80.0,
    #     "confidence_level": "HIGH"
    # }

    evidence_details: list[dict]


    # Evidence source names

    sources: list[str]


    # Overall RAG confidence

    rag_confidence: float


    # HIGH / MEDIUM / LOW

    rag_confidence_level: str


    # =========================================
    # EVIDENCE QUALITY GATE
    # =========================================

    # Overall evidence quality
    #
    # Possible values:
    #
    # SUFFICIENT
    # INSUFFICIENT

    evidence_quality: str


    # Explanation of evidence quality decision

    evidence_quality_message: str


    # Controls whether the workflow
    # is allowed to call the LLM
    #
    # True  → Continue to Prompt → Gemini
    #
    # False → Skip LLM and generate
    #          insufficient evidence response

    proceed_with_llm: bool


    # =========================================
    # LLM
    # =========================================

    prompt: str

    response: str


    # =========================================
    # OUTPUT SAFETY
    # =========================================

    output_safe: bool

    safety_message: str


    # =========================================
    # WORKFLOW
    # =========================================

    workflow_status: str