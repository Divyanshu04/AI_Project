import re


# =========================================
# HIGH-RISK CLINICAL TERMS
# =========================================

HIGH_RISK_TERMS = [
    "chest pain",
    "difficulty breathing",
    "shortness of breath",
    "severe bleeding",
    "unconscious",
    "stroke",
    "seizure",
    "suicidal",
]


# =========================================
# NEGATION TERMS
# =========================================

NEGATION_TERMS = [
    "no",
    "not",
    "without",
    "denies",
    "denied",
    "negative for",
]


# =========================================
# CHECK WHETHER TERM IS NEGATED
# =========================================

def is_negated(
        text: str,
        term: str,
) -> bool:

    # Find the location of the high-risk term

    term_position = text.find(
        term
    )

    if term_position == -1:

        return False


    # Look at the words immediately before
    # the detected high-risk term

    preceding_text = text[
        max(
            0,
            term_position - 50,
            ):
        term_position
    ]


    # Check for negation phrases

    for negation in NEGATION_TERMS:

        if negation in preceding_text:

            return True


    return False


# =========================================
# SAFETY CHECK
# =========================================

def run_safety_check(
        user_input: str,
) -> dict:

    """
    Perform a basic rule-based safety check.

    NOTE:
    This is a demonstration safety layer.
    It is not a clinically validated triage system.
    """

    text = (
        user_input
        .lower()
        .strip()
    )


    detected_terms = []

    negated_terms = []


    # =========================================
    # CHECK HIGH-RISK TERMS
    # =========================================

    for term in HIGH_RISK_TERMS:

        if term in text:

            if is_negated(
                    text,
                    term,
            ):

                negated_terms.append(
                    term
                )

            else:

                detected_terms.append(
                    term
                )


    # =========================================
    # HIGH-RISK CASE
    # =========================================

    if detected_terms:

        return {

            "risk_level": "HIGH",

            "requires_urgent_attention": True,

            "detected_terms": (
                detected_terms
            ),

            "negated_terms": (
                negated_terms
            ),

            "message": (
                "Potentially serious symptoms detected. "
                "The user should seek immediate professional "
                "medical evaluation."
            ),
        }


    # =========================================
    # LOW-RISK CASE
    # =========================================

    return {

        "risk_level": "LOW",

        "requires_urgent_attention": False,

        "detected_terms": [],

        "negated_terms": (
            negated_terms
        ),

        "message": (
            "No predefined high-risk symptoms "
            "detected."
        ),
    }