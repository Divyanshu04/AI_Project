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


def run_safety_check(user_input: str) -> dict:
    """
    Perform a basic safety check on the clinical input.
    """

    text = user_input.lower()

    detected_terms = [
        term
        for term in HIGH_RISK_TERMS
        if term in text
    ]

    if detected_terms:
        return {
            "risk_level": "HIGH",
            "requires_urgent_attention": True,
            "detected_terms": detected_terms,
            "message": (
                "Potentially serious symptoms detected. "
                "The user should seek immediate professional "
                "medical evaluation."
            ),
        }

    return {
        "risk_level": "LOW",
        "requires_urgent_attention": False,
        "detected_terms": [],
        "message": "No predefined high-risk symptoms detected.",
    }