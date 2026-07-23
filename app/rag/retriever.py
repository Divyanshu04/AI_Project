from app.rag.embeddings import (
    generate_embeddings,
)

from app.rag.vector_store import (
    search_documents,
)


# =========================================
# CONFIDENCE CALCULATION
# =========================================

def calculate_confidence(
        distance: float,
) -> float:

    """
    Convert ChromaDB distance into a simple
    prototype retrieval similarity score.

    Lower distance = higher similarity.

    NOTE:
    This is a prototype heuristic.
    It is NOT a clinically validated
    confidence score.
    """

    confidence = (
                         1.0 - float(distance)
                 ) * 100

    confidence = max(
        0.0,
        min(
            100.0,
            confidence,
        ),
    )

    return round(
        confidence,
        2,
    )


# =========================================
# CONFIDENCE LEVEL
# =========================================

def get_confidence_level(
        confidence: float,
) -> str:

    if confidence >= 70:

        return "HIGH"

    if confidence >= 40:

        return "MEDIUM"

    return "LOW"


# =========================================
# CLINICAL EVIDENCE RETRIEVER
# =========================================

def retrieve_clinical_evidence(
        query: str,
        n_results: int = 3,
        clinical_domain: str | None = None,
) -> dict:

    # =========================================
    # GENERATE QUERY EMBEDDING
    # =========================================

    query_embedding = generate_embeddings(
        [query]
    )[0]


    # =========================================
    # SEARCH VECTOR DATABASE
    # =========================================

    results = search_documents(
        query_embedding=query_embedding,
        n_results=n_results,
        clinical_domain=clinical_domain,
    )


    # =========================================
    # EXTRACT RESULTS
    # =========================================

    documents = results.get(
        "documents",
        [[]],
    )[0]

    metadatas = results.get(
        "metadatas",
        [[]],
    )[0]

    distances = results.get(
        "distances",
        [[]],
    )[0]


    # =========================================
    # BUILD EVIDENCE DETAILS
    # =========================================

    evidence_details = []

    confidence_scores = []


    for index, document in enumerate(
            documents
    ):

        # -----------------------------------------
        # Get Metadata
        # -----------------------------------------

        metadata = {}

        if index < len(
                metadatas
        ):

            metadata = (
                    metadatas[index]
                    or {}
            )


        # -----------------------------------------
        # Get Distance
        # -----------------------------------------

        distance = 0.0

        if index < len(
                distances
        ):

            distance = float(
                distances[index]
            )


        # -----------------------------------------
        # Calculate Retrieval Score
        # -----------------------------------------

        confidence = (
            calculate_confidence(
                distance
            )
        )


        confidence_level = (
            get_confidence_level(
                confidence
            )
        )


        confidence_scores.append(
            confidence
        )


        # -----------------------------------------
        # Evidence Details
        # -----------------------------------------

        evidence_details.append(
            {
                "text": document,

                "source": metadata.get(
                    "source",
                    "Unknown",
                ),

                "domain": metadata.get(
                    "domain",
                    clinical_domain
                    or "GENERAL_CLINICAL",
                    ),

                "document_id": metadata.get(
                    "document_id",
                    "",
                ),

                "evidence_level": metadata.get(
                    "evidence_level",
                    "DEMO",
                ),

                "publication_year": metadata.get(
                    "publication_year",
                    "N/A",
                ),

                "distance": distance,

                "confidence": confidence,

                "confidence_level": (
                    confidence_level
                ),
            }
        )


    # =========================================
    # OVERALL RETRIEVAL SCORE
    # =========================================

    if confidence_scores:

        overall_confidence = max(
            confidence_scores
        )

    else:

        overall_confidence = 0.0


    overall_confidence_level = (
        get_confidence_level(
            overall_confidence
        )
    )


    # =========================================
    # SOURCE LIST
    # =========================================

    sources = [

        metadata.get(
            "source",
            "Unknown",
        )

        for metadata in metadatas

    ]


    # =========================================
    # RETURN RETRIEVAL RESULT
    # =========================================

    return {

        # Detailed evidence objects

        "evidence": (
            evidence_details
        ),

        # Raw document text

        "documents": (
            documents
        ),

        # Evidence sources

        "sources": (
            sources
        ),

        # Overall retrieval score

        "confidence": (
            overall_confidence
        ),

        # HIGH / MEDIUM / LOW

        "confidence_level": (
            overall_confidence_level
        ),
    }