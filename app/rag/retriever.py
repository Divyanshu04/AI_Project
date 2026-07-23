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
    Convert ChromaDB vector distance into a
    prototype retrieval confidence score.

    Lower distance = better similarity.

    IMPORTANT:
    This is an engineering heuristic for the
    ClinicaSense AI prototype.

    It is NOT a clinically validated
    confidence score.
    """

    distance = float(
        distance
    )

    # -----------------------------------------
    # Very strong similarity
    # -----------------------------------------

    if distance <= 0.30:

        confidence = 90.0


    # -----------------------------------------
    # Strong similarity
    # -----------------------------------------

    elif distance <= 0.50:

        confidence = 80.0


    # -----------------------------------------
    # Moderate similarity
    # -----------------------------------------

    elif distance <= 0.70:

        confidence = 65.0


    # -----------------------------------------
    # Weak but potentially useful
    # -----------------------------------------

    elif distance <= 0.90:

        confidence = 45.0


    # -----------------------------------------
    # Poor similarity
    # -----------------------------------------

    else:

        confidence = 20.0


    return confidence


# =========================================
# CONFIDENCE LEVEL
# =========================================

def get_confidence_level(
        confidence: float,
) -> str:

    if confidence >= 75:

        return "HIGH"

    if confidence >= 50:

        return "MEDIUM"

    if confidence >= 30:

        return "LOW"

    return "INSUFFICIENT"


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
        # GET METADATA
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
        # GET DISTANCE
        # -----------------------------------------

        distance = 0.0

        if index < len(
                distances
        ):

            distance = float(
                distances[index]
            )


        # -----------------------------------------
        # CALCULATE RETRIEVAL CONFIDENCE
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
        # BUILD EVIDENCE OBJECT
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

                "chunk_id": metadata.get(
                    "chunk_id",
                    "",
                ),

                "chunk_index": metadata.get(
                    "chunk_index",
                    0,
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
    # OVERALL RETRIEVAL CONFIDENCE
    # =========================================

    if confidence_scores:

        # -----------------------------------------
        # Average confidence across retrieved chunks
        # -----------------------------------------

        overall_confidence = (
                sum(
                    confidence_scores
                )
                /
                len(
                    confidence_scores
                )
        )

    else:

        overall_confidence = 0.0


    overall_confidence_level = (
        get_confidence_level(
            overall_confidence
        )
    )


    # =========================================
    # UNIQUE SOURCE LIST
    # =========================================

    sources = list(
        dict.fromkeys(

            metadata.get(
                "source",
                "Unknown",
            )

            for metadata in metadatas

        )
    )


    # =========================================
    # RETURN RETRIEVAL RESULT
    # =========================================

    return {

        # -----------------------------------------
        # Detailed evidence objects
        # -----------------------------------------

        "evidence": (
            evidence_details
        ),


        # -----------------------------------------
        # Raw retrieved document chunks
        # -----------------------------------------

        "documents": (
            documents
        ),


        # -----------------------------------------
        # Unique evidence sources
        # -----------------------------------------

        "sources": (
            sources
        ),


        # -----------------------------------------
        # Overall retrieval confidence
        # -----------------------------------------

        "confidence": (
            round(
                overall_confidence,
                2,
            )
        ),


        # -----------------------------------------
        # HIGH / MEDIUM / LOW / INSUFFICIENT
        # -----------------------------------------

        "confidence_level": (
            overall_confidence_level
        ),

    }