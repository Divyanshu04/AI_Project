from app.rag.embeddings import (
    generate_embeddings,
)

from app.rag.vector_store import (
    search_documents,
)


# =========================================
# CLINICAL DOMAIN → SOURCE MAPPING
# =========================================

DOMAIN_SOURCE_MAP = {

    "CARDIOVASCULAR": [
        "cardiovascular_demo.txt",
    ],

    "HYPERTENSION": [
        "hypertension.txt",
    ],

    "DIABETES": [
        "diabetes.txt",
    ],

    "RESPIRATORY": [
        "respiratory_conditions.txt",
    ],

    "MEDICATION_SAFETY": [
        "medication_safety.txt",
    ],
}


# =========================================
# MEDICAL RAG RETRIEVER
# =========================================

def retrieve_clinical_evidence(
        query: str,
        n_results: int = 3,
        clinical_domain: str | None = None,
) -> dict:

    # -----------------------------------------
    # Generate query embedding
    # -----------------------------------------

    query_embedding = generate_embeddings(
        [query]
    )[0]


    # -----------------------------------------
    # Determine metadata filter
    # -----------------------------------------

    source_filter = None

    if clinical_domain:

        sources = DOMAIN_SOURCE_MAP.get(
            clinical_domain.upper(),
            [],
        )

        if len(sources) == 1:

            source_filter = {
                "source": sources[0]
            }


    # -----------------------------------------
    # Search ChromaDB
    # -----------------------------------------

    results = search_documents(
        query_embedding=query_embedding,
        n_results=n_results,
        where=source_filter,
    )


    # -----------------------------------------
    # Extract documents
    # -----------------------------------------

    documents = results.get(
        "documents",
        [[]],
    )[0]


    # -----------------------------------------
    # Extract metadata
    # -----------------------------------------

    metadatas = results.get(
        "metadatas",
        [[]],
    )[0]


    # -----------------------------------------
    # Return RAG results
    # -----------------------------------------

    return {
        "documents": documents,
        "sources": metadatas,
    }