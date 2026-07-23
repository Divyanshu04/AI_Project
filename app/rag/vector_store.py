import chromadb


# =========================================
# CHROMADB CLIENT
# =========================================

client = chromadb.PersistentClient(
    path="./chroma_db"
)


# =========================================
# CLINICAL KNOWLEDGE COLLECTION
# =========================================

collection = client.get_or_create_collection(
    name="clinical_knowledge"
)


# =========================================
# ADD DOCUMENTS
# =========================================

def add_documents(
        documents: list[dict],
        embeddings: list[list[float]],
):

    collection.add(

        ids=[
            f"doc_{i}"
            for i in range(
                len(documents)
            )
        ],

        documents=[
            doc["text"]
            for doc in documents
        ],

        embeddings=embeddings,

        metadatas=[
            {
                "source": doc.get(
                    "source",
                    "Unknown",
                ),

                "domain": doc.get(
                    "domain",
                    "GENERAL_CLINICAL",
                ),

                "document_id": doc.get(
                    "document_id",
                    "",
                ),

                "evidence_level": doc.get(
                    "evidence_level",
                    "DEMO",
                ),

                "publication_year": doc.get(
                    "publication_year",
                    "N/A",
                ),
            }

            for doc in documents
        ],
    )


# =========================================
# SEARCH DOCUMENTS
# =========================================

def search_documents(
        query_embedding: list[float],
        n_results: int = 3,
        clinical_domain: str | None = None,
):

    # =========================================
    # DOMAIN-AWARE SEARCH
    # =========================================

    if (
            clinical_domain
            and clinical_domain
            != "GENERAL_CLINICAL"
    ):

        results = collection.query(

            query_embeddings=[
                query_embedding
            ],

            n_results=n_results,

            # ---------------------------------
            # Metadata filtering
            # ---------------------------------

            where={
                "domain": clinical_domain
            },
        )

    else:

        # =====================================
        # GENERAL SEARCH
        # =====================================

        results = collection.query(

            query_embeddings=[
                query_embedding
            ],

            n_results=n_results,
        )


    return results