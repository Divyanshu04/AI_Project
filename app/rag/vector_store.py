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
            document.get(
                "chunk_id",
                f"doc_{i}",
            )

            for i, document in enumerate(
                documents
            )
        ],

        documents=[
            document["text"]

            for document in documents
        ],

        embeddings=embeddings,

        metadatas=[

            {
                "source": document.get(
                    "source",
                    "Unknown",
                ),

                "domain": document.get(
                    "domain",
                    "GENERAL_CLINICAL",
                ),

                "document_id": document.get(
                    "document_id",
                    "",
                ),

                "chunk_id": document.get(
                    "chunk_id",
                    "",
                ),

                "chunk_index": document.get(
                    "chunk_index",
                    0,
                ),

                "evidence_level": document.get(
                    "evidence_level",
                    "DEMO",
                ),

                "publication_year": document.get(
                    "publication_year",
                    "N/A",
                ),
            }

            for document in documents

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

    if clinical_domain:

        results = collection.query(

            query_embeddings=[
                query_embedding
            ],

            n_results=n_results,

            where={
                "domain": clinical_domain
            },

        )

    else:

        results = collection.query(

            query_embeddings=[
                query_embedding
            ],

            n_results=n_results,

        )


    return results