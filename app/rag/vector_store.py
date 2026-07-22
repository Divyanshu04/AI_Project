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
                "source": doc["source"]
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
        where: dict | None = None,
):

    # -----------------------------------------
    # Build query arguments
    # -----------------------------------------

    query_args = {
        "query_embeddings": [
            query_embedding
        ],
        "n_results": n_results,
    }


    # -----------------------------------------
    # Add metadata filter
    # -----------------------------------------

    if where:

        query_args[
            "where"
        ] = where


    # -----------------------------------------
    # Query ChromaDB
    # -----------------------------------------

    results = collection.query(
        **query_args
    )

    return results