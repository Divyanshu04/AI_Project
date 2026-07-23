from app.rag.document_loader import (
    load_documents,
)

from app.rag.embeddings import (
    generate_embeddings,
)

from app.rag.vector_store import (
    add_documents,
)


# =========================================
# MAIN INDEXING PIPELINE
# =========================================

def main():

    print(
        "Loading medical knowledge..."
    )


    # =========================================
    # LOAD AND CHUNK DOCUMENTS
    # =========================================

    documents = load_documents()


    print(
        f"Loaded {len(documents)} "
        "clinical chunks"
    )


    if not documents:

        print(
            "No medical knowledge documents "
            "found."
        )

        return


    # =========================================
    # DISPLAY CHUNK INFORMATION
    # =========================================

    print(
        "\nClinical Knowledge Chunks:"
    )


    for document in documents:

        print(
            f"  • "
            f"{document['domain']} | "
            f"{document['source']} | "
            f"{document['chunk_id']}"
        )


    # =========================================
    # GENERATE EMBEDDINGS
    # =========================================

    print(
        "\nGenerating embeddings..."
    )


    embeddings = generate_embeddings(

        [
            document["text"]
            for document in documents
        ]

    )


    # =========================================
    # ADD TO CHROMADB
    # =========================================

    print(
        "Adding chunks to ChromaDB..."
    )


    add_documents(
        documents=documents,
        embeddings=embeddings,
    )


    # =========================================
    # COMPLETED
    # =========================================

    print(
        "\nMedical knowledge indexing "
        "completed successfully."
    )

    print(
        f"Indexed {len(documents)} "
        "clinical chunks."
    )


# =========================================
# ENTRY POINT
# =========================================

if __name__ == "__main__":

    main()