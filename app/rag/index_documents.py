from app.rag.document_loader import (
    load_documents,
)

from app.rag.embeddings import (
    generate_embeddings,
)

from app.rag.vector_store import (
    add_documents,
)


KNOWLEDGE_PATH = (
    "app/data/medical_knowledge"
)


def main():

    print(
        "Loading medical knowledge..."
    )

    documents = load_documents(
        KNOWLEDGE_PATH
    )

    print(
        f"Loaded {len(documents)} documents"
    )

    texts = [
        doc["text"]
        for doc in documents
    ]

    print(
        "Generating embeddings..."
    )

    embeddings = generate_embeddings(
        texts
    )

    print(
        "Adding documents to ChromaDB..."
    )

    add_documents(
        documents,
        embeddings,
    )

    print(
        "Medical knowledge indexed successfully."
    )


if __name__ == "__main__":
    main()