from pathlib import Path


# =========================================
# KNOWLEDGE BASE DIRECTORY
# =========================================

KNOWLEDGE_BASE_DIR = Path(
    "app/data/medical_knowledge"
)


# =========================================
# DOMAIN MAPPING
# =========================================

DOMAIN_MAPPING = {

    "cardiovascular_demo.txt":
        "CARDIOVASCULAR",

    "hypertension.txt":
        "HYPERTENSION",

    "diabetes.txt":
        "DIABETES",

    "medication_safety.txt":
        "MEDICATION_SAFETY",

    "respiratory_conditions.txt":
        "RESPIRATORY",

}


# =========================================
# CHUNK CONFIGURATION
# =========================================

CHUNK_SIZE = 800

CHUNK_OVERLAP = 100


# =========================================
# CHUNK TEXT
# =========================================

def chunk_text(
        text: str,
        chunk_size: int = CHUNK_SIZE,
        chunk_overlap: int = CHUNK_OVERLAP,
) -> list[str]:

    text = text.strip()

    if not text:
        return []

    chunks = []

    start = 0

    text_length = len(text)

    while start < text_length:

        end = min(
            start + chunk_size,
            text_length,
            )

        chunk = text[
            start:end
        ].strip()

        if chunk:

            chunks.append(
                chunk
            )

        if end >= text_length:

            break

        start = (
                end - chunk_overlap
        )

    return chunks


# =========================================
# LOAD AND CHUNK DOCUMENTS
# =========================================

def load_documents():

    documents = []

    for file_path in sorted(
            KNOWLEDGE_BASE_DIR.glob(
                "*.txt"
            )
    ):

        text = file_path.read_text(
            encoding="utf-8"
        ).strip()

        if not text:

            continue

        domain = DOMAIN_MAPPING.get(
            file_path.name,
            "GENERAL_CLINICAL",
        )

        chunks = chunk_text(
            text
        )

        for chunk_index, chunk in enumerate(
                chunks
        ):

            documents.append(
                {
                    "text": chunk,

                    "source": file_path.name,

                    "domain": domain,

                    "document_id": (
                        file_path.stem
                    ),

                    "chunk_id": (
                        f"{file_path.stem}_"
                        f"chunk_{chunk_index}"
                    ),

                    "chunk_index": (
                        chunk_index
                    ),

                    "evidence_level": (
                        "DEMO"
                    ),

                    "publication_year": (
                        "N/A"
                    ),
                }
            )

    return documents