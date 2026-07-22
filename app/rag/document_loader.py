from pathlib import Path


def load_documents(
        directory: str,
) -> list[dict]:

    documents = []

    directory_path = Path(directory)

    for file_path in directory_path.glob("*.txt"):

        content = file_path.read_text(
            encoding="utf-8"
        )

        documents.append(
            {
                "text": content,
                "source": file_path.name,
            }
        )

    return documents