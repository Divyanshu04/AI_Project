from sentence_transformers import SentenceTransformer


MODEL_NAME = "all-MiniLM-L6-v2"


embedding_model = SentenceTransformer(
    MODEL_NAME
)


def generate_embeddings(
        texts: list[str],
) -> list[list[float]]:

    embeddings = embedding_model.encode(
        texts
    )

    return embeddings.tolist()