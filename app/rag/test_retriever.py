from app.rag.retriever import (
    retrieve_clinical_evidence,
)


query = """
A patient presents with chest pain
and shortness of breath.
"""


result = retrieve_clinical_evidence(
    query
)


print("\nRetrieved Evidence:\n")

for document in result["documents"]:

    print(
        "--------------------------------"
    )

    print(document)


print("\nSources:\n")

for source in result["sources"]:

    print(source)