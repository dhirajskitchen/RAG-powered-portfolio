from langchain_community.document_loaders import DirectoryLoader, TextLoader
from pathlib import Path

def load_documents():
    loader = DirectoryLoader(
    path="../context_documents",
    glob="**/*.md",
    loader_cls=TextLoader
    )

    documents = loader.load()

    for doc in documents:
        filepath = Path(doc.metadata["source"])

        doc.metadata.update({
            "filename": filepath.name,
            "category": filepath.stem,
            "filetype": "markdown"
        })

    return documents