from langchain_core.documents import Document
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings

def get_documents():
    embeddings = HuggingFaceEmbeddings(
        model_name="BAAI/bge-base-en-v1.5"
    )
    vectorstore = Chroma(
        persist_directory="./chroma_db",
        embedding_function=embeddings
    )

    collection = vectorstore._collection

    data = collection.get(
        include=["documents", "metadatas"]
    )

    return [
        Document(
            page_content=doc,
            metadata=meta
        )
        for doc, meta in zip(
            data["documents"],
            data["metadatas"]
        )
    ]