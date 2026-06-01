from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings

def embed_and_store(final_chunks):

    CHROMA_DB_PATH = "../RAG/chroma_langchain_db"
    embeddings = HuggingFaceEmbeddings(
        model_name="BAAI/bge-base-en-v1.5"
    )
    vector_store = Chroma(
        collection_name="portfolio",
        embedding_function=embeddings,
        persist_directory=CHROMA_DB_PATH 
    )
    batch_size = 50

    for i in range(0, len(final_chunks), batch_size):
        batch = final_chunks[i:i+batch_size]
        vector_store.add_documents(batch)
        print(f"Added batch {i//batch_size + 1}")
    