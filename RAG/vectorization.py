import os
from langchain_google_genai import ChatGoogleGenerativeAI,GoogleGenerativeAIEmbeddings
from langchain_chroma import Chroma
import bs4
import requests
from langchain_core.documents import Document
from dotenv import load_dotenv
from langchain_text_splitters import MarkdownHeaderTextSplitter,RecursiveCharacterTextSplitter
from pathlib import Path
from langchain_community.document_loaders import DirectoryLoader, TextLoader

load_dotenv()

model = ChatGoogleGenerativeAI(model="gemini-2.5-flash-lite")

embeddings = GoogleGenerativeAIEmbeddings(model="models/gemini-embedding-001")

vector_store = Chroma(
    collection_name="example_collection",
    embedding_function=embeddings,
    persist_directory="./chroma_langchain_db",  # Where to save data locally, remove if not necessary
)



loader = DirectoryLoader(
    path="context_documents",
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

print(documents[0].metadata)

print(f"Total characters in doc 1: {len(documents[0].page_content)}")

headers_to_split_on = [
    ("#", "h1"),
    ("##", "h2"),
    ("###", "h3"),
]

markdown_splitter = MarkdownHeaderTextSplitter(
    headers_to_split_on=headers_to_split_on
)

md_header_splits = []

for doc in documents:
    splits = markdown_splitter.split_text(doc.page_content)

    for split in splits:
        split.metadata.update(doc.metadata)
        md_header_splits.append(split)


text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=500,
    chunk_overlap=100,
    add_start_index=True
)

final_chunks = text_splitter.split_documents(md_header_splits)

print(f"Total chunks: {len(final_chunks)}")