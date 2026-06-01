from langchain_text_splitters import MarkdownHeaderTextSplitter,RecursiveCharacterTextSplitter

def markDownSplitter(documents):
    headers_to_split_on = [
        ("#", "section"),
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

    return md_header_splits

def characterTextSplitter(md_header_splits):
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1200,
        chunk_overlap=200,
        add_start_index=True
    )

    final_chunks = text_splitter.split_documents(md_header_splits)

    return final_chunks

def addMetadata(chunks):
    for idx, chunk in enumerate(chunks):

        chunk.metadata["citation_id"] = f"C{idx:04d}"
        filename = chunk.metadata.get("filename", "").lower()

        if "personality" in filename:
            chunk.metadata["type"] = "personality"

        elif "skills" in filename or "projects" in filename:
            chunk.metadata["type"] = "technical"

        elif "faq" in filename:
            chunk.metadata["type"] = "recruiter"

        elif "opinions" in filename:
            chunk.metadata["type"] = "beliefs"

        else:
            chunk.metadata["type"] = "general"
        return chunks

def chunk_documents(documents):
    md_header_splits=markDownSplitter(documents)
    chunks=characterTextSplitter(md_header_splits)
    final_chunks=addMetadata(chunks)
    return final_chunks
