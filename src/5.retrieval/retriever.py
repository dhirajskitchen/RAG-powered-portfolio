from sentence_transformers import CrossEncoder
from langchain_community.retrievers import BM25Retriever

def rerank(reranker,query, docs, top_k=5):

    pairs = [
        (query, doc.page_content)
        for doc in docs
    ]

    scores = reranker.predict(pairs)

    ranked_docs = sorted(
        zip(docs, scores),
        key=lambda x: x[1],
        reverse=True
    )

    return [
        doc
        for doc, score in ranked_docs[:top_k]
    ]


def hybrid_search(query,bm25_retriever,vector_retriever,reranker, k=5):

    bm25_docs = bm25_retriever.invoke(query)
    vector_docs = vector_retriever.invoke(query)

    scores = {}

    for rank, doc in enumerate(bm25_docs):
        scores[doc.page_content] = scores.get(doc.page_content, 0) + 1/(rank+60)

    for rank, doc in enumerate(vector_docs):
        scores[doc.page_content] = scores.get(doc.page_content, 0) + 1/(rank+60)

    doc_map = {
        doc.page_content: doc
        for doc in bm25_docs + vector_docs
    }

    ranked = sorted(
        scores.items(),
        key=lambda x: x[1],
        reverse=True
    )

    return [
        doc_map[content]
        for content, _ in ranked[:k]
    ]

def retrieve(query,final_chunks,vector_store):
    reranker = CrossEncoder(
    "cross-encoder/ms-marco-MiniLM-L-12-v2"
    )
    # BM25 retriever
    bm25_retriever = BM25Retriever.from_documents(final_chunks)
    bm25_retriever.k = 5

    # Vector retriever
    vector_retriever = vector_store.as_retriever(
        search_kwargs={"k": 5}
    )

    candidates = hybrid_search(
        query,
        bm25_retriever,
        vector_retriever,
        reranker,
        k=10
    )

    final_docs = rerank(
        reranker,
        query,
        candidates,
        top_k=5
    )

    return final_docs