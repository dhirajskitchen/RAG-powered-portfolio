Pipeline 1

embedding - GoogleGenerativeAIEmbeddings
Reranker - Cohere Reranker
Hybrid Retriever - EnsembleRetriever

Pipeline 2

embedding - HuggingFaceEmbeddings
Reranker - CrossEncoder Reranking using sentence_transformers
Hybrid Retriever - Manual Hybrid Retriever
