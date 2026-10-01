from langchain_huggingface import HuggingFaceEmbeddings


def create_embeddings(model_name: str) -> HuggingFaceEmbeddings:
    """Local sentence-transformers model with the "query: " / "passage: " prefixes E5 expects."""
    return HuggingFaceEmbeddings(
        model_name=model_name,
        model_kwargs={"prompts": {"query": "query: ", "passage": "passage: "}},
        encode_kwargs={"prompt_name": "passage", "normalize_embeddings": True},
        query_encode_kwargs={"prompt_name": "query", "normalize_embeddings": True},
    )
