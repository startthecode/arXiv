"""Dense embeddings with the local embedding model on the GPU (sentence-transformers)."""

from functools import lru_cache

import numpy as np

from rag.settings import Settings


@lru_cache
def load_model(name: str):
    """Load once per process. Heavy imports live here so other CLI commands stay fast."""
    import torch
    from sentence_transformers import SentenceTransformer

    if not torch.cuda.is_available():
        raise RuntimeError(f"CUDA not available (torch {torch.__version__}); expected the +cu130 build")
    model = SentenceTransformer(name, device="cuda", model_kwargs={"dtype": torch.bfloat16})
    model.max_seq_length = 1024  # chunks are <= 512 tokens; this only guards against outliers
    return model


def embed_documents(texts: list[str], s: Settings) -> np.ndarray:
    """Chunk texts -> (n, dim) unit-length float32 vectors. Documents get no instruction prompt."""
    model = load_model(s.embed_model)
    vecs = model.encode_document(texts, batch_size=s.embed_batch_size, normalize_embeddings=True, convert_to_tensor=True)
    return vecs.float().cpu().numpy()  # bf16 -> float32 (numpy has no bfloat16)


def embed_query(text: str, s: Settings) -> np.ndarray:
    """Query -> (dim,) vector, prefixed with the task instruction Qwen3 was trained with."""
    model = load_model(s.embed_model)
    vec = model.encode_query([text], prompt=s.embed_query_prompt, normalize_embeddings=True, convert_to_tensor=True)
    return vec[0].float().cpu().numpy()
