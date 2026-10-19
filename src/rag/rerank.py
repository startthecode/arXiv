"""Reranking with Qwen3-Reranker: reads question and passage *together* and scores relevance.

Embeddings squeeze each chunk into a vector before the question is known; a reranker sees both at once,
so it is slower but much more precise. We only run it on the top hybrid candidates.

The model ships its own chat template (<Instruct>/<Query>/<Document>) and is a small language model:
the score is logit("yes") - logit("no") for its next token, i.e. log-odds of "relevant".
"""

from functools import lru_cache

from rag.settings import Settings


@lru_cache
def load_reranker(name: str):
    import torch
    from sentence_transformers import CrossEncoder

    # question + one chunk (<= 512 tokens) fits easily in 1024
    return CrossEncoder(name, device="cuda", max_length=1024, model_kwargs={"dtype": torch.bfloat16})


def rerank(s: Settings, question: str, docs: list[str], top_k: int) -> list[tuple[int, float]]:
    """Return [(index into docs, P(relevant) 0..1)] for the top_k docs, best first."""
    import torch

    model = load_reranker(s.rerank_model)
    ranked = model.rank(
        question,
        docs,
        top_k=top_k,
        batch_size=s.rerank_batch_size,
        prompt=s.rerank_instruction,  # becomes the <Instruct> line of the model's chat template
        activation_fn=torch.nn.Sigmoid(),  # log-odds -> probability, so context_min_score is a real 0..1 threshold
    )
    return [(r["corpus_id"], float(r["score"])) for r in ranked]
