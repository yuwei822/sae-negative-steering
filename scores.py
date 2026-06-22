"""Concept-effect scoring: logit-lens tokens, rank-weighted p(l*), Gen Success.

Shared with Plan 1. An output feature's "concept tokens" l are the top-k tokens
its decoder direction promotes through the unembedding (logit lens).
"""
import torch


@torch.no_grad()
def logit_lens_tokens(model, sae, feature_idx: int, k: int = 20):
    """Top-k vocab tokens promoted by feature i's decoder direction.

    Returns (token_ids [k], token_strs [k]).
    """
    w_dec_i = sae.W_dec[feature_idx].to(model.W_U.dtype)   # [d_model]
    logits = w_dec_i @ model.W_U                            # [d_vocab]
    top = torch.topk(logits, k)
    ids = top.indices
    strs = [model.to_single_str_token(int(t)) for t in ids]
    return ids, strs


@torch.no_grad()
def p_concept(model, tokens, concept_ids, fwd_hooks=None, rank_weighted=False):
    """Single forward; probability mass on concept tokens at the final position.

    rank_weighted=False -> plain sum_j p(l_j)  (mirror of the output score).
    rank_weighted=True  -> sum_j w_j p(l_j), w_j = 1/(j+1) normalized, j=lens rank.
    """
    if fwd_hooks:
        logits = model.run_with_hooks(tokens, fwd_hooks=fwd_hooks)
    else:
        logits = model(tokens)
    last = logits[0, -1]                                    # [d_vocab]
    probs = torch.softmax(last.float(), dim=-1)
    ids = concept_ids.to(probs.device)
    if rank_weighted:
        w = 1.0 / torch.arange(1, len(ids) + 1, device=probs.device).float()
        w = w / w.sum()
        return float((w * probs[ids]).sum())
    return float(probs[ids].sum())


def gen_success(generated_ids, concept_ids):
    """Eq.7-style: fraction of generated tokens that are concept tokens, and a
    binary 'any concept token appeared'.
    """
    cset = set(int(t) for t in concept_ids)
    gen = [int(t) for t in generated_ids]
    if not gen:
        return 0.0, 0
    hits = sum(1 for t in gen if t in cset)
    return hits / len(gen), int(hits > 0)
