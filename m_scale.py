"""m_i: feature i's characteristic active magnitude (sets the units for -c*m_i).

Default: median of *active* activations over the feature's elicit prompts
(a gentler, self-contained unit than the borrowed a_max). Optionally override
with a hardcoded Neuronpedia max activation via config.
"""
import torch
from hook import feature_activation


@torch.no_grad()
def corpus_median_active(model, sae, hook_name, feature_idx, prompts):
    """Median of strictly-positive activations of feature i across prompts."""
    vals = []
    for p in prompts:
        toks = model.to_tokens(p)
        a_i = feature_activation(model, sae, hook_name, feature_idx, toks)  # [1,pos]
        active = a_i[a_i > 0]
        if active.numel():
            vals.append(active)
    if not vals:
        return 0.0
    allv = torch.cat(vals)
    return float(allv.median())


@torch.no_grad()
def corpus_max_active(model, sae, hook_name, feature_idx, prompts):
    """Max activation of feature i across prompts — the characteristic a_max (Eq.6)."""
    best = 0.0
    for p in prompts:
        toks = model.to_tokens(p)
        a_i = feature_activation(model, sae, hook_name, feature_idx, toks)
        best = max(best, float(a_i.max()))
    return best


def resolve_amp_ref(model, sae, hook_name, feature_idx, prompts, cfg):
    """a_max for amplify: Neuronpedia max if given, else corpus max-active."""
    np_max = cfg["m_scale"].get("fallback_neuronpedia_max")
    if np_max is not None:
        return float(np_max)
    return corpus_max_active(model, sae, hook_name, feature_idx, prompts)


def resolve_m(model, sae, hook_name, feature_idx, prompts, cfg):
    np_max = cfg["m_scale"].get("fallback_neuronpedia_max")
    if np_max is not None:
        return float(np_max), "neuronpedia_max"
    m = corpus_median_active(model, sae, hook_name, feature_idx, prompts)
    return m, "corpus_median_active"
