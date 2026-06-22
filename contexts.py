"""Floor-effect guard: keep only prompts where the concept is PRESENT (high
baseline p(l*)) AND the feature is ACTIVE (a_i > 0). Ablation does nothing if
a_i = 0, and there is nothing to suppress if the concept is absent.
"""
import torch
from hook import feature_activation
from scores import p_concept


@torch.no_grad()
def filter_prompts(model, sae, hook_name, feature_idx, concept_ids, prompts,
                   p_floor=0.0, require_active=True):
    """Return list of dicts for prompts passing the floor checks.

    Each kept dict: {prompt, baseline_p, a_i_max, active}.
    p_floor: minimum baseline p(l*) to count the concept as 'present'
             (0.0 keeps all; raise to be strict).
    """
    kept = []
    for p in prompts:
        toks = model.to_tokens(p)
        a_i = feature_activation(model, sae, hook_name, feature_idx, toks)
        a_max = float(a_i.max())
        active = a_max > 0
        base_p = p_concept(model, toks, concept_ids)
        ok = (base_p >= p_floor) and (active or not require_active)
        kept.append({
            "prompt": p, "baseline_p": base_p, "a_i_max": a_max,
            "active": bool(active), "passed": bool(ok),
        })
    return kept
