"""Load gemma-2-2b (HookedSAETransformer) + the Gemma-Scope SAE for plan3."""
import os
import yaml
import torch
from sae_lens import HookedSAETransformer, SAE


def pick_device(requested: str) -> str:
    if requested == "mps" and torch.backends.mps.is_available():
        return "mps"
    if requested == "cuda" and torch.cuda.is_available():
        return "cuda"
    return "cpu"


def load_config(path: str = "config.yaml") -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


def load_model_and_sae(cfg: dict):
    device = pick_device(cfg["device"])
    dtype = getattr(torch, cfg["model"]["dtype"])

    # Load the SAE first so we can load the model with the exact preprocessing the
    # SAE was trained against (sae_lens best practice — keeps activations faithful).
    sae = SAE.from_pretrained(
        release=cfg["sae"]["release"],
        sae_id=cfg["sae"]["sae_id"],
        device=device,
        dtype=cfg["model"]["dtype"],
    )
    if isinstance(sae, tuple):     # some versions return (sae, cfg_dict, sparsity)
        sae = sae[0]
    sae = sae.to(device=device, dtype=dtype)

    # Load WITHOUT transformer_lens processing (no LayerNorm folding / writing-weight
    # centering). Gemma-Scope (and the gpt2-res-jb) SAEs were trained on the model's
    # RAW residual stream; from_pretrained's processing shifts those activations, so
    # sae.encode(resid) would see off-distribution inputs (wrong m_i / a_max, less
    # faithful steering). sae_lens explicitly recommends no_processing here, passing
    # the SAE's own model_from_pretrained_kwargs.
    meta = getattr(sae.cfg, "metadata", None)
    model_kwargs = dict(getattr(meta, "model_from_pretrained_kwargs", None) or {})
    model = HookedSAETransformer.from_pretrained_no_processing(
        cfg["model"]["name"], device=device, dtype=dtype, **model_kwargs
    )
    return model, sae, device


def load_judge(cfg: dict, device: str):
    """Load an independent perplexity judge model, or None for the self-proxy.

    cfg['scoring']['perplexity_model']:
      'self'         -> return None (caller scores perplexity with the steered model)
      a model name   -> load that HookedTransformer (e.g. 'google/gemma-2-9b') as an
                        independent coherence judge. Heavier but unbiased by the
                        intervention. Re-tokenizes text under its own tokenizer.

    The judge is loaded in bfloat16 regardless of the steered model's dtype: a 9B in
    fp32 is ~36GB and would not coexist with the fp32 2B on 64GB, whereas bf16 is ~18GB.
    The judge only needs to rank token NLLs, so bf16 is ample. Override via
    scoring.perplexity_judge_dtype if you really want something else.
    """
    name = cfg.get("scoring", {}).get("perplexity_model", "self")
    if name in (None, "self"):
        return None
    from transformer_lens import HookedTransformer
    jdtype = getattr(torch, cfg.get("scoring", {}).get("perplexity_judge_dtype", "bfloat16"))
    # The judge can be pinned to a different device than the steered model (e.g. 'cpu')
    # so a 9B judge + 2B steered model don't compete for MPS memory on a 64GB box.
    jdevice = cfg.get("scoring", {}).get("perplexity_judge_device", device)
    print(f"[judge] loading independent perplexity judge: {name} "
          f"(dtype={jdtype}, device={jdevice}) ...")
    judge = HookedTransformer.from_pretrained(name, device=jdevice, dtype=jdtype)
    judge.eval()
    return judge


def sae_hook_name(sae, cfg: dict) -> str:
    """Resolve the resid hook name across sae_lens versions."""
    for attr in ("hook_name",):
        v = getattr(sae.cfg, attr, None)
        if v:
            return v
    meta = getattr(sae.cfg, "metadata", None)
    if meta is not None and getattr(meta, "hook_name", None):
        return meta.hook_name
    return cfg["sae"]["hook_name"]


if __name__ == "__main__":
    cfg = load_config()
    model, sae, device = load_model_and_sae(cfg)
    print(f"device={device}")
    print(f"model d_model={model.cfg.d_model} n_layers={model.cfg.n_layers}")
    print(f"sae d_sae={sae.cfg.d_sae} hook={sae.cfg.hook_name}")
