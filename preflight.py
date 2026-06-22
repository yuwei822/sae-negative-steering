"""Preflight: verify the model + SAE load, the feature index is what we expect, and
that +, 0, - hooks all run on MPS. Satisfies the plan's Definition-of-Done checks
(verify Gemma-Scope id + F8827 = hair; confirm hooks run on Metal).

  python preflight.py            # uses config.yaml (gemma + hair F8827)
  python preflight.py config_gpt2.yaml
"""
import sys
import torch

from load import load_config, load_model_and_sae, sae_hook_name, load_judge
from hook import make_steer_hook, make_random_hook
from scores import logit_lens_tokens
from m_scale import resolve_m, resolve_amp_ref


def main():
    config_path = sys.argv[1] if len(sys.argv) > 1 else "config.yaml"
    cfg = load_config(config_path)
    print(f"== preflight ({config_path}) ==")

    model, sae, device = load_model_and_sae(cfg)
    print(f"device={device}  (want mps)  | model={cfg['model']['name']}")
    print(f"model d_model={model.cfg.d_model} n_layers={model.cfg.n_layers}")
    hook_name = sae_hook_name(sae, cfg)
    print(f"sae d_sae={sae.cfg.d_sae} hook={hook_name}")
    assert device == "mps", "WARNING: not on MPS — check device/MPS availability"

    feat = cfg["features"][0]
    fi = feat["index"]
    if fi == "auto":
        print("feature index is 'auto' (gpt2 fallback) — skipping the hair-index check")
    else:
        ids, strs = logit_lens_tokens(model, sae, fi, k=15)
        print(f"\nF{fi} logit-lens top tokens: {strs}")
        if cfg["model"]["name"].startswith("google/gemma") and fi == 8827:
            ok = any("hair" in s.lower() for s in strs)
            print(f"hair-feature check (F8827): {'PASS ✅' if ok else 'FAIL ❌ — index may not match this SAE release'}")

    # resolve scales and run all three hooks once on the first elicit prompt
    if fi == "auto":
        from steer import auto_select_feature
        fi = auto_select_feature(model, sae, hook_name, feat["elicit_prompts"])
    ids, _ = logit_lens_tokens(model, sae, fi, k=cfg["scoring"]["top_k_logit_lens"])
    m_i, _ = resolve_m(model, sae, hook_name, fi, feat["elicit_prompts"], cfg)
    a_ref = resolve_amp_ref(model, sae, hook_name, fi, feat["elicit_prompts"], cfg)
    print(f"\nm_i={m_i:.4f}  a_max_char={a_ref:.4f}")

    toks = model.to_tokens(feat["elicit_prompts"][0])
    hooks = {
        "amplify(+)": make_steer_hook(sae, fi, "amplify", s=cfg["intervention"]["amplify_s"], a_max_char=a_ref),
        "ablate(0)": make_steer_hook(sae, fi, "coeff", c=0.0, m_i=m_i),
        "negative(-)": make_steer_hook(sae, fi, "coeff", c=2.0, m_i=m_i),
        "random(ctl)": make_random_hook(sae, fi, c=2.0, m_i=m_i, seed=0),
    }
    print("\nrunning +, 0, -, random hooks (checking Metal, no CPU fallback):")
    for name, hf in hooks.items():
        logits = model.run_with_hooks(toks, fwd_hooks=[(hook_name, hf)])
        assert str(logits.device).startswith("mps"), f"{name}: output not on MPS ({logits.device})"
        print(f"  {name:>12}: ok, logits on {logits.device}")

    # perplexity judge (loads the independent model if configured)
    pm = cfg.get("scoring", {}).get("perplexity_model", "self")
    print(f"\nperplexity judge: {pm}")
    if pm not in (None, "self"):
        judge = load_judge(cfg, device)
        jt = judge.to_tokens("The weather today is nice and")
        jl = judge(jt)
        assert str(jl.device).startswith("mps"), f"judge output not on MPS ({jl.device})"
        print(f"  judge {pm}: ok, logits on {jl.device}")
    else:
        print("  using self-perplexity proxy (no separate judge to load)")

    print("\nPREFLIGHT OK ✅  — ready to run steer.py")


if __name__ == "__main__":
    main()
