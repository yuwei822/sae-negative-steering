"""Run +, 0, -, and a random-direction control on output features; record concept
effect + coherence with N sampled generations per condition (mean +/- std).

For each feature x elicit prompt we run, each over N samples:
  - baseline   (no intervention)
  - amplify    (+: a_i + s*·a_max ; s* found by search if enabled)
  - c sweep    (0 = ablation, c>0 = negative -c·m_i)
  - rand sweep (magnitude-matched random direction, for each c>0 — guard #2)
and on neutral prompts the amplify "appears-from-nothing" demo.

p(l*) is a single-forward distributional metric (deterministic given the hook), so it
is computed once per condition. Gen-Success and perplexity depend on the sampled
continuation, so they are collected over N samples and written one row per sample.

Coherence (perplexity) is scored by an independent judge model if configured
(scoring.perplexity_model = e.g. google/gemma-2-9b), else by the steered model itself.

Outputs <results_dir>/results.csv and <results_dir>/generations.txt.
"""
import os
import csv
import torch

from load import load_config, load_model_and_sae, sae_hook_name, load_judge
from hook import make_steer_hook, make_random_hook, feature_activation
from scores import logit_lens_tokens, p_concept, gen_success
from m_scale import resolve_m, resolve_amp_ref
from contexts import filter_prompts


@torch.no_grad()
def auto_select_feature(model, sae, hook_name, prompts):
    """Pick the feature with the highest mean last-token activation across prompts."""
    d_sae = sae.W_dec.shape[0]
    acc = torch.zeros(d_sae, device=sae.W_dec.device)
    for p in prompts:
        toks = model.to_tokens(p)
        _, cache = model.run_with_cache(toks, names_filter=hook_name)
        a_last = sae.encode(cache[hook_name])[0, -1]   # [d_sae]
        acc += a_last
    return int(acc.argmax())


@torch.no_grad()
def self_perplexity(model, full_tokens, n_new):
    """Perplexity over the last n_new tokens under the (steered) model itself."""
    logits = model(full_tokens)
    logp = torch.log_softmax(logits[0].float(), dim=-1)
    start = full_tokens.shape[1] - n_new
    nll = [-logp[t - 1, int(full_tokens[0, t])].item()
           for t in range(start, full_tokens.shape[1])]
    if not nll:
        return float("nan")
    return float(torch.tensor(nll).mean().exp())


@torch.no_grad()
def judge_perplexity(judge, prompt_str, cont_str):
    """Perplexity of the continuation under an independent judge model.

    Re-tokenizes prompt and continuation under the judge's own tokenizer (prepend BOS
    only on the prompt) to avoid string-boundary merge artifacts, then scores NLL over
    the continuation positions. Unbiased by the intervention on the steered model.
    """
    p_tok = judge.to_tokens(prompt_str)                       # [1, P] (with BOS)
    c_tok = judge.to_tokens(cont_str, prepend_bos=False)      # [1, C]
    if c_tok.shape[1] == 0:
        return float("nan")
    full = torch.cat([p_tok, c_tok], dim=1)
    logits = judge(full)
    logp = torch.log_softmax(logits[0].float(), dim=-1)
    start = p_tok.shape[1]
    nll = [-logp[t - 1, int(full[0, t])].item() for t in range(start, full.shape[1])]
    return float(torch.tensor(nll).mean().exp())


@torch.no_grad()
def generate_with_hook(model, tokens, hook_fn, hook_name, gen_cfg, seed):
    torch.manual_seed(seed)
    hooks = [(hook_name, hook_fn)] if hook_fn is not None else []
    with model.hooks(fwd_hooks=hooks):
        out = model.generate(
            tokens,
            max_new_tokens=gen_cfg["max_new_tokens"],
            temperature=gen_cfg["temperature"],
            do_sample=gen_cfg["temperature"] > 0,
            verbose=False,
        )
    cont = out[:, tokens.shape[1]:]
    return out, cont


@torch.no_grad()
def search_s_star(model, sae, hook_name, fi, a_max_char, concept_ids,
                  neutral_prompts, cfg, gen_cfg, coherence):
    """Per-feature optimal amplify factor s* (paper-style search).

    Sweep s over intervention.s_grid on the neutral prompts; among the s whose mean
    perplexity stays within coherence_mult x baseline, pick the one with the largest
    concept effect p(l*). This is the strongest still-coherent amplification.

    Coherence is scored through the same `coherence` closure as the main run (the 9B
    judge when configured), so the s* gate and the reported perplexities are consistent.
    """
    s_grid = cfg["intervention"].get("s_grid", [cfg["intervention"]["amplify_s"]])
    coh_mult = cfg["intervention"].get("coherence_mult", 3.0)
    seed = gen_cfg["seed"]

    base_ppls = []
    for p in neutral_prompts:
        toks = model.to_tokens(p)
        out, cont = generate_with_hook(model, toks, None, hook_name, gen_cfg, seed)
        base_ppls.append(coherence(out, cont, p))
    base_ppl = float(torch.tensor(base_ppls).nanmean())

    rows = []
    for s in s_grid:
        effs, ppls = [], []
        for p in neutral_prompts:
            toks = model.to_tokens(p)
            hf = make_steer_hook(sae, fi, "amplify", s=s, a_max_char=a_max_char)
            effs.append(p_concept(model, toks, concept_ids,
                                  fwd_hooks=[(hook_name, hf)], rank_weighted=True))
            out, cont = generate_with_hook(model, toks, hf, hook_name, gen_cfg, seed)
            ppls.append(coherence(out, cont, p))
        rows.append((s, float(torch.tensor(effs).mean()),
                     float(torch.tensor(ppls).nanmean())))

    coherent = [r for r in rows if r[2] <= coh_mult * base_ppl]
    pick = max(coherent, key=lambda r: r[1]) if coherent else min(rows, key=lambda r: r[0])
    s_star = pick[0]
    print(f"  s* search (base_ppl={base_ppl:.1f}, <= {coh_mult}x): "
          + ", ".join(f"s={s}:p={e:.3f}/ppl={pp:.0f}" for s, e, pp in rows)
          + f"  -> s*={s_star}")
    return s_star


def run(config_path="config.yaml"):
    cfg = load_config(config_path)
    model, sae, device = load_model_and_sae(cfg)
    judge = load_judge(cfg, device)
    hook_name = sae_hook_name(sae, cfg)
    gen_cfg = cfg["generate"]
    n_samples = int(gen_cfg.get("n_samples", 1))
    base_seed = gen_cfg["seed"]
    s_default = cfg["intervention"]["amplify_s"]
    do_s_search = cfg["intervention"].get("search_s_star", False)
    c_grid = cfg["intervention"]["c_grid"]
    do_rand = cfg["intervention"].get("random_control", False)
    k = cfg["scoring"]["top_k_logit_lens"]

    def coherence(out_tokens, cont_tokens, prompt_str):
        """Perplexity via judge (re-tokenized) or self-proxy."""
        if judge is not None:
            cont_str = model.to_string(cont_tokens[0])
            return judge_perplexity(judge, prompt_str, cont_str)
        return self_perplexity(model, out_tokens, cont_tokens.shape[1])

    os.makedirs(cfg["paths"]["results_dir"], exist_ok=True)
    rows = []
    gen_log = open(os.path.join(cfg["paths"]["results_dir"], "generations.txt"), "w")

    for feat in cfg["features"]:
        fi = feat["index"]
        if fi == "auto":
            fi = auto_select_feature(model, sae, hook_name, feat["elicit_prompts"])
            print(f"[feat {feat['name']}] auto-selected feature F{fi}")
        concept_ids, concept_strs = logit_lens_tokens(model, sae, fi, k=k)
        gen_log.write(f"\n### feature {feat['name']} (F{fi}) concept tokens: {concept_strs}\n")
        print(f"[feat {feat['name']} F{fi}] logit-lens: {concept_strs}")

        m_i, m_src = resolve_m(model, sae, hook_name, fi, feat["elicit_prompts"], cfg)
        a_max_char = resolve_amp_ref(model, sae, hook_name, fi, feat["elicit_prompts"], cfg)
        print(f"  m_i={m_i:.4f} ({m_src})  a_max_char={a_max_char:.4f} (amplify ref)")

        neutral_prompts = feat.get("neutral_prompts", [])
        if do_s_search and neutral_prompts:
            s_used = search_s_star(model, sae, hook_name, fi, a_max_char,
                                   concept_ids, neutral_prompts, cfg, gen_cfg, coherence)
        else:
            s_used = s_default

        p_floor = cfg["scoring"].get("concept_floor", 0.0)
        checks = filter_prompts(model, sae, hook_name, fi, concept_ids,
                                feat["elicit_prompts"], p_floor=p_floor)
        for ch in checks:
            tag = "PASS" if ch["passed"] else "SKIP(floor)"
            print(f"  [{tag}] base_p={ch['baseline_p']:.4f} a_i_max={ch['a_i_max']:.3f} :: {ch['prompt']!r}")

        for ch in checks:
            if not ch["passed"]:
                rows.append(_row(feat, fi, m_i, s_used, "elicit", ch["prompt"],
                                 "skip_floor", None, 0, ch,
                                 float("nan"), 0.0, 0, float("nan")))
                continue
            prompt = ch["prompt"]
            toks = model.to_tokens(prompt)

            def score_condition(cond_label, hook_fn, c_val):
                # p(l*) is deterministic given the hook -> compute once.
                fwd = [(hook_name, hook_fn)] if hook_fn is not None else None
                pl = p_concept(model, toks, concept_ids, fwd_hooks=fwd, rank_weighted=True)
                gss, ppls = [], []
                for j in range(n_samples):
                    out, cont = generate_with_hook(model, toks, hook_fn, hook_name,
                                                   gen_cfg, base_seed + j)
                    gs_frac, gs_any = gen_success(cont[0], concept_ids)
                    ppl = coherence(out, cont, prompt)
                    gss.append(gs_frac)
                    ppls.append(ppl)
                    if j == 0:
                        gen_log.write(f"[{feat['name']}|{cond_label}] {prompt!r} -> "
                                      f"{model.to_string(cont[0])!r}\n")
                    rows.append(_row(feat, fi, m_i, s_used, "elicit", prompt, cond_label,
                                     c_val, j, ch, pl, gs_frac, gs_any, ppl))
                gs_m = float(torch.tensor(gss).mean())
                pp_m = float(torch.tensor(ppls).nanmean())
                print(f"    {cond_label:>12}: p(l*)={pl:.4f} genSucc={gs_m:.3f} "
                      f"ppl={pp_m:.2f}  (n={n_samples})")

            score_condition("baseline", None, None)
            score_condition("amplify",
                            make_steer_hook(sae, fi, "amplify", s=s_used, a_max_char=a_max_char),
                            None)
            for c in c_grid:
                lbl = "ablate(c=0)" if c == 0 else f"neg(c={c})"
                score_condition(lbl, make_steer_hook(sae, fi, "coeff", c=c, m_i=m_i), c)
                if do_rand and c > 0:
                    score_condition(f"rand(c={c})",
                                    make_random_hook(sae, fi, c=c, m_i=m_i, seed=base_seed),
                                    c)

        # neutral-prompt amplify demo (appears-from-nothing)
        for prompt in neutral_prompts:
            toks = model.to_tokens(prompt)
            hf = make_steer_hook(sae, fi, "amplify", s=s_used, a_max_char=a_max_char)
            for j in range(n_samples):
                out, cont = generate_with_hook(model, toks, hf, hook_name, gen_cfg,
                                               base_seed + j)
                gs_frac, gs_any = gen_success(cont[0], concept_ids)
                if j == 0:
                    gen_log.write(f"[{feat['name']}|neutral+amplify] {prompt!r} -> "
                                  f"{model.to_string(cont[0])!r}\n")
                    print(f"  [neutral+amplify] {prompt!r} -> {model.to_string(cont[0])!r}")
                rows.append(_row(feat, fi, m_i, s_used, "neutral", prompt, "amplify",
                                 None, j,
                                 {"baseline_p": float('nan'), "a_i_max": float('nan'),
                                  "active": False, "passed": True},
                                 float("nan"), gs_frac, gs_any, float("nan")))

    gen_log.close()
    out_csv = os.path.join(cfg["paths"]["results_dir"], "results.csv")
    _write_csv(rows, out_csv)
    print(f"\nWrote {len(rows)} rows -> {out_csv}")


def _row(feat, fi, m_i, s_used, ctx, prompt, cond, c_val, sample, ch,
         pl, gs_frac, gs_any, ppl):
    return {
        "feature": feat["name"], "feature_idx": fi, "m_i": m_i, "s_used": s_used,
        "context": ctx, "prompt": prompt, "condition": cond, "c": c_val,
        "sample": sample,
        "baseline_p": ch["baseline_p"], "a_i_max": ch["a_i_max"],
        "active": ch["active"], "passed_floor": ch["passed"],
        "p_concept": pl, "gen_success_frac": gs_frac, "gen_success_any": gs_any,
        "perplexity": ppl,
    }


def _write_csv(rows, path):
    if not rows:
        return
    cols = list(rows[0].keys())
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=cols)
        w.writeheader()
        w.writerows(rows)


if __name__ == "__main__":
    import sys
    run(sys.argv[1] if len(sys.argv) > 1 else "config.yaml")
