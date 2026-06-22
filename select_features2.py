"""ADDITIONAL benchmark v2 — select features by baseline p(l*), not by activation.

Fixes the floor effect that left only 1/12 features with the concept present. The two
fixes are the whole point of this file and are emphasised deliberately:

  1. PROMPT DESIGN ('concept-tail'). Every selection prompt is written so its natural
     NEXT token is a concept word ("He spent the entire" -> a time span; "...combed
     his" -> hair). That forces the concept into the OUTPUT distribution at the final
     position, which is exactly where p(l*) is read — so many more features clear the
     baseline-p(l*) floor than under generic prompts. selection_pstar.png visualises
     this: the per-prompt baseline p(l*) of the best feature, the direct evidence that
     the prompt design works.
  2. SELECTION CRITERION. For each prompt, among the ACTIVE features at the last
     position, pick the one whose logit-lens tokens capture the most next-token
     probability mass (= highest baseline p(l*)). That guarantees, by construction,
     concept-present AND feature-active — the clean substrate for measuring 0/negative.

Writes config_mvp2.yaml with results_dir=results_v2 so the original run is untouched,
and a selection_pstar.png diagnostic into results_v2/.
"""
import os
import sys
import yaml
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from load import load_config, load_model_and_sae, sae_hook_name

# concept-tail prompts: the natural continuation is a concept word.
CORPUS = [
    "He waited patiently for the entire",
    "The meeting dragged on for almost an entire",
    "She painted the bedroom wall a bright shade of",
    "At dusk the sunset turned the sky a deep",
    "They booked a one-way flight from New York to",
    "He was born and raised in the city of",
    "For breakfast she ordered a plate of scrambled",
    "On the way home he bought a fresh loaf of",
    "The first day of the week after Sunday is",
    "She counted slowly: five, six, seven,",
    "On stage the musician picked up his electric",
    "In the school orchestra she has always played the",
    "After the tests the doctor said he was suffering from",
    "Sadly she was diagnosed with an aggressive form of",
    "He stepped out of the shower and combed his",
    "Before the photo she tied back her long brown",
    "The defense attorney presented the case before the",
    "After deliberating for hours, the verdict came from the",
    "Having enlisted at eighteen, he proudly served in the",
    "Every Sunday the old priest read aloud from the holy",
    "In the final minute the striker scored the winning",
    "Overwhelmed and exhausted, tears streamed down her face as she felt so",
    "He clenched his fists, his chest tight with pure",
    "To finish the dish the chef sprinkled some freshly chopped",
    "She seasoned the soup with a pinch of salt and ground black",
    "The toddler laughed and reached for the fluffy little",
    "Across the savanna the photographer spotted a herd of",
    "The astronauts strapped in for the launch of the",
    "He invested his savings in shares of the rising tech",
    "On the map they traced the river all the way to the",
    "The librarian frowned and asked them to lower their",
    "He tightened the last bolt and started the car's",
    "The toddler scribbled all over the wall with a red",
    "The campers gathered dry wood and lit a roaring",
    "The senator stepped up to the podium to deliver his",
    "He poured the black coffee and added a splash of",
    "She wrapped the birthday present in shiny silver",
    "The mechanic knelt down to check the pressure in each",
    "After a long climb they finally reached the snowy",
    "He plugged in the amp and strummed a loud guitar",
    "The jury foreman rose to read aloud the final",
    "The surgeon held out a hand and asked for a clean",
    "He saddled up and rode his horse across the open",
    "At the bakery the warm smell came from a fresh loaf of",
    "She dipped her brush into the jar of bright yellow",
    "On the African plain the safari guide pointed at a lion and a",
    "The pianist sat down and began to play a gentle",
    "Late at night the old grandfather clock began to",
    "The toddler hugged the soft and fluffy stuffed",
]

# generic concept-absent prompts, shared by every selected feature so that s* search
# and the amplify "appears-from-nothing" demo have a neutral context to act on.
NEUTRAL = [
    "The weather today is",
    "My favorite thing about the city is",
    "Yesterday I went to the store and",
]


def _plot_selection_diagnostic(diag, p_floor, plot_dir):
    """Direct evidence that the concept-tail PROMPT DESIGN lifts baseline p(l*).

    One horizontal bar per concept-tail prompt = the baseline p(l*) of its best active
    feature (how much next-token mass already lands on a concept word). Bars above the
    floor (dashed line) are prompts that successfully put the concept in the OUTPUT and
    therefore yield a usable feature for the 0/negative sweep. The fraction clearing the
    floor is the headline: it is what 'concept-tail prompts beat generic prompts' means.
    """
    os.makedirs(plot_dir, exist_ok=True)
    rows = sorted(diag, key=lambda d: d[1])               # ascending p(l*)
    ys = range(len(rows))
    ps = [d[1] for d in rows]
    colors = ["#2a9d8f" if p >= p_floor else "#bbbbbb" for p in ps]
    labels = [f"{d[0][:42]:42s}  →{d[3]!r}" for d in rows]

    n_pass = sum(p >= p_floor for p in ps)
    fig, ax = plt.subplots(figsize=(11, max(5, 0.34 * len(rows))))
    ax.barh(list(ys), ps, color=colors)
    ax.axvline(p_floor, ls="--", c="red", label=f"floor p≥{p_floor}")
    ax.set_yticks(list(ys))
    ax.set_yticklabels(labels, fontsize=7, family="monospace")
    ax.set_xlabel("baseline p(l*) of best active feature  (next-token concept mass)")
    ax.set_title(f"Concept-tail prompt design: {n_pass}/{len(rows)} prompts clear the "
                 f"floor (concept lands in the OUTPUT)")
    ax.legend(loc="lower right")
    fig.tight_layout()
    path = os.path.join(plot_dir, "selection_pstar.png")
    fig.savefig(path, dpi=130)
    plt.close(fig)
    print(f"[select] wrote {path}  ({n_pass}/{len(rows)} prompts >= floor {p_floor})")


@torch.no_grad()
def select(cfg, n=15, p_floor=None, topk_active=120, prompts_per_feature=3,
           plot_dir="results_v2"):
    model, sae, dev = load_model_and_sae(cfg)
    hook_name = sae_hook_name(sae, cfg)
    W_U = model.W_U
    k = cfg["scoring"]["top_k_logit_lens"]
    # Use the SAME floor as the run-time guard (steer.py -> contexts.filter_prompts) so
    # selection and scoring agree on what counts as 'concept present'. Falls back to 0.03.
    if p_floor is None:
        p_floor = cfg.get("scoring", {}).get("concept_floor", 0.03)

    # feature_idx -> {"best": p, "strs": [...], "prompts": [(p, text), ...]}
    feats = {}
    diag = []          # (text, best_p, best_fi, concept_str) per prompt — for the diagnostic plot
    for text in CORPUS:
        toks = model.to_tokens(text)
        logits, cache = model.run_with_cache(toks, names_filter=hook_name)
        probs = torch.softmax(logits[0, -1].float(), dim=-1)     # next-token dist
        a_last = sae.encode(cache[hook_name])[0, -1]             # [d_sae]

        active = torch.nonzero(a_last > 0).flatten()
        if active.numel() == 0:
            continue
        # limit to the strongest-firing features for speed
        if active.numel() > topk_active:
            act_vals = a_last[active]
            keep = torch.topk(act_vals, topk_active).indices
            active = active[keep]

        # for each active feature: p(l*) = next-token mass on its logit-lens tokens
        best_fi, best_p, best_ids = None, -1.0, None
        for fi in active.tolist():
            ll = (sae.W_dec[fi].to(W_U.dtype) @ W_U)             # [d_vocab]
            ids = torch.topk(ll, k).indices
            pstar = float(probs[ids].sum())
            if pstar > best_p:
                best_fi, best_p, best_ids = fi, pstar, ids

        strs = [model.to_single_str_token(int(t)) for t in best_ids[:8]]
        print(f"  {text!r}\n    -> best F{best_fi} p(l*)={best_p:.3f} concept={strs}")
        diag.append((text, best_p, best_fi, strs[0] if strs else ""))
        if best_fi is not None and best_p >= p_floor:
            rec = feats.setdefault(best_fi, {"best": best_p, "strs": strs, "prompts": []})
            rec["prompts"].append((best_p, text))
            if best_p > rec["best"]:
                rec["best"], rec["strs"] = best_p, strs

    if plot_dir:
        _plot_selection_diagnostic(diag, p_floor, plot_dir)

    ranked = sorted(feats.items(), key=lambda kv: kv[1]["best"], reverse=True)[:n]
    features = []
    for fi, rec in ranked:
        prompts = [t for _, t in sorted(rec["prompts"], reverse=True)][:prompts_per_feature]
        features.append({"name": f"f{fi}", "index": fi, "elicit_prompts": prompts,
                         "neutral_prompts": list(NEUTRAL)})
        print(f"KEEP F{fi:5d}  baseline p(l*)={rec['best']:.3f}  {rec['strs']}  "
              f"({len(prompts)} prompt/s)")
    return features


def main():
    n = int(sys.argv[1]) if len(sys.argv) > 1 else 15
    config_path = sys.argv[2] if len(sys.argv) > 2 else "config.yaml"
    cfg = load_config(config_path)
    features = select(cfg, n=n)
    out = dict(cfg)
    out["features"] = features
    out["paths"] = {"results_dir": "results_v2"}     # separate from the original run
    with open("config_mvp2.yaml", "w") as fh:
        yaml.safe_dump(out, fh, sort_keys=False, allow_unicode=True)
    print(f"\nWrote {len(features)} features -> config_mvp2.yaml (results_dir=results_v2)")


if __name__ == "__main__":
    main()
