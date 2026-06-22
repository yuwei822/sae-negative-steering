"""Concept-vs-c and coherence-vs-c curves with error bars, plus the random-direction
control overlay, from <results_dir>/results.csv.

Rows are per-sample (generate.n_samples per condition), so means are over prompts x
samples and error bars are their spread. For each negative coefficient c>0 we overlay
the magnitude-matched random-direction control rand(c): the gap between neg(c) and
rand(c) is what tells us whether the negative DIRECTION buys suppression beyond a
perturbation of the same size (guard #2). c=0 (ablation) and the amplify/baseline
references are marked.
"""
import os
import math
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

RESULTS = "results"


def _sign_test(deltas, eps=1e-9):
    """Two-sided exact binomial sign test on paired per-feature deltas.

    Null: each feature's delta is equally likely positive or negative (p=0.5).
    Ties (|delta| <= eps) are dropped. Returns (n_pos, n_eff, p_value). No scipy:
    the two-sided p-value is summed directly from the Binomial(n_eff, 0.5) pmf.
    """
    pos = sum(1 for d in deltas if d > eps)
    neg = sum(1 for d in deltas if d < -eps)
    n = pos + neg
    if n == 0:
        return pos, 0, float("nan")
    k = min(pos, neg)
    tail = sum(math.comb(n, i) for i in range(0, k + 1)) / (2 ** n)
    p = min(1.0, 2 * tail)
    return pos, n, p


def _sweep(df, feature, prefixes):
    """Aggregate a sweep family (over prompts x samples) into a per-c curve."""
    sub = df[(df.feature == feature) & (df.context == "elicit")]
    sweep = sub[sub.condition.str.startswith(prefixes)].copy()
    sweep = sweep[sweep.c.notna()]
    if sweep.empty:
        return None
    return sweep.groupby("c").agg(
        p=("p_concept", "mean"), p_std=("p_concept", "std"),
        ppl=("perplexity", "mean"), ppl_std=("perplexity", "std"),
        gs=("gen_success_frac", "mean"),
    ).reset_index().sort_values("c")


def _refs(df, feature):
    sub = df[(df.feature == feature) & (df.context == "elicit")]
    base = sub[sub.condition == "baseline"]
    amp = sub[sub.condition == "amplify"]
    s_used = float(sub.s_used.iloc[0]) if "s_used" in sub and len(sub) else float("nan")
    return (base.p_concept.mean(), amp.p_concept.mean(),
            base.perplexity.mean(), amp.perplexity.mean(), s_used)


def plot_feature(df, feature, outdir=RESULTS):
    neg = _sweep(df, feature, ("ablate", "neg"))
    rand = _sweep(df, feature, ("rand",))
    if neg is None:
        print(f"[{feature}] no sweep rows, skipping plot")
        return
    base_p, amp_p, base_ppl, amp_ppl, s_used = _refs(df, feature)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4))

    ax1.errorbar(neg.c, neg.p, yerr=neg.p_std, fmt="o-", capsize=3,
                 label="neg  -c·m_i")
    if rand is not None:
        ax1.errorbar(rand.c, rand.p, yerr=rand.p_std, fmt="s--", capsize=3,
                     color="orange", label="random dir (matched |Δ|)")
    ax1.axhline(base_p, ls="--", c="gray", label=f"baseline p={base_p:.3f}")
    ax1.axhline(amp_p, ls=":", c="green", label=f"amplify p={amp_p:.3f} (s*={s_used:g})")
    ax1.axvline(0, c="red", alpha=0.4)
    ax1.annotate("c=0\nablation", (0, neg.p.iloc[0]),
                 textcoords="offset points", xytext=(8, 8), color="red")
    ax1.set_xlabel("c  (0=ablate, >0=negative/OOD)")
    ax1.set_ylabel("concept effect  p(l*)")
    ax1.set_title(f"{feature}: concept vs c")
    ax1.legend(fontsize=8)

    ax2.errorbar(neg.c, neg.ppl, yerr=neg.ppl_std, fmt="o-", c="purple", capsize=3,
                 label="neg perplexity")
    if rand is not None:
        ax2.errorbar(rand.c, rand.ppl, yerr=rand.ppl_std, fmt="s--", c="orange",
                     capsize=3, label="random dir perplexity")
    ax2.axhline(base_ppl, ls="--", c="gray", label=f"baseline ppl={base_ppl:.1f}")
    ax2.axvline(0, c="red", alpha=0.4)
    ax2.set_xlabel("c  (0=ablate, >0=negative/OOD)")
    ax2.set_ylabel("perplexity (coherence)")
    ax2.set_title(f"{feature}: coherence vs c (OOD collapse)")
    ax2.legend(fontsize=8)

    fig.tight_layout()
    path = os.path.join(outdir, f"curve_{feature}.png")
    fig.savefig(path, dpi=130)
    plt.close(fig)
    print(f"[{feature}] wrote {path}")

    abl_p = float(neg.p.iloc[0])
    deep = neg.iloc[-1]
    msg = (f"[{feature}] H2: ablation p={abl_p:.4f} | deepest neg (c={deep.c}) "
           f"p={deep.p:.4f} ppl={deep.ppl:.2f} (baseline p={base_p:.4f}, ppl={base_ppl:.2f})")
    if rand is not None:
        rdeep = rand.iloc[-1]
        msg += f" | random@c={rdeep.c} p={rdeep.p:.4f} ppl={rdeep.ppl:.2f}"
    print(msg)


def plot_ablation_vs_negative(df, outdir=RESULTS):
    """Dedicated 0-vs-negative figure: the core H1/H2 claim, drawn dead.

    Across the whole feature panel, three panels share the c axis (0 = ablation, >0 =
    negative/OOD):
      (1) concept p(l*) / baseline — violin per c + a thin line per feature. The drop
          happens AT c=0 (ablation is already clean); going more negative barely lowers
          it further. The flat region right of 0 is "0 is the clean off-state".
      (2) perplexity / baseline — violin per c + per-feature lines, ref line at 1.0.
          Stays ~1 at c=0 and climbs only as c grows: negative breaks coherence EARLIER
          than it adds suppression.
      (3) extra suppression of neg BEYOND ablation, (ablation_p - neg_p)/baseline, per
          c>0 (paired, within feature). Centred on 0 ⇒ negative buys ~nothing over 0.
    """
    feats = df.feature.unique()
    concept_by_c, ppl_by_c, extra_by_c = {}, {}, {}
    concept_traj, ppl_traj = [], []          # per-feature (cs, ratios)

    for feature in feats:
        neg = _sweep(df, feature, ("ablate", "neg"))
        base_p, _, base_ppl, _, _ = _refs(df, feature)
        if neg is None or base_ppl != base_ppl or base_p <= 1e-9:
            continue
        abl_p = float(neg.p.iloc[0])
        cs, cr, pr = [], [], []
        for _, r in neg.iterrows():
            cratio, pratio = r.p / base_p, r.ppl / base_ppl
            # skip NaN ratios (e.g. a degenerate/empty continuation) so violinplot
            # never receives a NaN-containing group mid-run.
            if cratio == cratio:
                concept_by_c.setdefault(r.c, []).append(cratio)
            if pratio == pratio:
                ppl_by_c.setdefault(r.c, []).append(pratio)
            cs.append(r.c); cr.append(cratio); pr.append(pratio)
            if r.c > 0 and (abl_p - float(r.p)) == (abl_p - float(r.p)):
                extra_by_c.setdefault(r.c, []).append((abl_p - float(r.p)) / base_p)
        concept_traj.append((cs, cr)); ppl_traj.append((cs, pr))

    if not concept_by_c:
        print("[ablation-vs-neg] no sweep rows, skipping")
        return

    cs_all = sorted(concept_by_c)
    pos = {c: i for i, c in enumerate(cs_all)}
    ecs = sorted(extra_by_c)

    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(17, 5))

    def _violin(ax, by_c, order):
        order = [c for c in order if by_c.get(c)]      # skip empty/all-NaN groups
        if order:
            parts = ax.violinplot([by_c[c] for c in order],
                                  positions=[pos[c] for c in order],
                                  showmeans=True, widths=0.7)
            for b in parts["bodies"]:
                b.set_facecolor("#8ecae6"); b.set_alpha(0.5)
        ax.set_xticks([pos[c] for c in cs_all])
        ax.set_xticklabels([("0\n(ablate)" if c == 0 else f"{c:g}") for c in cs_all])

    _violin(ax1, concept_by_c, cs_all)
    for cs, cr in concept_traj:
        ax1.plot([pos[c] for c in cs], cr, "-", color="gray", alpha=0.3, lw=0.8)
    ax1.axvline(pos[0], c="red", alpha=0.4)
    ax1.set_xlabel("c  (0 = ablation, >0 = negative / OOD)")
    ax1.set_ylabel("concept p(l*) / baseline")
    ax1.set_title("Concept vs c: keeps dropping past ablation (c=0)")

    _violin(ax2, ppl_by_c, cs_all)
    for cs, pr in ppl_traj:
        ax2.plot([pos[c] for c in cs], pr, "-", color="gray", alpha=0.3, lw=0.8)
    ax2.axhline(1.0, ls="--", c="gray")
    ax2.axvline(pos[0], c="red", alpha=0.4)
    ax2.set_xlabel("c  (0 = ablation, >0 = negative / OOD)")
    ax2.set_ylabel("perplexity / baseline")
    ax2.set_title("Coherence vs c: negative breaks it as c grows")

    if ecs:
        data = [extra_by_c[c] for c in ecs]
        parts = ax3.violinplot(data, positions=range(len(ecs)), showmeans=True, widths=0.7)
        for b in parts["bodies"]:
            b.set_facecolor("#ffb703"); b.set_alpha(0.5)
        ax3.axhline(0.0, ls="--", c="red")
        ax3.set_xticks(range(len(ecs)))
        ax3.set_xticklabels([f"{c:g}" for c in ecs])
        ax3.set_xlabel("c  (negative only)")
        ax3.set_ylabel("(ablation − neg) p(l*) / baseline")
        ax3.set_title("Extra suppression of neg BEYOND ablation (grows with c)")

    fig.tight_layout()
    path = os.path.join(outdir, "curve_ablation_vs_negative.png")
    fig.savefig(path, dpi=130); plt.close(fig)
    print(f"[ablation-vs-neg] wrote {path}  (features={len(concept_traj)})")


def aggregate(df, outdir=RESULTS):
    feats = df.feature.unique()
    per_c, ppl_c = {}, {}            # neg family, normalized
    rper_c, rppl_c = {}, {}          # random control, normalized
    h2_pts = []                      # (extra suppr vs ablation, ppl ratio at deepest c)
    rand_pts = []                    # (suppr of neg beyond random at matched |Δ|, ppl ratio)
    present_thr = 0.01
    present = []

    for feature in feats:
        neg = _sweep(df, feature, ("ablate", "neg"))
        rand = _sweep(df, feature, ("rand",))
        base_p, amp_p, base_ppl, amp_ppl, s_used = _refs(df, feature)
        if neg is None or base_ppl != base_ppl:
            continue
        denom = base_p if base_p > 1e-9 else 1e-9
        for _, r in neg.iterrows():
            per_c.setdefault(r.c, []).append(r.p / denom)
            ppl_c.setdefault(r.c, []).append(r.ppl / base_ppl)
        if rand is not None:
            for _, r in rand.iterrows():
                rper_c.setdefault(r.c, []).append(r.p / denom)
                rppl_c.setdefault(r.c, []).append(r.ppl / base_ppl)

        abl_p = float(neg.p.iloc[0])
        deep = neg.iloc[-1]
        h2_pts.append(((abl_p - float(deep.p)) / denom, float(deep.ppl) / base_ppl))
        if rand is not None:
            rdeep = rand.iloc[-1]
            rand_pts.append(((float(rdeep.p) - float(deep.p)) / denom,
                             float(deep.ppl) / base_ppl))
        if base_p >= present_thr:
            present.append((feature, base_p, abl_p, float(deep.p), base_ppl,
                            float(deep.ppl), s_used))

    print(f"[aggregate] concept-PRESENT features (baseline p>={present_thr}): "
          f"{len(present)}/{len(feats)}  <- floor effect (guard #3)")
    for f, bp, ap, dp, bppl, dppl, s_used in present:
        print(f"  {f}: baseline p={bp:.3f} -> ablation p={ap:.4f} "
              f"(removed {100*(bp-ap)/bp:.0f}%) | deepest-neg p={dp:.4f} "
              f"ppl {bppl:.1f}->{dppl:.1f}  s*={s_used:g}")

    cs = sorted(per_c)
    p_mean = [sum(per_c[c]) / len(per_c[c]) for c in cs]
    ppl_mean = [sum(ppl_c[c]) / len(ppl_c[c]) for c in cs]
    rcs = sorted(rper_c)
    rp_mean = [sum(rper_c[c]) / len(rper_c[c]) for c in rcs]
    rppl_mean = [sum(rppl_c[c]) / len(rppl_c[c]) for c in rcs]

    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(16, 4))
    ax1.plot(cs, p_mean, "o-", label="neg")
    if rcs:
        ax1.plot(rcs, rp_mean, "s--", color="orange", label="random dir")
    ax1.axvline(0, c="red", alpha=0.4)
    ax1.set_xlabel("c (0=ablate, >0=neg)"); ax1.set_ylabel("concept p(l*) / baseline")
    ax1.set_title(f"Aggregated concept vs c  (n={len(feats)})"); ax1.legend(fontsize=8)

    ax2.plot(cs, ppl_mean, "o-", c="purple", label="neg")
    if rcs:
        ax2.plot(rcs, rppl_mean, "s--", color="orange", label="random dir")
    ax2.axvline(0, c="red", alpha=0.4); ax2.axhline(1.0, ls="--", c="gray")
    ax2.set_xlabel("c (0=ablate, >0=neg)"); ax2.set_ylabel("perplexity / baseline")
    ax2.set_title("Aggregated coherence vs c (OOD collapse)"); ax2.legend(fontsize=8)

    if rand_pts:
        xs = [p[0] for p in rand_pts]; ys = [p[1] for p in rand_pts]
        ax3.scatter(xs, ys)
        ax3.axvline(0, c="gray", ls="--")
        ax3.set_xlabel("neg suppression beyond random (matched |Δ|), frac of baseline")
        ax3.set_ylabel("perplexity ratio at deepest c")
        ax3.set_title("Is the negative DIRECTION special? (>0 ⇒ yes)")
    elif h2_pts:
        xs = [p[0] for p in h2_pts]; ys = [p[1] for p in h2_pts]
        ax3.scatter(xs, ys); ax3.axvline(0, c="gray", ls="--")
        ax3.set_xlabel("extra suppression from neg vs ablation, frac of baseline")
        ax3.set_ylabel("perplexity ratio at deepest c")
        ax3.set_title("H2: does going negative buy suppression?")

    fig.tight_layout()
    path = os.path.join(outdir, "curve_aggregate.png")
    fig.savefig(path, dpi=130); plt.close(fig)
    print(f"[aggregate] wrote {path}  (features={len(feats)})")

    if h2_pts:
        h2_deltas = [p[0] for p in h2_pts]
        me = sum(h2_deltas) / len(h2_deltas)
        pos, n, pv = _sign_test(h2_deltas)
        print(f"[aggregate] H2: mean extra suppression from neg over ablation = "
              f"{me:.3f} of baseline (≈0 ⇒ on/off, not bidirectional)")
        print(f"[aggregate] H2 sign test: {pos}/{n} features suppress more under neg "
              f"than ablation, two-sided p={pv:.3g}")
    if rand_pts:
        rand_deltas = [p[0] for p in rand_pts]
        mr = sum(rand_deltas) / len(rand_deltas)
        pos, n, pv = _sign_test(rand_deltas)
        print(f"[aggregate] guard#2: mean suppression of neg BEYOND a matched random "
              f"push = {mr:.3f} of baseline (≈0 ⇒ effect is just perturbation size, "
              f"not the negative direction)")
        print(f"[aggregate] guard#2 sign test: {pos}/{n} features suppress more under "
              f"neg than a matched random push, two-sided p={pv:.3g}")


def main():
    import sys
    results_dir = sys.argv[1] if len(sys.argv) > 1 else RESULTS
    df = pd.read_csv(os.path.join(results_dir, "results.csv"))
    for feature in df.feature.unique():
        plot_feature(df, feature, outdir=results_dir)
    if df.feature.nunique() > 1:
        plot_ablation_vs_negative(df, outdir=results_dir)
        aggregate(df, outdir=results_dir)


if __name__ == "__main__":
    main()
