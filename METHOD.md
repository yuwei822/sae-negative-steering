# Plan 3 — Method

**Amplify vs. Off vs. Negative steering on a single SAE output feature.**

Extends Arad, Mueller & Belinkov (EMNLP 2025), *SAEs Are Good for Steering — If You Select the Right Features*. The paper only **amplifies** features (positive steering factor). This document specifies the full method for a **three-way** comparison — `+` amplify, `0` ablate, `−` negative — on a single output feature, with particular care given to **where the negative coefficient comes from and why its construction is defensible**.

This is the **full experimental version**, not a smoke MVP. Concretely that means: every condition is run over **N sampled generations** (mean ± spread, not a single sample); each negative coefficient is paired with a **magnitude-matched random-direction control** (so the negative *direction* is separated from raw perturbation size); the amplify factor is found by a **per-feature `s*` search**; coherence is judged by an **independent Gemma-2-9B**, not a self-proxy; and the feature set is ~15 concept-present output features, not one. The four design choices that make this more than a demo are spelled out in §3.7 (random control), §2.3 (`s*`), §5 (9B judge), and §8 (statistical design).

---

## 0. Central question

For an SAE **output feature** (a feature whose decoder direction writes a human-nameable concept into the residual stream), is its causal handle **bidirectional**, or is it essentially **on/off**?

Concretely, given target feature `i` at layer `ℓ`:

- **`+` amplify** — `ã_i = a_i + s·a_max` (the paper's steering, Eq. 6).
- **`0` ablate** — `ã_i = 0`: the feature's natural in-distribution off-state.
- **`−` negative** — `ã_i = −c·m_i` with `c > 0`: push the coefficient *past* off, into the regime the SAE was never trained on.

Because the SAE is non-negative (JumpReLU for Gemma-Scope; ReLU for the GPT-2 fallback), the negative side is **out-of-distribution (OOD) by construction**. So a second question follows: does going negative **buy any concept suppression over simple ablation**, or does it only work by pushing the model OOD and breaking coherence?

**Hypotheses.**
- **H1** — `0` (ablation) cleanly reduces the concept; it is the feature's in-distribution off-state.
- **H2** — `−` (past-off) buys **little extra** suppression over `0`, and/or **costs coherence** as `c` grows. I.e. the handle is closer to on/off than bidirectional.
- **H3** — ablation may **not fully remove** the concept if redundant pathways also produce it; the residual concept under `0` measures how much *this* feature actually controls.

---

## 1. Model and SAE (what runs on the Mac Studio, and why)

### 1.1 Primary — the plan-faithful target

| Component | Value |
|---|---|
| Model | `google/gemma-2-2b` (HookedSAETransformer) |
| SAE release | `gemma-scope-2b-pt-res-canonical` |
| SAE id | `layer_22/width_16k/canonical` |
| Hook point | `blocks.22.hook_resid_post` (residual stream, post-block, layer 22) |
| SAE type | **JumpReLU** (non-negative activations), width 16 384 |
| Smoke feature | **hair**, F8827 (paper output score 0.808) |
| dtype / device | `float32` on `mps` (Apple Metal) |

**Why this model.**

1. **It is the exact target the plan and the source paper use.** The paper's reproducible result — amplifying the *hair* feature (Gemma-2-2B, L22, F8827) to make neutral prompts produce hair-heavy coherent text (Fig. 2b) — is the single cheapest end-to-end validation of the whole pipeline. Using the same model + SAE + feature means our `+` condition is a **direct reproduction**, not an analogy. If `+` reproduces, the hook, the SAE wiring, and the scoring are all proven at once before any novel `−` measurement is trusted.
2. **Gemma-Scope is a high-quality, canonical, public SAE suite** trained densely across every layer of Gemma-2-2B. The `canonical` width-16k SAE at layer 22 is the one the paper reports the *hair* feature against, so the feature index, the logit-lens tokens, and the output score are all directly comparable to published numbers.
3. **It is non-negative (JumpReLU).** This is essential to the experiment, not incidental: the "negative is OOD" argument (Section 3) only has teeth because the SAE genuinely never emits a negative coefficient during training. A model with a signed feature basis would defeat the point.
4. **It fits the hardware comfortably.** The paper's runs use a Mac Studio (M3 Ultra, 512 GB unified memory): the 2B model in fp32 is ~10 GB and the SAE adds a few hundred MB, with ample headroom to also load Gemma-2-9B as a heavier perplexity judge. On a smaller (e.g. 64 GB) machine the same configuration still fits; `bfloat16` halves memory and roughly doubles generation throughput with negligible quality loss, and is a supported drop-in.

**Coherence judge.** Coherence (perplexity) is scored by an **independent Gemma-2-9B** (`scoring.perplexity_model`), loaded as a second model and re-tokenizing each continuation under its own tokenizer (§5). This removes the bias of letting the steered model grade its own output. The 2B (steered, fp32) + 9B (judge, bf16) coexist comfortably in unified memory; even on a 64 GB machine they fit. Set `perplexity_model: self` to fall back to the lighter 2B self-proxy.

**Gating caveat.** Both `gemma-2-2b` **and** the `gemma-2-9b` judge are gated Hugging Face models. Before running you must accept each license (<https://huggingface.co/google/gemma-2-2b>, <https://huggingface.co/google/gemma-2-9b>) and authenticate on the machine (`huggingface-cli login`, or `export HF_TOKEN=...`). `preflight.py` verifies both loads, checks that F8827's logit-lens actually reads as *hair*, and exercises the `+ / 0 / − / random` hooks before any sweep runs.

### 1.2 Fallback — non-gated

| Component | Value |
|---|---|
| Model | `gpt2-small` |
| SAE | `gpt2-small-res-jb`, hook `blocks.7.hook_resid_pre` |
| SAE type | ReLU (also non-negative) |
| Feature | auto-selected (no hand-labelled *hair* analogue) |

Used only when no HF token is available. ReLU is non-negative too, so the **"negative is OOD"** argument still holds; what is lost is the published *hair* reproduction (the feature is discovered automatically instead of being the known F8827).

---

## 2. The three interventions

### 2.1 SAE-coefficient framing

All three interventions act on the **SAE activation `a`**, not on the residual stream directly. This is deliberate and keeps us inside the paper's framing: intervening on `a` is *feature steering*; an arbitrary residual-direction projection would no longer be a statement about a feature.

At the hook point we encode the residual, modify exactly one coordinate of the activation vector, and write the change back along that feature's decoder direction.

```
a   = SAE.encode(resid)          # [batch, pos, d_sae]
a_i = a[..., i]                  # the target feature's current coefficient
```

- **`+` amplify** — `ã_i = a_i + s·a_max`
- **`0` ablate** — `ã_i = 0`, the `c = 0` special case of the line below
- **`−` negative** — `ã_i = −c·m_i`, `c > 0`

### 2.2 Single-feature delta steering (how the change is written back)

We do **not** re-decode the whole residual (`W_dec·ã + b_dec`) and replace it, because that would discard the SAE's reconstruction error `ε` and silently perturb every other feature. Instead we move the residual **only along feature `i`'s decoder column** by exactly the coefficient delta:

```
resid' = resid + (ã_i − a_i) · W_dec[:, i]
```

Consequences, all of which matter for a clean causal claim:

- Every other feature's contribution is untouched.
- The SAE reconstruction error `ε` is preserved exactly.
- The **no-intervention baseline is bit-for-bit the original model** (`ã_i = a_i ⇒ delta = 0`), so any measured change is attributable to feature `i` alone.

The hook stays active across **all** generation steps, so the intervention shapes the whole continuation, not just the first token.

### 2.3 The amplify reference `a_max`

`a_max` in Eq. 6 is the feature's **characteristic** maximum activation — a fixed scalar derived from data (corpus max-active, or Neuronpedia's published max) — **not** the current prompt's `a.max()`.

This distinction is load-bearing for the **appears-from-nothing** demo: on a *neutral* prompt the feature is off, so the current-prompt maximum is `0`, and `a_i + s·0 = a_i` would amplify by nothing. Using the characteristic `a_max` lets `+` inject the concept into a prompt where it was absent — which is exactly how the paper produced Fig. 2b. (Implemented as `a_max_char`, resolved by `m_scale.resolve_amp_ref`.)

**Per-feature `s*` search.** The amplify factor `s` is not fixed at a demo value. For each feature we sweep `s` over `intervention.s_grid` on the neutral prompts and pick `s*` = the **largest `s` whose mean perplexity stays within `coherence_mult ×` the neutral baseline**, maximizing concept effect `p(ℓ*)` among those — i.e. the strongest still-coherent amplification, mirroring the paper's `s*`. (`steer.search_s_star`; the chosen `s*` is recorded per feature in `results.csv`.)

---

## 3. The negative coefficient — origin and justification

This is the novel part of the method and the part most easily mis-stated, so it is specified in full.

### 3.1 Definition

```
ã_i = −c · m_i ,      c ≥ 0
```

- `m_i` — feature `i`'s **own characteristic active magnitude**, a positive scalar.
- `c` — a **dimensionless sweep coefficient**, the x-axis of the experiment. `c = 0` is ablation; `c > 0` is negative.

`m_i` is obtained one of two ways (config `m_scale.method`):

- **`corpus_median_active`** — the **median of the feature's activations over positions where it is genuinely active** (`a_i > 0`) across the elicit corpus. A gentle, robust unit: "the typical magnitude when this feature is on."
- **Neuronpedia max** (`fallback_neuronpedia_max`) — the feature's published maximum activation; a consistent satiation-level anchor.

The default grid is `c ∈ {0, 0.25, 0.5, 1, 2, 4}`.

### 3.2 Why scale by `m_i` at all (rather than an absolute number)

Different features live on wildly different activation scales: an activation of `2` may be saturation for one feature and negligible for another. A raw negative constant like `−2` is therefore **not comparable across features**, and an aggregate curve over many features would be meaningless.

Tying the magnitude to the feature's **own** characteristic scale `m_i` makes `c` a **dimensionless, cross-feature-comparable knob**: `c = 1` always means *"one characteristic active-magnitude below zero,"* for every feature. This is what lets the per-feature curves be **aligned and aggregated**, and it mirrors exactly how the paper treats its amplification factor as a sweepable x-axis.

### 3.3 Why `median-active` (or Neuronpedia max) specifically

`median-active` is the feature's typical magnitude **when it is actually firing**. So `−c·m_i` reads as *"how many typical-on magnitudes past zero are we pushing."* The unit is anchored in the feature's **real operating range**, not chosen arbitrarily. Neuronpedia's max is an alternative anchor at the saturation end; the two differ only in step coarseness, not in kind.

### 3.4 Why `c = 0` is a defensible origin for the axis

`c = 0 ⇒ ã_i = 0 ⇒` ablation, and ablation is the feature's **genuine in-distribution off-state**: because the SAE is sparse, it sees `a_i = 0` constantly during training. The sweep therefore **starts at a meaningful, in-distribution zero** and moves **monotonically outward** into OOD as `c` grows. The coefficient `c` parametrizes a **continuous path from the in-distribution off-state into the out-of-distribution regime** — which is precisely what lets us locate *where* coherence breaks (the perplexity knee). Finding that collapse point is a goal of the experiment, not a nuisance.

### 3.5 Why `−c·m_i` and not `−c·a_i` (reflecting the current activation)

- A **fixed** `m_i` **decouples the intervention magnitude from the current prompt's incidental activation**: the same `c` applies the same push at every prompt and position, preserving comparability.
- `−c·a_i` would **collapse to 0 wherever `a_i ≈ 0`**, conflating "off" with "negative" and destroying the distinction the experiment is built to measure.

### 3.6 The honest limit of the justification

**`m_i` justifies the *magnitude and units* of the negative push. It does *not* make the *sign* in-distribution.** JumpReLU/ReLU SAEs are trained with non-negative activations only; the decoder has never seen a negative coefficient. Therefore:

> The reasonableness of `−c·m_i` is that it places an **inevitably-OOD probe** on a **meaningful, comparable axis** — it is **not** a claim that negative coefficients are a valid SAE operation.

Direct consequences, enforced throughout:

- The **clean, in-distribution suppression anchor is `0` (ablation)**, never `−`.
- `−` is reported as a **past-off OOD probe**, read together with the perplexity curve. We never claim "negative steering is a clean suppression tool."

### 3.7 Coupling caveat and the random-direction control (implemented)

Larger `c` is simultaneously *"more negative"* **and** *"larger-magnitude intervention"* — the two are coupled by construction. So a bare negative sweep answers *"does pushing past off add suppression, and at what coherence cost,"* **not** *"is the negative regime intrinsically special."*

To disentangle those, a **magnitude-matched random-direction control** is run for **every** `c > 0` (`hook.make_random_hook`). The negative intervention writes a delta of per-position L2 norm

```
||(ã_i − a_i)·W_dec[:,i]||  =  |(−c·m_i) − a_i| · ||W_dec[:,i]||
```

along the feature's decoder direction. The control writes the **same per-position L2 magnitude** along a **fixed random unit direction** in residual space (seeded by the feature, so the direction is constant across the `c`-sweep while the magnitude scales with `c`). Then:

- If `neg(c)` suppresses the concept **more** than `rand(c)` **and** costs no more coherence, the **negative direction itself buys something** — bidirectionality has real content.
- If `neg(c)` and `rand(c)` look the same, the apparent "suppression" is just **perturbation size**, not the negative direction — a stronger, more honest null than H2 alone.

`analyze.py` overlays `rand(c)` on both the concept-vs-`c` and perplexity-vs-`c` curves, and reports the aggregate *"suppression of neg beyond a matched random push"* (≈ 0 ⇒ the effect is perturbation magnitude, not direction). This is the single highest-value hardening of the `−` conclusion, and it is now part of the standard run.

---

## 4. Measurement context (the floor effect — do not skip)

The three conditions **do not share a natural context**:

- **`+`** is naturally demonstrated on a **neutral** prompt (concept absent → it appears). This is the Fig. 2b reproduction.
- **`0` and `−`** only have something to act on where the **feature is active** and the **concept is already present**. On a neutral prompt the feature is off, so ablation does nothing and negative merely injects an OOD value to suppress an absent concept — meaningless.

**Therefore the clean three-way comparison runs on a CONCEPT-ELICITING prompt** (feature fires, baseline already contains the concept), against a no-intervention baseline. Separately, `+` is shown on a neutral prompt (the appears-from-nothing demo), with the explicit note that `0/−` there are uninformative — which itself illustrates the floor effect.

### 4.1 Concept-tail prompt design (how we make `0` and `−` measurable)

Because the whole point of `0`/`−` is *suppression*, they can only be measured where the concept is genuinely about to be emitted. We engineer that on purpose with **concept-tail prompts**: every selection prompt is written so its **natural next token is a concept word** —

- `"He waited patiently for the entire ___"` → a time span,
- `"He stepped out of the shower and combed his ___"` → *hair*,
- `"To finish the dish the chef sprinkled some freshly chopped ___"` → an herb.

Reading `p(ℓ*)` at the **final position** (§5) then lands squarely on the concept, so many more features clear the baseline-`p(ℓ*)` floor than under generic prompts. This directly fixes the original failure mode (selecting features by raw activation left only ~1 in 12 with the concept actually present in the output). The corpus lives in `select_features2.CORPUS`, and **`select_features2.py` emits `selection_pstar.png`** — a per-prompt bar of the best feature's baseline `p(ℓ*)` with the floor line — as direct evidence that the prompt design lifts `p(ℓ*)` (the headline is "N of M prompts clear the floor").

### 4.2 Selection by baseline `p(ℓ*)`, not by activation

For each concept-tail prompt we take the **active** features at the last position and keep the one whose logit-lens tokens capture the most next-token mass (highest baseline `p(ℓ*)`). By construction the selected feature is **both** concept-present **and** feature-active — the clean substrate the `0`/`−` sweep needs. We keep ~15 such features (`select_features2.py 15`).

### 4.3 The floor guard at run time

**Floor guard (two conditions, both required).** To measure suppression you need the concept **present** (high baseline `p(ℓ*)`) **and** the feature **active** (`a_i > 0`). Prompts/features failing either are **excluded**, not scored as "perfectly suppressed." This is enforced by `scoring.concept_floor` (the minimum baseline `p(ℓ*)` for a prompt to count the concept as present), passed into `contexts.filter_prompts`; prompts below the floor are written as `skip_floor` rows and dropped from every curve. The gemma config sets `concept_floor: 0.02`; the gpt2 auto-feature fallback uses `0.0`. (Feature *selection* by baseline `p(ℓ*)` — `select_features2.py` — is what makes the aggregate honest: it ensures the features being swept actually emit their concept, fixing the case where a sweep "works" only because nothing was there to suppress.)

**Selection-circularity caveat (regression to the mean).** Both the per-prompt floor and `select_features2`'s ranking select on **high baseline `p(ℓ*)`**. Quantities selected for being extreme are partly extreme by noise, so on a *fresh* measurement they regress downward — meaning some of the apparent drop from baseline to ablation is regression to the mean, not a causal effect of the intervention. Two things keep the headline conclusion robust to this: (i) the conclusions about `−` are stated as **paired, within-feature contrasts** — `neg(c)` vs `ablate(c=0)` and `neg(c)` vs the magnitude-matched `rand(c)` — and the *same* selected baseline sits on both sides of each contrast, so selection bias cancels in the difference; (ii) the `s*` search and the neutral "appears-from-nothing" demo are evaluated on prompts that were **not** selected for high `p(ℓ*)`. Read the absolute baseline→ablation gap with this caveat in mind; read the paired `neg − ablate` and `neg − rand` deltas as the trustworthy quantities.

---

## 5. Metrics (per condition, averaged over prompts)

**Concept effect** (two readouts):

- **Rank-weighted `p(ℓ*)`** — the probability mass the next-token distribution places on the feature's top-`k` logit-lens tokens `ℓ` (the feature's "concept" vocabulary, from `W_dec[i] @ W_U`). A cheap single-forward mirror of the paper's output score. Default `k = 20`. **Scope:** this is read at the **final prompt position only** — it is the model's *immediate next-token* concept mass, not an average over the continuation. It is therefore a sensitive, deterministic, low-variance probe of how strongly the feature is about to express its concept, but it does **not** by itself prove the concept persists through the generated text. **Gen-Success** (below) is the complementary multi-token, behavioral readout; the two are reported together precisely because the single-position metric and the whole-continuation metric can disagree (e.g. a strong push that the model immediately recovers from).
- **Gen Success** — whether the feature's logit-lens tokens appear in the generated continuation (behavioral, over `max_new_tokens = 32`, `temperature = 0.7`).

**Coherence:**

- **Perplexity** — by default scored by an **independent Gemma-2-9B judge** (`scoring.perplexity_model`). The judge re-tokenizes the prompt and continuation under its own tokenizer (BOS on the prompt only, to avoid string-boundary merge artifacts) and returns the NLL-perplexity over the continuation positions (`steer.judge_perplexity`). Using a separate, larger model removes the bias of letting the steered model grade its own (possibly degenerate) output. The same judge closure also gates the `s*` search (§2.3), so the amplify-coherence budget and the reported perplexities use one consistent yardstick. Set `perplexity_model: self` for the lighter 2B self-proxy (`steer.self_perplexity`). The perplexity-vs-`c` curve is what exposes the OOD-collapse knee.

  **Memory / OOM escape hatches.** The judge is loaded in `bfloat16` regardless of the steered model's dtype (a 9B in fp32 is ~36 GB and would not coexist with an fp32 2B on 64 GB; bf16 is ~18 GB). If the 2B (on `mps`) and the 9B judge still contend for unified memory, pin the judge to CPU with `scoring.perplexity_judge_device: cpu` (slower but never OOMs), or override its precision with `scoring.perplexity_judge_dtype`. The cheapest escape is `perplexity_model: self`, which loads no second model at all.

`p(ℓ*)` is a single deterministic forward given the hook, so it is computed **once** per condition. Gen-Success and perplexity depend on the **sampled** continuation, so they are collected over **`n_samples`** generations (§8) and reported as mean ± spread.

**Deliverable is the curve, not a single `c`.** If one number per feature is needed (e.g. a scatter), pick the `c` maximizing `suppression / perplexity`.

---

## 6. Critical guards (summary)

1. **Negative is OOD regardless.** `m_i` makes the *scale* principled (feature's own units), not the *sign*. Frame `−` as a past-off OOD probe; `0` is the clean suppression. (§3.6)
2. **More-negative = bigger push.** `c` couples "more negative" with "larger intervention"; the comparison is about pushing past off, not about a special negative regime. The **magnitude-matched random-direction control** (run for every `c>0`) disentangles them. (§3.7)
3. **Floor effect, two conditions.** Concept present (high baseline `p(ℓ*)`) **and** feature active (`a_i > 0`). Exclude failures; never score them as suppressed. (§4)
4. **`+` and `−` scales are not directly comparable** (`a_max`-based vs `m_i`-based). Compare them only through outcome metrics (concept effect, perplexity), never through raw coefficient magnitudes.

---

## 7. Settings reference (`config.yaml`)

| Knob | Value | Meaning |
|---|---|---|
| `model.name` | `google/gemma-2-2b` | primary, plan-faithful |
| `model.dtype` | `float32` | `bfloat16` is a fine, faster drop-in |
| `sae.release` / `sae.sae_id` | `gemma-scope-2b-pt-res-canonical` / `layer_22/width_16k/canonical` | Gemma-Scope JumpReLU, width 16k |
| `sae.hook_name` | `blocks.22.hook_resid_post` | intervention point |
| smoke feature | hair, index `8827` | paper output score 0.808 |
| `intervention.amplify_s` | `10.0` | fallback amplify factor if `search_s_star` is off |
| `intervention.search_s_star` | `true` | per-feature `s*` search on neutral prompts (§2.3) |
| `intervention.s_grid` | `[2, 5, 10, 20, 40]` | amplify factors searched for `s*` |
| `intervention.coherence_mult` | `3.0` | `s*` must keep perplexity ≤ this × neutral baseline |
| `intervention.c_grid` | `[0, 0.25, 0.5, 1, 2, 4]` | `0` = ablation; `>0` = negative (OOD); extend top end until perplexity clearly breaks |
| `intervention.random_control` | `true` | run magnitude-matched random-direction control per `c>0` (§3.7) |
| `m_scale.method` | `corpus_median_active` | `m_i` = median active magnitude; or set `fallback_neuronpedia_max` |
| `generate` | 32 tokens, temp 0.7, seed 0 | continuation settings |
| `generate.n_samples` | `10` | sampled generations per condition → mean ± spread (§9) |
| `scoring.top_k_logit_lens` | `20` | size of the concept token set `ℓ` |
| `scoring.concept_floor` | `0.02` | min baseline `p(ℓ*)` for a prompt to count the concept as present (floor guard, §4); `0.0` keeps all |
| `scoring.perplexity_model` | `google/gemma-2-9b` | independent coherence judge; `self` for the lighter 2B proxy |
| `scoring.perplexity_judge_device` | `mps` | device for the judge; set `cpu` if the 9B + 2B risk MPS OOM (§5) |

---

## 8. Statistical design

**The headline result is a paired, within-feature contrast — read it first.** Because feature selection conditions on high baseline `p(ℓ*)` (§4, regression-to-mean caveat), the *absolute* baseline→ablation gap is partly selection artifact. The trustworthy quantities are the two **paired deltas**, where the same selected baseline appears on both sides and selection bias cancels:

1. **`neg(c) − rand(c)`** — does the negative *direction* suppress more than a same-magnitude random push? (guard #2, the cleanest test of "is `−` special".)
2. **`neg(c) − ablate(c=0)`** — does pushing past off buy suppression over the in-distribution off-state? (H2.)

`analyze.py` aggregates each across the feature panel and runs an **exact binomial sign test** (`_sign_test`, no scipy — the two-sided p-value is summed directly from `Binomial(n_eff, 0.5)`): under the null that the per-feature delta is equally likely either sign, it reports how many of the `n` features fall on the "neg is special / neg buys suppression" side and the two-sided p-value. A near-zero mean delta **with** a non-significant sign test is H2 confirmed; a positive mean **with** a significant sign test is genuine bidirectional content.

The dedicated figure for this is **`curve_ablation_vs_negative.png`** (`analyze.plot_ablation_vs_negative`): a violin (plus a thin line per feature) of concept-`p(ℓ*)/baseline` and of `perplexity/baseline` across the whole `c` sweep, and a third panel showing the paired *extra suppression of neg beyond ablation* per `c>0`. Read together it makes the H1/H2 story visual: the concept drop happens **at `c=0`** (ablation is the clean off-state), perplexity stays ~1 at `c=0` and only climbs as `c` grows (negative breaks coherence *earlier* than it adds suppression), and the third panel sits on ~0 (negative buys ~nothing over ablation).

Supporting design:

- **Replication.** Every condition is generated `n_samples` times (different seeds). `p(ℓ*)` is deterministic given the hook (computed once); **Gen-Success** and **perplexity** are random under sampling and are reported as **mean ± standard deviation** over `prompts × samples`. `results.csv` is written **one row per sample**, so any downstream test (CI, paired comparison, the sign test above) can be computed from it.
- **Multiple prompts per feature.** The hair feature uses ~10 elicit prompts; auto-discovered features keep up to 3 concept-tail prompts each (`select_features2.prompts_per_feature`). Effects are averaged across prompts before being read as a feature-level number.
- **Feature panel.** ~15 concept-present output features (selected by baseline `p(ℓ*)`, §4), not one — so the aggregate curves and the paired sign tests have real `n`.
- **Paired controls.** For each `c>0`, `neg(c)` and its `rand(c)` share magnitude and prompt, so the *neg-minus-random* difference is a **paired** quantity (§3.7) — the cleanest test of whether the negative direction is special.

---

## 9. Reading the results — what each output means and how to draw the conclusion

This section is the **interpretation key**: it says, for every figure and printout, exactly what to look at, what each pattern means, and which hypothesis it supports. Read it with the outputs open.

### 9.1 `selection_pstar.png` — did the prompt design work?

- **What it is.** One horizontal bar per concept-tail prompt = the baseline `p(ℓ*)` of that prompt's best active feature. Green = clears the floor; grey = below it. Title = "N of M prompts clear the floor."
- **How to read it.** Look at the **green fraction**.
  - **Most bars green (say ≳ 60–80%)** ⇒ the concept-tail prompts successfully put concepts at the next-token position, so the selected features are genuinely concept-present. **This is the precondition for everything else** — `0`/`−` are now measuring real suppression, not a floor artifact.
  - **Many bars grey** ⇒ the prompts are not eliciting their concept; rewrite the weak ones (the grey labels tell you which) before trusting the sweep.
- **Verdict it supports.** A high green fraction is the evidence that "selecting by baseline `p(ℓ*)` on concept-tail prompts" fixed the original 1-in-12 floor problem.

### 9.2 `curve_ablation_vs_negative.png` — the central 0-vs-`−` verdict (read this one first)

Three panels, shared x = `c` (`0` = ablation, `>0` = negative/OOD). Each violin is the spread across the ~15 features; the thin grey lines are individual features.

- **Panel 1 — concept `p(ℓ*)` / baseline.** *Where does the concept drop?*
  - **Big drop already at `c = 0`, then ~flat for `c > 0`** ⇒ **H1 + H2**: ablation is the clean off-state, and pushing negative barely removes more. The handle is **on/off**, not bidirectional.
  - **Keeps falling well past `c = 0`** ⇒ the negative regime is removing *additional* concept — evidence of **bidirectional** content (against H2). Quantify it in Panel 3.
- **Panel 2 — perplexity / baseline.** *What does it cost?* Reference line at 1.0 = baseline coherence.
  - **≈ 1 at `c = 0`** ⇒ ablation is coherent (a usable suppression tool).
  - **Climbs as `c` grows** ⇒ negative breaks fluency. **Compare the two panels:** if perplexity in Panel 2 starts climbing at a smaller `c` than where Panel 1 stops dropping, then **negative pays a coherence cost before it buys any extra suppression** — the strongest form of H2.
- **Panel 3 — extra suppression of neg beyond ablation, `(ablation − neg)/baseline`, per `c > 0`.** *Does negative buy anything over `0`?* Red line at 0.
  - **Violins centred on ~0** ⇒ negative buys essentially **nothing** over ablation ⇒ **H2 confirmed**.
  - **Violins clearly above 0** ⇒ negative does add suppression; read Panel 2 to see whether the coherence cost was worth it.

### 9.3 `curve_aggregate.png` — is the negative *direction* special? (vs a random push)

- **Left (concept/baseline vs `c`): the `neg` line vs the `random dir` line.**
  - **`neg` ≈ `random`** ⇒ the suppression is just **perturbation magnitude**, not the negative direction — going negative is not special.
  - **`neg` clearly below `random`** ⇒ the negative *direction itself* suppresses more than a same-sized random push ⇒ the direction is special.
- **Middle (perplexity/baseline vs `c`):** compares the coherence cost of `neg` vs `random`. If `neg` costs *more* perplexity for the *same or less* suppression, negative is strictly worse than a random perturbation.
- **Right (scatter, one dot per feature):** x = "neg suppression beyond random (matched |Δ|)", y = perplexity ratio at the deepest `c`. **Dots with x > 0** = features where negative beats random; **x ≈ 0** = no special direction; **high y** = expensive in coherence. A cloud sitting at **x ≈ 0** is the clean "not special" result.

### 9.4 Console printouts — the statistical verdict

After `analyze.py` the aggregate prints two **paired sign tests** (these are the headline numbers, §8):

- **`H2 sign test: k/n features … p=…`** (`neg` − `ablate`).
  - **`p > 0.05` and mean delta ≈ 0** ⇒ **H2 confirmed**: negative does not reliably out-suppress ablation → on/off handle.
  - **`p < 0.05` and the mean positive** ⇒ negative *reliably* adds suppression → genuine bidirectional content.
- **`guard#2 sign test: k/n features … p=…`** (`neg` − `random`).
  - **`p > 0.05`** ⇒ the negative direction is **not** distinguishable from a matched random push → "suppression" is perturbation size, not direction.
  - **`p < 0.05` and positive** ⇒ the negative direction is genuinely special.
- The aggregate also prints, per concept-present feature, `baseline p → ablation p (removed X%)` and the deepest-neg `p` + perplexity — read `removed X%` as **how much of the concept this single feature actually controls** (H3: if ablation removes little, redundant pathways also produce the concept).

### 9.5 Per-feature `curve_f*.png` — the same logic, one feature at a time

Each shows concept-vs-`c` and perplexity-vs-`c` with error bars and the `rand(c)` overlay, plus the baseline and `s*`-amplify reference lines. Use them to spot **heterogeneity**: most features can be on/off (flat after `c=0`) while a few are genuinely bidirectional (keep dropping). The aggregate hides that; these reveal it.

### 9.6 One-line verdicts (the decision table)

| Pattern you observe | Conclusion |
|---|---|
| Panel-3 violins ~0 **and** H2 sign test `p > 0.05` | **H2 confirmed** — the handle is on/off; `0` (ablation) is the clean suppression, `−` adds nothing |
| Panel-1 keeps dropping past 0 **and** H2 sign test `p < 0.05`, positive | Negative adds real suppression — **bidirectional** content |
| `neg` ≈ `random` in aggregate **and** guard#2 `p > 0.05` | The push is just **magnitude**, not the negative direction |
| `neg` < `random` **and** guard#2 `p < 0.05` | The negative **direction** is special |
| Perplexity climbs (Panel 2) before concept drops further (Panel 1) | Negative **costs coherence before buying suppression** — strongest H2 |
| Ablation removes only a little of the concept | **H3** — redundant pathways also produce it; this feature is not the sole cause |

## 10. Known failure modes

- **"Perfectly suppressed" everywhere** ⇒ floor control missing (concept wasn't present / feature wasn't active). Re-check §4.
- **`−` looks identical to `0`** ⇒ the handle really is on/off — that is **H2 confirmed**, a finding, not a bug. Confirm via the perplexity curve.
- **Perplexity flat even at large `c`** ⇒ the negative push is too small relative to the residual; check `m_i` and extend the `c` grid (but expect OOD eventually).
- **`+` and `−` look "symmetric"** ⇒ suspicious given non-negativity; verify the negative branch actually drives `ã_i < 0` and isn't being clamped.
