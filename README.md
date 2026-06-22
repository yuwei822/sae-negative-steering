# Plan 3 — Amplify vs Off vs Negative on output features

Three-way SAE-coefficient steering on a single **output feature**:
`+` amplify (`a_i + s·a_max`), `0` ablate, `−` negative (`-c·m_i`, OOD).
Question: is an output feature's causal handle **bidirectional** or essentially **on/off**,
and does going negative buy suppression over ablation — or only break coherence?

This is the **full experimental version**: N sampled generations per condition (mean±std),
a **magnitude-matched random-direction control** for every `c>0`, a per-feature **`s*` search**,
an independent **Gemma-2-9B perplexity judge**, and ~15 concept-present features. See
[`METHOD.md`](METHOD.md) for the complete method (and the negative coefficient's justification).

Extends Arad, Mueller & Belinkov (EMNLP 2025), *SAEs Are Good for Steering*.

## Quickstart (Apple Silicon / MPS)

The paper's results were produced on a Mac Studio (Apple M3 Ultra, 512 GB unified
memory). It also runs on smaller Apple Silicon machines (e.g. a 64 GB M5 Pro) via the
`bfloat16` / CPU-judge / 2B-self-proxy escape hatches noted below.


```bash
cd plan3
bash setup.sh                      # venv + deps, verifies MPS

# gemma-2-2b AND the gemma-2-9b judge are GATED — accept both licenses then log in:
#   https://huggingface.co/google/gemma-2-2b
#   https://huggingface.co/google/gemma-2-9b   (perplexity judge; or set perplexity_model: self)
./.venv/bin/huggingface-cli login  # or: export HF_TOKEN=hf_xxx

bash run_all.sh                    # full pipeline on gemma-2-2b + hair F8827 (~overnight)
```

No HF token / don't want the gated model? Non-gated fallback (gpt2-small):

```bash
bash run_all.sh gpt2
```

## What the model choice is

- **Primary (`config.yaml`)** — `google/gemma-2-2b` + Gemma-Scope SAE
  `gemma-scope-2b-pt-res-canonical` / `layer_22/width_16k/canonical`, hook
  `blocks.22.hook_resid_post`. Smoke feature: **hair, F8827** (paper output score 0.808).
  This is the plan-faithful target; on the 512GB Mac Studio the 2B (fp32, ~10GB) and the
  9B judge (bf16, ~18GB) co-load with ample headroom (a 64GB machine also fits comfortably).
- **Fallback (`config_gpt2.yaml`)** — `gpt2-small` + `gpt2-small-res-jb`
  (`blocks.7.hook_resid_pre`). Non-gated, no token. ReLU SAE is also non-negative, so the
  "negative is OOD" argument still holds; feature is auto-selected instead of hair.

## Pipeline (what `run_all.sh` does)

1. **`preflight.py`** — loads model+SAE (+ the 9B judge), prints F8827's logit-lens (checks it
   says "hair"), runs `+/0/-/random` once, asserts everything stays on **MPS** (no silent CPU fallback).
2. **`steer.py config.yaml`** — `+/0/-` three-way + the `rand(c)` control on the hair-eliciting
   prompts (with a no-intervention baseline), `s*`-searched amplify, N samples per condition, plus
   the neutral-prompt amplify "appears-from-nothing" demo → `results/`.
3. **`select_features2.py 15`** — discovers ~15 **concept-present** output features by ranking on
   baseline `p(ℓ*)` over concept-tail prompts (keeps up to 3 prompts each) → `config_mvp2.yaml`.
4. **`steer.py config_mvp2.yaml`** + **`analyze.py results_v2`** — runs the full `c`-sweep (with
   random control + error bars) across all features and emits per-feature + aggregate curves → `results_v2/`.

## Files

| File | Role |
|---|---|
| `config.yaml` / `config_gpt2.yaml` | primary (gemma+hair+9B judge) / fallback (gpt2, self-proxy) settings |
| `config_mvp2.yaml` | **frozen 15-feature discovery panel** behind the paper (Table 2); reconstructed from `results_v2/results.csv` so the panel reproduces without re-running selection |
| `load.py` | load `HookedSAETransformer` + SAE + optional perplexity judge, resolve hook name |
| `hook.py` | the `+ / 0 / −` single-feature delta-steering hook **and** the magnitude-matched random control |
| `scores.py` | logit-lens tokens, rank-weighted `p(ℓ*)`, Gen Success |
| `m_scale.py` | `m_i` (median-active) and `a_max` (max-active / Neuronpedia) scales |
| `contexts.py` | floor-effect guard (concept-present + feature-active) |
| `steer.py` | run conditions (N samples, `s*` search, random control, judge perplexity), write `results.csv` + `generations.txt` |
| `select_features2.py` | **v2 discovery (by baseline `p(ℓ*)` on concept-tail prompts)** — fixes the floor effect, keeps ≤3 prompts/feature, emits `selection_pstar.png` |
| `analyze.py` | concept-vs-`c`, perplexity-vs-`c` (error bars + random-control overlay), the dedicated **ablation-vs-negative** figure, aggregate/H2/guard-#2 curves + paired sign tests |
| `preflight.py` | MPS + feature-index + hook (incl. random) + judge sanity checks |
| `make_table1.py` | render the qualitative-continuations table (paper Appendix Table 1) → `results_v2/table1_qualitative.png` |
| `FIGURES.md` | map of every paper figure/table → the file and script that produce it |

The committed `results/` and `results_v2/` directories hold the exact artifacts behind
the paper (CSV + figures + sampled generations), so every number and figure can be
checked without the overnight rerun. See [`FIGURES.md`](FIGURES.md) for the mapping.

## Outputs

- `results/` — hair three-way + `rand(c)` control + curves (with error bars).
- `results_v2/`:
  - `selection_pstar.png` — **prompt-design diagnostic**: per concept-tail prompt, the
    baseline `p(ℓ*)` of its best feature, and how many clear the floor (evidence that
    putting the concept word at the next-token position lifts `p(ℓ*)`).
  - `curve_ablation_vs_negative.png` — **the core 0-vs-`−` figure**: violin + per-feature
    lines for concept and coherence across the `c` sweep, plus the paired "extra
    suppression of neg beyond ablation (≈0 ⇒ on/off)" panel.
  - `curve_aggregate.png` — concept-vs-`c`, coherence-vs-`c`, and the "is the negative
    direction special?" scatter; the console also prints paired sign tests.
  - per-feature `curve_*.png`, `results.csv` (one row per sample), `generations.txt`.

## How to read the outputs (interpretation)

This is the short interpretation key — for every output, what to look at and what to conclude.
The long version (panel-by-panel, with hypothesis mapping) is [`METHOD.md`](METHOD.md) §9.

**Read the figures in this order:**

1. **`selection_pstar.png` — did the prompt design work?** Look at the **green fraction**
   (prompts whose best feature clears the floor). **Most bars green** ⇒ the concept-tail prompts
   really put the concept at the next-token position, so `0`/`−` below are measuring *real*
   suppression, not a floor artifact. Many grey ⇒ rewrite those prompts before trusting anything.

2. **`curve_ablation_vs_negative.png` — the core 0-vs-`−` verdict (read first).** Three panels,
   x = `c` (`c=0` is ablation, `c>0` is negative/OOD):
   - **Panel 1 (concept):** big drop **already at `c=0`** then ~flat ⇒ **on/off handle** (`0` is the
     clean off-state, negative adds little). Keeps falling past `c=0` ⇒ genuinely **bidirectional**.
   - **Panel 2 (perplexity, 1.0 = baseline):** ≈1 at `c=0` ⇒ ablation stays coherent. Climbs with `c`
     ⇒ negative breaks fluency. If it climbs **before** Panel 1 drops further ⇒ negative *costs
     coherence before it buys suppression* (the strongest "on/off" evidence).
   - **Panel 3 (extra suppression of `−` beyond `0`):** violins centred on **~0** ⇒ negative buys
     **nothing** over ablation. Clearly **above 0** ⇒ negative does add suppression.

3. **`curve_aggregate.png` — is the negative *direction* special, vs a same-sized random push?**
   `neg` line ≈ `random dir` line ⇒ the suppression is just **perturbation magnitude** (not special).
   `neg` clearly **below** `random` ⇒ the negative **direction itself** suppresses more (special).
   Right-panel scatter at **x ≈ 0** = clean "not special"; **x > 0** = negative beats random.

4. **Console sign tests** (printed by `analyze.py`) — the headline statistics:
   - `H2 sign test (neg − ablate)`: **`p > 0.05`, mean ≈ 0** ⇒ **H2 confirmed**, on/off handle.
     `p < 0.05`, mean positive ⇒ negative reliably adds suppression (bidirectional).
   - `guard#2 sign test (neg − random)`: **`p > 0.05`** ⇒ direction not distinguishable from a
     random push. `p < 0.05`, positive ⇒ negative direction is special.
   - Per feature: `baseline p → ablation p (removed X%)` = how much of the concept **this single
     feature** controls. Small `X%` ⇒ **H3**: redundant pathways also produce the concept.

5. **Per-feature `curve_f*.png`** — same logic, one feature at a time. Use them to spot
   **heterogeneity** the aggregate hides: most features on/off (flat after `c=0`) while a few keep
   dropping (bidirectional).

**One-line verdicts:**

| You observe | Conclusion |
|---|---|
| Panel-3 ~0 **and** H2 `p > 0.05` | **H2 confirmed** — on/off handle; `0` is the clean suppression, `−` adds nothing |
| Panel-1 keeps dropping past 0 **and** H2 `p < 0.05` (positive) | Negative adds real suppression — **bidirectional** |
| `neg` ≈ `random` **and** guard#2 `p > 0.05` | The push is just **magnitude**, not the negative direction |
| `neg` < `random` **and** guard#2 `p < 0.05` | The negative **direction** is special |
| Perplexity (Panel 2) climbs before concept (Panel 1) drops further | Negative **costs coherence before buying suppression** — strongest H2 |
| Ablation removes only a little of the concept | **H3** — redundant pathways also produce it |

## Tuning knobs (`config.yaml`)

- `generate.n_samples` — sampled generations per condition (error bars). Lower it to trade rigor for speed.
- `intervention.search_s_star` / `s_grid` / `coherence_mult` — per-feature `s*` search (paper-style);
  set `search_s_star: false` to use the fixed `amplify_s`.
- `intervention.c_grid` — the negative sweep; push the top end until perplexity clearly breaks
  (finding the OOD-collapse point is the point).
- `intervention.random_control` — toggle the magnitude-matched random-direction control.
- `m_scale.fallback_neuronpedia_max` — set to hair's Neuronpedia max activation to use it as the
  principled `a_max` / `m_i` instead of the corpus estimate.
- `scoring.perplexity_model` — `google/gemma-2-9b` (independent judge, default) or `self` (2B proxy, cheap).
- `scoring.perplexity_judge_device` — `mps` (default) or `cpu`; pin the 9B judge to CPU if it + the 2B risk MPS OOM.
- `scoring.concept_floor` — min baseline `p(ℓ*)` for a prompt to count the concept as present (floor guard). `0.02` for gemma/hair; `0.0` keeps all.

## Notes / caveats

- **Floor effect is real:** suppression is only measurable where the concept is genuinely in the
  output (`p(ℓ*)` high) *and* the feature fires. v2 selection enforces this; v1 does not.
- **`−` is OOD regardless.** `m_i` makes the *scale* principled, not the sign — JumpReLU/ReLU SAEs
  were never trained on negative coefficients. Treat `−` as a past-off OOD probe, `0` as the clean
  suppression. The `rand(c)` control tests whether the negative *direction* (not just the push size) matters.
- **Cost:** the full run (N samples × 15 features × random control × 9B judge) is an overnight job.
  Drop `n_samples`, `random_control`, or use `perplexity_model: self` to shorten it.
