# Paper figures/tables → code & data

Every figure and table in the paper is reproducible from this repo. The committed
`results/` and `results_v2/` directories already contain the exact artifacts behind the
paper; the table below maps each paper element to the file that produces it and the script
that emits it. (Paper figures were renamed when imported into the LaTeX source — the
left column is the paper's filename, the middle column is what this code emits.)

| Paper element | Emitted file (in this repo) | Produced by |
|---|---|---|
| **Figure 1** `fig_aggregate.png` (aggregated negative sweep + random control) | `results_v2/curve_aggregate.png` | `analyze.py results_v2` → `aggregate()` |
| **Figure 2** `fig_ablation_vs_negative.png` (per-feature violins) | `results_v2/curve_ablation_vs_negative.png` | `analyze.py results_v2` → `plot_ablation_vs_negative()` |
| **Appendix Figure** `fig_hair.png` (reproduction hair feature F8827) | `results/curve_hair.png` | `analyze.py results` → `plot_feature()` |
| **Appendix Table 1** `fig_table1.png` (qualitative continuations) | `results_v2/table1_qualitative.png` | `make_table1.py` |
| **Appendix Table 2** (15-feature panel: m_i, a_max, s*, sel. p) | `results_v2/results.csv` + console log of `analyze.py` | `steer.py config_mvp2.yaml` |
| **Table 6** (fixed run settings) | `config.yaml` | — (configuration) |
| Prompt-design diagnostic (not in paper) | `results_v2/selection_pstar.png` | `select_features2.py` |
| Per-feature curves (heterogeneity, §4.5) | `results_v2/curve_f*.png` | `analyze.py results_v2` → `plot_feature()` |

## Headline numbers (all from `results_v2/results.csv`)

- Ablation removes ~25% of concept on average; deep negative (c=4) ~88%.
- Direction-specificity: 14/15 features suppressed more by negative than the matched random push.
- Sign tests (printed by `analyze.py results_v2`): H2 (neg − ablate) p = 6.1e-5;
  guard#2 (neg − random) p = 9.8e-4.

## Data ↔ paper

- **Concept-tail prompt pool (49 prompts, Appendix A.1)** — `select_features2.py` `CORPUS`.
- **Neutral prompts (3, Appendix A.2)** — `select_features2.py` `NEUTRAL` (also in every config).
- **15-feature panel (Appendix A.3 / Table 2)** — frozen in `config_mvp2.yaml`
  (reconstructed from `results_v2/results.csv`; see header of that file).
- **hair reproduction feature F8827** — `config.yaml`, with its own elicit prompts.
