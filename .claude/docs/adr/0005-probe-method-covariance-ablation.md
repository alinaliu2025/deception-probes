# ADR 0005 — Probe Method Variants & the Covariance Ablation

**Status:** Accepted
**Date:** 2026-06-28

## Context

ADR 0004 locked in two probe families, `mms` (difference-of-means) and `lr`
(logistic regression), and left the *default* open. On the first scaled
sycophancy run (5000 examples, 1731 grouped pairs, Qwen2.5-0.5B, grouped held-out
split) the two methods diverged sharply:

| Method | Best-layer AUROC | Run |
|---|---|---|
| `mms` (raw diff-of-means) | **0.534** | `..._mms_1b8fcab_JACKPC` |
| `lr` (standardized) | **0.839** | `..._lr_8719e35_JACKPC` |

Both used identical activations and the identical grouped split, so the gap is
purely the fitter. `mms` was near-chance and **flat across all 25 layers**; `lr`
showed the expected mid-late deception hump (peak layer 21).

The first hypothesis was **feature scale**: residual-stream dims have a few
massive-activation channels whose raw norm dominates a raw mean-difference. `lr`
already neutralizes this via `StandardScaler` (folded back to raw space). So we
added a standardized diff-of-means to test whether scale alone explained the gap.

## Decision

**ACCEPTED** — add two diagnostic probe variants alongside `mms`/`lr`, selectable
via `--method`, all four folded into the same raw-space `Probe` so directions stay
comparable for the transfer study. Raw `mms` and `lr` are unchanged.

| Method | Direction | Covariance used |
|---|---|---|
| `mms` | `mean_pos − mean_neg` | none |
| `mms_std` | per-dim z-scored diff-of-means, scaler folded back | diagonal only (per-dim variance) |
| `lda` | `Σ⁻¹ (mean_pos − mean_neg)`, Ledoit-Wolf shrinkage (`shrinkage="auto"`, lsqr) | **full** within-class covariance |
| `lr` | logistic-regression weight (standardized, folded back) | full, discriminative fit |

This is a deliberate **ablation ladder over how much covariance structure the
fitter uses** — none → diagonal → full closed-form → full discriminative.

## Empirical result

Same data and split as above:

| Method | Best-layer AUROC |
|---|---|
| `mms` | 0.534 |
| `mms_std` | **0.536** |
| `lda` | *(pending run)* |
| `lr` | 0.839 |

**Per-dim standardization did essentially nothing** (0.534 → 0.536). This
falsifies the scale hypothesis: if rogue high-variance channels were the killer,
diagonal whitening would have closed most of the gap. It did not. The signal that
`lr` exploits lives in **correlated combinations of dimensions**, which only a
**full-covariance** method can credit. `lda` is the principled diff-of-means
analogue that uses `Σ⁻¹`; comparing it to `lr` separates "covariance" from
"discriminative fit" as the source of `lr`'s edge.

## Consequences

- `mms_std` and `lda` are **diagnostic / ablation** methods, not new canonical
  probes. They exist to attribute the `mms`↔`lr` gap, and to give a fair
  diff-of-means baseline (`lda`) if a closed-form, non-iterative probe is wanted.
- `Σ` is hidden×hidden (~896² at 0.5B) and ill-conditioned at this n; `lda` relies
  on Ledoit-Wolf shrinkage to stay invertible. Watch this if/when the model or
  feature dimension grows.
- All four methods record their name in `meta.json` (ADR 0003), so ablation runs
  are self-identifying.
- This sharpens ADR 0004's open question on the default method: the choice is no
  longer `mms` vs `lr` on taste — there is now evidence the raw diff-of-means is
  near-chance on sycophancy at this scale, and that the gap is covariance, not
  scale. Resolve the canonical Phase 1 method (and revisit project memory's
  "diff-in-means locked" note) once the `lda` run lands.

## Open questions

- Final `lda` number and whether it reaches `lr`. If `lda ≈ lr`, the gap is purely
  covariance; if `lda < lr`, the discriminative fit adds real signal.
- Whether to promote `lda` to a reported method or keep it diagnostic-only.
- Does the same ladder hold for sandbagging and concealment, or is the covariance
  dependence sycophancy-specific?
