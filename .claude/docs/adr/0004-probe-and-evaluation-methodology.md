# ADR 0004 — Probe Training & Evaluation Methodology

**Status:** Accepted
**Date:** 2026-06-28

## Context

The headline result is a cross-type **transfer matrix**: train a probe on one
deception type, test it on another. For that comparison to mean anything, every
probe has to be built and scored the same defensible way, and the held-out numbers
have to be trustworthy rather than artifacts of how the data was split. Several
methodology choices are baked into the code (`probes.py`, `evaluate.py`,
`data/sandbagging.py`) but were only recorded in comments and project memory. This
ADR makes them explicit so the pipeline can be reproduced without reading the
source.

## Decision

**ACCEPTED** — the methodology below. (The choice of a *default* probe method,
`lr` vs `mms`, is deliberately left open — see Open questions.)

| Decision | Choice | Rationale |
|---|---|---|
| Probe shape | One **unit direction** in hidden space + a **bias**; `score(X) = X·dir − bias` | A probe is portable: train on type A, call `.score()` on type B activations. This is what makes the transfer study clean. |
| Probe methods | **`mms`** (difference-of-means) and **`lr`** (logistic regression), both selectable via `--method` | Two standard linear-probe families; agreement between them is itself evidence the signal is real, not a quirk of one fitter. |
| `lr` standardization | Fit on `StandardScaler`-transformed activations, then **fold the scaler back** into a raw-space direction | Residual dims have wildly different scales, so standardizing is what lets lbfgs converge and the L2 penalty hit features evenly. Folding it back keeps every stored direction in the **same raw space**, so cosines and cross-type scoring are comparable. |
| Direction normalization | Both methods store a **unit** direction; the threshold lives in `bias` | Magnitude is not comparable across types/methods; only the direction and a calibrated boundary are. |
| Held-out split | **Grouped** split: a matched pair's two halves share a `user` key and are kept on the **same side** of the split (`GroupShuffleSplit`) | A pair differs only in the honest/deceptive nudge. If one half trains and its twin tests, the probe can memorize the topic and the AUROC is inflated. Grouping forbids that leak. |
| Label balance | Pairs are matched 1:1, so the split is balanced without `stratify` | `1 = deceptive, 0 = control`, never flipped (project convention). |
| Layer selection | Sweep **every** layer, keep the best-AUROC layer; the `Probe` carries its own `.layer` | Deception is not known a priori to live in one layer; the sweep finds it and the probe stays self-describing. |
| Leak check | **`--permute`** shuffles labels and retrains; a leak-free pipeline collapses to **AUROC ≈ 0.5** | Cheap standing guard against the grouped split (or anything else) leaking signal. See ADR 0003 and the Phase 1 null-result gate. |
| Sandbagging capability filter | **`--filter`**: only count a wrong answer as sandbagging if the model answers correctly under the control prompt | Without it, "wrong" conflates *pretending to be worse* with *actually doesn't know*, and the result is meaningless. Sandbagging-specific. |

## Consequences

- A new deception type is a new module under `src/dprobe/data/` exposing
  `build()`, plus one line in `data/__init__.py`. It inherits all of the above for
  free — no probe/eval code is special-cased per type.
- Because directions are unit vectors in shared raw space, the transfer matrix and
  the **cosine-similarity** sanity check are directly comparable across types and,
  for cosines, across `lr` and `mms`.
- The best-layer-by-AUROC selection takes a max over ~25 layers on the same data it
  reports, which is mildly **optimistic**. Acceptable while datasets are tiny and
  the point is relative comparison; revisit with a nested/held-out layer choice
  once datasets are large enough for the bias to matter.
- Every run records the method, split, flags, and per-layer AUROCs in its
  `meta.json` (ADR 0003), so which methodology produced a number is never ambiguous.

## Open questions

- **Default probe method (`lr` vs `mms`) — left unspecified for now.** The scripts
  currently default to `lr`; project notes have at times treated `mms`
  (difference-of-means) as the locked Phase 1 method. Resolve which is canonical
  before headline results, and update this ADR (and the script default) to match.
- Whether to replace best-layer-by-AUROC with a leakage-free layer choice
  (e.g. pick the layer on a validation fold, report on a separate test fold), or a
  multi-layer ensemble, per the Phase 1 null-result gate.
- Whether `lr`'s `C` (L2 strength) should be tuned per type/layer or held fixed at
  the current default for comparability.
