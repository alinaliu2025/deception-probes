# ADR 0001 — Activation Extraction Position

**Status:** Accepted  
**Date:** 2026-06-27

## Context

Probes need a position in the token sequence to extract residual-stream activations. Two candidates:

| Position | What it captures | Dataset fit |
|---|---|---|
| **Prompt-final** (last token before generation) | Model's internal state *before* speaking — "is it about to capitulate?" | Current seed design; hard to pair with CAA |
| **Response-final** (last token of completion) | Model's state *after* speaking — "did it capitulate?" | CAA contrastive pairs plug in directly |
| **Deception-onset token** | First token where sycophantic vs honest responses diverge | Scientifically strongest; requires token-level alignment across pairs |

Literature note: some steering-vector work targets the token where divergence *starts* (e.g. the first disagreement token), not the final token. This is harder to implement but captures the earliest internal signal of deception.

## Decision

**ACCEPTED** — response-final (last token of assistant completion).

Rationale: CAA (nrimsky/CAA) provides 1000 contrastive pairs structured for response-final extraction. Prompt-final would require discarding the completion structure entirely, losing the primary advantage of the dataset.

**Future experiment — deception-onset token:** Find the first token where sycophantic and honest completions diverge; extract activations there. Theoretically captures the earliest internal signal of deception rather than its aftermath. Stronger scientific claim, harder to implement (requires token-level alignment across pairs). Good candidate for a follow-up contribution once response-final baseline is established.

## Consequences

- `activations.py::build_prompt` must accept an optional `completion` argument
- `Example` gains `completion: str | None` field (None → prompt-final, backward-compat with seed data)
- Data source: `Anthropic/model-written-evals` sycophancy JSONL files via HuggingFace `datasets` library; fixed-seed 95/5 split (~9500 train / ~500 test), split by prompt to prevent leakage
- Fields used: `question` → `Example.user`, `answer_matching_behavior` → `Example.completion` (label=1), `answer_not_matching_behavior` → `Example.completion` (label=0)
- Deception-onset experiment noted for future phase; worth revisiting when scaling to Gemma 2 2B on OSC

## Open questions

None.
