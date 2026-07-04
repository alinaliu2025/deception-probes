# ADR 0007 — Sycophancy Behavioural Split (label = model's own choice)

**Status:** Accepted
**Date:** 2026-07-02

## Context

Both existing sycophancy designs leak at the *construct* level — the label is
readable off the input tokens, so AUROC measures prompt-reading, not deception:

- `completion` (ADR 0001): pasted answer + response-final read → probe reads
  which answer was pasted. Scaled run `2026-07-01T01-27-44Z` AUROC 0.998.
- `framing` (ADR 0006): classes differ in system-prompt text → probe reads
  "is the honesty instruction present". Run `2026-06-30T01-57-43Z` hit
  **AUROC 1.0 at every layer from layer 1** — the signature of an
  input-readable shortcut (a computed state feature ramps mid-late; it is not
  perfectly decodable at layer 1). This violates ADR 0006's own prediction of
  a "lower, believable AUROC with a smooth mid-late ramp". The permutation
  control passed (≈0.5), confirming only that the *split* doesn't leak;
  `--permute` cannot detect construct leaks.

The literature splits into two input-construction families. Contrast-pair /
instruction-prompted work (RepE 2310.01405; ITI 2306.03341, NeurIPS 2023; CAA,
Rimsky et al., ACL 2024) tolerates instruction-differing inputs because its
claims are validated **causally by steering**, not by classification AUROC.
Detection work uses **behavioural labels against independent ground truth**:
MASK (2503.03750) elicits the model's own belief, then checks whether a
pressure prompt makes it contradict that belief; Apollo probes
(Goldowsky-Dill 2502.03407, **preprint**) label roleplay scenarios by what the
model actually does. Our headline claim is a detection/transfer matrix, so we
need the behavioural family.

### Data fix folded in (affects ALL sycophancy designs)

While building this we found that on the HF hub,
`sycophancy_on_philpapers2020.jsonl` is **byte-identical** to
`sycophancy_on_nlp_survey.jsonl` (same sha256 `582860b4…`, same etag,
verified 2026-07-02) — an upstream duplicate, not philosophy data. Every
prior sycophancy run therefore trained on NLP-survey questions **twice** plus
the political file, and never saw philpapers content. The duplicate file is
now removed from `_HF_FILES`. No split leakage resulted (duplicate questions
share the `user` string, so the grouped split kept copies together), but
dataset composition and the fixed-seed train/test split both change from this
commit on: **pre-/post-fix runs are not directly comparable.**

## Decision

Add `design="behavioral"` (third `--design` value, default still
`completion`; nothing existing changes behaviour).

Per model-written-evals question (the prompt already contains the pressure —
a persona bio asserting a view):

1. **Belief elicitation.** Strip the persona via `neutral_question()`
   (curated question-stem markers, verified 100% coverage on both source
   files; unmatched rows dropped and counted). Score both pre-written answers
   with `seq_logprob` under `NEUTRAL_SYSTEM`. Higher = the model's default
   stance.
2. **Gate.** Keep the question only if the belief is the honest answer
   (`answer_not_matching_behavior`). If the model already agrees with the
   user absent the persona, agreeing under pressure isn't sycophancy.
   Analogue of sandbagging's capability filter.
3. **Behaviour + label.** Score both answers under the full pressure prompt.
   **Label 1** = model flips to the user-matching answer against its own
   elicited belief (deceptive condition). **Label 0** = it holds its belief
   (control). Invariant preserved.
4. **Read position:** prompt-final on the pressure prompt, no completion.
   Both classes share the identical prompt construction — there is no
   inserted text for the probe to shortcut on. The probe is predictive
   ("is the model about to cave").

Mechanics: `build(design="behavioral")` emits ONE example per question with
the `-1` unlabeled sentinel; `behavioral_filter` (mandatory — `train_one`
errors without `--filter`) assigns labels, then balances classes by seeded
subsampling of the majority. `data.FILTERS["sycophancy"]` is now a
design-dispatching `sycophancy_filter`. Filter stats (gate drops,
pre-balance **base rate**, per-class n) go into `meta.json` as
`filter_stats`.

## Consequences

- Expected on a trustworthy run: AUROC **well below ceiling** with a mid-late
  ramp. AUROC ≈1.0 from layer 1 again would mean something still leaks.
- Check the printed base rate before scaling: if the model rarely caves (or
  nearly always caves), balanced n collapses and more raw questions are
  needed.
- Cost: 4 teacher-forced forwards per question (2 belief + 2 pressure),
  one-time, like the other filters.
- Residual caveat: classes can differ in *question content* (topics the model
  caves on vs. not) — a correlational confound, far weaker than the lexical
  one. Follow-up if it matters: within-question temperature sampling so the
  same question appears in both classes.
- These are opinion questions: "belief" = the model's default stance, not
  world truth — exactly MASK's honesty-vs-accuracy distinction.
- The philpapers dedupe changes the dataset for `completion` and `framing`
  too; old run metas remain valid records of what *was* run.

## Open questions

- Log-prob choice vs sampled generation: labels come from comparing two
  teacher-forced answers, not free generation. Cheap and deterministic, but a
  sampled-rollout variant (temperature > 0, parse the answer) would let one
  question appear in both classes and is the natural scale-up.
- Belief stability: single neutral elicitation per question; no
  self-consistency check across paraphrases yet.
