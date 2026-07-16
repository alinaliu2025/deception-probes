# Session review — July 10–11, 2026 (with Claude)

Personal notes. Lives in `docs/` which is gitignored (my private folder), so this
never clutters the branch.

## Where things stood at the start

- Jack had taken sycophancy through four designs (completion → framing →
  behavioral → rollout), killing the first three with controls (permutation,
  neutral-read). Current pipeline: within-question rollouts on ARC "factual"
  source, capability/belief gate, letter balance, commit-prefix read (ADR
  0008/0009). Headline: mms AUROC 1.0 @ L13 (2026-07-06T11-07-48Z).
- Steering (ADR 0010) had run twice; Jack's verdict since: the direction is
  **correlational, not causal** — the add-pass advantage over a random
  direction is confounded by parse-rate collapse, and the ablate pass is null.
- Sandbagging and omission were untouched 12-item seeds. My only run was a
  laptop smoke test. I felt behind; turned out the other ⅔ of the project
  (and the headline transfer matrix) were simply unstarted.

## What I built (branch `sandbagging-rollout`, pushed)

1. **ADR 0011 + sandbagging rollout pipeline.** The old sandbagging design was
   a control-vs-sandbag system-prompt contrast = the exact instruction-token
   leak that killed sycophancy's framing design. Rebuilt it on Jack's rollout
   mechanics with pressure in the SYSTEM prompt and the capability filter as
   the gate:
   - `src/dprobe/data/mcq.py` — shared ARC loader, seed-identical to
     sycophancy's factual source (same 95/5 split + letter sides,
     byte-for-byte, tested) so the transfer matrix can't contaminate
     train/test across types.
   - `src/dprobe/data/rollout.py` — Jack's rollout engine generalized;
     parameterized only by how the unpressured gate prompt is built.
     `sycophancy.py` untouched (only `parse_choice` imported).
   - `src/dprobe/data/sandbagging.py` — rollout design + legacy baseline kept.
   - `train_one.py` plumbing, tests (`test_sandbagging_rollout.py`),
     `.claude/docs/adr/0011-...md` (Proposed).
   - Smoke run passed on my laptop: 36 examples end-to-end, gate dropped 3.
   - Run: `python -m scripts.train_one --type sandbagging --design rollout
     --source factual-small --filter`
2. **crossread — the 2×2 read comparison.** `scripts/crossread.py` (+ tests,
   `scripts/osc/crossread.sbatch`, `scripts/crossread_plot.py`): one rollout
   pass, snapshot every rollout under BOTH reads (text = full generated answer
   cut at the "(X)" ≈ generation-time state by replay determinism; commit =
   bare " (X)" reconstruction), train a probe on each, test each on both,
   per layer, same grouped split.
3. **Fixes found along the way:**
   - `.gitignore`: my bare `docs` line was also re-ignoring `.claude/docs`
     (the team's ADRs). Root-anchored to `/docs/`.
   - `scripts/osc/train_one.sbatch` exported `DPROBE_MODEL` but never passed
     it — my 07-09 OSC "7B" smoke silently ran 0.5B. Now passes `--model`.
   - `tests/conftest.py` so pytest can import from `scripts/`.
   - Recurring `.git/*.lock` failures (VS Code / sandbox sessions):
     `rm -f .git/*.lock` when no git command is actually running.

## Discoveries (things nobody knew before this session)

1. **The read position is the `)` after the letter, not the letter itself** —
   in both prefix modes. `verify_read_positions` passes anyway because it only
   asserts the letter is within the last 4 tokens.
2. **The two reads tokenize the answer differently.** Sampled text: `(A` + `)`
   (paren+letter merged). Reconstructed commit prefix: ` (` + `A` + `)`. So the
   commit "reenactment" doesn't just drop the preamble; it reads a
   differently-tokenized answer. Concrete mechanism for generation-time vs
   reconstructed reads to disagree.
3. **Crossread results** (7B, factual, 1500q, mms, 476 examples / 142 test
   rows; run `2026-07-11T20-11-22Z_crossread_mms_0c4d6ee_p0333`, OSC job
   50377142, 13 min):
   - Diagonals reproduce Jack's runs (C→C 1.0 @ L15; T→T = his 0.98 curve).
   - **Early/mid text signal is a read-specific artifact**: T→T 0.93 @ L5
     while T→C ≈ 0.67 and C→C ≈ chance. That's the preamble-wording shortcut,
     worth ~+0.27 AUROC below ~L12.
   - **The deep signal is shared**: C→T 0.999 @ L15 — commit-trained probes
     transfer to generation-time states nearly perfectly from L13 up.
   - **Asymmetry**: text-trained probe transfers 0.857 @ L15 but 0.96–0.995 @
     L19–21 (at L15 it learns signal+wording blend; deeper, wording fades and
     it converges on the shared feature).
   - Practical: train on the commit read; probes from generated-token reads
     should come from L19–21, not L15. Jack's "which token am I reading"
     question: matters below ~L12, doesn't for the deep signal.

## Concepts I actually understand now

- A **probe** is a stored direction (list of numbers) + threshold; score =
  dot product. mms = mean(deceptive) − mean(honest), normalized.
- The **hidden state at a token is not the token**: it's everything the model
  computed about the whole context up to that instant (why letter-balanced
  data can still carry signal at the commit token).
- **Nothing is captured during generation** — both reads are replays.
  Transformers recompute identical states given identical tokens; that's why
  the text read ≈ generation-time (and why the tokenization drift matters).
- The **leak taxonomy**: instruction-contrast leak (probe reads which prompt),
  content confound (question predicts label → within-question rollouts fix),
  wording shortcut (preamble phrasing → commit prefix fix), letter shortcut
  (→ letter balance), split leak (→ permutation control ≈ 0.5).
- **Layer fingerprint**: high AUROC at L1–3 = lexical/surface shortcut;
  chance early + rise mid-stack = computed/abstract signal.
- **Transfer cells** = train probe on type A, AUROC on type B (detection
  transfer). Direction cosines are a related but different analysis.
- **The correctness confound** (the current big open problem): in both
  sycophancy-factual and my sandbagging design, label 1 ≡ wrong answer. A
  probe reading "my answer is wrong" scores perfectly on both AND transfers —
  the headline transfer matrix could measure wrongness, not deception.
  Proposed control: **honest-error test** — probe unpressured wrong answers
  (no deception present); if the probe fires, it's reading wrongness.
- Jack's steering verdict: detection ≠ causation; the L15 direction elicits
  wrong answers only alongside parse-rate collapse, and ablation ≈ random
  control → treat all current probes as correlational.
- Logistics: commit = save to local logbook, push = publish to GitHub; OSC
  login nodes are for sbatch/squeue/git only; OSC bills actual runtime
  (crossread cost ≈ 0.2 GPU-hours), `OSCusage` shows the balance.

## Open next steps

1. Send Dr. Zhu the ADR 0011 message (draft ready; every claim now true:
   branch pushed, smoke passed, sycophancy.py untouched). Ask about the two
   flagged open questions: instructed-vs-spontaneous sandbagging, and whether
   the sandbag prompt should state only the evaluation incentive.
2. Post the crossread result + figure to Slack (draft ready), including the
   scope line: this closes the read question, NOT the wrongness question.
3. Build + run the **honest-error control** (next experiment; tests Jack's
   own hypothesis cheaply; also pre-registers the control my transfer matrix
   will need).
4. After sign-off: 7B sandbagging run on OSC (~1500q) + `--permute` control +
   `--rollout-prefix text` ablation, per ADR 0011's validation plan.
5. Then: first sycophancy↔sandbagging transfer cells (`scripts/compare.py`
   needs pointing at rollout-design datasets), steering for sandbagging later.
6. Housekeeping: crossread currently lives on `sandbagging-rollout` (kept
   deliberately — it depends on the shared engine + sbatch files); offer to
   split it if the team wants to merge it separately.
