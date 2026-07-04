# Run catalog — commands, flags, and past results

Single reference for **every command the pipeline can run** and **every recorded run to
date**. Generated from the git-tracked `meta.json` files across the sycophancy branches
(`sycophancy-factual`, `sycophancy-no-leak`) at the time they were merged into
`first-sycophancy`.

- Label convention: **1 = deceptive, 0 = control** (never flipped — see `CLAUDE.md`).
- Metric: **AUROC** (0.5 = chance, 1.0 = perfect).
- Design rationale lives in the ADRs (`.claude/docs/adr/0006`–`0009`); this file is the
  operational "what can I run / what has been run" index.

---

## 1. Commands & flags

### `scripts.train_one` — train + report one deception type

```bash
python -m scripts.train_one --type {sycophancy|sandbagging|omission} [flags]
```

| Flag | Values | Meaning |
|---|---|---|
| `--type` | `sycophancy` \| `sandbagging` \| `omission` | which deception type (required) |
| `--method` | `mms` \| `mms_std` \| `lr` | probe fit: diff-in-means / standardized diff-in-means / logistic regression (`lr` default, needed for the comparison grid) |
| `--max-examples N` | int | cap dataset; drops **whole pairs** so balance is kept |
| `--permute` | flag | permutation control: shuffle labels, expect AUROC ≈ 0.5 (higher ⇒ leak) |
| `--filter` | flag | behaviour filter. **Mandatory for sandbagging** (capability filter); sycophancy `framing` keep-if-flips; sycophancy `behavioral` assigns the labels |
| `--design` | `completion` \| `framing` \| `behavioral` \| `rollout` | sycophancy construction (ADR 0006/0007/0008) |
| `--read-prompt` | `neutral` | behavioral-design confound control (neutral read) — ADR 0008 |
| `--source` | `factual` | sycophancy behavioral/rollout only: ARC factual MCQs, user asserts a wrong answer (ADR 0009) |
| `--rollouts N` | int | rollout-design: samples per question |
| `--temperature T` | float | rollout-design sampling temperature |
| `--max-new-tokens N` | int | rollout-design generation length (ADR 0009 addendum) |
| `--rollout-prefix` | `commit` \| `text` | read-prefix mode; `text` = wording-shortcut ablation (ADR 0009 addendum) |

### `scripts.compare` — the cross-type transfer study

```bash
python -m scripts.compare --method lr
```

Builds the cross-type **transfer matrix** + direction cosines across all types.

### Design status (sycophancy)

| Design | Status | One-line verdict |
|---|---|---|
| `completion` | rejected | leaks (AUROC ≈ 0.998–0.999) — ADR 0006 |
| `framing` | rejected | leaks (AUROC 1.000; permute control 0.593) — ADR 0007 |
| `behavioral` | rejected | content-confounded; **neutral-read control hit AUROC 1.0** — ADR 0008 |
| `rollout` | Proposed, pending team sign-off | within-question fix — ADR 0008 |
| `--source factual` | Proposed, pending sign-off | ARC factual source for rollout — ADR 0009 |

---

## 2. Past runs (all recorded `meta.json`)

Every tracked run, oldest → newest. `model` is the Qwen2.5-Instruct size. `filter`/`permute`
blank = off. AUROC is the best-layer value.

| # | date (UTC) | design | source | read | method | model | filter | permute | N | layer | AUROC | sha | on-branch |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 1 | 2026-06-28 19:53 | - | - | - | mms | 0.5B |  |  | 5000 | 20 | 0.534 | `9e09e4a` | factual |
| 2 | 2026-06-28 20:15 | - | - | - | lr | 0.5B |  |  | 5000 | 21 | 0.839 | `8719e35` | factual |
| 3 | 2026-06-28 20:27 | - | - | - | mms | 0.5B |  |  | 5000 | 20 | 0.534 | `1b8fcab` | factual |
| 4 | 2026-06-28 21:30 | - | - | - | mms_std | 0.5B |  |  | 5000 | 20 | 0.536 | `e3b80f4` | factual |
| 5 | 2026-06-28 22:11 | - | - | - | lr | 3B |  |  | 200 | 29 | 0.875 | `b5c3232` | factual |
| 6 | 2026-06-29 00:25 | - | - | - | lr | 7B |  |  | 5000 | 26 | 0.999 | `3da7279` | factual |
| 7 | 2026-06-30 01:44 | - | - | - | lr | 7B |  |  | 200 | 19 | 0.960 | `14fa1e3` | factual |
| 8 | 2026-06-30 01:57 | framing | - | - | lr | 7B | yes |  | 224 | 1 | 1.000 | `683bea3` | factual |
| 9 | 2026-06-30 02:18 | framing | - | - | lr | 7B | yes | yes | 224 | 1 | 0.593 | `683bea3` | factual |
| 10 | 2026-07-01 01:27 | completion | - | - | lr | 7B |  |  | 20000 | 18 | 0.998 | `683bea3` | factual |
| 11 | 2026-07-02 16:54 | behavioral | - | - | lr | 0.5B | yes |  | 18 | 2 | 1.000 | `90b3404` | factual |
| 12 | 2026-07-02 17:06 | behavioral | - | - | lr | 7B | yes |  | 418 | 23 | 0.985 | `76e41fe` | factual |
| 13 | 2026-07-02 17:10 | behavioral | - | - | lr | 7B | yes | yes | 196 | 24 | 0.560 | `601710d` | factual |
| 14 | 2026-07-02 17:30 | behavioral | - | neutral | lr | 0.5B | yes |  | 18 | 1 | 1.000 | `f3a1d8e` | factual |
| 15 | 2026-07-02 17:34 | behavioral | - | neutral | lr | 7B | yes |  | 104 | 1 | 1.000 | `0c88872` | o/no-leak |
| 16 | 2026-07-02 17:58 | rollout | - | pressure | lr | 0.5B | yes |  | 24 | 1 | 1.000 | `0c88872` | factual |
| 17 | 2026-07-02 18:05 | rollout | - | pressure | lr | 7B | yes |  | 8 | 1 | 1.000 | `3c4d610` | o/no-leak |
| 18 | 2026-07-02 18:07 | rollout | - | pressure | lr | 7B | yes |  | 6 | 1 | 1.000 | `3c4d610` | o/no-leak |
| 19 | 2026-07-02 18:15 | rollout | - | pressure | lr | 7B | yes |  | 16 | 22 | 0.556 | `3c4d610` | o/no-leak |
| 20 | 2026-07-02 18:48 | rollout | factual | pressure | lr | 0.5B | yes |  | 84 | 5 | 1.000 | `3c4d610` | factual |
| 21 | 2026-07-02 18:56 | rollout | factual | pressure | lr | 7B | yes |  | 16 | 1 | 1.000 | `7cec263` | factual |
| 22 | 2026-07-02 19:38 | rollout | factual | pressure | lr | 7B | yes |  | 96 | 4 | 1.000 | `72bd7d9` | factual |
| 23 | 2026-07-02 19:58 | rollout | factual | pressure | lr | 0.5B | yes |  | 64 | 5 | 1.000 | `e69a7a2` | factual |
| 24 | 2026-07-02 20:21 | rollout | factual | pressure | lr | 7B | yes |  | 456 | 4 | 1.000 | `0ae9fad` | factual |
| 25 | 2026-07-02 20:31 | rollout | factual | pressure | lr | 7B | yes |  | 400 | 15 | 1.000 | `13b243f` | factual |

### Reading the table — caveats

- **Chance vs. leak.** The early un-designed `lr` runs on the 7B model (rows 6–7) and every
  `completion`/`framing` run score ~1.0 — these are the leaks the ADRs were written to kill,
  **not** evidence of a working probe.
- **Behavioral is confounded.** Rows 14–15 are the neutral-read control: with the deceptive
  framing stripped out, AUROC is still **1.000**, so the behavioral probe keys on question
  content, not deception (ADR 0008).
- **Rollout results are not yet trustworthy.** The pressure-read rollout runs mostly hit
  1.000; the one that dropped to 0.556 (row 19) differs only in N. Combined with the tiny
  N (6–96 for most), treat these as pipeline checks, not findings.
- **`13b243f` = "bad factual run".** Row 25 (factual, N=400, AUROC 0.9997) ran on
  **Qwen2.5-7B**, not the locked 0.5B model, and is flagged bad in its commit message.
- **Seed datasets are 12 items** (`CLAUDE.md`): any small-N AUROC can look great or terrible
  by luck. Scale before drawing conclusions.

To regenerate this table after new runs, re-read the `results/runs/*/meta.json` files (only
`meta.json` is git-tracked; `.npy`/`.png` are regenerable and gitignored).
