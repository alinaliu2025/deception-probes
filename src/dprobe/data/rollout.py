"""Generic within-question rollout engine (ADR 0008 mechanics, ADR 0011 extraction).

This generalises sycophancy's `rollout_filter` so other deception types can use
the within-question contrast without duplicating it. `sycophancy.py` is left
untouched (it keeps its own copy for now); the ONLY thing imported from it is
`parse_choice`, so there is a single implementation of the answer parser.

The engine is parameterised by ONE thing: how to build the UNPRESSURED prompt
for the capability/belief gate. For sycophancy the pressure lives in the user
turn (persona + asserted view; neutral = stripped user text). For sandbagging
it lives in the system prompt (sandbag instruction; neutral = control system,
same user text). Everything downstream -- gate, sampling, parsing, ambivalence
keep, within-question balance, letter balance, stats, run log -- is identical.

Contract for input examples (built by the type's `build(design='rollout')`):
  label == UNLABELED (-1)  sentinel; running the engine is MANDATORY
  ex.system / ex.user      the PRESSURE prompt
  meta['matching']         " (X)" tag of the DECEPTIVE answer (label 1)
  meta['not_matching']     " (X)" tag of the honest/correct answer (label 0)
  meta['neutral_user']     unpressured question text (for logs + gate builders)

Module-level knobs mirror sycophancy's and are set by scripts/train_one.py for
non-sycophancy types (sycophancy's own knobs still live in its module).
"""

from __future__ import annotations

import random
import re

from .base import Example
from .sycophancy import parse_choice

UNLABELED = -1
_SEED = 42

# sampling knobs (same defaults/rationale as sycophancy.py, see its comments)
ROLLOUT_N = 8
ROLLOUT_TEMPERATURE = 1.0
ROLLOUT_MAX_NEW_TOKENS = 24
ROLLOUT_PREFIX_MODE = "commit"  # 'commit' = bare " (X)"; 'text' = wording ablation

# gate knobs ('logprob' = one teacher-forced comparison; 'sampled' = generate
# GATE_N answers unpressured, keep if correct on >= GATE_THRESHOLD)
GATE_MODE = "logprob"
GATE_N = 20
GATE_THRESHOLD = 0.9
GATE_TEMPERATURE = 0.7

# 'strict' = sycophancy.parse_choice, first "(X)" only (ADR 0008). 'lenient'
# adds fallbacks for answers that commit without the parenthesised format
# (" B) ...", "Answer: B", "The answer is B") -- motivated by the 2026-07-13
# instructed-arm run where 56% of rollouts parsed as nothing because the model,
# told to "sound like it is trying", wrote prose past the token cap. Opt-in;
# never changes what strict WOULD have parsed (strict is tried first).
PARSE_MODE = "strict"
UNPARSED_LOG_SAMPLES = 20  # raw unparsed rollout texts kept for the run log

_CHOICE_RE = re.compile(r"\(([A-Z])\)")
# lenient fallbacks, tried in order after the strict "(X)" pattern misses
_LENIENT_RES = (
    re.compile(r"^\s*([A-Z])[\.\):,\s]"),                        # " B) nectar" / "B."
    re.compile(r"\banswer\s*(?:is|:)?\s*\(?([A-Z])\b", re.I),    # "the answer is B"
    re.compile(r"\b([A-Z])\s*\)"),                               # "B )" / mid-text "B)"
)


def parse_choice_lenient(text: str, matching: str, not_matching: str):
    """Strict parse first; on a miss, try the fallback patterns. Same return
    contract as sycophancy.parse_choice. A fallback letter must belong to one of
    the two known tags, otherwise the rollout stays unparsed."""
    choice, end = parse_choice(text, matching, not_matching)
    if choice is not None:
        return choice, end
    for pat in _LENIENT_RES:
        m = pat.search(text)
        if not m:
            continue
        tag = f"({m.group(1)})"
        if tag in matching:
            return "matching", m.end()
        if tag in not_matching:
            return "not_matching", m.end()
    return None, None


def _parse(text: str, matching: str, not_matching: str):
    fn = parse_choice_lenient if PARSE_MODE == "lenient" else parse_choice
    return fn(text, matching, not_matching)


def _gate_logprob(model, tokenizer, device, neutral_ex: Example,
                  matching: str, not_matching: str) -> bool:
    """Deterministic gate: on the unpressured prompt, does the model rate the
    honest/correct answer above the deceptive one?"""
    from ..activations import build_prompt, seq_logprob

    p = build_prompt(tokenizer, neutral_ex)
    return (seq_logprob(model, tokenizer, p, not_matching, device)
            > seq_logprob(model, tokenizer, p, matching, device))


def _gate_sampled(model, tokenizer, device, neutral_ex: Example,
                  matching: str, not_matching: str) -> bool:
    """Sampled gate: GATE_N unpressured answers; keep the question only if the
    correct answer wins on >= GATE_THRESHOLD of the parseable samples."""
    import torch

    from ..activations import build_prompt

    p = build_prompt(tokenizer, neutral_ex)
    inputs = tokenizer(p, return_tensors="pt").to(device)
    with torch.no_grad():
        gen = model.generate(
            **inputs, do_sample=True, temperature=GATE_TEMPERATURE,
            num_return_sequences=GATE_N, max_new_tokens=ROLLOUT_MAX_NEW_TOKENS,
            pad_token_id=tokenizer.pad_token_id,
        )
    new = gen[:, inputs.input_ids.shape[1]:]
    decoded = tokenizer.batch_decode(new, skip_special_tokens=True)
    correct = parseable = 0
    for t in decoded:
        choice, _ = parse_choice(t, matching, not_matching)
        if choice is None:
            continue
        parseable += 1
        if choice == "not_matching":
            correct += 1
    return parseable > 0 and correct / parseable >= GATE_THRESHOLD


def passes_gate(model, tokenizer, device, ex: Example, neutral_example_fn) -> bool:
    """Capability/belief gate on the UNPRESSURED prompt built by neutral_example_fn."""
    neutral_ex = neutral_example_fn(ex)
    args = (model, tokenizer, device, neutral_ex,
            ex.meta["matching"], ex.meta["not_matching"])
    return _gate_sampled(*args) if GATE_MODE == "sampled" else _gate_logprob(*args)


def run_rollout_filter(model, tokenizer, device, examples: list[Example], *,
                       deception_type: str, neutral_example_fn,
                       gate_fail_reason: str) -> list[Example]:
    """Label rollout-design examples per SAMPLED rollout (ADR 0008 mechanics).

    Per question: gate, then ROLLOUT_N samples at ROLLOUT_TEMPERATURE under the
    identical pressure prompt. Each parsed rollout becomes its own Example:
    label 1 = chose meta['matching'] (the deceptive answer), 0 = chose
    meta['not_matching']; the read prefix rides in meta['assistant_prefix'].
    Only AMBIVALENT questions survive, balanced within question, then across
    the deceptive-answer letter. Stats land in run_rollout_filter.last_stats,
    the per-question trace in .last_log (train_one picks both up).
    """
    import torch

    from ..activations import build_prompt, default_batch_size

    torch.manual_seed(_SEED)  # seeded, but GPU sampling is not portable-exact
    rng = random.Random(_SEED)

    # stage 1: gate
    gated: list[Example] = []
    gate_failed: list[Example] = []
    total = len(examples)
    for n, ex in enumerate(examples, 1):
        (gated if passes_gate(model, tokenizer, device, ex, neutral_example_fn)
         else gate_failed).append(ex)
        if n % 500 == 0:
            print(f"  rollout gate {n}/{total} questions", end="\r", flush=True)
    print()

    # stage 2: batched sampling under the pressure prompt
    gen_batch = max(1, default_batch_size(device) // max(1, ROLLOUT_N))
    texts_per_q: list[list[str]] = []
    for start in range(0, len(gated), gen_batch):
        chunk = gated[start:start + gen_batch]
        prompts = [build_prompt(tokenizer, ex) for ex in chunk]
        inputs = tokenizer(prompts, return_tensors="pt", padding=True).to(device)
        with torch.no_grad():
            out = model.generate(
                **inputs, do_sample=True, temperature=ROLLOUT_TEMPERATURE,
                num_return_sequences=ROLLOUT_N,
                max_new_tokens=ROLLOUT_MAX_NEW_TOKENS,
                pad_token_id=tokenizer.pad_token_id,
            )
        new = out[:, inputs.input_ids.shape[1]:]
        decoded = tokenizer.batch_decode(new, skip_special_tokens=True)
        for i in range(len(chunk)):
            texts_per_q.append(decoded[i * ROLLOUT_N:(i + 1) * ROLLOUT_N])
        done = min(start + gen_batch, len(gated))
        print(f"  rollout sampling {done}/{len(gated)} questions", end="\r", flush=True)
    print()

    # stage 3: parse, keep ambivalent questions, balance within question
    n_unparsed = 0
    n_deterministic = 0
    unparsed_samples: list[str] = []
    pairs_by_letter: dict[str, list[tuple[Example, Example]]] = {}
    q_records: list[dict] = []
    for ex, texts in zip(gated, texts_per_q):
        deceptive, honest = [], []
        parsed = []
        seen: set[tuple[int, str]] = set()
        for t in texts:
            choice, end = _parse(t, ex.meta["matching"], ex.meta["not_matching"])
            if choice is None:
                n_unparsed += 1
                parsed.append("unparsed")
                if len(unparsed_samples) < UNPARSED_LOG_SAMPLES:
                    unparsed_samples.append(t)
                continue
            label = 1 if choice == "matching" else 0
            parsed.append("deceptive" if label == 1 else "honest")
            if ROLLOUT_PREFIX_MODE == "commit":
                prefix = ex.meta["matching"] if label == 1 else ex.meta["not_matching"]
            else:  # 'text': wording-shortcut ablation arm (ADR 0009 addendum)
                prefix = t[:end]
            if (label, prefix) in seen:
                continue
            seen.add((label, prefix))
            child = Example(
                ex.system, ex.user, label, deception_type,
                meta={**ex.meta, "assistant_prefix": prefix},
            )
            (deceptive if label == 1 else honest).append(child)
        k = min(len(deceptive), len(honest))
        m = _CHOICE_RE.search(ex.meta["matching"])
        letter = m.group(1) if m else "?"
        q_records.append({
            "question": ex.meta.get("neutral_user", ex.user),
            "user": ex.user,
            "deceptive_letter": letter,
            "parsed": parsed,
            "kept_pairs": k,
            "outcome": "ambivalent" if k > 0 else "single-class",
        })
        if k == 0:
            n_deterministic += 1
            continue
        pairs_by_letter.setdefault(letter, []).extend(
            zip(rng.sample(deceptive, k), rng.sample(honest, k)))

    # stage 4: letter-side balance
    pairs_before = sum(len(v) for v in pairs_by_letter.values())
    out: list[Example] = []
    if pairs_by_letter:
        floor = min(len(v) for v in pairs_by_letter.values())
        for letter, pairs in sorted(pairs_by_letter.items()):
            for dec, hon in rng.sample(pairs, floor):
                out.extend((dec, hon))

    run_rollout_filter.last_stats = {
        # type-specific build variants (e.g. sandbagging's pressure arm,
        # ADR 0011 addendum) ride in meta; surface them in the log SUMMARY
        **({"pressure": examples[0].meta["pressure"]}
           if examples and "pressure" in examples[0].meta else {}),
        "questions_in": total,
        "dropped_gate_failed": len(gate_failed),
        "sampled_questions": len(gated),
        "gate_mode": GATE_MODE,
        "gate_n": GATE_N if GATE_MODE == "sampled" else None,
        "gate_threshold": GATE_THRESHOLD if GATE_MODE == "sampled" else None,
        "n_rollouts": ROLLOUT_N,
        "temperature": ROLLOUT_TEMPERATURE,
        "max_new_tokens": ROLLOUT_MAX_NEW_TOKENS,
        "prefix_mode": ROLLOUT_PREFIX_MODE,
        "parse_mode": PARSE_MODE,
        "unparsed_rollouts": n_unparsed,
        "single_class_questions": n_deterministic,
        "ambivalent_questions": len(gated) - n_deterministic,
        "pairs_before_letter_balance": pairs_before,
        "pairs_per_letter": {k: len(v) for k, v in sorted(pairs_by_letter.items())},
        "n_examples": len(out),
    }
    print(f"  rollout filter [{deception_type}]: {total} in | {len(gate_failed)} "
          f"gate-dropped | {n_deterministic} single-class dropped | "
          f"{pairs_before} pairs from {len(gated) - n_deterministic} ambivalent "
          f"questions | letter-balanced to {len(out)} examples")
    kept_users = {e.user for e in out}
    run_rollout_filter.last_log = render_log(
        run_rollout_filter.last_stats, gate_failed, q_records, kept_users,
        gate_fail_reason, unparsed_samples=unparsed_samples)
    return out


def render_log(stats, gate_failed, q_records, kept_users,
               gate_fail_reason: str, unparsed_samples: list[str] | None = None) -> str:
    """Human-readable per-question trace (written to run_dir/run_log.txt)."""
    def short(q, n=90):
        q = " ".join(q.split())
        return q if len(q) <= n else q[:n - 1] + "…"

    lines = ["=" * 78, "ROLLOUT FILTER RUN LOG", "=" * 78, ""]
    lines.append("SUMMARY")
    for k, v in stats.items():
        lines.append(f"  {k}: {v}")
    lines.append("")

    lines.append(f"GATE FAILURES ({len(gate_failed)} questions dropped -- "
                 f"{gate_fail_reason})")
    for ex in gate_failed:
        lines.append(f"  [gate-fail] {short(ex.meta.get('neutral_user', ex.user))}")
    lines.append("")

    lines.append(f"GATED QUESTIONS ({len(q_records)} sampled under pressure)")
    lines.append("  legend: rollouts parsed as deceptive(1)/honest(0)/unparsed; "
                 "'kept' = examples used after within-question balance; "
                 "USED = survived letter balance, DROPPED = trimmed by it")
    for r in q_records:
        counts = {c: r["parsed"].count(c) for c in ("deceptive", "honest", "unparsed")}
        if r["outcome"] == "single-class":
            status = "SINGLE-CLASS"
        elif r["user"] in kept_users:
            status = "USED"
        else:
            status = "AMBIVALENT-BUT-TRIMMED"
        lines.append(
            f"  [{status}] deceptive-letter={r['deceptive_letter']} "
            f"rollouts={counts['deceptive']}d/{counts['honest']}h/{counts['unparsed']}u "
            f"kept_pairs={r['kept_pairs']} :: {short(r['question'])}")
    lines.append("")
    if unparsed_samples:
        lines.append(f"UNPARSED ROLLOUT SAMPLES (first {len(unparsed_samples)}; "
                     "raw generated text that produced no recognisable choice -- "
                     "read these before changing the parser or token cap)")
        for t in unparsed_samples:
            lines.append(f"  [unparsed] {short(t, 160)!r}")
        lines.append("")
    lines.append("=" * 78)
    return "\n".join(lines)
