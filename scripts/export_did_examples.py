"""Export the exact examples a DiD sycophancy run trained on, as JSONL + Markdown.

The run dir only persists ``meta.json`` in git; the per-question verdicts live in
the run's ``run_log.txt`` (the filter trace, ADR 0011) and the example text is
never stored at all -- but the factual builder is deterministic (seeded per ARC
index, ADR 0009), so rebuilding ``build(design='did', source='factual')`` locally
and joining on the trace's truncated question label recovers the full prompts.

Usage (after scp-ing the run's run_log.txt back from OSC into its run dir):

    python -m scripts.export_did_examples results/runs/<run>/run_log.txt

Writes ``did_examples_used.jsonl`` (one record per USED example, machine-readable)
and ``did_examples_used.md`` (human-readable, grouped caved/held) next to the
run_log. ``--include-trimmed`` also exports labeled questions the class balance
dropped. The join key is the trace's 90-char whitespace-collapsed question label
(``tracefmt.short``); any ambiguous or unmatched lines are reported, not guessed.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

from dprobe.data import sycophancy
from dprobe.tracefmt import short

# body lines carry render()'s 2-space indent; see _render_did_log
_VERDICT_RE = re.compile(
    r"^  \[(?P<status>USED|TRIMMED) (?P<outcome>caved|held)\((?P<label>[01])\)\] "
    r"(?P<gate>.*?) :: (?P<q>.*)$")
_ANSWER_RE = re.compile(r"^      pressure-answer (?P<json>.*)$")


def parse_run_log(path: Path) -> list[dict]:
    """Return one record per labeled question in a DiD filter trace, in trace
    order: {status, outcome, label, gate, q_short, pressure_answer}."""
    records: list[dict] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        m = _VERDICT_RE.match(line)
        if m:
            records.append({
                "status": m["status"],
                "outcome": m["outcome"],
                "label": int(m["label"]),
                "gate": m["gate"],
                "q_short": m["q"],
                "pressure_answer": None,
            })
            continue
        m = _ANSWER_RE.match(line)
        if m and records and records[-1]["pressure_answer"] is None:
            try:
                records[-1]["pressure_answer"] = json.loads(m["json"])
            except json.JSONDecodeError:
                records[-1]["pressure_answer"] = m["json"]
    return records


def build_join_index(source: str, split: str) -> dict[str, list]:
    """Rebuild the deterministic dataset and index Examples by the same 90-char
    label the trace printed for them (short of meta['neutral_user'])."""
    examples = sycophancy.build(split=split, design="did", source=source)
    index: dict[str, list] = {}
    for ex in examples:
        index.setdefault(short(ex.meta["neutral_user"]), []).append(ex)
    return index


def to_record(i: int, rec: dict, ex) -> dict:
    return {
        "i": i,
        "label": rec["label"],
        "outcome": rec["outcome"],
        "status": rec["status"],
        "system": ex.system,
        "pressure_user": ex.user,
        "neutral_user": ex.meta["neutral_user"],
        "user_asserted_wrong": ex.meta["matching"],
        "correct_answer": ex.meta["not_matching"],
        "belief_gate": rec["gate"],
        "pressure_answer": rec["pressure_answer"],
    }


def render_markdown(rows: list[dict], run_log: Path, source: str, split: str) -> str:
    caved = [r for r in rows if r["label"] == 1]
    held = [r for r in rows if r["label"] == 0]
    lines = [
        "# DiD sycophancy run -- examples used",
        "",
        f"- trace: `{run_log}`",
        f"- source: `{source}` (split `{split}`), design `did`, gate from trace",
        f"- exported: {len(rows)} examples ({len(caved)} caved / label 1, "
        f"{len(held)} held / label 0)",
        f"- system prompt (both conditions): {rows[0]['system']!r}" if rows else "",
        "",
        "Label 1 = model caved to the user's wrong assertion under pressure; "
        "label 0 = it held the correct answer. Every question passed the sampled "
        "belief gate (model answers correctly unpressured).",
        "",
    ]
    for title, group in (("Caved (label 1)", caved), ("Held (label 0)", held)):
        lines += [f"## {title} -- {len(group)} examples", ""]
        for r in group:
            lines += [
                f"### {r['i']}. [{r['status']}] {r['belief_gate']}",
                "",
                "```",
                r["pressure_user"],
                "```",
                f"- user asserted (wrong): `{r['user_asserted_wrong'].strip()}` | "
                f"correct: `{r['correct_answer'].strip()}`",
                f"- model under pressure: {json.dumps(r['pressure_answer'], ensure_ascii=False)}",
                "",
            ]
    return "\n".join(lines)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("run_log", type=Path,
                    help="path to the run's run_log.txt (or its run dir)")
    ap.add_argument("--source", default="factual",
                    choices=["factual", "factual-small"],
                    help="dataset source the run used (meta.json 'source')")
    ap.add_argument("--split", default="train", choices=["train", "test"])
    ap.add_argument("--include-trimmed", action="store_true",
                    help="also export labeled questions dropped by class balance")
    ap.add_argument("--out-stem", default="did_examples_used",
                    help="basename for the .jsonl/.md written next to the trace")
    args = ap.parse_args(argv)

    run_log = args.run_log / "run_log.txt" if args.run_log.is_dir() else args.run_log
    if not run_log.exists():
        sys.exit(f"no run_log.txt at {run_log} -- scp it back from the run dir on OSC")

    records = parse_run_log(run_log)
    wanted = [r for r in records
              if args.include_trimmed or r["status"] == "USED"]
    if not wanted:
        sys.exit("no USED verdict lines found -- is this a DiD run_log.txt?")

    index = build_join_index(args.source, args.split)
    rows, unmatched, ambiguous = [], [], []
    for rec in wanted:
        candidates = index.get(rec["q_short"], [])
        if len(candidates) == 1:
            rows.append(to_record(len(rows) + 1, rec, candidates[0]))
        elif not candidates:
            unmatched.append(rec["q_short"])
        else:
            ambiguous.append(rec["q_short"])

    out_dir = run_log.parent
    jsonl_path = out_dir / f"{args.out_stem}.jsonl"
    md_path = out_dir / f"{args.out_stem}.md"
    with jsonl_path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
    md_path.write_text(render_markdown(rows, run_log, args.source, args.split),
                       encoding="utf-8")

    n1 = sum(r["label"] == 1 for r in rows)
    print(f"exported {len(rows)} examples ({n1} caved / {len(rows) - n1} held) "
          f"-> {jsonl_path} and {md_path}")
    if unmatched:
        print(f"WARNING: {len(unmatched)} trace lines had no rebuilt match "
              f"(wrong --source/--split, or dataset drift?); first: "
              f"{unmatched[0]!r}")
    if ambiguous:
        print(f"WARNING: {len(ambiguous)} trace labels matched >1 rebuilt "
              f"question (90-char collision); skipped; first: {ambiguous[0]!r}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
