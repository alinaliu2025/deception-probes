#!/usr/bin/env python3
"""
run_roleplay_through_model.py

1. Calls scheming_data_parser.py to (re)build data/roleplaying/dataset_mcq.jsonl
2. Loops over that jsonl and runs every prompt through a local HF model
3. Parses the (A)/(B) choice out of each generation and writes the results

Running smoke test command line: 
python src/dprobe/scheming/run_roleplay_through_model.py \
  --parser-script src/dprobe/scheming/scheming_data_parser.py \
  --model 0.5b --full --limit 20

Running on OSC sbatch command line:

Usage:
    python run_roleplay_through_model.py --model 0.5b
    python run_roleplay_through_model.py --model 7b --batch-size 16
    python run_roleplay_through_model.py --skip-parse --limit 20
"""

from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys
from pathlib import Path

DEFAULT_YAML = "data/roleplaying/dataset.yaml"
DEFAULT_JSONL = "data/roleplaying/dataset_mcq.jsonl" # Parsed model input
DEFAULT_OUT = "runs/roleplay_choices.jsonl" # Parsed model output
PARSER_SCRIPT = "scheming_data_parser.py"

MODEL_ALIASES = {
    "0.5b": "Qwen/Qwen2.5-0.5B-Instruct",
    "7b": "Qwen/Qwen2.5-7B-Instruct",
}
DEFAULT_MODEL = "0.5b"


# --------------------------------------------------------------------------- #
# step 1: build the jsonl
# --------------------------------------------------------------------------- #
def build_jsonl(yaml_path: str, jsonl_path: str, parser_script: str, full: bool) -> None:
    """Shell out to scheming_data_parser.py."""
    script = Path(parser_script)
    if not script.exists():
        raise FileNotFoundError(f"{script} not found (pass --parser-script)")

    cmd = [sys.executable, str(script), "-i", yaml_path, "-o", jsonl_path, "-f", "jsonl"]
    if full:
        cmd.append("--full")
    print(f"$ {' '.join(cmd)}")
    subprocess.run(cmd, check=True)


def load_prompts(jsonl_path: str, limit: int | None = None) -> list[dict]:
    path = Path(jsonl_path)
    if not path.exists():
        raise FileNotFoundError(f"{path} not found; run without --skip-parse to build it")

    records = []
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if line:
            records.append(json.loads(line))
    if limit is not None:
        records = records[:limit]
    if not records:
        raise ValueError(f"{path} contained no records")
    print(f"loaded {len(records)} prompts from {path}")
    return records


# --------------------------------------------------------------------------- #
# step 2: the model
# --------------------------------------------------------------------------- #
def load_model(model_name: str):
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    if torch.cuda.is_available():
        device, dtype = "cuda", torch.bfloat16
    elif getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        device, dtype = "mps", torch.float16
    else:
        device, dtype = "cpu", torch.float32

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=dtype)
    model.to(device)
    model.eval()

    # Left padding is required: the continuation is sliced at a single prompt
    # length for the whole batch, which is only correct if every row's prompt
    # ends at the same index.
    tokenizer.padding_side = "left"
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token

    print(f"model: {model_name} | device: {device} | dtype: {dtype}")
    return model, tokenizer, device


# --------------------------------------------------------------------------- #
# choice parsing
# --------------------------------------------------------------------------- #
# Parenthesised form first, bare letter only as a fallback with word boundaries,
# so "Answer: (B)" does not match the A in "Answer". A generation that mentions
# both letters is treated as uncommitted rather than resolved by first match.
_PAREN_RE = re.compile(r"\(\s*([AB])\s*\)", re.I)
_BARE_RE = re.compile(r"\b(A|[Bb])\b")


def _single_letter(matches: list[str]) -> str | None:
    letters = {m.upper() for m in matches}
    return letters.pop() if len(letters) == 1 else None


def parse_choice(text: str) -> str | None:
    """The committed A or B, or None if the generation never commits."""
    if not text:
        return None
    paren = _PAREN_RE.findall(text)
    if paren:
        return _single_letter(paren)
    return _single_letter(_BARE_RE.findall(text))


# --------------------------------------------------------------------------- #
# step 3: the loop
# --------------------------------------------------------------------------- #
def run_prompts(
    model,
    tokenizer,
    records: list[dict],
    device: str,
    max_new_tokens: int = 8,
    batch_size: int = 8,
    temperature: float = 0.0,
    system: str | None = None,
) -> list[dict]:
    import torch

    gen_kwargs = {"max_new_tokens": max_new_tokens, "pad_token_id": tokenizer.pad_token_id}
    if temperature > 0:
        gen_kwargs.update(do_sample=True, temperature=temperature)
    else:
        gen_kwargs.update(do_sample=False)

    out: list[dict] = []
    for start in range(0, len(records), batch_size):
        batch = records[start : start + batch_size]

        chats = []
        for r in batch:
            messages = []
            if system:
                messages.append({"role": "system", "content": system})
            messages.append({"role": "user", "content": r["prompt"]})
            chats.append(
                tokenizer.apply_chat_template(
                    messages, add_generation_prompt=True, tokenize=False
                )
            )

        # add_special_tokens=False: apply_chat_template already emitted them.
        enc = tokenizer(
            chats, return_tensors="pt", padding=True, add_special_tokens=False
        ).to(device)

        with torch.no_grad():
            gen = model.generate(**enc, **gen_kwargs)
        new = gen[:, enc["input_ids"].shape[1] :]

        for r, ids in zip(batch, new):
            text = tokenizer.decode(ids, skip_special_tokens=True)
            letter = parse_choice(text)
            rec = {"index": r.get("index"), "raw": text, "letter": letter}
            # only present when the jsonl was built with --full
            if r.get("honest_label"):
                rec["honest_label"] = r["honest_label"]
                rec["deceptive"] = None if letter is None else letter != r["honest_label"]
            out.append(rec)

        print(f"  {min(start + batch_size, len(records))}/{len(records)}", end="\r", flush=True)
    print()
    return out


def summarise(results: list[dict]) -> dict:
    parsed = [r for r in results if r["letter"] is not None]
    stats = {
        "n": len(results),
        "n_parsed": len(parsed),
        "unparsed_rate": 1 - len(parsed) / max(len(results), 1),
        "picked_A": sum(1 for r in parsed if r["letter"] == "A"),
        "picked_B": sum(1 for r in parsed if r["letter"] == "B"),
    }
    labelled = [r for r in parsed if r.get("deceptive") is not None]
    if labelled:
        stats["deception_rate"] = sum(r["deceptive"] for r in labelled) / len(labelled)
    return stats


# --------------------------------------------------------------------------- #
# cli
# --------------------------------------------------------------------------- #
def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(
        description="Build the MCQ jsonl and run it through a local Qwen model.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    ap.add_argument(
        "--model",
        default=DEFAULT_MODEL,
        help=f"HF model id, or a shortcut: {', '.join(MODEL_ALIASES)}",
    )
    ap.add_argument("--yaml", default=DEFAULT_YAML, help="source dataset for the parser")
    ap.add_argument("--jsonl", default=DEFAULT_JSONL, help="MCQ jsonl the parser writes")
    ap.add_argument("--out", default=DEFAULT_OUT, help="where to write generations")
    ap.add_argument("--parser-script", default=PARSER_SCRIPT)
    ap.add_argument("--skip-parse", action="store_true", help="reuse the existing jsonl")
    ap.add_argument(
        "--full",
        action="store_true",
        help="build the jsonl with --full so honest_label is available and a deception rate can be computed",
    )
    ap.add_argument("--batch-size", type=int, default=8)
    ap.add_argument("--max-new-tokens", type=int, default=8)
    ap.add_argument("--temperature", type=float, default=0.0, help="0 = greedy")
    ap.add_argument("--system", default=None, help="optional system prompt")
    ap.add_argument("--limit", type=int, default=None, help="only run the first N prompts")
    args = ap.parse_args(argv)

    if not args.skip_parse:
        build_jsonl(args.yaml, args.jsonl, args.parser_script, args.full)

    records = load_prompts(args.jsonl, args.limit)

    model_name = MODEL_ALIASES.get(args.model.lower(), args.model)
    model, tokenizer, device = load_model(model_name)

    print("\nexample prompt:")
    print("-" * 70)
    print(records[0]["prompt"])
    print("-" * 70 + "\n")

    results = run_prompts(
        model,
        tokenizer,
        records,
        device,
        max_new_tokens=args.max_new_tokens,
        batch_size=args.batch_size,
        temperature=args.temperature,
        system=args.system,
    )

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(
        "\n".join(json.dumps(r, ensure_ascii=False) for r in results) + "\n", encoding="utf-8"
    )

    stats = summarise(results)
    print("\n=== results ===")
    for k, v in stats.items():
        print(f"  {k}: {v}")
    print(f"\nsaved -> {out_path}")

    meta = {"model": model_name, "device": device, **vars(args), **stats}
    Path(str(out_path) + ".meta.json").write_text(
        json.dumps(meta, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
