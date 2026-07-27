"""LLM judge backend for the concept-vector pipeline (ADR 0014).

The judge scores how strongly a response expresses the target trait. It is the
step that separates a persona vector from an instruction-contrast vector, and it
is also the one live risk named in `SCOPE.md`: the direction inherits whatever
the judge thinks the trait is. So this module keeps the judge boring, cached and
deterministic, and makes every score auditable.

Backend is OSU's LiteLLM proxy, which is OpenAI-compatible:

    export LITELLM_API_KEY=...        # or put it in .env, which is gitignored
    python -m scripts.concept_vector --judge-model <model>

The key is read from the environment, never passed on the command line (shell
history) and never written to `meta.json` or `run_log.txt`.

NOTE ON WHERE THIS RUNS. OSC compute nodes have no internet, so judging cannot
happen inside the extraction job. Judge on a login node or locally, commit the
scored records, then run extraction. `openai` is an optional dependency for
exactly this reason: the OSC environment does not need it.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from .config import RESULTS_DIR

DEFAULT_BASE_URL = "https://litellm.cloud.osu.edu/v1"
# Judge choice is the ONE live risk in ADR 0014, so the default is argued, not
# picked. Requirements, in order:
#   1. supports temperature=0 -> deterministic -> a re-run reproduces the study.
#      Rules out the o-series and other reasoning models.
#   2. stable, not a preview. A model deprecated mid-study invalidates the cache
#      and there is no way to regenerate the old scores. Rules out gemini
#      *-preview and anything released in the last month.
#   3. different family from the SUBJECT model (Qwen2.5-7B). A judge sharing the
#      subject's blind spots cannot flag a wrong answer they both believe.
#      Rules out qwen3-32b.
#   4. strong factual knowledge. The judge has to know Mercury has the shortest
#      day before it can tell a sandbagged answer from a correct one.
# The second-judge control (ADR 0014 control 6) must differ in family from BOTH
# this default and the subject model: gpt-5.4-2026-03-05.
DEFAULT_MODEL = "claude-sonnet-5"
SECOND_JUDGE = "gpt-5.4-2026-03-05"
CACHE_DIR = RESULTS_DIR / ".judge_cache"  # under results/, already gitignored


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


def _api_key() -> str:
    """Environment first, then a gitignored .env at the repo root.

    Returns the key. NEVER print, log, or put it in meta.json.
    """
    for var in ("LITELLM_API_KEY", "OPENAI_API_KEY"):
        if os.environ.get(var):
            return os.environ[var]

    env_file = _repo_root() / ".env"
    if env_file.exists():
        for line in env_file.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            k, v = line.split("=", 1)
            if k.strip() in ("LITELLM_API_KEY", "OPENAI_API_KEY"):
                return v.strip().strip("'\"")

    raise RuntimeError(
        "no judge API key. Set LITELLM_API_KEY in your environment:\n"
        "    export LITELLM_API_KEY=...\n"
        "or put LITELLM_API_KEY=... in a .env file at the repo root (.env is "
        "gitignored). Do not pass it as a command-line argument, it ends up in "
        "shell history."
    )


def list_models(base_url: str = DEFAULT_BASE_URL) -> list[str]:
    """What the proxy actually offers. Run this before picking --judge-model."""
    from openai import OpenAI

    client = OpenAI(api_key=_api_key(), base_url=base_url)
    return sorted(m.id for m in client.models.list().data)


# o-series reasoning models reject `temperature` outright. LiteLLM only strips
# unsupported params if the proxy was configured with drop_params, which we cannot
# assume, so skip it here and fall back on the error too (see _supports_temperature
# and the retry loop). Matches o1/o3/o4 with or without a date suffix.
_REASONING_RE = re.compile(r"(^|/)o[134](-|$)")

_SCORE_RE = re.compile(r"-?\d+")


def _supports_temperature(model: str) -> bool:
    return _REASONING_RE.search(model) is None


def parse_score(text: str) -> int:
    """First integer in the reply, clamped to 0-100.

    The rubric asks for a bare number, and models mostly comply, but "Score: 85"
    and "85/100" both show up. Taking the FIRST integer handles both. A reply with
    no digits is a judge failure, not a zero, so it raises rather than quietly
    scoring 0 and dragging the positive pole down.
    """
    m = _SCORE_RE.search(text or "")
    if m is None:
        raise ValueError(f"no number in judge reply: {text!r}")
    return max(0, min(100, int(m.group())))


def make_judge(model: str = DEFAULT_MODEL, base_url: str = DEFAULT_BASE_URL,
               cache_dir: Path | None = None, temperature: float = 0.0,
               max_retries: int = 4, verbose: bool = True):
    """Return `score_fn(prompt) -> int`, the callable `concept.judge_responses` wants.

    Deterministic (temperature 0) and cached on disk, keyed by model + prompt. The
    cache is what makes rubric iteration affordable: changing the rubric changes
    every prompt and so misses by design, but re-running after an unrelated bug fix
    is free. Cached scores also mean the hand-agreement check in ADR 0014 scores the
    same numbers the run used, not a fresh sample.
    """
    from openai import OpenAI

    cache_dir = Path(cache_dir or CACHE_DIR)
    cache_dir.mkdir(parents=True, exist_ok=True)
    client = OpenAI(api_key=_api_key(), base_url=base_url)
    stats = {"hits": 0, "calls": 0}

    def _path(prompt: str) -> Path:
        key = hashlib.sha256(f"{model}\0{temperature}\0{prompt}".encode()).hexdigest()
        return cache_dir / f"{key}.json"

    use_temp = _supports_temperature(model)
    if verbose and not use_temp:
        print(f"judge: {model} is a reasoning model, omitting temperature. "
              "Scores are NOT fully deterministic, so the cache is what keeps a "
              "re-run reproducible. Do not clear it mid-study.")

    def score_fn(prompt: str) -> int:
        nonlocal use_temp
        p = _path(prompt)
        if p.exists():
            stats["hits"] += 1
            return json.loads(p.read_text(encoding="utf-8"))["score"]

        last = None
        for attempt in range(max_retries):
            try:
                kw = {"temperature": temperature} if use_temp else {}
                resp = client.chat.completions.create(
                    model=model, messages=[{"role": "user", "content": prompt}],
                    **kw)
                score = parse_score(resp.choices[0].message.content)
                p.write_text(json.dumps({"score": score, "model": model}),
                             encoding="utf-8")
                stats["calls"] += 1
                return score
            except Exception as exc:  # transient proxy errors, rate limits, bad parse
                last = exc
                # belt and braces: some proxies reject temperature for models the
                # name pattern does not catch. Drop it once and retry rather than
                # burning all 4 attempts on the same rejection.
                if use_temp and "temperature" in str(exc).lower():
                    use_temp = False
                    continue
                if attempt < max_retries - 1:
                    time.sleep(2 ** attempt)
        raise RuntimeError(
            f"judge failed after {max_retries} attempts: {last}. Scoring this "
            "response 0 would silently bias the positive pole, so the run stops "
            "instead."
        ) from last

    score_fn.stats = stats  # type: ignore[attr-defined]
    score_fn.model = model  # type: ignore[attr-defined]
    return score_fn


def prefetch(score_fn, prompts: list[str], workers: int = 8,
             verbose: bool = True) -> None:
    """Warm the cache concurrently, so the serial scoring loop is all cache hits.

    `concept.judge_responses` scores serially, which keeps it simple and testable.
    400 sequential round trips is several minutes of waiting; the same 400 across 8
    threads is well under one. Failures surface here, before extraction starts.
    """
    todo = list(dict.fromkeys(prompts))  # dedupe, preserve order
    if verbose:
        print(f"judge: warming cache for {len(todo)} unique prompts "
              f"({workers} workers)")
    with ThreadPoolExecutor(max_workers=workers) as pool:
        for i, _ in enumerate(pool.map(score_fn, todo), 1):
            if verbose and i % 25 == 0:
                print(f"  {i}/{len(todo)}", end="\r", flush=True)
    if verbose:
        print(f"  done. {score_fn.stats['calls']} calls, "
              f"{score_fn.stats['hits']} cache hits")
