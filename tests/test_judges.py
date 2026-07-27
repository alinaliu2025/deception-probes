"""Fast checks for the judge backend (ADR 0014). No network, no API key.

Everything here runs against a fake client, because the parts worth pinning are
the parts that fail quietly: score parsing, the cache key, and the refusal to
turn a judge failure into a zero.
"""

import json

import pytest

from dprobe import concept, judges
from dprobe.concept import ConceptSpec


# --- score parsing ---------------------------------------------------------

@pytest.mark.parametrize("text,expected", [
    ("85", 85),
    ("  85\n", 85),
    ("Score: 85", 85),
    ("85/100", 85),
    ("0", 0),
    ("100", 100),
    ("150", 100),      # clamped
    ("-20", 0),        # clamped
])
def test_parse_score_handles_the_shapes_models_actually_emit(text, expected):
    assert judges.parse_score(text) == expected


def test_parse_score_raises_rather_than_scoring_zero():
    """A reply with no number is a judge FAILURE. Scoring it 0 would drag the
    positive pole below pos_min and silently shrink that pole, which looks like a
    weak fixture rather than a broken judge."""
    for bad in ("", "I cannot score this.", None):
        with pytest.raises(ValueError, match="no number"):
            judges.parse_score(bad)


# --- fake backend ----------------------------------------------------------

class _FakeCompletions:
    def __init__(self, replies, calls):
        self._replies, self._calls = replies, calls

    def create(self, model, messages, **kw):
        self._calls.append({"prompt": messages[0]["content"], "model": model,
                            "kw": kw})
        r = self._replies.pop(0)
        if isinstance(r, Exception):
            raise r

        class _M:
            message = type("m", (), {"content": r})()
        return type("R", (), {"choices": [_M()]})()


class _FakeClient:
    def __init__(self, replies, calls):
        self.chat = type("c", (), {"completions": _FakeCompletions(replies, calls)})()


def _make(monkeypatch, tmp_path, replies, **kw):
    """make_judge with the OpenAI client and the key lookup swapped out."""
    calls: list[str] = []
    monkeypatch.setattr(judges, "_api_key", lambda: "not-a-real-key")
    import sys
    import types
    stub = types.ModuleType("openai")
    stub.OpenAI = lambda api_key, base_url: _FakeClient(replies, calls)
    monkeypatch.setitem(sys.modules, "openai", stub)
    return judges.make_judge(cache_dir=tmp_path, **kw), calls


def test_scores_and_caches(monkeypatch, tmp_path):
    judge, calls = _make(monkeypatch, tmp_path, ["77", "77"])
    assert judge("prompt one") == 77
    assert judge("prompt one") == 77          # second call must hit the cache
    assert len(calls) == 1, "cache miss on an identical prompt"
    assert judge.stats == {"hits": 1, "calls": 1}


def test_different_prompts_do_not_share_a_cache_entry(monkeypatch, tmp_path):
    judge, calls = _make(monkeypatch, tmp_path, ["10", "90"])
    assert judge("a") == 10
    assert judge("b") == 90
    assert len(calls) == 2


def test_retries_then_succeeds(monkeypatch, tmp_path):
    monkeypatch.setattr(judges.time, "sleep", lambda _s: None)
    judge, calls = _make(monkeypatch, tmp_path,
                         [RuntimeError("502 from proxy"), "42"])
    assert judge("p") == 42
    assert len(calls) == 2


def test_gives_up_loudly_instead_of_returning_zero(monkeypatch, tmp_path):
    """A dead proxy must stop the run. Returning 0 would quietly reshape both
    poles and the direction would still get built."""
    monkeypatch.setattr(judges.time, "sleep", lambda _s: None)
    judge, _ = _make(monkeypatch, tmp_path,
                     [RuntimeError("boom")] * 4, max_retries=4)
    with pytest.raises(RuntimeError, match="judge failed after 4 attempts"):
        judge("p")


def test_cache_file_records_the_model(monkeypatch, tmp_path):
    judge, _ = _make(monkeypatch, tmp_path, ["55"], model="some-model")
    judge("p")
    written = json.loads(next(tmp_path.glob("*.json")).read_text())
    assert written == {"score": 55, "model": "some-model"}


# --- reasoning-model temperature handling ---------------------------------

@pytest.mark.parametrize("model", ["o4-mini-2025-04-16", "o3-mini-2025-01-31",
                                   "openai/o1"])
def test_reasoning_models_are_called_without_temperature(monkeypatch, tmp_path,
                                                         model):
    """The o-series rejects `temperature` outright. Sending it fails every call,
    which on the proxy looks like an outage rather than a parameter problem."""
    judge, calls = _make(monkeypatch, tmp_path, ["70"], model=model,
                         verbose=False)
    judge("p")
    assert "temperature" not in calls[0]["kw"]


@pytest.mark.parametrize("model", ["claude-sonnet-5", "gpt-5.4-2026-03-05",
                                   "qwen3-32b", "nova-pro-v1", "minimax-m2.5"])
def test_ordinary_models_still_get_temperature(monkeypatch, tmp_path, model):
    """Determinism matters for every model that supports it. The pattern must not
    over-match: 'o3' inside another name should not strip temperature."""
    judge, calls = _make(monkeypatch, tmp_path, ["70"], model=model)
    judge("p")
    assert calls[0]["kw"]["temperature"] == 0.0


def test_drops_temperature_when_the_proxy_rejects_it(monkeypatch, tmp_path):
    """Fallback for models the name pattern does not catch. Retrying without the
    parameter must not consume the retry budget meant for real outages."""
    monkeypatch.setattr(judges.time, "sleep", lambda _s: None)
    judge, calls = _make(
        monkeypatch, tmp_path,
        [RuntimeError("Unsupported value: 'temperature' is not supported"), "64"],
        model="some-new-reasoner")
    assert judge("p") == 64
    assert "temperature" in calls[0]["kw"]
    assert "temperature" not in calls[1]["kw"]


def test_cache_key_separates_models(monkeypatch, tmp_path):
    """Two models must not share a cached score, or swapping --judge-model would
    silently reuse the old judge's opinions."""
    j1, c1 = _make(monkeypatch, tmp_path, ["11"], model="model-a")
    j1("same prompt")
    j2, c2 = _make(monkeypatch, tmp_path, ["99"], model="model-b")
    assert j2("same prompt") == 99
    assert len(c2) == 1


# --- integration with the pipeline ----------------------------------------

def test_prefetch_warms_what_judge_responses_looks_up(monkeypatch, tmp_path):
    """The cache key is the prompt string, so `prefetch` and `judge_responses`
    must build that string identically. They share `concept.build_judge_prompt`
    for exactly this reason; if either inlines its own format call, prefetch
    warms keys nothing ever reads and every score becomes a live API call."""
    spec = ConceptSpec.load()
    records = ([{"pole": "pos", "system": "s", "user": "u", "response": f"r{i}"}
                for i in range(25)]
               + [{"pole": "neg", "system": "s", "user": "u", "response": f"n{i}"}
                  for i in range(25)])
    judge, calls = _make(monkeypatch, tmp_path, ["90"] * 25 + ["5"] * 25)

    judges.prefetch(judge, [concept.build_judge_prompt(spec, r) for r in records],
                    workers=1, verbose=False)
    n_after_warm = len(calls)

    kept = concept.judge_responses(records, spec, judge)
    assert len(calls) == n_after_warm, "judge_responses missed the warmed cache"
    assert len(kept) == 50


def test_api_key_error_names_the_env_var(monkeypatch, tmp_path):
    monkeypatch.delenv("LITELLM_API_KEY", raising=False)
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    monkeypatch.setattr(judges, "_repo_root", lambda: tmp_path)  # no .env here
    with pytest.raises(RuntimeError, match="LITELLM_API_KEY"):
        judges._api_key()


def test_env_file_is_read_but_only_for_known_keys(monkeypatch, tmp_path):
    monkeypatch.delenv("LITELLM_API_KEY", raising=False)
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    (tmp_path / ".env").write_text(
        "# comment\nSOMETHING_ELSE=nope\nLITELLM_API_KEY='sk-xyz'\n")
    monkeypatch.setattr(judges, "_repo_root", lambda: tmp_path)
    assert judges._api_key() == "sk-xyz"
