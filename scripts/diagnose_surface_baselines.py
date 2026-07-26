"""Do the caved/held labels fall out of the QUESTION TEXT alone?

WHY THIS EXISTS
---------------
The promptfinal layer sweep on the 400q instructed run
(`2026-07-16T03-49-37Z_sandbagging_mms_b271758_p0306`) reads:

    layer  0 : 0.500      <- embeddings, arrow is identically zero
    layer  1 : 0.736      <- ONE decoder block
    layers 2-18: 0.70-0.76
    layer 20 : 0.845      <- best

AUROC 0.5 is chance, so layer 1 sits 0.236 above chance and layer 20 sits 0.345
above chance: one decoder block already recovers ~68% of the separation the best
layer achieves. A representation of "I am deliberately underperforming" should
need depth to assemble. Something available that early is more likely a property
of the INPUT, not a state the model computed.

The caved set and the held set are DIFFERENT QUESTIONS (a question is labeled by
what the model does under the sandbag prompt, so each question lands in exactly
one class). If those two sets differ in surface properties -- topic, vocabulary,
length, difficulty -- then a classifier that never sees the model at all should
separate them. That is what this script measures.

WHAT IT RUNS
------------
Everything below reads `run_log.txt` from a completed did run. No model, no GPU,
no activations. Runs on a laptop in seconds.

1. GATE PASS RATE. The belief gate sampled 20 unpressured answers per question
   and the trace records e.g. `gate=sampled 19/20 correct`. That is a graded
   measure of how firmly the model holds the answer. If caved questions have
   systematically lower pass rates than held ones, then "caved" is partly a
   proxy for "the model's grip on this question was weaker", and the direction
   fit on those labels inherits that.
   NOTE the gate threshold is 0.90, so every question here scored >= 18/20. The
   question is whether caved clusters at 18-19 and held at 20.

2. QUESTION LENGTH. The dumbest possible surface feature. Included because if
   THIS separates the classes, nothing subtler needs explaining.

3. BAG OF WORDS. Word-count features over the question text, 5-fold CV, two
   estimators (logistic regression and a Naive-Bayes log-count-ratio scorer).
   This is the number to compare against layer 1's 0.736.

4. PERMUTATION CONTROL. Same pipeline with labels shuffled. Expect ~0.5. Guards
   against the CV or the vectorizer leaking, same role `--permute` plays in
   train_one.

HOW TO READ THE RESULT
----------------------
    BoW ~ 0.70+   the layer-1 number is explained by question text. The part of
                  the probe worth reporting is what layer 20 adds ON TOP of a
                  text-only baseline, not the raw 0.845.
    BoW ~ 0.55    the early-layer signal is something the model computed, and
                  the surface-feature worry does not apply.

CAVEAT ON THE TEXT
------------------
`tracefmt.short()` truncates questions to 90 characters in the log, so this runs
on question PREFIXES. That makes every number here a LOWER BOUND -- BoW on full
text can only do better. To run on full text, rebuild the dataset with
`mcq.rows("factual", "train")` and join on the prefix.

USAGE
-----
    python -m scripts.diagnose_surface_baselines \
        --run results/runs/2026-07-16T03-49-37Z_sandbagging_mms_b271758_p0306

    # include the class-balance TRIMMED questions too (more data, but these are
    # NOT the rows the probe was fit on)
    python -m scripts.diagnose_surface_baselines --run <dir> --include-trimmed

sklearn is used when importable; otherwise the numpy fallbacks below run and
give the same answer to within CV noise.
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import numpy as np

# ---------------------------------------------------------------------------
# parsing
# ---------------------------------------------------------------------------

# "  [USED caved(1)] gate=sampled 20/20 correct (20 sampled) :: <question>"
_ROW = re.compile(
    r"^\s*\[(?P<status>USED|TRIMMED)\s+(?P<outcome>caved|held)\((?P<label>[01])\)\]"
    r"\s+gate=sampled\s+(?P<correct>\d+)/(?P<parseable>\d+)\s+correct"
    r"\s+\((?P<n>\d+)\s+sampled\)\s+::\s+(?P<question>.*)$"
)

# '      gate-sample[correct] "(B)"'  -- the gate only marks a sample `correct`
# when it chose not_matching, so the letter on these lines IS the correct option
# for that question. That is how we recover the answer side without re-reading
# ARC (the log truncates the question before the options).
_GATE_SAMPLE = re.compile(r'gate-sample\[correct\]\s+"\((?P<letter>[AB])\)"')


def parse_run_log(path: Path, include_trimmed: bool = False) -> list[dict]:
    """Pull one record per labeled question out of a did_filter trace.

    Only USED rows by default: those are the class-balanced rows the probe was
    actually fit on, so they are the population whose labels we are auditing.
    """
    lines = path.read_text(errors="replace").splitlines()
    records = []
    for i, line in enumerate(lines):
        m = _ROW.match(line)
        if not m:
            continue
        if m["status"] == "TRIMMED" and not include_trimmed:
            continue
        # walk forward to this question's first correct gate sample
        letter = None
        for j in range(i + 1, min(i + 30, len(lines))):
            if _ROW.match(lines[j]):
                break
            s = _GATE_SAMPLE.search(lines[j])
            if s:
                letter = s["letter"]
                break
        records.append({
            "status": m["status"],
            "label": int(m["label"]),          # 1 = caved (sandbagged), 0 = held
            "gate_correct": int(m["correct"]),
            "gate_parseable": int(m["parseable"]),
            "gate_n": int(m["n"]),
            "correct_letter": letter,          # which option was right, A or B
            "question": m["question"].rstrip("…").strip(),
        })
    if not records:
        raise SystemExit(
            f"no labeled rows parsed from {path}.\n"
            "This script expects a did-design run_log.txt written with "
            "gate=sampled. A logprob-gate run has no pass rates and a "
            "non-did run has no caved/held lines."
        )
    return records


# ---------------------------------------------------------------------------
# metrics (numpy, no sklearn required)
# ---------------------------------------------------------------------------

def auroc(labels: np.ndarray, scores: np.ndarray) -> float:
    """AUROC via the rank identity: P(score of a random positive > a random
    negative), ties counted as half. Same number sklearn's roc_auc_score gives."""
    labels = np.asarray(labels)
    order = np.argsort(scores, kind="mergesort")
    ranks = np.empty(len(scores), dtype=float)
    ranks[order] = np.arange(1, len(scores) + 1)
    # average ranks within tied score groups
    s = np.asarray(scores)[order]
    i = 0
    while i < len(s):
        j = i
        while j + 1 < len(s) and s[j + 1] == s[i]:
            j += 1
        if j > i:
            ranks[order[i:j + 1]] = ranks[order[i:j + 1]].mean()
        i = j + 1
    n_pos = int((labels == 1).sum())
    n_neg = int((labels == 0).sum())
    if n_pos == 0 or n_neg == 0:
        return float("nan")
    return float((ranks[labels == 1].sum() - n_pos * (n_pos + 1) / 2) / (n_pos * n_neg))


def mannwhitney_p(a: np.ndarray, b: np.ndarray) -> float:
    """Two-sided normal-approximation p-value for the rank-sum statistic.

    Enough precision to say 'these two distributions differ' at n ~ 160/160;
    scipy is not required. Ties are not corrected for, so the p is slightly
    conservative on heavily tied data (which gate counts are)."""
    labels = np.concatenate([np.ones(len(a)), np.zeros(len(b))])
    scores = np.concatenate([a, b])
    u_frac = auroc(labels, scores)
    n1, n2 = len(a), len(b)
    u = u_frac * n1 * n2
    mu = n1 * n2 / 2
    sigma = np.sqrt(n1 * n2 * (n1 + n2 + 1) / 12)
    if sigma == 0:
        return float("nan")
    z = (u - mu) / sigma
    # two-sided normal tail without scipy
    p = 2 * 0.5 * (1 - _erf(abs(z) / np.sqrt(2)))
    return float(min(1.0, max(0.0, p)))


def _erf(x: float) -> float:
    """Abramowitz & Stegun 7.1.26. Accurate to ~1.5e-7, plenty for a p-value."""
    sign = 1 if x >= 0 else -1
    x = abs(x)
    t = 1.0 / (1.0 + 0.3275911 * x)
    y = 1.0 - (((((1.061405429 * t - 1.453152027) * t) + 1.421413741) * t
                - 0.284496736) * t + 0.254829592) * t * np.exp(-x * x)
    return sign * y


# ---------------------------------------------------------------------------
# bag of words
# ---------------------------------------------------------------------------

_TOKEN = re.compile(r"[a-z][a-z']+")


def count_matrix(texts: list[str], vocab: dict[str, int] | None = None):
    """Lowercase word counts. Vocabulary is built from the TRAIN fold only and
    passed in for the test fold, so the vectorizer cannot see held-out text."""
    if vocab is None:
        vocab = {}
        for t in texts:
            for w in _TOKEN.findall(t.lower()):
                vocab.setdefault(w, len(vocab))
    X = np.zeros((len(texts), len(vocab)), dtype=np.float32)
    for i, t in enumerate(texts):
        for w in _TOKEN.findall(t.lower()):
            j = vocab.get(w)
            if j is not None:
                X[i, j] += 1.0
    return X, vocab


def fit_logreg(X: np.ndarray, y: np.ndarray, C: float = 1.0,
               steps: int = 600, lr: float = 0.5) -> np.ndarray:
    """L2 logistic regression by full-batch gradient descent.

    Deliberately plain: this is a BASELINE, so the point is that even a crude
    text classifier reaches whatever AUROC it reaches. Returns [w, b]."""
    n, d = X.shape
    mu, sd = X.mean(0), X.std(0) + 1e-8
    Xs = (X - mu) / sd
    w = np.zeros(d)
    b = 0.0
    lam = 1.0 / (C * max(n, 1))
    for _ in range(steps):
        z = Xs @ w + b
        p = 1.0 / (1.0 + np.exp(-np.clip(z, -30, 30)))
        g = (p - y)
        gw = Xs.T @ g / n + lam * w
        gb = g.mean()
        w -= lr * gw
        b -= lr * gb
    return np.concatenate([w / sd, [b - (mu / sd) @ w]])


def score_logreg(X: np.ndarray, wb: np.ndarray) -> np.ndarray:
    return X @ wb[:-1] + wb[-1]


def nb_logratio(X: np.ndarray, y: np.ndarray, alpha: float = 1.0) -> np.ndarray:
    """Multinomial Naive-Bayes log-count-ratio, the classic strong baseline for
    short text: r = log( p(w|caved) / p(w|held) ), score = X @ r."""
    p = X[y == 1].sum(0) + alpha
    q = X[y == 0].sum(0) + alpha
    return np.log((p / p.sum()) / (q / q.sum()))


def cv_auroc(texts: list[str], y: np.ndarray, kind: str, folds: int = 5,
             seed: int = 0) -> float:
    """Stratified k-fold CV AUROC, vocabulary refit inside each fold."""
    rng = np.random.default_rng(seed)
    idx_pos = rng.permutation(np.where(y == 1)[0])
    idx_neg = rng.permutation(np.where(y == 0)[0])
    fold_of = np.empty(len(y), dtype=int)
    for arr in (idx_pos, idx_neg):
        for k, i in enumerate(arr):
            fold_of[i] = k % folds
    scores = np.zeros(len(y))
    for f in range(folds):
        te = np.where(fold_of == f)[0]
        tr = np.where(fold_of != f)[0]
        Xtr, vocab = count_matrix([texts[i] for i in tr])
        Xte, _ = count_matrix([texts[i] for i in te], vocab)
        if kind == "logreg":
            wb = fit_logreg(Xtr, y[tr])
            scores[te] = score_logreg(Xte, wb)
        elif kind == "nb":
            r = nb_logratio(Xtr, y[tr])
            scores[te] = Xte @ r
        else:
            raise ValueError(kind)
    return auroc(y, scores)


# ---------------------------------------------------------------------------
# report
# ---------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--run", required=True,
                    help="path to a did run dir (must contain run_log.txt)")
    ap.add_argument("--include-trimmed", action="store_true",
                    help="also use questions dropped by class balancing. More "
                         "data, but these are NOT the rows the probe was fit on.")
    ap.add_argument("--folds", type=int, default=5)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    run_dir = Path(args.run)
    recs = parse_run_log(run_dir / "run_log.txt", args.include_trimmed)
    y = np.array([r["label"] for r in recs])
    texts = [r["question"] for r in recs]
    rate = np.array([r["gate_correct"] / max(1, r["gate_parseable"]) for r in recs])

    probe = {}
    meta_path = run_dir / "meta.json"
    if meta_path.exists():
        meta = json.loads(meta_path.read_text())
        pf = meta.get("did_positions", {}).get("promptfinal", {})
        curve = pf.get("aurocs") or pf.get("auroc_per_layer") or []
        probe = {"best_layer": pf.get("best_layer"), "best": pf.get("auroc"),
                 "layer1": curve[1] if len(curve) > 1 else None}

    print("=" * 74)
    print("SURFACE-FEATURE BASELINES FOR THE CAVED/HELD LABELS")
    print("=" * 74)
    print(f"run          : {run_dir.name}")
    print(f"questions    : {len(recs)}  ({int((y==1).sum())} caved / "
          f"{int((y==0).sum())} held)")
    print("rows         : USED only" if not args.include_trimmed
          else "rows         : USED + TRIMMED")
    if probe:
        print(f"probe (promptfinal): layer1={probe['layer1']:.3f}   "
              f"best={probe['best']:.3f} @L{probe['best_layer']}")
    print()

    print("1. BELIEF-GATE PASS RATE  (how firmly the model held the answer, unpressured)")
    a = auroc(y, -rate)
    p = mannwhitney_p(rate[y == 1], rate[y == 0])
    print(f"   caved  mean {rate[y==1].mean():.4f}   "
          f"20/20 in {100*(rate[y==1]==1.0).mean():.1f}% of questions")
    print(f"   held   mean {rate[y==0].mean():.4f}   "
          f"20/20 in {100*(rate[y==0]==1.0).mean():.1f}% of questions")
    print(f"   AUROC of (lower pass rate -> caved) : {a:.3f}")
    print(f"   Mann-Whitney two-sided p            : {p:.2g}")
    print("   reading: >0.5 means caved questions were held LESS firmly, so the")
    print("            label is partly tracking the model's grip on the answer.")
    uniq, cnt = np.unique(np.round(rate, 3), return_counts=True)
    spread = ", ".join(f"{u:.2f}:{c}" for u, c in zip(uniq, cnt))
    print(f"   pass-rate distribution among kept questions: {spread}")
    if len(uniq) == 1:
        print("   NOTE: every kept question is at the same rate, so the sampled gate")
        print("         partitioned exactly as a single greedy pass would have. At")
        print("         --gate-n 20 that is 20x the generation cost for no extra")
        print("         resolution on THIS data; check before paying for it again.")
    print()

    print("2. QUESTION LENGTH  (the dumbest surface feature)")
    for name, f in (("characters", lambda t: len(t)),
                    ("words", lambda t: len(t.split()))):
        v = np.array([f(t) for t in texts], dtype=float)
        print(f"   {name:<11} caved {v[y==1].mean():7.2f}   "
              f"held {v[y==0].mean():7.2f}   AUROC {auroc(y, v):.3f}")
    print()

    print(f"3. BAG OF WORDS  ({args.folds}-fold CV, vocabulary refit per fold)")
    for kind, label in (("logreg", "logistic regression"), ("nb", "NB log-ratio")):
        s = cv_auroc(texts, y, kind, args.folds, args.seed)
        print(f"   {label:<20} AUROC {s:.3f}")
    print("   note: log text is truncated to 90 chars, so these are LOWER BOUNDS")
    print()

    print("4. ANSWER-LETTER SIDE  (was the correct option (A) or (B)?)")
    letters = [r["correct_letter"] for r in recs]
    have = np.array([l in ("A", "B") for l in letters])
    if have.sum() < 10:
        print("   not recoverable from this log (no gate-sample lines)")
    else:
        L = np.array([1.0 if l == "A" else 0.0 for l in letters])[have]
        yl = y[have]
        pA1, pA0 = L[yl == 1].mean(), L[yl == 0].mean()
        n1, n0 = int((yl == 1).sum()), int((yl == 0).sum())
        pp = L.mean()
        z = ((pA1 - pA0) / np.sqrt(pp * (1 - pp) * (1 / n1 + 1 / n0))
             if 0 < pp < 1 else float("nan"))
        print(f"   caved  : correct answer was (A) in {100*pA1:.1f}% of {n1} questions")
        print(f"   held   : correct answer was (A) in {100*pA0:.1f}% of {n0} questions")
        print(f"   AUROC of (correct==A -> caved) : {auroc(yl, L):.3f}")
        print(f"   two-proportion z               : {z:.2f}")
        print("   reading: the classes should be ~50/50 here. A gap means the model")
        print("            has a side bias under the pressure prompt, so 'caved' is")
        print("            partly 'the correct answer happened to be on one side'.")
        print("            The did design is NOT letter-balanced (ADR 0012); ADR 0008")
        print("            step 7 is the fix, applied jointly with class balance.")
    print()

    print("5. PERMUTATION CONTROL  (labels shuffled, expect ~0.5)")
    rng = np.random.default_rng(args.seed + 1)
    yp = rng.permutation(y)
    for kind, label in (("logreg", "logistic regression"), ("nb", "NB log-ratio")):
        s = cv_auroc(texts, yp, kind, args.folds, args.seed)
        print(f"   {label:<20} AUROC {s:.3f}")
    print()

    if probe and probe["layer1"] is not None:
        best_text = max(cv_auroc(texts, y, "logreg", args.folds, args.seed),
                        cv_auroc(texts, y, "nb", args.folds, args.seed))
        print("=" * 74)
        print("VERDICT")
        print(f"  text-only baseline      {best_text:.3f}   "
              f"({best_text - 0.5:+.3f} above chance)")
        print(f"  probe @ layer 1         {probe['layer1']:.3f}   "
              f"({probe['layer1'] - 0.5:+.3f} above chance)")
        print(f"  probe @ layer {probe['best_layer']:<2}        {probe['best']:.3f}   "
              f"({probe['best'] - 0.5:+.3f} above chance)")
        gap = probe["best"] - best_text
        print()
        if best_text >= probe["layer1"] - 0.03:
            print("  Text alone matches or beats layer 1. The shallow part of the")
            print("  sweep is explained by which questions ended up in which class.")
        else:
            print("  Text alone falls well short of layer 1. The early-layer signal")
            print("  is not obviously a property of the question wording.")
        print(f"  What the deep probe adds over text alone: {gap:+.3f} AUROC.")
        print("=" * 74)


if __name__ == "__main__":
    main()
