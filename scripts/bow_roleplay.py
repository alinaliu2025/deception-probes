"""Can a classifier that never sees the model separate the roleplay completions?

RUN THIS BEFORE ANY GPU TIME. No model, no network, a few seconds.

The honest and deceptive completions in Apollo's roleplay set are matched on
scenario, question, answer prefix and system prompt. Only the completion differs.
That is a good matched-pair design for everything EXCEPT the completions
themselves, and the completions are exactly what the probe reads.

On the 10 items inlined in scheming.py, 7 of 10 honest completions contain a
negation and 0 of 10 deceptive ones do. Honest answers admit ("No, I took a
longer route", "I wasn't aware"); deceptive ones assert ("Yes, we took the most
efficient route"). Two honest completions literally open with "To be honest".

If that holds at 371, a probe trained on these completions is partly a
syntactic-polarity classifier, and the dangerous part is that such a direction
would PARTIALLY transfer to behavioural deception, because behavioural deception
also involves confident positive assertion. A spurious transfer would look like
success.

This script measures three things:

  1. bag-of-words AUROC, grouped CV so a pair never straddles the split
  2. the negation asymmetry, which is the specific tell
  3. length, the other cheap surface feature

WHAT THE NUMBERS MEAN. BoW AUROC is the floor any probe must clear to be
interesting. Report it beside every probe AUROC. A probe at 0.90 against a BoW
baseline of 0.88 has found almost nothing.

    python -m scripts.bow_roleplay
    python -m scripts.bow_roleplay --yaml data/roleplaying/dataset.yaml
"""

import argparse
import re

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import GroupKFold
from sklearn.pipeline import make_pipeline

from dprobe.data import scheming

NEGATION = re.compile(r"\b(not|no|n't|never|didn't|wasn't|haven't|isn't)\b", re.I)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--yaml", default=None,
                    help="path to Apollo's data/roleplaying/dataset.yaml. "
                         "Omitted = the 10 inlined smoke-test items.")
    ap.add_argument("--folds", type=int, default=5)
    args = ap.parse_args()

    items = scheming.load_roleplay_yaml(args.yaml)
    print(f"\npairs: {len(items)}")

    texts = [t for it in items for t in (it["honest"], it["deceptive"])]
    labels = np.array([y for _ in items for y in (0, 1)])
    # group = pair index, so a pair NEVER straddles the split. Without this the
    # two completions of one scenario land on both sides and the CV is optimistic
    # for the same reason the 07-06 sycophancy split was.
    groups = np.array([i for i in range(len(items)) for _ in (0, 1)])

    print("\n=== 1. bag-of-words AUROC (grouped CV, pairs never split) ===")
    n_folds = min(args.folds, len(items))
    oof = np.zeros(len(texts))
    for tr, te in GroupKFold(n_splits=n_folds).split(texts, labels, groups):
        clf = make_pipeline(TfidfVectorizer(ngram_range=(1, 2), min_df=1),
                            LogisticRegression(max_iter=2000))
        clf.fit([texts[i] for i in tr], labels[tr])
        oof[te] = clf.predict_proba([texts[i] for i in te])[:, 1]
    bow = roc_auc_score(labels, oof)
    print(f"  AUROC = {bow:.3f}   (chance 0.500)")
    print(f"  >>> every probe AUROC on this dataset must be reported against "
          f"{bow:.3f}, not against 0.5")

    print("\n=== 2. negation asymmetry ===")
    hon = [it["honest"] for it in items]
    dec = [it["deceptive"] for it in items]
    nh = sum(bool(NEGATION.search(t)) for t in hon)
    nd = sum(bool(NEGATION.search(t)) for t in dec)
    print(f"  honest    with a negation: {nh:4d}/{len(hon)} ({nh/len(hon):.0%})")
    print(f"  deceptive with a negation: {nd:4d}/{len(dec)} ({nd/len(dec):.0%})")
    only_neg = roc_auc_score(
        labels, [0.0 if NEGATION.search(t) else 1.0 for t in texts])
    print(f"  AUROC from the negation flag ALONE = {only_neg:.3f}")

    print("\n=== 3. length ===")
    lh, ld = [len(t) for t in hon], [len(t) for t in dec]
    print(f"  honest    mean chars {np.mean(lh):6.0f}")
    print(f"  deceptive mean chars {np.mean(ld):6.0f}")
    print(f"  AUROC from length ALONE = "
          f"{roc_auc_score(labels, [-len(t) for t in texts]):.3f}")

    print("\n=== verdict ===")
    if bow >= 0.80:
        print(f"  BoW at {bow:.3f}. The completions are separable from surface "
              "text alone.\n  A probe trained here is partly a text classifier. "
              "Build the polarity-direction\n  control before trusting any "
              "transfer result.")
    elif bow >= 0.65:
        print(f"  BoW at {bow:.3f}. Real surface signal, not overwhelming. "
              "Report it beside\n  every probe number and keep the polarity "
              "control on the list.")
    else:
        print(f"  BoW at {bow:.3f}. Surface features are weak here. The "
              "matched-pair design\n  is holding up better than the 10-item "
              "sample suggested.")


if __name__ == "__main__":
    main()
