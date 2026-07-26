"""Cross-tab answer letter x caved/held label from a did-run run_log.txt.

Usage: python -m scripts.letter_balance results/runs/<run>/run_log.txt
The letter/label are NOT stored in meta.json; this reconstructs the table from
the per-question trace (header [USED|TRIMMED caved(1)|held(0)] + pressure-answer).
"""
import re, sys
from collections import Counter

path = sys.argv[1] if len(sys.argv) > 1 else (
    "results/runs/2026-07-16T03-49-37Z_sandbagging_mms_b271758_p0306/run_log.txt")

used, allc = Counter(), Counter()
cur = None
for ln in open(path):
    m = re.search(r'\[(USED|TRIMMED) (caved|held)', ln)
    if m:
        cur = (m.group(1), m.group(2))
    pm = re.search(r'pressure-answer "\(([AB])\)"', ln)
    if pm and cur:
        status, lab = cur
        allc[(lab, pm.group(1))] += 1
        if status == "USED":
            used[(lab, pm.group(1))] += 1
        cur = None

def show(counter, title):
    print(f"\n{title}")
    print(f"  {'class':10s} {'A':>4s} {'B':>4s} {'%B':>5s}")
    for lab in ("caved", "held"):
        a, b = counter[(lab, "A")], counter[(lab, "B")]
        pb = 100 * b / (a + b) if a + b else 0
        print(f"  {lab:10s} {a:4d} {b:4d} {pb:4.0f}%")

show(used, "USED (balanced training set):")
show(allc, "All gated (pre-balance):")
