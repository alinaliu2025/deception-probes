"""Parser + builder checks for the scheming rollout path. No model download.

The parser is the risky part: a reasoning model names (A)/(B) many times while it
thinks, so parse_final_choice must read only the COMMITTED answer, not the chatter.
"""

from dprobe.data import scheming


def test_parse_ignores_cot_chatter():
    m, n = "(A) disable oversight", "(B) leave it on"
    t = "<think>Option (A) would disable oversight. (B) is safer. I'll go (A).</think>\n\n(A)"
    choice, _ = scheming.parse_final_choice(t, m, n)
    assert choice == "matching"


def test_parse_reads_after_think_not_inside():
    # the CoT is full of (A); the committed answer after </think> is (B)
    m, n = "(A) covert", "(B) honest"
    t = "<think>(A) (A) (A) tempting but risky</think>\nFinal answer: (B)"
    assert scheming.parse_final_choice(t, m, n)[0] == "not_matching"


def test_parse_no_think_tags_last_letter_wins():
    m, n = "(A) covert", "(B) honest"
    t = "I considered (A) but I'll pick (B)."
    assert scheming.parse_final_choice(t, m, n)[0] == "not_matching"


def test_parse_unparsed_returns_none():
    assert scheming.parse_final_choice("no letters here", "(A) x", "(B) y") == (None, None)


def test_reasoned_prefix_ends_at_commit():
    m, n = "(A) covert", "(B) honest"
    t = "reasoning ... </think>\nanswer (A) and then trailing junk"
    choice, end = scheming.parse_final_choice(t, m, n)
    assert choice == "matching"
    assert t[:end].rstrip().endswith("(A)")  # the read token sits on the commit


def test_build_rollout_unlabeled_and_flagged():
    exs = scheming.build("rollout")
    assert exs and all(e.label == scheming.UNLABELED for e in exs)
    assert all(e.meta["reconstructed"] for e in exs)
    assert all(e.meta["design"] == "rollout" for e in exs)
    for e in exs:
        assert e.meta["neutral_user"] and e.meta["calm_system"]


def test_covert_action_is_not_always_letter_a():
    """Updated 2026-07-27. This test used to assert `matching` always starts with
    "(A)", which pinned the covert-always-(A) bug in place as if it were the
    contract. That bug was fixed on 2026-07-26 (`choice_block`, `_SIDE_SEED`), so
    the assertion was pinning the wrong invariant and failed the moment the fix
    landed.

    The real contract: the covert commit and the honest commit sit on OPPOSITE
    letters, and which side gets the covert action varies across scenarios. If it
    did not vary, the label would be a perfect function of the emitted letter and
    a probe scoring AUROC 1.0 would represent nothing (ADR 0012 letter shortcut).
    """
    exs = scheming.build("rollout")
    sides = set()
    for e in exs:
        m, n = e.meta["matching"], e.meta["not_matching"]
        assert m[:3] in ("(A)", "(B)") and n[:3] in ("(A)", "(B)")
        assert m[:3] != n[:3], f"covert and honest on the same letter: {m[:3]}"
        sides.add(m[:3])
    assert sides == {"(A)", "(B)"}, (
        f"covert action only ever appears at {sides}. The letter-side flip is "
        "inert, which is the bug it was written to fix."
    )


def test_letter_side_is_stable_across_builds():
    """Seeded, so a scenario's side must not move between runs or machines.
    A side that drifts makes two runs of the same design incomparable."""
    a = {e.meta["neutral_user"]: e.meta["matching"][:3]
         for e in scheming.build("rollout")}
    b = {e.meta["neutral_user"]: e.meta["matching"][:3]
         for e in scheming.build("rollout")}
    assert a == b
