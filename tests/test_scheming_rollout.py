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
        assert e.meta["matching"].startswith("(A)")
        assert e.meta["not_matching"].startswith("(B)")
        assert e.meta["neutral_user"] and e.meta["calm_system"]
