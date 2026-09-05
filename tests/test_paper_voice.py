"""The paper states what is true now, positively, and does not narrate its own history.

The rules are `skills/doc-current-state`: no change history, failure narration, self-justification,
hedging, defensive framing, or spurious negation. A paper that says a result was withdrawn, or
that a package "was wrong", spends its credibility describing its drafts instead of its evidence.

The vocabulary below is banned outright rather than judged in context. Each entry was in the paper
and each was replaceable by a positive statement of the same fact -- "the read on $J$" for "the
uncorrected read", "what bounds the measurement" for "the library's own limits". A word that has
an honest use here would need an exemption, and none has needed one yet.
"""
from __future__ import annotations

import re
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
PAPER = ROOT / "research" / "PAPER.md"
# The sdist ships this suite but not every repository file it reads. A shipped test that fails
# for an absent repository file reads as a broken package rather than a file that was never
# part of the distribution, so the module skips instead.
if not PAPER.exists():                                           # pragma: no cover - sdist
    pytest.skip("the paper is a repository file and is not shipped", allow_module_level=True)

#: phrase -> what to write instead.
BANNED = {
    "withdrawn": "state what the section establishes now",
    "retraction": "state what the section establishes now",
    "uncorrected": r"say which matrix is read: 'the read on $J$'",
    "the corrected read": r"'the read on $M$'",
    "anyway": "name the condition the section applies under",
    "fallen into": "state the requirement, not the history of missing it",
    "was wrong": "state what is true",
    "misread": "state what is true",
    "never once": "state the property positively",
    "not as criticism": "drop the defence and state the bound",
    "hopefully": "state the property or drop it",
    "for now": "state the present constraint",
    "no longer": "state what is true now",
    "used to": "state what is true now",
    "previously": "state what is true now",
    "formerly": "state what is true now",
    # A scope limit is a fact about where a result holds. Written as a confession it reads as
    # commentary on the paper rather than on the measurement, and a reviewer who notices the
    # register is not reading the science. State the range the result covers instead.
    "honest": "state the result and the range it covers",
    "does not establish": "state what the measurement bounds",
    "not evidence": "state what the quantity is",
    "says nothing about": "state the scope positively",
    "is not a measurement": "state what the quantity needs to be one",
    "no claim is made": "state the question as open",
    "nothing here shows": "state what the reads cover",
    "is a judgement": "state which quantity carries the result",
}


def text() -> str:
    return PAPER.read_text(encoding="utf-8")


def test_the_paper_is_there_to_check():
    """The premise. A missing or truncated file would pass every check below."""
    t = text()
    assert len(t.splitlines()) > 500, f"PAPER.md is {len(t.splitlines())} lines; expected the paper"
    assert "## Abstract" in t and "## References" in t


@pytest.mark.parametrize("phrase", sorted(BANNED))
def test_the_paper_does_not_narrate_its_own_history(phrase: str):
    t = text().lower()
    hits = [m.start() for m in re.finditer(re.escape(phrase), t)]
    where = [t[max(0, i - 60):i + 60].replace("\n", " ") for i in hits[:3]]
    assert not hits, (
        f"{len(hits)} use(s) of {phrase!r}; instead {BANNED[phrase]}. Context: {where}")


def test_the_title_states_a_mechanism_rather_than_a_fault_or_a_headline_ratio():
    """The supported finding is what the identity does to a derived floor. A title phrased as a
    deficit reads as a fault in the lens, and the lens is not at fault -- the identity is what the
    architecture puts there. A title carrying the raw ratio overstates it: most of the 2x-to-21x
    range is reproduced by a structure-free surrogate (§1.2), so the ratio is not a property of
    transformers and does not belong in a title."""
    first = text().splitlines()[0]
    assert first.startswith("# "), f"first line is not a title: {first!r}"
    assert "noise floor" in first.lower(), f"title no longer states the mechanism: {first!r}"
    for word in ("understates", "wrong", "fails", "recovers"):
        assert word not in first.lower(), (
            f"title frames the result as a fault or as the unadjusted ratio: {first!r}")


def test_the_recovered_directions_section_carries_its_evidence_and_its_control():
    """§1.7 rests on the two-fit reproduction, and that reproduction only means something beside
    the two controls a review supplied: the position control (block Jaccard decays as a power law
    in mode index, so a two-block comparison measures position) and the decode-matched null
    (random directions through the same readout overlap far more than uniform token draws)."""
    t = text()
    assert "### 1.7 What the recovered directions contain" in t, "the evidence section is gone"
    for token in ("0.390", "0.038", "power law", "no knee",
                  "exp53_hidden_modes_reproduce.py", "exp52_what_the_hidden_modes_name.py"):
        assert token in t, f"§1.7 no longer carries {token!r}"


def test_the_structure_free_control_is_reported_beside_the_gain():
    """The headline ratio is mostly what a variance-estimating floor does to any matrix carrying a
    scalar identity. Reporting the gain without the surrogate that reproduces most of it states a
    property of transformers that the measurement does not support."""
    t = text()
    for token in ("structure-free", "1.87", "Haar"):
        assert token in t, f"the structure-free control is no longer reported: {token!r} missing"


def test_the_crossing_reports_its_matched_control():
    """§8's published arms cannot separate 'this direction is structure' from 'a pseudo-inverse
    returns what was put into it'. The arm that can crosses an unresolved direction, and it scores
    higher than the resolved one."""
    t = text()
    assert "unresolved" in t, "§8 no longer reports the matched control"
    assert "+0.313" in t, "§8 no longer reports what the unresolved direction scores"


def test_the_two_claim_tables_name_the_catalogues_own_fit():
    """qwen3.5-4b publishes two fits. The published row has to be the one
    `entroptics-jlens fetch qwen3.5-4b` returns -- the n=1000 file, which reads 183 -- or a reader
    running the documented command gets a different number from the table."""
    for doc in (PAPER, ROOT / "README.md"):
        row = [ln for ln in doc.read_text(encoding="utf-8").splitlines()
               if ln.startswith("| qwen3.5-4b |")]
        assert row, f"{doc.name} has no qwen3.5-4b row"
        assert "183" in row[0] and "0.790" in row[0], (
            f"{doc.name} row is the n=417 fit, which is not what fetch returns: {row[0]}")
