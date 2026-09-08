"""Every prose surface states what is true now, positively, and does not narrate its own history.

The rules are `skills/doc-current-state`: no change history, failure narration, self-justification,
hedging, defensive framing, or spurious negation. Text that says a result was withdrawn, or that a
read "was wrong", spends its credibility describing its drafts instead of its evidence. What a
reader needs is what each read measures and the range it holds over.

Checked across the paper, the README, CONTRIBUTING, the experiments index and the package's own
docstrings. Guarding the paper alone leaves the surface most readers actually meet unguarded.

The vocabulary below is banned outright rather than judged in context. Each entry appeared in one
of these files and each was replaceable by a positive statement of the same fact -- "the read on
$J$" for "the uncorrected read", "what bounds the measurement" for "the library's own limits".
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


def prose_files() -> list[Path]:
    """Every file whose wording a reader meets: the prose, and the docstrings behind `help()`."""
    docs = [PAPER, ROOT / "README.md", ROOT / "CONTRIBUTING.md",
            ROOT / "research" / "experiments" / "README.md"]
    return [p for p in docs + sorted((ROOT / "src" / "entroptics_jlens").glob("*.py"))
            if p.exists()]

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
    # Narrating what did not work. The measurement that came of it is worth stating; the account
    # of arriving at it describes the work rather than the result.
    "does not work": "state the regime each read is right about",
    "doesn't work": "state the regime each read is right about",
    "went wrong": "state what the read measures",
    "measured the wrong thing": "state what the present read measures",
    "obvious fix": "name the method and where it applies",
    # "failure mode" is deliberately absent. It is a technical term with a real use here -- the
    # paper probes opposite failure modes of a degradation, and cites a known one from the
    # literature -- and banning it would forbid describing a method's behaviour.
    "not hypothetical": "give the measurement without the preamble",
    "explained away": "state the mechanism",
    "has actually happened": "state what the check covers",
    "devastating": "state the number and its cause",
    "nothing here fixes": "state what the read establishes",
    "nothing here closes": "state what each read is exact about",
}


def text() -> str:
    return PAPER.read_text(encoding="utf-8")


def test_the_paper_is_there_to_check():
    """The premise. A missing or truncated file would pass every check below."""
    t = text()
    assert len(t.splitlines()) > 500, f"PAPER.md is {len(t.splitlines())} lines; expected the paper"
    assert "## Abstract" in t and "## References" in t


def test_every_prose_surface_is_there_to_check():
    """The same premise for the wider sweep. A collection that came back short would pass the
    check below while leaving most of the words a reader meets unread."""
    files = prose_files()
    names = {p.name for p in files}
    assert {"PAPER.md", "README.md", "CONTRIBUTING.md"} <= names, sorted(names)
    assert len(files) >= 18, f"only {len(files)} prose files collected: {sorted(names)}"


@pytest.mark.parametrize("phrase", sorted(BANNED))
def test_no_prose_surface_narrates_its_own_history(phrase: str):
    """The paper and everything else a reader meets, including the docstrings behind `help()`."""
    found = []
    for path in prose_files():
        t = path.read_text(encoding="utf-8").lower()
        for m in list(re.finditer(re.escape(phrase), t))[:2]:
            i = m.start()
            found.append(f"{path.name}: ...{t[max(0, i - 55):i + 55]!r}...".replace("\n", " "))
    assert not found, (
        f"{len(found)} use(s) of {phrase!r}; instead {BANNED[phrase]}.\n  " + "\n  ".join(found))


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
