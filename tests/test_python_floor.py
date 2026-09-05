"""Every source file must parse under the Python version `pyproject.toml` claims to support.

A version claim cannot fail on a runner newer than the floor it claims, so a source using syntax
the floor cannot parse ships unseen. This checks the claim directly instead.

`ast.parse(..., feature_version=)` parses against a target grammar regardless of what is
executing, so it closes part of that gap on any interpreter.

**Only part.** `feature_version` gates the GRAMMAR and not the tokenizer, so what it rejects also
depends on what the running interpreter can lex. Measured on 3.12, asking for `feature_version=(3, 10)`:

    except* (3.11)              rejected   yes
    type X = int (3.12)         rejected   yes
    def f[T](x: T) (3.12)       rejected   yes
    f-string nested quotes      rejected   NO
    f-string with a backslash   rejected   NO

The two f-string rows flip on a 3.10 interpreter, where the tokenizer refuses them before the
grammar is consulted, so the gap opens exactly when the suite runs on something newer than the
floor. That is why the CI matrix carries a 3.10 leg alongside this file.

`ruff` covers the tokenizer half: its `target-version = "py310"` in `pyproject.toml` has its own
lexer and reports a newer f-string as `invalid-syntax`. The two checks are complementary and
neither is sufficient --

    ruff target-version   tokenizer-level: f-strings
    this test             grammar-level: except*, type aliases, PEP 695 generics
    the CI 3.10 matrix    the standard library, which neither of the above can see

and all three are cheap, so all three run.
"""
from __future__ import annotations

import ast
import re
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
PRUNE = {".git", ".venv", "build", "dist", "lenses", "out", "results", "streams",
         "__pycache__", ".pytest_cache", ".ruff_cache", "node_modules"}


def declared_floor() -> tuple[int, int]:
    """The floor `pyproject.toml` claims, read from the file rather than typed here.

    Parsed with a regex rather than a TOML library on purpose: `tomllib` is itself 3.11+, and a
    test about supporting 3.10 that cannot run on 3.10 would be the same joke twice.
    """
    text = (ROOT / "pyproject.toml").read_text(encoding="utf-8")
    m = re.search(r'requires-python\s*=\s*"[^"]*?(\d+)\.(\d+)', text)
    assert m, "pyproject.toml does not declare requires-python"
    return int(m.group(1)), int(m.group(2))


def sources() -> list[Path]:
    return sorted(p for p in ROOT.rglob("*.py") if not (PRUNE & set(p.relative_to(ROOT).parts)))


def test_the_floor_is_declared_and_plausible():
    major, minor = declared_floor()
    assert major == 3 and 8 <= minor <= 20, f"implausible floor {major}.{minor}"


@pytest.mark.parametrize("path", sources(), ids=lambda p: str(p.relative_to(ROOT)))
def test_every_source_parses_at_the_declared_floor(path: Path):
    # utf-8-sig: some sources in this workspace carry a BOM, which survives a bare utf-8 read as
    # U+FEFF and makes `ast.parse` raise on a file that is perfectly valid.
    source = path.read_text(encoding="utf-8-sig")
    try:
        ast.parse(source, filename=str(path), feature_version=declared_floor())
    except SyntaxError as exc:                       # pragma: no cover - only on a defect
        raise AssertionError(
            f"{path.relative_to(ROOT)}:{exc.lineno} does not parse under Python "
            f"{'.'.join(map(str, declared_floor()))}, which pyproject.toml claims to support: "
            f"{exc.msg}") from exc


@pytest.mark.parametrize("source,added_in", [
    ("try:\n    pass\nexcept* ValueError:\n    pass\n", (3, 11)),
    ("type X = int\n", (3, 12)),
    ("def f[T](x: T) -> T: return x\n", (3, 12)),
])
def test_the_check_actually_rejects_newer_grammar(source, added_in):
    """Proof that the parametrised test above can fail, on the class of syntax it does catch.

    The three cases are grammar constructs, which `feature_version` does gate. An f-string is
    not, which is why the module docstring states the scope and ruff covers the rest.
    """
    with pytest.raises(SyntaxError):
        ast.parse(source, feature_version=(3, 10))

    # `feature_version` only ever LOWERS the grammar the parser accepts; it cannot raise it above
    # the running interpreter's own. So the acceptance half is checkable only on an interpreter
    # that already has the syntax, and asserting it unconditionally fails on the 3.10 leg of the
    # matrix.
    if sys.version_info[:2] >= added_in:
        ast.parse(source, feature_version=added_in)  # accepted at the version that added it


def test_ruff_is_the_half_of_this_that_catches_f_strings():
    """The scope note, as an assertion. `feature_version` does not gate the tokenizer, so on an
    interpreter that can already lex the construct an f-string newer than the floor passes here.
    That makes ruff's pin load-bearing rather than stylistic.

    The gap is a property of the *running* interpreter, not of `feature_version`. On 3.12 the
    tokenizer accepts the nested quotes and `feature_version=(3, 10)` does not object; on 3.10
    the tokenizer rejects them outright. So the gap opens on an interpreter newer than the
    floor, which is why the CI matrix carries a 3.10 leg.
    """
    fstring_3_12 = 'x = {"k": 1}\ny = f"{x["k"]}"\n'
    if sys.version_info[:2] >= (3, 12):
        ast.parse(fstring_3_12, feature_version=(3, 10))   # accepted: the documented gap
    else:
        with pytest.raises(SyntaxError):                   # the tokenizer catches it here
            ast.parse(fstring_3_12, feature_version=(3, 10))

    major, minor = declared_floor()
    pyproject = (ROOT / "pyproject.toml").read_text(encoding="utf-8")
    assert f'target-version = "py{major}{minor}"' in pyproject, (
        f"ruff must be pinned to the declared floor py{major}{minor}; it is the only check here "
        f"that sees an f-string newer than the floor")
