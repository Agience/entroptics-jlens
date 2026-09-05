"""Results files that say whether they are finished.

Several experiments write their JSON incrementally, so a long run can be watched and a killed one
keeps what it computed. That convenience created a silent truncation: a `--help` invocation on a
script that lacked an argument parser re-ran the experiment, a timeout killed it at layer 26 of 31,
and the file it left behind was well-formed JSON that a later script read as a complete result.
The failure surfaced as `KeyError: 27` in an unrelated experiment.

A partial file is fine. A partial file that reads as whole is the bug. So every incremental write
stamps ``complete: false`` and the final write stamps ``complete: true``, and readers use
``load_complete``, which refuses anything else with a non-recoverable error rather than working
from part of a run.

The convention originated in ``exp4_stream_complement.py``, which had carried its own
``write_out(complete)`` since the section 7 measurements were first recorded. This module is that
idea lifted out of one script and given a reader, and exp4 now writes through it.
"""
from __future__ import annotations

import json
import subprocess
from functools import lru_cache
from pathlib import Path


class IncompleteResults(RuntimeError):
    """A results file whose run was interrupted, read as though it had finished."""


@lru_cache(maxsize=1)
def code_version() -> str:
    """The commit these numbers were produced by, with a marker when the tree is dirty.

    Cached for the life of the process. It spawns two ``git`` subprocesses, measured at 117-197 ms
    per call, and ``dump`` is called once per layer by design -- a 31-layer sweep was paying about
    four seconds for the same answer 31 times. Caching also pins the stamp to what the tree held
    when the run began, which is the more honest reading of "the commit these numbers were
    produced by" for a run that outlives an edit.

    Recorded because the alternative was measured. A committed exp4 result was rebuilt from its
    own provenance block -- same model, same lens hash, same prompt count, same library versions,
    same seed -- and 8 of 220 numeric fields came back different, all of them from the one read
    whose implementation had changed since. The block recorded every input except the arithmetic.

    Returns ``"unknown"`` rather than raising when git is unavailable: a results file that cannot
    name its commit is still worth writing, and saying so beats guessing.
    """
    try:
        root = Path(__file__).resolve().parents[2]
        sha = subprocess.run(["git", "-C", str(root), "rev-parse", "--short", "HEAD"],
                             capture_output=True, text=True, timeout=10)
        if sha.returncode != 0:
            return "unknown"
        dirty = subprocess.run(["git", "-C", str(root), "status", "--porcelain"],
                               capture_output=True, text=True, timeout=10)
        suffix = "+dirty" if dirty.returncode == 0 and dirty.stdout.strip() else ""
        return sha.stdout.strip() + suffix
    except (OSError, subprocess.SubprocessError):
        return "unknown"


#: The keys this module owns. They are removed from the payload before the merge rather than
#: merged over, because ``{**payload, "complete": v}`` keeps the key's FIRST insertion position:
#: a payload already carrying ``complete`` -- which is exactly what ``load_complete`` returns, so
#: read-modify-write produces one -- put the stamp at byte 4 of the file with the right value and
#: the wrong position, silently voiding the truncation guarantee below.
_STAMPED = ("code_commit", "complete")


def dump(path, payload: dict, *, complete: bool) -> None:
    """Write ``payload`` with its completion state and the producing commit stamped on it.

    Call it inside the loop with ``complete=False`` and once more after the loop with
    ``complete=True``. The completion key is written last so a truncated *write* is also invalid
    JSON rather than a valid file claiming completion -- and it is written last even when the
    payload already carries one, which needs the explicit removal above rather than a merge.

    ``code_version`` is stamped on every file, because the inputs alone do not determine a result.
    A ``code_commit`` in the payload is replaced by the real one for the same reason.
    """
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    body = {k: v for k, v in payload.items() if k not in _STAMPED}
    p.write_text(json.dumps({**body, "code_commit": code_version(),
                             "complete": bool(complete)}, indent=2),
                 encoding="utf-8", newline="\n")


def load_complete(path) -> dict:
    """Read a results file, refusing one whose run was interrupted.

    Files written before this convention existed carry no ``complete`` key, and some hold a bare
    list rather than an object. Both are refused: an unstamped file gives no evidence either way,
    and guessing is what this module exists to stop.
    """
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"{p}: no results file; run the experiment that writes it")
    doc = json.loads(p.read_text(encoding="utf-8"))
    if not isinstance(doc, dict):
        raise IncompleteResults(
            f"{p}: holds a bare {type(doc).__name__}, so it predates the completion stamp and "
            f"carries no evidence that its run finished. Re-run the experiment that produces it.")
    if doc.get("complete") is not True:
        raise IncompleteResults(
            f"{p}: written by a run that did not finish (complete="
            f"{doc.get('complete')!r}). Re-run the experiment that produces it; reading part of a "
            f"sweep as though it were the whole sweep is how a truncated file becomes a result.")
    return doc
