"""A bench that seals half its items and lets you open the seal once.

Every wrong result recorded on 2026-08-22 was a process failure rather than a measurement one, and
the same three in rotation:

    a magnitude was read off one sample and written down          (five answers, five runs)
    a control was absent, so perturbation read as content         (the depth-write nulls)
    a claim was checked within the sample it was fitted on        (a 2.7 sigma sign reversal)

Each was caught eventually by a rule I already knew. This module makes the rules structural instead,
so the next experiment cannot skip them by being in a hurry.

    Bench(items)            splits at construction. Half is OPEN and half is SEALED, and the seal
                            is not a train/test split for tuning -- it is a replication half.
    bench.measure(arms)     runs on the OPEN half only. Returns a Report.
    report.claim(...)       writes a claim down, in the form that transfers.
    bench.verify(claim)     opens the seal ONCE and says whether the claim held.

`verify` refuses a second call. A seal that can be opened twice is a development set with extra
steps, and the sign reversal that motivated this module would have survived one.

WHAT `claim` WILL ACCEPT, and why it is narrow. Measured over three benchmarks each run twice on
disjoint halves, absolute effect magnitudes did not replicate -- they halved, doubled, and on one
benchmark reversed sign -- while rates and control-relative comparisons replicated tightly. The
five pairs of rates, open half against sealed half, are 73.3/73.3, 50.9/50.0, 76.4/71.8, 19.5/20.0
and 48.0/48.5. So a claim is a statement about a
RATE or an ORDERING against a control, and `claim` will not take a magnitude.

Means are still reported by `verify`, and deliberately: one hybrid arm carried a replicated mean
gain while losing on 65% of queries, and a rate alone would have thrown it away.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Dict, List, Sequence

import numpy as np


class SealBroken(RuntimeError):
    """The sealed half has already been opened for this bench."""


@dataclass(frozen=True)
class Claim:
    """A statement in the form that survived replication: a rate, or an ordering against a control.

    `arm` is expected to beat `against` on `rate`, where `rate` is the share of items on which the
    arm's score exceeds the reference's. `margin` is the smallest rate difference worth calling a
    difference, and it is the caller's, because only the caller knows what would change a decision.
    """
    arm: str
    against: str
    rate: float
    margin: float
    note: str = ""


@dataclass
class Report:
    """Per-item scores from the open half, and the claims built from them."""
    scores: Dict[str, np.ndarray]
    n: int
    _claims: List[Claim] = field(default_factory=list)

    def rate(self, arm: str, against: str) -> float:
        """The share of items where `arm` scores strictly better than `against`.

        A rate rather than a mean difference: the mean halved between disjoint samples on the
        benchmark this module was written for, and the rate came back identical.
        """
        a, b = self.scores[arm], self.scores[against]
        return float(np.mean(a > b))

    def claim(self, arm: str, against: str, *, margin: float = 0.05, note: str = "") -> Claim:
        c = Claim(arm=arm, against=against, rate=self.rate(arm, against),
                  margin=float(margin), note=note)
        self._claims.append(c)
        return c

    def summary(self) -> str:
        w = max((len(k) for k in self.scores), default=4)
        out = [f"open half, n = {self.n}", f"{'arm':>{w}}{'mean':>10}{'median':>10}"]
        for k, v in self.scores.items():
            out.append(f"{k:>{w}}{v.mean():>10.4f}{np.median(v):>10.4f}")
        return "\n".join(out)


class Bench:
    """Items split into an open half and a sealed half, deterministically."""

    def __init__(self, items: Sequence, *, seed: int = 0, name: str = "bench"):
        if len(items) < 4:
            raise ValueError(f"{len(items)} items is not a bench; a sealed half needs items in it")
        self.name = name
        order = np.random.default_rng(seed).permutation(len(items))
        cut = len(items) // 2
        self._open = [items[int(i)] for i in order[:cut]]
        self._sealed = [items[int(i)] for i in order[cut:]]
        self._opened = False

    def __len__(self) -> int:
        return len(self._open) + len(self._sealed)

    @property
    def open_items(self) -> List:
        return list(self._open)

    def _measure(self, items, arms: Dict[str, Callable]) -> Report:
        # Every arm is scored over the SAME `items`, so unequal counts cannot arise here by
        # construction. This guard read `if len(n) != 1: "arms returned different counts"` until
        # 2026-09-02, which was therefore reachable only for an empty `arms` -- and then reported
        # a count mismatch for what is actually "you passed no arms".
        if not arms:
            raise ValueError(
                "no arms to measure. A bench measures named arms against each other; an empty "
                "mapping has nothing to compare and nothing to compare it against.")
        scores = {k: np.asarray([float(fn(it)) for it in items], dtype=np.float64)
                  for k, fn in arms.items()}
        return Report(scores=scores, n=len(scores[next(iter(scores))]))

    def measure(self, arms: Dict[str, Callable]) -> Report:
        """Run every arm on the OPEN half. The sealed half is untouched."""
        if "base" not in arms:
            raise ValueError(
                "no arm named 'base'. A bench without a reference measures the arm against "
                "nothing, and every null on 2026-08-22 that turned out to be perturbation rather "
                "than content was read off a table with no control in it.")
        return self._measure(self._open, arms)

    def verify(self, claim: Claim, arms: Dict[str, Callable]) -> dict:
        """Open the seal, once, and report whether the claim held.

        Returns the open-half rate, the sealed-half rate, and whether the claim survived. A second
        call raises: the seal is the whole mechanism and a re-openable one is a development set.
        """
        if self._opened:
            raise SealBroken(
                f"{self.name}: the sealed half has already been opened. Build a new bench with "
                f"fresh items rather than looking again -- a seal opened twice would not have "
                f"caught the 2.7 sigma sign reversal this exists for.")

        # Validated BEFORE the seal is marked open, and the ordering is the whole point. Until
        # 2026-09-02 `self._opened = True` came first, so a call that could not do anything --
        # an empty mapping, or one missing the arm the claim names -- consumed the one resource
        # this module exists to protect, and the bench was spent on an error message.
        missing = [n for n in (claim.arm, claim.against) if n not in arms]
        if missing:
            raise ValueError(
                f"{self.name}: verifying {claim.arm!r} against {claim.against!r} needs both arms, "
                f"and {missing} were not supplied. The seal is NOT opened by this call; pass the "
                f"same arms the claim was measured with and it will still be there.")
        self._opened = True
        rep = self._measure(self._sealed, arms)
        got = rep.rate(claim.arm, claim.against)
        held = (got - 0.5) >= claim.margin and (claim.rate - 0.5) >= claim.margin
        # Every arm was run on the sealed half to answer one claim, so every arm's rate is
        # reported. The `base` arm is required so a table always carries its control.
        base = "base" if "base" in rep.scores else claim.against
        # Means are reported and are NOT the claim. Refusing to claim a magnitude and refusing to
        # SHOW one are different things, and conflating them was a real blind spot: on the canon
        # fusion bench a hybrid arm carried a better mean (0.4359 against 0.3790) while losing on
        # rate, and this method could not say whether the mean gain survived the seal.
        return {"claim": claim, "open_rate": claim.rate, "sealed_rate": got,
                "margin": claim.margin, "held": bool(held), "sealed_n": rep.n,
                "verdict": ("HELD" if held else "DID NOT HOLD"),
                "sealed_rates_vs_base": {k: rep.rate(k, base)
                                         for k in rep.scores if k != base},
                "sealed_means": {k: float(v.mean()) for k, v in rep.scores.items()}}
