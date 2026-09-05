"""The sensitivity curve of B4' as inline SVG, for embedding in the report.

A resolution limit is a shape -- accuracy climbing out of chance as the true gap grows -- and the
table hides it. Two coverage series, one per degradation mode, against a chance line, on a log
gap axis because the gaps span three decades.

    python experiments/plot_sensitivity.py results/ranking_sensitivity.json --out results/sens.svg
"""
from __future__ import annotations

import argparse
import math
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

import entroptics_jlens as je                                          # noqa: E402

W, H = 720, 330
L, R, T, B = 62, 152, 22, 46
LO, HI = 1e-4, 1e-1

COL = {"noise": "#F0B45C", "rank": "#5CC9CD", "participation": "#7C86A0"}


def sx(g):
    t = (math.log10(max(g, LO)) - math.log10(LO)) / (math.log10(HI) - math.log10(LO))
    return L + t * (W - L - R)


def sy(a):
    return T + (1.0 - a) * (H - T - B)


def series(rows, kind, key, col, dash=""):
    pts = [(sx(r.get("true_gap", r.get("gap"))), sy(r["acc"][key]))
           for r in rows if r["kind"] == kind]
    pts.sort()
    d = " ".join(f"{'M' if i == 0 else 'L'}{x:.1f},{y:.1f}" for i, (x, y) in enumerate(pts))
    out = [f'<path d="{d}" fill="none" stroke="{col}" stroke-width="2.2"'
           f'{f" stroke-dasharray={chr(34)}{dash}{chr(34)}" if dash else ""}/>']
    for x, y in pts:
        out.append(f'<circle cx="{x:.1f}" cy="{y:.1f}" r="3.6" fill="{col}"/>')
    return "\n".join(out)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("src", type=Path, nargs="?",
                    default=Path("results/recalibrate.json"))
    ap.add_argument("--out", type=Path, default=Path("results/sensitivity.svg"))
    a = ap.parse_args(argv)
    rows = je.load_complete(a.src)["rows"]        # refuse a partial sweep, see plot_reach

    p = [f'<svg viewBox="0 0 {W} {H}" xmlns="http://www.w3.org/2000/svg" '
         f'font-family="ui-sans-serif,system-ui,sans-serif" font-size="11" role="img" '
         f'aria-label="ordering accuracy against true transport-error gap">']
    for a_ in (0.0, 0.25, 0.5, 0.75, 1.0):                       # horizontal grid + y labels
        y = sy(a_)
        p.append(f'<line x1="{L}" y1="{y:.1f}" x2="{W-R}" y2="{y:.1f}" stroke="#1E2740"/>')
        p.append(f'<text x="{L-9}" y="{y+3.5:.1f}" text-anchor="end" fill="#7C86A0">'
                 f'{a_:.2f}</text>')
    for g, lab in ((1e-4, "0.01%"), (1e-3, "0.1%"), (1e-2, "1%"), (1e-1, "10%")):
        x = sx(g)
        p.append(f'<line x1="{x:.1f}" y1="{T}" x2="{x:.1f}" y2="{H-B}" stroke="#1E2740"/>')
        p.append(f'<text x="{x:.1f}" y="{H-B+16}" text-anchor="middle" fill="#7C86A0">'
                 f'{lab}</text>')
    p.append(f'<line x1="{L}" y1="{sy(0.5):.1f}" x2="{W-R}" y2="{sy(0.5):.1f}" '
             f'stroke="#E0776D" stroke-width="1.3" stroke-dasharray="5 4"/>')
    p.append(f'<text x="{W-R-4}" y="{sy(0.5)-6:.1f}" text-anchor="end" fill="#E0776D">'
             f'chance</text>')
    p.append(series(rows, "noise", "coverage", COL["noise"]))
    p.append(series(rows, "rank", "coverage", COL["rank"]))
    p.append(series(rows, "rank", "participation", COL["participation"], dash="3 3"))
    for i, (lab, col) in enumerate(((f"coverage {chr(0x2014)} added noise", COL["noise"]),
                                    (f"coverage {chr(0x2014)} lost rank", COL["rank"]),
                                    ("participation ratio", COL["participation"]))):
        y = T + 12 + i * 19
        p.append(f'<line x1="{W-R+6}" y1="{y}" x2="{W-R+26}" y2="{y}" stroke="{col}" '
                 f'stroke-width="2.2"/>')
        p.append(f'<text x="{W-R+31}" y="{y+3.5}" fill="#B6BECD">{lab}</text>')
    p.append(f'<text x="{L}" y="{H-8}" fill="#7C86A0">true gap, centred cosine to the '
             f'pre-norm residual (log)</text>')
    p.append(f'<text transform="translate(15,{sy(0.5):.1f}) rotate(-90)" text-anchor="middle" '
             f'fill="#7C86A0">layers ordered correctly</text>')
    p.append("</svg>")
    a.out.parent.mkdir(parents=True, exist_ok=True)
    a.out.write_text("\n".join(p), encoding="utf-8", newline="\n")
    print(f"wrote {a.out}")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (FileNotFoundError, je.IncompleteResults) as exc:
        print(f"refused: {exc}", file=sys.stderr)
        raise SystemExit(2)
