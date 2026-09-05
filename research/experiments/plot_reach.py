"""How far a mean-Jacobian linearisation carries, by depth, as inline SVG.

Explained variance of the transport against the PRE-norm final residual direction, the target
`J` actually maps into (`exp18_prenorm_target.py`). The shape is the
claim: flat and near zero through the layers a lens is most wanted for, climbing only at the end.

    python experiments/plot_reach.py results/prenorm_target.json --out results/reach.svg
"""
from __future__ import annotations

import argparse
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

import entroptics_jlens as je                                          # noqa: E402

W, H = 720, 320
L, R, T, B = 58, 26, 20, 46


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("src", type=Path, nargs="*",
                    default=[Path("results/reach_4b.json"), Path("results/second_model.json")],
                    help="one or more reach files; plotted against RELATIVE depth so models of "
                         "different depth compare")
    ap.add_argument("--out", type=Path, default=Path("results/reach.svg"))
    a = ap.parse_args(argv)
    # `load_complete` rather than `json.load`: a figure built from a partial sweep looks
    # finished, and a figure is the artefact that gets published.
    series = [(f.stem, je.load_complete(f)) for f in a.src]

    def sx(rel):
        return L + rel * (W - L - R)

    def sy(v):
        return T + (1.0 - v) * (H - T - B)

    def val(r):
        """cos^2 on the readout-space metric; older files carry other keys."""
        if "cos2" in r:
            return r["cos2"]
        if "cos_a" in r:
            return max(r["cos_a"], r["cos_b"]) ** 2
        return 1.0 - min(r["aff_a"], r["aff_b"]) ** 2

    p = [f'<svg viewBox="0 0 {W} {H}" xmlns="http://www.w3.org/2000/svg" '
         f'font-family="ui-sans-serif,system-ui,sans-serif" font-size="11" role="img" '
         f'aria-label="explained variance of the transport against relative depth">']
    for v in (0.0, 0.25, 0.5, 0.75, 1.0):
        y = sy(v)
        p.append(f'<line x1="{L}" y1="{y:.1f}" x2="{W-R}" y2="{y:.1f}" stroke="#1E2740"/>')
        p.append(f'<text x="{L-9}" y="{y+3.5:.1f}" text-anchor="end" fill="#7C86A0">'
                 f'{v:.2f}</text>')
    for rel in (0.0, 0.25, 0.5, 0.75, 1.0):
        p.append(f'<text x="{sx(rel):.1f}" y="{H-B+16}" text-anchor="middle" fill="#7C86A0">'
                 f'{rel:.2f}</text>')
    p.append(f'<line x1="{L}" y1="{sy(0.5):.1f}" x2="{W-R}" y2="{sy(0.5):.1f}" '
             f'stroke="#E0776D" stroke-width="1.2" stroke-dasharray="5 4"/>')
    p.append(f'<text x="{L+8}" y="{sy(0.5)-6:.1f}" fill="#E0776D">half the variance</text>')
    for i, (name, doc) in enumerate(series):
        col = ("#5CC9CD", "#F0B45C")[i % 2]
        pts = [(sx(r["rel_depth"]), sy(val(r))) for r in doc["rows"]]
        if i == 0:
            area = (f'M{pts[0][0]:.1f},{sy(0.0):.1f} '
                    + " ".join(f"L{x:.1f},{y:.1f}" for x, y in pts)
                    + f' L{pts[-1][0]:.1f},{sy(0.0):.1f} Z')
            p.append(f'<path d="{area}" fill="{col}" fill-opacity="0.12"/>')
        p.append('<path d="' + " ".join(f"{'M' if j == 0 else 'L'}{x:.1f},{y:.1f}"
                                        for j, (x, y) in enumerate(pts))
                 + f'" fill="none" stroke="{col}" stroke-width="2.4"/>')
        for x, y in pts:
            p.append(f'<circle cx="{x:.1f}" cy="{y:.1f}" r="3.0" fill="{col}"/>')
        lab = f'{doc.get("model", name)} (d={doc.get("d_model", "?")})'
        p.append(f'<line x1="{L+14}" y1="{T+12+i*19}" x2="{L+34}" y2="{T+12+i*19}" '
                 f'stroke="{col}" stroke-width="2.4"/>')
        p.append(f'<text x="{L+39}" y="{T+15.5+i*19}" fill="#B6BECD">{lab}</text>')
    p.append(f'<text x="{L}" y="{H-8}" fill="#7C86A0">relative depth</text>')
    p.append(f'<text transform="translate(14,{sy(0.5):.1f}) rotate(-90)" text-anchor="middle" '
             f'fill="#7C86A0">explained variance of the final state</text>')
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
