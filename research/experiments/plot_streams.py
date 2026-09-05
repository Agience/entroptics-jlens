"""A self-contained comparison page for the stream reads of experiment 4.

The etendue result is a claim about a SHAPE -- an interior peak in one model, a strictly
monotone decline in the others -- and a shape belongs in a picture. Tables make "0.158, 0.127,
0.102, 0.032, ..." look like an interior structure that is not there.

Plots against relative depth so models of different depth compare, and marks each curve's
maximum. A curve whose maximum sits at relative depth 0 has no interior peak, which is the whole
question.

    python experiments/plot_streams.py results/etendue_gpt2.json \\
        results/etendue_pythia.json results/streams_qwen08_n12.json \\
        --out results/etendue.html

No server, no CDN, inline SVG.
"""
from __future__ import annotations

import argparse
import html
import math
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

import entroptics_jlens as je                                          # noqa: E402

PALETTE = ["#6ea8fe", "#f0a35e", "#5ec97a", "#d98bd0", "#e0575b", "#5fd0d0", "#c9b458"]

CSS = """
:root { color-scheme: dark light; --bg:#0f1115; --fg:#e6e6e6; --mut:#8b93a7; --line:#252a35; }
* { box-sizing:border-box; }
body { margin:0; padding:28px 32px; background:var(--bg); color:var(--fg);
       font:14px/1.5 ui-monospace,SFMono-Regular,Menlo,Consolas,monospace; }
h1 { font-size:18px; margin:0 0 4px; letter-spacing:-.2px; }
h2 { font-size:13px; margin:26px 0 8px; color:var(--mut); font-weight:500;
     text-transform:uppercase; letter-spacing:.7px; }
.sub { color:var(--mut); font-size:12.5px; margin-bottom:16px; max-width:92ch; }
.wrap { overflow-x:auto; border:1px solid var(--line); border-radius:8px; margin-bottom:6px; }
svg { display:block; }
table { border-collapse:collapse; width:100%; font-size:12.5px; }
th,td { padding:5px 10px; text-align:right; border-bottom:1px solid var(--line);
        white-space:nowrap; }
th { color:var(--mut); font-weight:500; }
th:first-child, td:first-child { text-align:left; }
.legend { margin:8px 0 4px; }
.legend span { margin-right:18px; font-size:12px; }
.dot { display:inline-block; width:9px; height:9px; border-radius:2px; margin-right:5px; }
.note { color:var(--mut); font-size:12px; margin:10px 0 0; max-width:92ch; }
"""


def curves(series, key, label, w=980, h=300, pad=56, logy=False):
    vals = [r[key] for _, rows in series for r in rows if r[key] > 0]
    if not vals:
        return "<div style='padding:40px;color:#8b93a7'>no data</div>"
    lo, hi = min(vals), max(vals)
    f = (lambda v: math.log10(max(v, 1e-12))) if logy else (lambda v: v)
    lo_t, hi_t = (f(lo), f(hi)) if logy else (0.0, hi)
    span = (hi_t - lo_t) or 1.0

    def px(d): return pad + d * (w - pad - 20)
    def py(v): return h - pad - (f(v) - lo_t) / span * (h - pad - 22)

    out = [f'<svg viewBox="0 0 {w} {h}" width="100%" height="{h}">']
    for frac in (0, 0.25, 0.5, 0.75, 1.0):
        y = h - pad - frac * (h - pad - 22)
        v = (10 ** (lo_t + span * frac)) if logy else (lo_t + span * frac)
        out.append(f'<line x1="{pad}" y1="{y:.1f}" x2="{w-20}" y2="{y:.1f}" stroke="#252a35"/>')
        out.append(f'<text x="{pad-8}" y="{y+4:.1f}" fill="#8b93a7" font-size="11" '
                   f'text-anchor="end">{v:.3g}</text>')
    for d in (0.0, 0.25, 0.5, 0.75, 1.0):
        out.append(f'<text x="{px(d):.1f}" y="{h-pad+18}" fill="#8b93a7" font-size="11" '
                   f'text-anchor="middle">{d:.2f}</text>')
    out.append(f'<text x="{w/2:.0f}" y="{h-8}" fill="#8b93a7" font-size="11" '
               f'text-anchor="middle">relative depth</text>')
    out.append(f'<text x="14" y="16" fill="#8b93a7" font-size="11">{html.escape(label)}</text>')
    for i, (name, rows) in enumerate(series):
        c = PALETTE[i % len(PALETTE)]
        last = rows[-1]["layer"] or 1
        pts = " ".join(f"{px(r['layer']/last):.1f},{py(r[key]):.1f}"
                       for r in rows if r[key] > 0)
        out.append(f'<polyline points="{pts}" fill="none" stroke="{c}" stroke-width="2"/>')
        pk = max(rows, key=lambda r: r[key])
        out.append(f'<circle cx="{px(pk["layer"]/last):.1f}" cy="{py(pk[key]):.1f}" r="4.5" '
                   f'fill="none" stroke="{c}" stroke-width="2"><title>{html.escape(name)} peak '
                   f'{pk[key]:.3f} at layer {pk["layer"]}</title></circle>')
    out.append("</svg>")
    return "".join(out)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("runs", nargs="+", type=Path)
    ap.add_argument("--out", type=Path, required=True)
    a = ap.parse_args(argv)

    series, meta = [], []
    for path in a.runs:
        if not path.is_file():
            raise SystemExit(f"refusing: {path} does not exist. This page reports measurements; "
                             f"it does not invent a curve for a run that was not made.")
        run = je.load_complete(path)              # refuse a partial sweep
        rows = sorted(run["layers"], key=lambda r: r["layer"])
        # one rank spec per page, else two curves for one model would be plotted as if they
        # were two models
        specs = {r.get("rank_spec", "pr") for r in rows}
        if len(specs) > 1:
            rows = [r for r in rows if r.get("rank_spec") == "pr"]
            if not rows:
                raise SystemExit(f"refusing: {path} holds several rank specs {sorted(specs)} and "
                                 f"none is 'pr'; pick one before plotting.")
        name = run["model"].split("/")[-1]
        series.append((name, rows))
        last = rows[-1]["layer"] or 1
        pk = max(rows, key=lambda r: r["etendue_match"])
        mono = all(rows[i]["etendue_match"] <= rows[i - 1]["etendue_match"] + 1e-9
                   for i in range(1, len(rows)))
        meta.append({"name": name, "d": run["d_model"], "L": len(rows),
                     "prompts": run["prompts"], "peak": pk["etendue_match"],
                     "peak_layer": pk["layer"], "peak_depth": pk["layer"] / last,
                     "ratio": pk["etendue_match"] / rows[0]["etendue_match"]
                              if rows[0]["etendue_match"] > 0 else float("nan"),
                     "monotone": mono})

    legend = "".join(
        f'<span><i class="dot" style="background:{PALETTE[i % len(PALETTE)]}"></i>'
        f'{html.escape(m["name"])} <span style="color:#8b93a7">(d={m["d"]}, {m["L"]}L, '
        f'n={m["prompts"]})</span></span>' for i, m in enumerate(meta))
    rows_html = "".join(
        f'<tr><td>{html.escape(m["name"])}</td><td>{m["d"]}</td><td>{m["L"]}</td>'
        f'<td>{m["peak"]:.3f}</td><td>{m["peak_layer"]}</td><td>{m["peak_depth"]:.2f}</td>'
        f'<td>{m["ratio"]:.2f}</td>'
        f'<td>{"yes" if m["monotone"] else "<b>no</b>"}</td></tr>' for m in meta)

    page = f"""<!doctype html><meta charset="utf-8">
<title>stream reads across models</title>
<style>{CSS}</style>
<h1>What the transport carries, on real residual streams</h1>
<div class="sub">Etendue match is the ratio of the two sides' phase space &mdash; how much of the
stream's phase space the transport can carry, a bound rather than an observation. A curve whose
maximum sits at relative depth 0 has no interior band; only a non-monotone curve does. Circles
mark each maximum.</div>
<div class="legend">{legend}</div>

<h2>Etendue match (log scale)</h2>
<div class="wrap">{curves(series, "etendue_match", "etendue match", logy=True)}</div>

<h2>Directions actually carried</h2>
<div class="wrap">{curves(series, "K_transport_dirs", "resolved directions of the transported frame")}</div>

<h2>Modes the stream resolves</h2>
<div class="wrap">{curves(series, "K_stream", "K(stream)")}</div>

<h2>Summary</h2>
<div class="wrap"><table>
<tr><th>model</th><th>d</th><th>layers</th><th>peak match</th><th>at layer</th>
<th>rel. depth</th><th>peak/first</th><th>monotone?</th></tr>
{rows_html}</table></div>
<p class="note">A peak at relative depth 0 with peak/first = 1.00 is a monotone decline: no
interior structure. The complement result (essentially all of the stream's resolved modes lying
outside what the transport carries) holds in every model here regardless of that shape.</p>
"""
    a.out.parent.mkdir(parents=True, exist_ok=True)
    a.out.write_text(page, encoding="utf-8")
    print(f"wrote {a.out.resolve().as_uri()}")
    for m in meta:
        print(f"  {m['name']:<18} peak {m['peak']:.3f} @ layer {m['peak_layer']} "
              f"(rel {m['peak_depth']:.2f})  peak/first {m['ratio']:.2f}  "
              f"monotone={m['monotone']}")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (FileNotFoundError, je.IncompleteResults) as exc:
        print(f"refused: {exc}", file=sys.stderr)
        raise SystemExit(2)
