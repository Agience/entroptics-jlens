"""A self-contained local page that fills in while the run is going.

No server, no external assets, no dependencies. The page is rewritten after every layer and
carries a meta-refresh while the run is in flight; the final write drops the refresh, so an
open tab stops repainting the moment the run finishes. Open it once:

    start results/live.html          # Windows
    open  results/live.html          # macOS

Inline SVG, because a chart that needs a CDN is a chart that does not render offline.
"""
from __future__ import annotations

import html
import json
from pathlib import Path

CSS = """
:root { color-scheme: dark light; --bg:#0f1115; --fg:#e6e6e6; --mut:#8b93a7; --line:#252a35;
        --mp:#6ea8fe; --rob:#f0a35e; --ok:#5ec97a; --bad:#e0575b; }
* { box-sizing: border-box; }
body { margin:0; padding:28px 32px; background:var(--bg); color:var(--fg);
       font:14px/1.5 ui-monospace,SFMono-Regular,Menlo,Consolas,monospace; }
h1 { font-size:18px; margin:0 0 2px; letter-spacing:-.2px; }
.sub { color:var(--mut); font-size:12.5px; margin-bottom:20px; }
.badge { display:inline-block; padding:2px 8px; border-radius:10px; font-size:11px;
         margin-left:8px; vertical-align:2px; }
.running { background:#3a2f10; color:#f0c35e; }
.done { background:#12331d; color:var(--ok); }
.grid { display:grid; grid-template-columns:repeat(auto-fit,minmax(150px,1fr)); gap:10px;
        margin-bottom:22px; }
.card { border:1px solid var(--line); border-radius:8px; padding:10px 12px; }
.card .k { color:var(--mut); font-size:11px; text-transform:uppercase; letter-spacing:.6px; }
.card .v { font-size:20px; margin-top:2px; }
.wrap { overflow-x:auto; border:1px solid var(--line); border-radius:8px; margin-bottom:22px; }
svg { display:block; }
table { border-collapse:collapse; width:100%; font-size:12.5px; }
th,td { padding:5px 10px; text-align:right; border-bottom:1px solid var(--line);
        white-space:nowrap; }
th { color:var(--mut); font-weight:500; text-align:right; position:sticky; top:0;
     background:var(--bg); }
th:first-child, td:first-child { text-align:left; }
.mp { color:var(--mp); } .rob { color:var(--rob); }
.note { color:var(--mut); font-size:12px; margin:14px 0 0; max-width:80ch; }
.legend span { margin-right:16px; font-size:12px; }
.dot { display:inline-block; width:9px; height:9px; border-radius:2px; margin-right:5px; }
"""


def _chart(rows, w=920, h=280, pad=44):
    """K vs layer, both providers. Inline SVG, linear axes, no library."""
    if not rows:
        return '<div style="padding:40px;color:#8b93a7">waiting for the first layer…</div>'
    xs = [r["layer"] for r in rows]
    ks = [r["K_mp"] for r in rows] + [r["K_robust"] for r in rows]
    x0, x1 = min(xs), max(xs)
    y1 = max(ks) or 1
    span_x = (x1 - x0) or 1

    def px(l): return pad + (l - x0) / span_x * (w - pad - 16)
    def py(k): return h - pad - (k / y1) * (h - pad - 18)

    out = [f'<svg viewBox="0 0 {w} {h}" width="100%" height="{h}">']
    # gridlines + y labels
    for frac in (0, 0.25, 0.5, 0.75, 1.0):
        v = y1 * frac
        y = py(v)
        out.append(f'<line x1="{pad}" y1="{y:.1f}" x2="{w-16}" y2="{y:.1f}" stroke="#252a35"/>')
        out.append(f'<text x="{pad-8}" y="{y+4:.1f}" fill="#8b93a7" font-size="11" '
                   f'text-anchor="end">{v:.0f}</text>')
    # x labels
    step = max(1, len(xs) // 12)
    for l in xs[::step]:
        out.append(f'<text x="{px(l):.1f}" y="{h-pad+18}" fill="#8b93a7" font-size="11" '
                   f'text-anchor="middle">{l}</text>')
    out.append(f'<text x="{w/2:.0f}" y="{h-6}" fill="#8b93a7" font-size="11" '
               f'text-anchor="middle">layer</text>')
    for key, colour in (("K_mp", "#6ea8fe"), ("K_robust", "#f0a35e")):
        pts = " ".join(f"{px(r['layer']):.1f},{py(r[key]):.1f}" for r in rows)
        out.append(f'<polyline points="{pts}" fill="none" stroke="{colour}" stroke-width="2"/>')
        for r in rows:
            out.append(f'<circle cx="{px(r["layer"]):.1f}" cy="{py(r[key]):.1f}" r="2.5" '
                       f'fill="{colour}"><title>layer {r["layer"]}: {key}={r[key]}</title></circle>')
    out.append("</svg>")
    return "".join(out)


def render(path: Path, meta: dict, rows: list, done: bool) -> None:
    """Write the page. Called after every layer; cheap."""
    refresh = "" if done else '<meta http-equiv="refresh" content="2">'
    status = ('<span class="badge done">complete</span>' if done else
              '<span class="badge running">running…</span>')
    ks = [r["K_mp"] for r in rows]
    cards = [
        ("model", meta.get("name", "—")),
        ("d_model", meta.get("d_model", "—")),
        ("layers read", f"{len(rows)} / {meta.get('n_layers', '?')}"),
        ("n_prompts", meta.get("n_prompts", "—")),
        ("far", meta.get("far", "—")),
        ("K_mp range", f"{min(ks)}–{max(ks)}" if ks else "—"),
        ("peak layer", str(max(rows, key=lambda r: r["K_mp"])["layer"]) if rows else "—"),
        ("elapsed", f"{meta.get('elapsed', 0):.1f}s"),
    ]
    card_html = "".join(
        f'<div class="card"><div class="k">{html.escape(str(k))}</div>'
        f'<div class="v">{html.escape(str(v))}</div></div>' for k, v in cards)

    head = ("<tr><th>layer</th><th>K_mp</th><th>K_robust</th><th>E_mp</th><th>E_robust</th>"
            "<th>floor_mp</th><th>floor_rob</th><th>&sigma;&#8321;</th>"
            "<th>controls mp/rob</th><th>s</th></tr>")
    body = []
    for r in rows:
        c = r.get("controls")
        ctl = ("  ".join(f"{c[k]['K_mp']}/{c[k]['K_robust']}"
                         for k in ("gaussian", "shuffled", "matched_spectrum")) if c else "")
        body.append(
            f'<tr><td>{r["layer"]}</td><td class="mp">{r["K_mp"]}</td>'
            f'<td class="rob">{r["K_robust"]}</td>'
            f'<td>{r["energy_resolved_mp"]:.3f}</td><td>{r["energy_resolved_robust"]:.3f}</td>'
            f'<td>{r["floor_mp"]:.4g}</td><td>{r["floor_robust"]:.4g}</td>'
            f'<td>{r["sigma_top"]:.4g}</td><td>{ctl}</td>'
            f'<td>{r.get("seconds", 0):.1f}</td></tr>')

    page = f"""<!doctype html><meta charset="utf-8">{refresh}
<title>resolved rank — {html.escape(str(meta.get('name', 'lens')))}</title>
<style>{CSS}</style>
<h1>Resolved rank of the Jacobian transport{status}</h1>
<div class="sub">{html.escape(str(meta.get('path', '')))}</div>
<div class="grid">{card_html}</div>
<div class="legend">
  <span><i class="dot" style="background:#6ea8fe"></i>K under <b>mp</b> (default null)</span>
  <span><i class="dot" style="background:#f0a35e"></i>K under <b>robust</b> null</span>
</div>
<div class="wrap">{_chart(rows)}</div>
<div class="wrap"><table>{head}{''.join(body)}</table></div>
<p class="note">K = #(&sigma;<sub>k</sub> &gt; floor), the modes the transport resolves above a
derived Tracy&ndash;Widom edge. No fitted constant, no corpus, no forward pass. The two
providers estimate the noise variance differently and neither dominates on real transports, so
both are reported. Controls: gaussian / shuffled / matched-spectrum &mdash; the first two must
read 0, the third preserves K by construction.</p>
"""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(page, encoding="utf-8")


def dump_json(path: Path, run: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(run, indent=2), encoding="utf-8")
