"""The depth-profile page: one curve per model, plotted against relative depth.

Self-contained, no server, no CDN. Rewritten after every model and meta-refreshing while the run
is in flight, same contract as ``live_report``.

Relative depth (layer / last layer) is the x-axis because the question is whether the arc in
PR(J - alpha I) is a property of one model or of the architecture, and models of different depth
are only comparable once depth is normalised.
"""
from __future__ import annotations

import html
from pathlib import Path

CSS = """
:root { color-scheme: dark light; --bg:#0f1115; --fg:#e6e6e6; --mut:#8b93a7; --line:#252a35; }
* { box-sizing:border-box; }
body { margin:0; padding:28px 32px; background:var(--bg); color:var(--fg);
       font:14px/1.5 ui-monospace,SFMono-Regular,Menlo,Consolas,monospace; }
h1 { font-size:18px; margin:0 0 2px; letter-spacing:-.2px; }
h2 { font-size:14px; margin:26px 0 8px; color:var(--mut); font-weight:500;
     text-transform:uppercase; letter-spacing:.7px; }
.sub { color:var(--mut); font-size:12.5px; margin-bottom:18px; max-width:88ch; }
.badge { display:inline-block; padding:2px 8px; border-radius:10px; font-size:11px;
         margin-left:8px; vertical-align:2px; }
.running { background:#3a2f10; color:#f0c35e; } .done { background:#12331d; color:#5ec97a; }
.wrap { overflow-x:auto; border:1px solid var(--line); border-radius:8px; margin-bottom:8px; }
svg { display:block; }
table { border-collapse:collapse; width:100%; font-size:12.5px; }
th,td { padding:5px 10px; text-align:right; border-bottom:1px solid var(--line); white-space:nowrap; }
th { color:var(--mut); font-weight:500; }
th:first-child, td:first-child { text-align:left; }
.legend { margin:10px 0 4px; }
.legend span { margin-right:18px; font-size:12px; }
.dot { display:inline-block; width:9px; height:9px; border-radius:2px; margin-right:5px; }
.note { color:var(--mut); font-size:12px; margin:12px 0 0; max-width:88ch; }
"""

PALETTE = ["#6ea8fe", "#f0a35e", "#5ec97a", "#d98bd0", "#e0575b", "#5fd0d0", "#c9b458"]


def _curves(profiles, key, w=980, h=320, pad=52, logy=True):
    live = [p for p in profiles if p.get("layers")]
    if not live:
        return '<div style="padding:40px;color:#8b93a7">waiting for the first layer…</div>'
    import math
    vals = [r[key] for p in live for r in p["layers"] if r[key] > 0]
    if not vals:
        return '<div style="padding:40px;color:#8b93a7">no data</div>'
    lo, hi = min(vals), max(vals)
    if logy:
        def f(v):
            return math.log10(max(v, 1e-9))
        lo_t, hi_t = f(lo), f(hi)
    else:
        def f(v):
            return v
        lo_t, hi_t = 0.0, hi
    span = (hi_t - lo_t) or 1.0

    def px(d): return pad + d * (w - pad - 18)
    def py(v): return h - pad - (f(v) - lo_t) / span * (h - pad - 20)

    out = [f'<svg viewBox="0 0 {w} {h}" width="100%" height="{h}">']
    for frac in (0, 0.25, 0.5, 0.75, 1.0):
        t = lo_t + span * frac
        y = h - pad - frac * (h - pad - 20)
        v = (10 ** t) if logy else t
        lab = f"{v:.3g}"
        out.append(f'<line x1="{pad}" y1="{y:.1f}" x2="{w-18}" y2="{y:.1f}" stroke="#252a35"/>')
        out.append(f'<text x="{pad-8}" y="{y+4:.1f}" fill="#8b93a7" font-size="11" '
                   f'text-anchor="end">{lab}</text>')
    for d in (0.0, 0.25, 0.5, 0.75, 1.0):
        out.append(f'<text x="{px(d):.1f}" y="{h-pad+18}" fill="#8b93a7" font-size="11" '
                   f'text-anchor="middle">{d:.2f}</text>')
    out.append(f'<text x="{w/2:.0f}" y="{h-8}" fill="#8b93a7" font-size="11" '
               f'text-anchor="middle">relative depth (layer / last layer)</text>')
    for i, p in enumerate(live):
        c = PALETTE[i % len(PALETTE)]
        pts = " ".join(f"{px(r['depth']):.1f},{py(r[key]):.1f}"
                       for r in p["layers"] if r[key] > 0)
        out.append(f'<polyline points="{pts}" fill="none" stroke="{c}" stroke-width="2"/>')
        peak = max(p["layers"], key=lambda r: r[key])
        if peak[key] > 0:
            out.append(f'<circle cx="{px(peak["depth"]):.1f}" cy="{py(peak[key]):.1f}" r="4.5" '
                       f'fill="none" stroke="{c}" stroke-width="2"><title>{html.escape(p["name"])}'
                       f' peak: layer {peak["layer"]}, {peak[key]:.1f}</title></circle>')
    out.append("</svg>")
    return "".join(out)



#: Built outside the f-string: a backslash inside one is a syntax error before Python 3.12, and
#: this package supports 3.10.
def _partial_badge(p) -> str:
    if not p.get("partial"):
        return ""
    return (' <span style="color:#f0c35e">(partial, '
            f'{len(p["layers"])}/{p["n_layers"]})</span>')


def render(path: Path, profiles: list, elapsed: float, done: bool) -> None:
    live = [p for p in profiles if p.get("layers")]
    refresh = "" if done else '<meta http-equiv="refresh" content="3">'
    status = ('<span class="badge done">complete</span>' if done
              else '<span class="badge running">running…</span>')
    legend = "".join(
        f'<span><i class="dot" style="background:{PALETTE[i % len(PALETTE)]}"></i>'
        f'{html.escape(p["name"])} <span style="color:#8b93a7">(d={p["d_model"]}, '
        f'{p["n_layers"]}L)</span></span>' for i, p in enumerate(live))

    rows = "".join(
        f'<tr><td>{html.escape(p["name"])}'
        f'{_partial_badge(p)}'
        f'</td><td>{p["d_model"]}</td><td>{p["n_layers"]}</td>'
        f'<td>{p["n_prompts"]}</td><td>{p["peak_layer"]}</td>'
        f'<td>{p["peak_depth"]:.2f}</td><td>{p["peak_PR_M"]:.1f}</td>'
        f'<td>{p["layers"][-1]["identity_energy"]:.3f}</td></tr>' for p in live)

    page = f"""<!doctype html><meta charset="utf-8">{refresh}
<title>transport depth profile</title>
<style>{CSS}</style>
<h1>Depth profile of the residual transport{status}</h1>
<div class="sub">Participation ratio and Shannon effective rank of <b>J &minus; &alpha;I</b>, where
&alpha; = tr(J)/d is the exact Frobenius projection onto span(I). A residual stream adds
h<sub>l</sub> downstream, so J &rarr; I with depth and the identity's flat spectral block
otherwise dominates every read. No threshold, no null, no fitted constant.
&nbsp;&nbsp;{elapsed:.0f}s elapsed.</div>
<div class="legend">{legend}</div>

<h2>PR(J &minus; &alpha;I) &mdash; effective rank of the residual transport (log scale)</h2>
<div class="wrap">{_curves(profiles, "PR_M")}</div>

<h2>PR(J) &mdash; undecomposed, for contrast</h2>
<div class="wrap">{_curves(profiles, "PR_J")}</div>

<h2>Identity energy &mdash; &Vert;&alpha;I&Vert;&sup2; / &Vert;J&Vert;&sup2; (linear)</h2>
<div class="wrap">{_curves(profiles, "identity_energy", h=240, logy=False)}</div>

<h2>Peaks</h2>
<div class="wrap"><table>
<tr><th>model</th><th>d_model</th><th>layers</th><th>n_prompts</th><th>peak layer</th>
<th>rel. depth</th><th>peak PR(M)</th><th>final identity energy</th></tr>
{rows}</table></div>
<p class="note">Circles mark each model's peak. If the arc is architectural rather than a
property of one model, the peaks line up in relative depth. PR(J) undecomposed climbs
monotonically and shows no band &mdash; that curve is the identity growing, not the transport.</p>
"""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(page, encoding="utf-8")
