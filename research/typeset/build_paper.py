"""Typeset a Markdown paper as a two-column arXiv-style preprint: <paper>.md -> <paper>.tex -> <paper>.pdf.

    python typeset/build_paper.py <paper>.md             # <paper>.tex and <paper>.pdf beside the Markdown
    python typeset/build_paper.py <paper>.md --tex-only  # the .tex only (needs pandoc, not TeX)
    python typeset/build_paper.py <paper>.md --keep-log  # also keep the TeX log (<paper>.log) beside it

Needs pandoc (3.x) and tectonic on PATH, or set PANDOC / TECTONIC to their executables; with
--engine xelatex any XeLaTeX installation is used instead of tectonic. Python standard library only.

The template (arxiv-twocolumn.latex) and filter (arxiv-twocolumn.lua) sit beside this script. The
Markdown file is never written to. It is read as GitHub-flavoured Markdown with $...$ math, with
typographic quotes, and with inline math allowed to close right before a digit ("$\\sim$3 dB"), which
pandoc's dollar rule would otherwise print as source. Its front matter is read from the text itself:

  * the first level-1 heading is the title, and a heading right under it is the subtitle;
  * a line naming the author is replaced by the template's author block (name, affiliation, e-mail,
    and the ORCID iD when the line carries one);
  * an italic line such as "*Pre-print. November 2026.*" is the date;
  * the "Abstract" section becomes the abstract, set across both columns under the title when the two
    leave room for the body on page 1, and as the opening of the columns otherwise;
  * anything else above the first section stays, as the first text of the body.

A YAML block at the top of the Markdown overrides any of these (title, subtitle, date, abstract,
author-name, author-affiliation, author-email, orcid, number-sections, figure-width).

The section level is the shallowest heading level below the title that holds more than one heading;
a lone heading above it (a part of the paper) is set as a divider. Sections are numbered by LaTeX
only when most headings at the section level carry no number of their own; --number-sections yes/no
overrides that.

The build reports the page count, the smallest scale applied to any display or table (nothing is
set below the template's 8 pt floor), anything that overflows, and every page that is less than a
quarter full and is neither a float page nor the last page.

The .tex is a function of the inputs alone. The PDF carries the date in SOURCE_DATE_EPOCH when that
is set, and 1970-01-01 otherwise, so it carries no build time.
"""

from __future__ import annotations

import argparse
import os
import re
import shutil
import subprocess
import sys
import tempfile
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
TEMPLATE = HERE / "arxiv-twocolumn.latex"
FILTER = HERE / "arxiv-twocolumn.lua"

AUTHOR = {
    "author-name": "Ikailo John Sessford",
    "author-affiliation": "Ikailo Inc.",
    "author-email": "john@ikailo.com",
}
AUTHOR_MARK = "Sessford"

# Characters every font of the template has (ASCII, Latin-1, Latin Extended-A, the common dashes,
# quotes, daggers, bullet and ellipsis). Every other character the paper uses gets a \typesetmap line,
# which in text takes it from the first font that has it.
# DERIVED: the Unicode block bounds of printable ASCII (0x20-0x7E), Latin-1 Supplement and Latin
# Extended-A (0xA0-0x17F), and the code points of the dashes, quotes, daggers, bullet and ellipsis,
# all of which Latin Modern and DejaVu Sans Mono carry (checked by a glyph probe under tectonic).
NATIVE = set(range(0x20, 0x7F)) | set(range(0xA0, 0x180)) | {
    0x2013, 0x2014, 0x2018, 0x2019, 0x201C, 0x201D, 0x2020, 0x2021, 0x2022, 0x2026}

HEADING = re.compile(r"^(#{1,6})\s+(.*?)\s*#*\s*$")
FENCE = re.compile(r"^(```|~~~)")
# A heading that carries its own number: "3", "3.2", "RF-3", "M-4.1", "§3", "A.", "Appendix B".
NUMBERED = re.compile(r"^(§\s*)?(([A-Z]{1,4}-)?\d+(\.\d+)*\.?|[A-Z]\.(\d+(\.\d+)*\.?)?|Appendix\b)\s")
UNNUMBERED_MATTER = re.compile(r"^(abstract|references|bibliography|acknowledg|declaration|keywords|"
                               r"mathematics subject|data availability|appendix$)", re.I)


def headings(lines: list[str]) -> list[tuple[int, int, str]]:
    """(line index, level, text) of every ATX heading outside fenced code."""
    out, fence = [], None
    for i, line in enumerate(lines):
        m = FENCE.match(line)
        if m:
            fence = None if fence == m.group(1) else (fence or m.group(1))
            continue
        if fence:
            continue
        h = HEADING.match(line)
        if h:
            out.append((i, len(h.group(1)), h.group(2)))
    return out


def split_yaml(text: str) -> tuple[str, str]:
    if text.startswith("---\n"):
        end = text.find("\n---\n", 4)
        # DERIVED: str.find returns -1 when the closing fence is absent.
        if end < 0:
            end = text.find("\n...\n", 4)
        # DERIVED: str.find's position of the closing fence; -1 when absent.
        if end > 0:
            return text[: end + 5], text[end + 5:]
    return "", text


def yaml_keys(block: str) -> set[str]:
    return {m.group(1) for m in re.finditer(r"^([A-Za-z][\w-]*)\s*:", block, re.M)}


def front_matter(body: str) -> tuple[dict, str]:
    """Metadata read from the paper's own front matter, and the body without it."""
    lines = body.split("\n")
    hs = headings(lines)
    meta: dict = {}
    # DERIVED: 1 is the level of a "# " heading, the title.
    if not hs or hs[0][1] != 1:
        return meta, body
    t_i, _, title = hs[0]
    meta["title"] = title
    drop = set(range(0, t_i + 1))
    # The front block runs from the title to the first heading of the body (the subtitle aside).
    rest = hs[1:]
    first = next((j for j in range(t_i + 1, len(lines)) if lines[j].strip()), -1)
    if (rest and rest[0][0] == first and not NUMBERED.match(rest[0][2])
            and rest[0][2].strip().rstrip(".").lower() != "abstract"):
        meta["subtitle"] = rest[0][2]
        drop.add(rest[0][0])
        rest = rest[1:]
    end = rest[0][0] if rest else len(lines)
    for j in range(t_i + 1, end):
        s = lines[j].strip()
        if j in drop:
            continue
        # CHOSEN: an author line is one short line; 200 characters keeps a body paragraph that names
        # the author from being read as the author line.
        if AUTHOR_MARK in s and len(s) < 200 and not s.startswith(">"):
            drop.add(j)
            k = j + 1
            while k < end and lines[k].strip() and "ORCID" in lines[k]:
                drop.add(k)
                k += 1
            m = re.search(r"ORCID\s*\[?(\d{4}-\d{4}-\d{4}-\d{3}[\dX])", "\n".join(lines[j:k]))
            if m:
                meta["orcid"] = m.group(1)
        elif re.fullmatch(r"\*[^*].*[^*]\*|_[^_].*[^_]_", s) and re.search(r"(pre-?print|\b(19|20)\d\d\b)", s, re.I):
            meta["date"] = s[1:-1]
            drop.add(j)
    # The abstract: a heading "Abstract" anywhere before the body's first numbered section.
    for idx, (i, level, text) in enumerate(rest):
        if text.strip().rstrip(".").lower() == "abstract":
            # DERIVED: idx + 1 is the next heading, which ends the abstract section.
            stop = rest[idx + 1][0] if idx + 1 < len(rest) else len(lines)
            k = i + 1
            while k < stop and lines[k].strip() not in ("---", "***", "___"):
                k += 1
            meta["abstract"] = "\n".join(lines[i + 1: k]).strip()
            drop.update(range(i, k))
            # The rule closing the abstract goes with it.
            while k < stop and (not lines[k].strip() or lines[k].strip() in ("---", "***", "___")):
                drop.add(k)
                k += 1
            break
        if NUMBERED.match(text):
            break
    kept = [line for j, line in enumerate(lines) if j not in drop]
    # Rules that only separated the dropped front matter go too.
    out: list[str] = []
    for line in kept:
        if line.strip() in ("---", "***", "___") and not "".join(out).strip():
            continue
        out.append(line)
    return meta, "\n".join(out)


def strip_leading_rules(text: str) -> str:
    lines = text.split("\n")
    while lines and (not lines[0].strip() or lines[0].strip() in ("---", "***", "___")):
        lines.pop(0)
    return "\n".join(lines)


def math_before_digit(body: str) -> str:
    """Let inline math end right before a digit, as in "$\\sim$3 dB".

    Pandoc's dollar-math rule refuses a closing $ followed by a digit and would print the TeX source.
    An empty HTML comment after the closing $ (which the LaTeX output drops) keeps it math. Code spans
    and fenced code are left alone.
    """
    out, fence = [], None
    for line in body.split("\n"):
        m = FENCE.match(line)
        if m:
            fence = None if fence == m.group(1) else (fence or m.group(1))
        if fence or m or line.startswith("    ") or "$" not in line:
            out.append(line)
            continue
        parts = re.split(r"(`+[^`]*`+)", line)
        parts = [p if p.startswith("`") else _close_before_digit(p) for p in parts]
        out.append("".join(parts))
    return "\n".join(out)


def _close_before_digit(s: str) -> str:
    """Pair the $ signs as pandoc does (an opener is followed by a non-space, a closer preceded by one;
    $$ and \\$ are not single dollars), and mark each closer that a digit follows."""
    # DERIVED: every 1 and 2 below is a character offset: a backslash escapes the one character after
    # it (skip 2), "$$" is two characters (skip 2), and i - 1 / i + 1 are the characters either side of
    # a "$".
    marks, i, opener = [], 0, None
    while i < len(s):
        c = s[i]
        if c == "\\":
            i += 2
            continue
        if c == "$":
            if s.startswith("$$", i):
                i += 2
                continue
            if opener is None:
                # DERIVED: i + 1 is the character after the "$".
                if i + 1 < len(s) and not s[i + 1].isspace():
                    opener = i
            # DERIVED: i - 1 is the character before the "$"; the math must hold at least one.
            elif not s[i - 1].isspace() and i - 1 > opener:
                # DERIVED: i + 1 is the character after the closing "$".
                if i + 1 < len(s) and s[i + 1].isdigit():
                    marks.append(i + 1)
                opener = None
        i += 1
    for m in reversed(marks):
        s = s[:m] + "<!-- -->" + s[m:]
    return s


def section_level(body: str) -> int:
    """The heading level the paper's sections are written at (the title is already out of the body).

    The shallowest level that holds more than one heading; a level with a single heading above it is a
    part divider ("# Further reads" between numbered "## 3." sections), not the section level.
    """
    levels = [lvl for _, lvl, _ in headings(body.split("\n"))]
    # DERIVED: with no headings there is nothing to shift, so the section level is the first, 1.
    if not levels:
        return 1
    # DERIVED: more than one heading, 1, is what makes a level a sequence of sections rather than a
    # single divider.
    several = sorted(lvl for lvl in set(levels) if levels.count(lvl) > 1)
    return several[0] if several else min(levels)


def numbering(body: str) -> bool:
    """True when LaTeX should number the sections: most of the paper's own section headings carry no
    number of their own."""
    hs = [(lvl, t) for _, lvl, t in headings(body.split("\n"))]
    if not hs:
        return False
    top = section_level(body)
    sections = [t for lvl, t in hs if lvl == top and not UNNUMBERED_MATTER.match(t)]
    if not sections:
        return False
    carried = sum(1 for t in sections if NUMBERED.match(t))
    # DERIVED: a majority test, carried > len / 2, written without division; a paper that numbers
    # most of its own sections is not numbered again.
    return carried * 2 < len(sections)


def glyph_map(texts: list[str]) -> str:
    chars = sorted({ord(c) for t in texts for c in t} - NATIVE)
    # DERIVED: 0x7F is the last ASCII code point; 0xD800-0xDFFF is the UTF-16 surrogate block, never
    # a character; 0xFE0F is the emoji variation selector, which selects a presentation and prints
    # nothing.
    chars = [c for c in chars if c > 0x7F and not (0xD800 <= c <= 0xDFFF) and c not in (0xFE0F,)]
    if not chars:
        return ""
    return "\\AtBeginDocument{%\n" + "".join(f'\\typesetmap{{"{c:04X}}}% {chr(c)}\n' for c in chars) + "}\n"


def yaml_literal(meta: dict) -> str:
    """A YAML metadata block holding each value as a literal block scalar (read as Markdown by pandoc)."""
    if not meta:
        return ""
    out = ["---"]
    for k, v in meta.items():
        out.append(f"{k}: |-")
        out.extend(("  " + line) if line.strip() else "" for line in v.split("\n"))
    return "\n".join(out) + "\n---\n\n"


def tool(env: str, name: str) -> str:
    exe = os.environ.get(env) or shutil.which(name)
    if not exe:
        sys.exit(f"{name} not found: put it on PATH or set {env}")
    return exe


def overfull(log: str, limit: float) -> list[str]:
    out = []
    for m in re.finditer(r"Overfull \\hbox \(([\d.]+)pt too wide\)[^\n]*", log):
        if float(m.group(1)) > limit:
            out.append(m.group(0))
    return out


def sparse_pages(log: str) -> list[tuple[int, float]]:
    """Pages of text less than a quarter full, from the template's "typeset-column" lines.

    A page's fill is the mean over its columns of (text + floats) / text height; a column the page
    never started counts as empty. Float-only pages are made without a column and are not logged, and
    the last page is allowed to be short.
    """
    cols: dict[int, list[float]] = {}
    width: dict[int, int] = {}
    for m in re.finditer(r"typeset-column: (\d+) (\d) (-?\d+) (-?\d+) (\d+)", log):
        page, n, text, floats, height = (int(g) for g in m.groups())
        cols.setdefault(page, []).append((text + floats) / height)
        width[page] = n
    if not cols:
        return []
    last = max(cols)
    # CHOSEN: a quarter of a page, the brief's line for a near-empty page; it decides only what the
    # build reports, never the layout.
    quarter = 0.25
    return [(p, sum(f) / width[p]) for p, f in sorted(cols.items())
            if p != last and sum(f) / width[p] < quarter]


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("markdown", type=Path, help="the paper's Markdown file")
    ap.add_argument("--out", type=Path, help="output stem (default: the Markdown's path without .md)")
    ap.add_argument("--number-sections", choices=("auto", "yes", "no"), default="auto")
    ap.add_argument("--tex-only", action="store_true", help="write the .tex and stop")
    ap.add_argument("--engine", choices=("tectonic", "xelatex"), default="tectonic")
    ap.add_argument("--keep-log", action="store_true", help="keep the TeX log next to the PDF")
    # CHOSEN: 20 pt is the report threshold the brief set for an overfull line: about two characters
    # past the column edge, visible on the page, while TeX's sub-point overshoots are not reported.
    # It only decides what is printed, never the layout.
    ap.add_argument("--overfull-pt", type=float, default=20.0, help="report overfull lines wider than this")
    a = ap.parse_args(argv)

    md = a.markdown.resolve()
    root = md.parent
    stem = (a.out.resolve() if a.out else md.with_suffix(""))
    tex_path, pdf_path = stem.with_suffix(".tex"), stem.with_suffix(".pdf")

    text = md.read_text(encoding="utf-8").replace("\r\n", "\n")
    yaml_block, body = split_yaml(text)
    given = yaml_keys(yaml_block)
    meta, body = front_matter(body)
    body = math_before_digit(strip_leading_rules(body))
    meta = {k: math_before_digit(v) for k, v in meta.items() if k not in given}
    # Plain values (not Markdown, so an e-mail address is not turned into a link) go in as variables.
    plain = {k: v for k, v in AUTHOR.items() if k not in given}
    if "orcid" in meta:
        plain["orcid"] = meta.pop("orcid")

    top = section_level(body)
    numbered = {"yes": True, "no": False}.get(a.number_sections)
    if numbered is None:
        flag = re.search(r"^number-sections\s*:\s*(\S+)", yaml_block, re.M)
        numbered = flag.group(1).lower() in ("true", "yes") if flag else numbering(body)

    pandoc = tool("PANDOC", "pandoc")
    ver = re.search(r"(\d+)\.(\d+)", subprocess.run([pandoc, "--version"], capture_output=True,
                                                     text=True).stdout)
    # DERIVED: (0, 0) stands for an unreadable version, below every release, so it fails the check below.
    ver = (int(ver.group(1)), int(ver.group(2))) if ver else (0, 0)
    # DERIVED: pandoc 3.0 introduced the Figure block and the Lua API the filter uses.
    if ver < (3, 0):
        sys.exit(f"pandoc 3.x is needed; {pandoc} is {ver[0]}.{ver[1]}")
    # DERIVED: pandoc 3.8 replaced --no-highlight with --syntax-highlighting=none.
    plain_code = "--syntax-highlighting=none" if ver >= (3, 8) else "--no-highlight"
    t0 = time.time()
    with tempfile.TemporaryDirectory(prefix="typeset-") as tmp:
        tmpd = Path(tmp)
        src = tmpd / "body.md"
        src.write_text(yaml_literal(meta) + yaml_block + body, encoding="utf-8")
        glyphs = tmpd / "glyphs.tex"
        glyphs.write_text(glyph_map([text]), encoding="utf-8")
        cmd = [pandoc, str(src), "-f", "gfm+implicit_figures+ascii_identifiers+smart", "-t", "latex",
               "--template", str(TEMPLATE), "--lua-filter", str(FILTER),
               "-H", str(glyphs),
               "-M", f"typeset-section-level={top}", plain_code,
               "--resource-path", str(root), "--wrap=preserve", "-o", str(tex_path)]
        for k, v in plain.items():
            cmd += ["-V", f"{k}={v}"]
        if numbered:
            cmd.append("--number-sections")
        r = subprocess.run(cmd, cwd=root)
        if r.returncode:
            return r.returncode
        print(f"wrote {tex_path} ({'numbered' if numbered else 'unnumbered'} sections)")
        if a.tex_only:
            return 0

        env = dict(os.environ)
        # DERIVED: 0 is the Unix epoch, a date that names no build; a caller's own value wins.
        env.setdefault("SOURCE_DATE_EPOCH", "0")
        outd = tmpd / "out"
        outd.mkdir()
        if a.engine == "tectonic":
            cmd = [tool("TECTONIC", "tectonic"), "--keep-logs", "--chatter", "minimal",
                   "--outdir", str(outd), str(tex_path)]
            r = subprocess.run(cmd, cwd=tex_path.parent, env=env)
        else:
            xe = tool("XELATEX", "xelatex")
            # CHOSEN: three passes, the usual LaTeX count: the second resolves page references and
            # PDF bookmarks written by the first, the third settles any page break the second moved.
            # Tectonic decides its own reruns.
            for _ in range(3):
                r = subprocess.run([xe, "-interaction=nonstopmode", "-halt-on-error",
                                    f"-output-directory={outd}", str(tex_path)],
                                   cwd=tex_path.parent, env=env, stdout=subprocess.DEVNULL)
                if r.returncode:
                    break
        logf = outd / (tex_path.stem + ".log")
        log = logf.read_text(encoding="utf-8", errors="replace") if logf.exists() else ""
        if a.keep_log and logf.exists():
            shutil.copyfile(logf, stem.with_suffix(".log"))
        if r.returncode:
            # CHOSEN: the last 4000 characters of the log, enough to hold a TeX error with its context
            # lines; the whole log is kept with --keep-log.
            print(log[-4000:], file=sys.stderr)
            return r.returncode
        shutil.copyfile(outd / (tex_path.stem + ".pdf"), pdf_path)
    m = re.search(r"Output written on \S+ \((\d+) pages?", log)
    pages = m.group(1) if m else "?"
    print(f"wrote {pdf_path} ({pages} pages, {time.time() - t0:.0f} s)")
    wide = overfull(log, a.overfull_pt)
    missing = sorted(set(re.findall(r"Missing character: There is no (.) ", log)))
    for w in wide:
        print("  " + w)
    if missing:
        print("  missing glyphs: " + " ".join(missing))
    print(f"  {len(wide)} overfull lines over {a.overfull_pt:g} pt; {len(missing)} missing glyphs")
    scales = [int(s) for s in re.findall(r"typeset-scale: (\d+)", log)]
    strips = re.findall(r"typeset-strip: (\w+)", log)
    spills = re.findall(r"typeset-overflow: ([\d.]+pt) into ([\d.]+pt)", log)
    # DERIVED: the template logs scales in thousandths.
    smallest = f"{min(scales) / 1000:.3f}" if scales else "none (nothing scaled)"
    print(f"  smallest scale: {smallest} over {len(scales)} scaled; {len(strips)} set across both "
          f"columns ({', '.join(sorted(set(strips))) or '-'}); {len(spills)} too wide at the floor")
    for w, into in spills:
        print(f"    {w} wide into {into}")
    sparse = sparse_pages(log)
    print(f"  pages under a quarter full (not float pages, not the last): "
          f"{', '.join(f'p{p} ({f:.0%})' for p, f in sparse) or 'none'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
