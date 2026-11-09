-- arxiv-twocolumn.lua: the pandoc filter that goes with arxiv-twocolumn.latex.
--
-- It changes layout only, never text:
--   * inline code becomes \texttt with break points after punctuation, so long identifiers wrap, and
--     a word of more than 24 characters in prose (a path, a commit hash) gets break points too;
--   * a figure becomes a column-width float, or a full-width one when the image is landscape; a
--     paragraph right after it that starts "Figure N" becomes its caption, and a .pdf sibling of a
--     raster image is used when one exists;
--   * a display equation goes to \typesetdisplay, which sets it in the column, scaled to fit no
--     smaller than the template's floor, or else across both columns (one with a \tag, \label,
--     numbered environment or top-level line break is left to TeX); \displaystyle inside inline math
--     is set as \textstyle;
--   * a table is laid out from the length of its cells: at its natural width in the column when it
--     fits, wrapped in the column when it has at most three columns, otherwise full width; a table too
--     long to float breaks across columns or pages instead. A "Table N" paragraph right before or
--     after a floated table travels with it as its caption;
--   * headings are shifted so the paper's section level is \section; a heading above it is a part
--     divider;
--   * a horizontal rule right before a section heading is dropped (the heading separates them).
--
-- Widths are counted in characters of the table type (\footnotesize, Latin Modern): about 58 to a
-- column and 120 across the page for the geometry the template sets.

local stringify = pandoc.utils.stringify

-- DERIVED: footnotesize Latin Modern sets about 4.2 pt a character on average; the column is 248 pt
-- and the text 514 pt, less a margin of a few characters for the estimate's error.
local COL_CHARS = 58
local FULL_CHARS = 120
-- DERIVED: 2\tabcolsep = 8 pt, two characters.
local CELL_PAD = 2
-- DERIVED: a full-width float holds at most the text height, about 75 footnotesize lines; 70 leaves
-- room for the caption and rules.
local FLOAT_LINES = 70
-- CHOSEN: an unbreakable in-column table may take about two thirds of a column (45 of 66 lines);
-- longer ones break across columns, so a table never pushes most of a column of text away.
local INLINE_LINES = 45

local meta_figure_width = 'auto'

local function ulen(s)
  return utf8.len(s) or #s
end

local function to_latex_inlines(inlines)
  local doc = pandoc.Pandoc({ pandoc.Plain(inlines) })
  local out = pandoc.write(doc, 'latex')
  return (out:gsub('%s+$', ''))
end

local function to_latex_blocks(blocks)
  local out = pandoc.write(pandoc.Pandoc(blocks), 'latex')
  out = out:gsub('%s+$', '')
  return out
end

-- Inline code -----------------------------------------------------------------------------------------

local ESC = {
  ['\\'] = '\\textbackslash{}', ['{'] = '\\{', ['}'] = '\\}', ['$'] = '\\$', ['&'] = '\\&',
  ['#'] = '\\#', ['^'] = '\\^{}', ['_'] = '\\_', ['%'] = '\\%', ['~'] = '\\textasciitilde{}',
}
local BREAK_AFTER = { ['/'] = true, ['.'] = true, [','] = true, ['_'] = true, ['='] = true, ['-'] = true,
  [':'] = true, ['('] = true, [')'] = true, ['['] = true, [']'] = true, ['|'] = true, [';'] = true,
  ['>'] = true, ['<'] = true, ['+'] = true, ['*'] = true }

local function code_latex(text)
  local parts = {}
  for _, cp in utf8.codes(text) do
    local ch = utf8.char(cp)
    local e = ESC[ch]
    if ch == '-' then e = '-{}' end
    if ch == "'" then e = '\\textquotesingle{}' end
    if ch == '`' then e = '\\textasciigrave{}' end
    if ch == '<' then e = '\\textless{}' end
    if ch == '>' then e = '\\textgreater{}' end
    if ch == '|' then e = '\\textbar{}' end
    if ch == ' ' then e = '\\ ' end
    parts[#parts + 1] = e or ch
    if BREAK_AFTER[ch] then parts[#parts + 1] = '\\allowbreak{}' end
  end
  return '\\texttt{' .. table.concat(parts) .. '}'
end

local function Code(el)
  return pandoc.RawInline('latex', code_latex(el.text))
end

-- A word too long to hyphenate (a path, a hash, an identifier in prose) gets break points after
-- its punctuation and every eight characters of a long alphanumeric run.
-- CHOSEN: 24 characters is about half a column of 10 pt text; an English word that long is rare
-- enough that only paths, hashes and identifiers reach it. Eight characters between break points keep
-- a hash's pieces long enough to read and short enough to fill a line.
local LONG_WORD = 24
local function Str(el)
  if ulen(el.text) <= LONG_WORD then return nil end
  local out, piece, run = pandoc.List(), {}, 0
  local function flush()
    if #piece > 0 then out:insert(pandoc.Str(table.concat(piece))) end
    piece = {}
    out:insert(pandoc.RawInline('latex', '\\allowbreak{}'))
  end
  for _, cp in utf8.codes(el.text) do
    local ch = utf8.char(cp)
    piece[#piece + 1] = ch
    if ch:match('%w') then
      run = run + 1
      if run >= 8 and run % 8 == 0 then flush() end
    else
      run = 0
      if BREAK_AFTER[ch] then flush() end
    end
  end
  if #piece > 0 then out:insert(pandoc.Str(table.concat(piece))) end
  if out[#out].t == 'RawInline' then out:remove(#out) end
  return out
end

-- The display broken into lines at its top-level \qquad separators (outside braces and \left..\right),
-- as a gathered block; a part that is only \text{...} stays on the line before it. Empty when the
-- display has no such separator.
local function split_display(t)
  local parts, depth, start, i = {}, 0, 1, 1
  while i <= #t do
    local c = t:sub(i, i)
    if c == '\\' then
      if t:sub(i, i + 5) == '\\qquad' and not t:sub(i + 6, i + 6):match('%a') and depth == 0 then
        parts[#parts + 1] = t:sub(start, i - 1)
        start = i + 6
        i = i + 6
      elseif t:sub(i, i + 4) == '\\left' and not t:sub(i + 5, i + 5):match('%a') then
        depth = depth + 1; i = i + 5
      elseif t:sub(i, i + 5) == '\\right' and not t:sub(i + 6, i + 6):match('%a') then
        depth = depth - 1; i = i + 6
      else
        i = i + 2
      end
    else
      if c == '{' then depth = depth + 1 elseif c == '}' then depth = depth - 1 end
      i = i + 1
    end
  end
  parts[#parts + 1] = t:sub(start)
  if #parts < 2 then return '' end
  local lines = {}
  for _, p in ipairs(parts) do
    local s = p:gsub('^%s+', ''):gsub('%s+$', '')
    if s ~= '' then
      if #lines > 0 and s:match('^\\text%b{}[%s,.;]*$') then
        lines[#lines] = lines[#lines] .. '\\qquad ' .. s
      else
        lines[#lines + 1] = s
      end
    end
  end
  if #lines < 2 then return '' end
  return '\\begin{gathered}' .. table.concat(lines, '\\\\ ') .. '\\end{gathered}'
end

local function DisplayMath(el)
  if el.mathtype == 'InlineMath' then
    -- Inline math is set at text size: a \displaystyle written into it would push the line apart.
    if el.text:find('\\displaystyle', 1, true) then
      return pandoc.Math('InlineMath', (el.text:gsub('\\displaystyle', '\\textstyle')))
    end
    return nil
  end
  local t = el.text
  for _, s in ipairs({ '\\tag', '\\label', '\\begin{align', '\\begin{gather', '\\begin{multline',
                       '\\begin{equation', '\\begin{eqnarray' }) do
    if t:find(s, 1, true) then return nil end
  end
  -- A line break outside an inner environment needs display mode proper.
  local outer = t:gsub('\\begin{(%a+%*?)}.-\\end{%1}', '')
  if outer:find('\\\\', 1, true) then return nil end
  return pandoc.RawInline('latex', '\\typesetdisplay{' .. t .. '}{' .. split_display(t) .. '}')
end

-- Headings: the paper's section level (build_paper.py reads it from the Markdown) becomes \section.
-- A heading above that level is a part of the paper and is set as a divider, not a section.
local section_level = 1
local function Header(el)
  local level = el.level - (section_level - 1)
  if level >= 1 then
    el.level = level
    return el
  end
  return pandoc.RawBlock('latex', '\\typesetpart{' .. to_latex_inlines(el.content) .. '}{'
    .. stringify(el.content):gsub('[\\{}%%#&$^_~]', '') .. '}')
end

-- Lengths --------------------------------------------------------------------------------------------

-- The visible length of math source: commands and grouping removed.
local function math_len(tex)
  local s = tex:gsub('\\[a-zA-Z]+', 'x'):gsub('[{}%^_]', '')
  -- CHOSEN: 0.8, math glyphs with their sub- and superscripts set narrower than the same count of
  -- text characters; only the table-width estimate uses it, and TeX checks the real width.
  return math.max(1, math.floor(ulen(s) * 0.8 + 0.5))
end

-- DERIVED: DejaVu Sans Mono's advance is 0.6 em, about 4.9 pt at footnotesize after the template's
-- lowercase match, against about 4.2 pt for an average Latin Modern character.
local MONO = 1.2

local function visible(blocks)
  local n, longest, cur = 0, 0, 0
  -- k: the run's width; seg: its longest unbreakable piece (nil: the whole run).
  local function word(k, seg)
    n = n + k
    if seg then
      cur = cur + seg
      if cur > longest then longest = cur end
      cur = 0
    else
      cur = cur + k
      if cur > longest then longest = cur end
    end
  end
  local function space() n = n + 1; cur = 0 end
  local function code(t)
    local seg = 0
    for piece in (t .. ' '):gmatch('(.-)[%s/%.,_=:%(%)%[%]|;<>%+%*%-]') do
      seg = math.max(seg, ulen(piece) + 1)
    end
    word(ulen(t) * MONO, seg * MONO)
  end
  pandoc.walk_block(pandoc.Div(blocks), {
    Str = function(s) word(ulen(s.text)) end,
    Space = space, SoftBreak = space, LineBreak = space,
    Math = function(m) word(math_len(m.text)) end,
    Code = function(c) code(c.text) end,
    RawInline = function(r)
      local t = r.text:match('^\\texttt{(.*)}$')
      if t then
        t = t:gsub('\\allowbreak{}', ''):gsub('\\textbackslash{}', '\\'):gsub('\\[a-z]+{}', 'x'):gsub('\\(.)', '%1')
        code(t)
      end
    end,
  })
  return n, longest
end

-- Tables ---------------------------------------------------------------------------------------------

local function starts_with_label(blk, label)
  if not blk or blk.t ~= 'Para' then return false end
  local s = stringify(blk)
  return s:match('^' .. label .. ' [%dA-Z][%w.%-]*[.:]') ~= nil
end

-- Share `avail` characters among columns. Every column first gets its longest unbreakable word (or
-- its whole text, if shorter); a column that fits its fair share of the rest keeps its natural width,
-- and the remaining columns share what is left in proportion to their natural widths.
local function water_fill(natural, floor, avail)
  local w, fixed, flexible = {}, 0, {}
  for i = 1, #natural do flexible[i] = true end
  local changed = true
  while changed do
    changed = false
    local rem, k = avail - fixed, 0
    for i in pairs(flexible) do k = k + 1 end
    if k == 0 then break end
    local share = rem / k
    for i in pairs(flexible) do
      if natural[i] <= share then
        w[i] = natural[i]; fixed = fixed + natural[i]; flexible[i] = nil; changed = true
      end
    end
  end
  local rem, tot = avail - fixed, 0
  for i in pairs(flexible) do tot = tot + natural[i] end
  for i in pairs(flexible) do w[i] = rem * natural[i] / tot end
  -- Raise any column below its floor, taking the difference from the columns above theirs.
  local deficit, spare = 0, 0
  for i = 1, #natural do
    local f = math.min(floor[i], natural[i])
    if w[i] < f then deficit = deficit + f - w[i]; w[i] = f
    else spare = spare + (w[i] - f) end
  end
  if deficit > 0 and spare > 0 then
    local take = math.min(1, deficit / spare)
    for i = 1, #natural do
      local f = math.min(floor[i], natural[i])
      if w[i] > f then w[i] = w[i] - (w[i] - f) * take end
    end
  end
  return w
end

local function align_char(a, wrapped, frac, dim)
  local base = 'l'
  if a == 'AlignRight' then base = 'r' elseif a == 'AlignCenter' then base = 'c' end
  if not wrapped then return base end
  local P = ({ l = 'L', r = 'R', c = 'C' })[base]
  return string.format('%s{%.3f%s}', P, frac, dim)
end

local function cell_tex(blocks, wrapped)
  local t = to_latex_blocks(blocks)
  if wrapped then
    t = t:gsub('\n\n+', '\\par ')
  else
    t = t:gsub('\n\n+', ' ')
  end
  t = t:gsub('\\\\', wrapped and '\\newline ' or ' ')
  return t
end

local function table_latex(tbl, caption_inlines)
  local st = pandoc.utils.to_simple_table(tbl)
  local ncol = #st.aligns
  local rows = {}
  local has_head = false
  for _, c in ipairs(st.headers) do if #c > 0 then has_head = true end end
  if has_head then rows[#rows + 1] = st.headers end
  for _, r in ipairs(st.rows) do rows[#rows + 1] = r end

  local natural, longest, lens = {}, {}, {}
  for i = 1, ncol do natural[i] = 1; longest[i] = 1 end
  for ri, r in ipairs(rows) do
    lens[ri] = {}
    for i = 1, ncol do
      local n, l = visible(r[i] or {})
      lens[ri][i] = n
      if n > natural[i] then natural[i] = n end
      if l > longest[i] then longest[i] = l end
    end
  end
  local sum = 0
  for i = 1, ncol do sum = sum + natural[i] end

  local function layout(chars)
    local avail = chars - CELL_PAD * ncol
    if sum <= avail then return nil, true end           -- natural widths fit
    local w = water_fill(natural, longest, avail)
    -- It fits when every column holds its longest word within the space there is.
    local total = 0
    for i = 1, ncol do total = total + w[i] end
    return w, total <= avail + 1e-6
  end
  local function height(w)
    local h = 0
    for ri = 1, #rows do
      local m = 1
      for i = 1, ncol do
        if w then m = math.max(m, math.ceil(lens[ri][i] / math.max(w[i], 1))) end
      end
      h = h + m
    end
    return h
  end

  local mode, w
  local wcol, okcol = layout(COL_CHARS)
  if wcol == nil then
    mode, w = 'column', nil
  elseif okcol and ncol <= 3 then
    mode, w = 'column', wcol
  else
    local wfull = layout(FULL_CHARS)
    if height(wfull) <= FLOAT_LINES then
      mode, w = 'full', wfull
    elseif okcol then
      mode, w = 'column', wcol
    else
      mode, w = 'page', wfull
    end
  end
  local wrapped = w ~= nil
  local dim = (mode == 'column') and '\\linewidth' or '\\textwidth'
  local chars = (mode == 'column') and COL_CHARS or FULL_CHARS

  local spec = {}
  for i = 1, ncol do
    local frac = wrapped and (w[i] / chars) or 0
    spec[#spec + 1] = align_char(st.aligns[i], wrapped, frac, dim)
  end
  spec = table.concat(spec)

  local function row_tex(r)
    local cells = {}
    for i = 1, ncol do cells[i] = cell_tex(r[i] or {}, wrapped) end
    return table.concat(cells, ' & ') .. ' \\\\'
  end
  local head = has_head and row_tex(rows[1]) or nil
  local body = {}
  for ri = (has_head and 2 or 1), #rows do body[#body + 1] = row_tex(rows[ri]) end
  body = table.concat(body, '\n')

  local cap = ''
  if caption_inlines and #caption_inlines > 0 then
    cap = '\\caption*{' .. to_latex_inlines(caption_inlines) .. '}\n'
  end

  local function tabular(env)
    local s = '\\begin{' .. env .. '}{' .. spec .. '}\n\\toprule\n'
    if head then s = s .. head .. '\n\\midrule\n' end
    return s .. body .. '\n\\bottomrule\n\\end{' .. env .. '}'
  end
  local function breakable(size)
    local s = '{' .. size .. '\n'
    if head then
      s = s .. '\\tablefirsthead{\\toprule\n' .. head .. '\n\\midrule}\n'
           .. '\\tablehead{\\toprule\n' .. head .. '\n\\midrule}\n'
    else
      s = s .. '\\tablefirsthead{\\toprule}\n\\tablehead{\\toprule}\n'
    end
    s = s .. '\\tabletail{\\midrule}\n\\tablelasttail{\\bottomrule}\n'
    s = s .. '\\begin{xtabular}{' .. spec .. '}\n' .. body .. '\n\\end{xtabular}\n}'
    return s
  end

  if mode == 'full' then
    local t = '\\begin{table*}[tp]\n\\centering\\footnotesize\n' .. cap
    if wrapped then
      t = t .. tabular('tabular')
    else
      t = t .. '\\typesetfit{\\textwidth}{%\n' .. tabular('tabular') .. '}'
    end
    return t .. '\n\\end{table*}', true
  elseif mode == 'page' then
    -- Too long for one page: a run of full-width floats, each a page's worth of rows under the
    -- repeated header, so the text keeps flowing in its columns around them.
    local chunks, cur, lines = {}, {}, has_head and 2 or 0
    for ri = (has_head and 2 or 1), #rows do
      local m = 1
      for i = 1, ncol do m = math.max(m, math.ceil(lens[ri][i] / math.max(w[i], 1))) end
      if #cur > 0 and lines + m > FLOAT_LINES then
        chunks[#chunks + 1] = cur; cur = {}; lines = has_head and 2 or 0
      end
      cur[#cur + 1] = row_tex(rows[ri]); lines = lines + m
    end
    if #cur > 0 then chunks[#chunks + 1] = cur end
    local out = {}
    for k, c in ipairs(chunks) do
      body = table.concat(c, '\n')
      out[#out + 1] = '\\begin{table*}[tp]\n\\centering\\footnotesize\n' .. (k == 1 and cap or '')
        .. tabular('tabular') .. '\n\\end{table*}'
    end
    return table.concat(out, '\n'), true
  else
    if height(w) <= INLINE_LINES then
      -- In the column if it fits there at its own size, else across both columns (decided by TeX).
      return '\\typesetcolumntable{%\n' .. tabular('tabular') .. '}', false
    end
    return '\\begin{center}\n' .. breakable('\\footnotesize') .. '\n\\end{center}', false
  end
end

-- Figures --------------------------------------------------------------------------------------------

local function file_exists(p)
  local f = io.open(p, 'rb')
  if f then f:close(); return true end
  return false
end

local function resolve(src)
  if file_exists(src) then return src end
  for _, d in ipairs(PANDOC_STATE.resource_path or {}) do
    local p = d .. '/' .. src
    if file_exists(p) then return p end
  end
  return nil
end

local function image_src(src)
  local stem = src:match('^(.*)%.[Pp][Nn][Gg]$') or src:match('^(.*)%.[Jj][Pp][Ee]?[Gg]$')
  if stem and resolve(stem .. '.pdf') then return stem .. '.pdf' end
  return src
end

local function is_landscape(src)
  local p = resolve(src)
  if not p then return false end
  local f = io.open(p, 'rb')
  local data = f:read('a')
  f:close()
  local ok, w, h = pcall(function()
    local size = pandoc.image.size(data)
    return size.width, size.height
  end)
  if not ok or not w or not h or h == 0 then return false end
  return w > h
end

local function figure_latex(img, caption_inlines)
  local src = image_src(img.src)
  local wide
  if meta_figure_width == 'full' then wide = true
  elseif meta_figure_width == 'column' then wide = false
  else wide = is_landscape(img.src) end
  local env = wide and 'figure*' or 'figure'
  local width = wide and '\\textwidth' or '\\linewidth'
  local s = '\\begin{' .. env .. '}[tbp]\n\\centering\n\\includegraphics[width=' .. width
            .. ',height=0.8\\textheight,keepaspectratio]{' .. src .. '}\n'
  if caption_inlines and #caption_inlines > 0 then
    s = s .. '\\caption*{' .. to_latex_inlines(caption_inlines) .. '}\n'
  end
  return s .. '\\end{' .. env .. '}'
end

-- The image of a figure-like block: a Figure, or a paragraph holding one image and nothing else.
local function lone_image(blk)
  local inl
  if blk.t == 'Figure' then
    if #blk.content ~= 1 or (blk.content[1].t ~= 'Plain' and blk.content[1].t ~= 'Para') then return nil end
    inl = blk.content[1].content
  elseif blk.t == 'Para' then
    inl = blk.content
  else
    return nil
  end
  local img
  for _, x in ipairs(inl) do
    if x.t == 'Image' then
      if img then return nil end
      img = x
    elseif x.t ~= 'Space' and x.t ~= 'SoftBreak' then
      return nil
    end
  end
  return img
end

local function Blocks(blocks)
  local out = pandoc.List()
  local i = 1
  while i <= #blocks do
    local b, nxt = blocks[i], blocks[i + 1]
    local img = lone_image(b)
    if img then
      local cap
      if starts_with_label(nxt, 'Figure') or starts_with_label(nxt, 'Fig%.') then
        cap = nxt.content; i = i + 1
      elseif b.t == 'Figure' then
        cap = pandoc.utils.blocks_to_inlines(b.caption.long)
      else
        cap = img.caption
      end
      out:insert(pandoc.RawBlock('latex', figure_latex(img, cap)))
    elseif b.t == 'Table' then
      -- The caption: the table's own, else a "Table N" paragraph right before it, else one right after.
      local cap = pandoc.utils.blocks_to_inlines(b.caption.long or {})
      local before, after = false, false
      if #cap == 0 and #out > 0 and starts_with_label(out[#out], 'Table') then
        cap = out[#out].content; before = true
      elseif #cap == 0 and starts_with_label(nxt, 'Table') then
        cap = nxt.content; after = true
      end
      local tex, floats = table_latex(b, cap)
      if floats then
        if before then out:remove(#out) end
        if after then i = i + 1 end
      elseif before or after then
        tex = table_latex(b, nil)       -- an in-column table keeps its caption paragraph where it is
      end
      out:insert(pandoc.RawBlock('latex', tex))
    elseif b.t == 'HorizontalRule' and (nxt == nil or nxt.t == 'Header') then
      -- A rule that only separates sections: the section heading already does that.
    else
      out:insert(b)
    end
    i = i + 1
  end
  return out
end

local function Meta(m)
  if m['figure-width'] then meta_figure_width = stringify(m['figure-width']) end
  if m['typeset-section-level'] then
    section_level = tonumber(stringify(m['typeset-section-level'])) or 1
  end
end

return {
  { Meta = Meta },
  { Code = Code, Math = DisplayMath, Str = Str, Header = Header },
  { Blocks = Blocks },
}
