/** A small markdown renderer for model output.

    Escape first, transform second. That order is the whole security model
    here: the text being rendered came out of a language model, and a model
    that has just read an attacker-supplied document is exactly the thing you
    do not hand raw innerHTML. Nothing in this file ever passes an HTML tag
    through from the source — every `<` in the input is already `&lt;` before
    a single rule runs, and the only tags in the output are ones written
    literally below.

    Supports what the assistant actually emits: headings, bold, italic,
    inline and fenced code, links, bullet and numbered lists, pipe tables,
    blockquotes, and rules. Deliberately not a full CommonMark implementation. */

const ESCAPES = { "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" };
const escape = (s) => String(s ?? "").replace(/[&<>"']/g, (c) => ESCAPES[c]);

/* Only http(s). A `javascript:` href is the other way model output becomes
   script, and it survives escaping because it is an attribute value. */
const SAFE_URL = /^https?:\/\//i;

function link(url, text) {
  if (!SAFE_URL.test(url)) return text;      // already escaped by the caller
  return `<a href="${url}" target="_blank" rel="noopener noreferrer">${text}</a>`;
}

/* A placeholder escaped text cannot contain: `escape()` has already turned
   every angle bracket into an entity, so nothing in the source can collide
   with these brackets. */
const HOLD = /〈(\d+)〉/g;

/** Inline formatting for one already-escaped run of text.

    Code spans and links are lifted out before the emphasis rules run. They
    used to be rewritten in place, which meant an underscore inside a URL
    turned `target="_blank"` into `target="<em>blank"` and broke the link —
    emphasis has no business editing markup this function just generated. */
function inline(raw) {
  let s = escape(raw);

  const holds = [];
  const hold = (html) => `〈${holds.push(html) - 1}〉`;

  s = s.replace(/`([^`]+)`/g, (_, code) => hold(`<code>${code}</code>`));
  s = s.replace(/\[([^\]\n]+)\]\((https?:\/\/[^\s)]+)\)/g,
                (_, text, url) => hold(link(url, text)));
  s = s.replace(/(^|[\s(])(https?:\/\/[^\s<)]+)/g,
                (_, pre, url) => pre + hold(link(url, url)));

  s = s.replace(/\*\*([^*\n]+)\*\*/g, "<strong>$1</strong>");
  s = s.replace(/(^|[^*\w])\*([^*\n]+)\*/g, "$1<em>$2</em>");
  s = s.replace(/(^|[^_\w])_([^_\n]+)_/g, "$1<em>$2</em>");
  s = s.replace(/~~([^~\n]+)~~/g, "<del>$1</del>");

  return s.replace(HOLD, (_, n) => holds[+n]);
}

const RULE = /^\s*([-*_])(?:\s*\1){2,}\s*$/;
const HEADING = /^(#{1,6})\s+(.*)$/;
const BULLET = /^(\s*)[-*+]\s+(.*)$/;
const NUMBER = /^(\s*)(\d+)[.)]\s+(.*)$/;
const QUOTE = /^\s*>\s?(.*)$/;
const FENCE = /^\s*```/;

/** A pipe table needs a delimiter row — that is what separates it from a
    sentence that happens to contain a vertical bar. */
const isDelimiter = (line) =>
  /\|/.test(line) && /^[\s|:-]+$/.test(line) && /-/.test(line);

const cells = (line) =>
  line.trim().replace(/^\||\|$/g, "").split("|").map((c) => c.trim());

function alignments(line) {
  return cells(line).map((c) => {
    const left = c.startsWith(":");
    const right = c.endsWith(":");
    if (left && right) return ' style="text-align:center"';
    if (right) return ' style="text-align:right"';
    return "";
  });
}

export function renderMarkdown(source) {
  const lines = String(source ?? "").replace(/\r\n?/g, "\n").split("\n");
  const out = [];
  let i = 0;

  const listBlock = (depth) => {
    const ordered = NUMBER.test(lines[i]);
    const items = [];
    while (i < lines.length) {
      const m = ordered ? NUMBER.exec(lines[i]) : BULLET.exec(lines[i]);
      if (!m) break;
      const indent = m[1].length;
      if (indent < depth) break;
      if (indent > depth) {
        // Deeper than this list — it belongs under the item already collected.
        const nested = listBlock(indent);
        if (items.length) items[items.length - 1] += nested;
        else out.push(nested);
        continue;
      }
      i += 1;
      let text = ordered ? m[3] : m[2];
      while (i < lines.length && lines[i].trim() && !BULLET.test(lines[i])
             && !NUMBER.test(lines[i]) && !HEADING.test(lines[i])
             && !isDelimiter(lines[i]) && !FENCE.test(lines[i])) {
        text += " " + lines[i].trim();
        i += 1;
      }
      items.push(`<li>${inline(text)}</li>`);
    }
    const tag = ordered ? "ol" : "ul";
    return `<${tag}>${items.join("")}</${tag}>`;
  };

  while (i < lines.length) {
    const line = lines[i];

    if (!line.trim()) { i += 1; continue; }

    if (FENCE.test(line)) {
      i += 1;
      const body = [];
      while (i < lines.length && !FENCE.test(lines[i])) { body.push(lines[i]); i += 1; }
      i += 1;                                    // closing fence
      out.push(`<pre><code>${escape(body.join("\n"))}</code></pre>`);
      continue;
    }

    if (RULE.test(line)) { out.push("<hr>"); i += 1; continue; }

    const heading = HEADING.exec(line);
    if (heading) {
      // Clamped to h4–h6: this renders inside a chat turn, not a document.
      const level = Math.min(6, 3 + heading[1].length);
      out.push(`<h${level}>${inline(heading[2])}</h${level}>`);
      i += 1;
      continue;
    }

    if (QUOTE.test(line)) {
      const body = [];
      while (i < lines.length && QUOTE.test(lines[i])) {
        body.push(QUOTE.exec(lines[i])[1]);
        i += 1;
      }
      out.push(`<blockquote>${inline(body.join(" "))}</blockquote>`);
      continue;
    }

    if (line.includes("|") && i + 1 < lines.length && isDelimiter(lines[i + 1])) {
      const head = cells(line);
      const align = alignments(lines[i + 1]);
      i += 2;
      const body = [];
      while (i < lines.length && lines[i].includes("|") && lines[i].trim()) {
        body.push(cells(lines[i]));
        i += 1;
      }
      const thead = head.map((c, n) => `<th${align[n] || ""}>${inline(c)}</th>`).join("");
      const tbody = body.map((row) =>
        `<tr>${head.map((_, n) =>
          `<td${align[n] || ""}>${inline(row[n] ?? "")}</td>`).join("")}</tr>`).join("");
      // Wrapped so a wide table scrolls itself rather than the whole page.
      out.push(`<div class="table-wrap"><table><thead><tr>${thead}</tr></thead>`
               + `<tbody>${tbody}</tbody></table></div>`);
      continue;
    }

    if (BULLET.test(line) || NUMBER.test(line)) {
      out.push(listBlock((BULLET.exec(line) || NUMBER.exec(line))[1].length));
      continue;
    }

    const para = [];
    while (i < lines.length && lines[i].trim() && !HEADING.test(lines[i])
           && !BULLET.test(lines[i]) && !NUMBER.test(lines[i]) && !QUOTE.test(lines[i])
           && !RULE.test(lines[i]) && !FENCE.test(lines[i])
           && !(lines[i].includes("|") && isDelimiter(lines[i + 1] || ""))) {
      para.push(lines[i]);
      i += 1;
    }
    // Single newlines inside a paragraph were meant as line breaks.
    out.push(`<p>${para.map(inline).join("<br>")}</p>`);
  }

  return out.join("");
}
