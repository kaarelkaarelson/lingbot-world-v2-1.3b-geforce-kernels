#!/usr/bin/env python3
"""Render the result tables from bench/summary.json into README.md (Markdown) and the write-up (HTML rows).

Each target file carries marker comments; only the text between them is replaced, the styling around it is hand-made:
  Markdown:  <!-- table:engines -->  ...  <!-- /table:engines -->
  HTML:      <!-- table:engines -->  ...  <!-- /table:engines -->   (inside the <tbody>)

  python tools/tables.py                       # rewrite README.md in place
  python tools/tables.py --html path/to/index.html   # also rewrite the write-up's rows
  python tools/tables.py --check [--html ...]  # exit 1 if any target is out of date
"""
import argparse
import json
import os
import re
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA = json.load(open(os.path.join(ROOT, "bench", "summary.json")))


def speedup(fps):
    ours = DATA["engines"]["rows"][0]["fps"]
    return f"{ours / fps:.1f}×"


# ---------- Markdown ----------

def md_engines():
    out = ["| Engine | s per chunk | FPS as played | Ours vs it | What it ran on the 5090 |", "|---|---|---|---|---|"]
    for r in DATA["engines"]["rows"]:
        name = f"**{r['engine']}**" if r.get("ours") else r["engine"] + (f" `{r['version']}`" if r.get("version") and r["version"] != "single GPU" else (", single GPU" if r.get("version") == "single GPU" else ""))
        fps = f"**{r['fps']}**" if r.get("ours") else f"{r['fps']}"
        sp = "—" if r.get("ours") else f"**{speedup(r['fps'])}**"
        out.append(f"| {name} | {r['s_per_chunk']:.2f} | {fps} | {sp} | {r['ran']} |")
    return "\n".join(out)


def md_baseline():
    out = ["| | Original paper's code | Ours |", "|---|---|---|"]
    for r in DATA["baseline_vs_ours"]["rows"]:
        out.append(f"| {r['metric']} | {r['before']} | **{r['after']}** |")
    return "\n".join(out)


def md_ladder():
    L = DATA["ladder"]
    out = ["| Step | Before | After | s per chunk |", "|---|---|---|---|"]
    prev = L["start"]
    for r in L["rows"]:
        after = r["after"]
        if r.get("after_url"):
            after = after.replace(r["after_link_text"], f"[{r['after_link_text']}]({r['after_url']})")
        before = r["before"]
        if r.get("before_url"):
            t = r.get("before_link_text", before)
            before = before.replace(t, f"[{t}]({r['before_url']})")
        out.append(f"| {r['step']} | {before} | {after} | {prev:.2f} → {r['after_s']:.2f} |")
        prev = r["after_s"]
    t = L["total"]
    out.append(f"| **Total** | {t['before']} | **{t['after']}** | **{t['before_s']:.2f} → {t['after_s']:.2f}** |")
    return "\n".join(out)


def md_peaks():
    out = ["| Kernel | Reached | Peak on RTX 5090 | of peak |", "|---|---|---|---|"]
    for r in DATA["peaks"]["rows"]:
        out.append(f"| {r['kernel']} | {r['reached']} | {r['peak']} | **{r['pct']}** |")
    return "\n".join(out)


def md_quality():
    out = ["| | Original paper's code | Ours |", "|---|---|---|"]
    for r in DATA["quality"]["rows"]:
        m = f"[{r['metric']}]({r['url']})" if r.get("url") else r["metric"]
        out.append(f"| {m} | {r['before']} | **{r['after']}** |")
    return "\n".join(out)


# ---------- HTML rows (the write-up) ----------

def html_engines():
    out = []
    for r in DATA["engines"]["rows"]:
        if r.get("ours"):
            out.append(f'  <tr class="ours"><td>Ours</td><td class="num">{r["s_per_chunk"]:.2f}</td><td class="num">{r["fps"]}</td><td class="num">—</td></tr>')
        else:
            cite = f' <span class="cite">[<a href="{r["url"]}">{r["cite"]}</a>]</span>' if r.get("cite") else ""
            out.append(f'  <tr><td>{r["engine"]}{cite}</td><td class="num">{r["s_per_chunk"]:.2f}</td><td class="num">{r["fps"]}</td><td class="num">{speedup(r["fps"])}</td></tr>')
    return "\n".join(out)


def html_baseline_small():
    out = []
    for r in DATA["baseline_vs_ours"]["rows"]:
        if r.get("big"):
            continue
        out.append(f'    <tr><td>{r["metric"]}</td><td class="num">{r["before"]}</td><td class="arr">→</td><td class="num hi">{r["after"]}</td></tr>')
    return "\n".join(out)


def html_ladder():
    L = DATA["ladder"]
    out = []
    prev = L["start"]
    for r in L["rows"]:
        after = r["after"]
        if r.get("after_url"):
            after = after.replace(r["after_link_text"], f'<a href="{r["after_url"]}">{r["after_link_text"]}</a>')
        before = r["before"]
        if r.get("before_url"):
            t = r.get("before_link_text", before)
            before = before.replace(t, f'<a href="{r["before_url"]}">{t}</a>')
        w = round(r["after_s"] / L["start"] * 100)
        out.append(f'  <tr><td>{r["step"]}</td><td class="before">{before}</td><td class="after">{after}</td><td class="num"><span class="bar"><i style="width:{w}%"></i></span><span class="b">{prev:.2f}</span> → {r["after_s"]:.2f}</td></tr>')
        prev = r["after_s"]
    t = L["total"]
    w = round(t["after_s"] / L["start"] * 100)
    out.append(f'  <tr class="ours"><td>Total</td><td class="before">{t["before"]}</td><td class="after">{t["after"]}</td><td class="num"><span class="bar"><i style="width:{w}%"></i></span><span class="b">{t["before_s"]:.2f}</span> → {t["after_s"]:.2f}</td></tr>')
    return "\n".join(out)


def html_peaks():
    return "\n".join(f'  <tr><td>{r["kernel"]}</td><td class="what">{r["reached"]}</td><td class="what">{r["peak"]}</td><td class="num hi">{r["pct"]}</td></tr>' for r in DATA["peaks"]["rows"])


def html_quality():
    out = []
    for r in DATA["quality"]["rows"]:
        m = r["metric"]
        if r.get("cite"):
            m = f'{m} <span class="cite">[<a href="{r["url"]}">{r["cite"]}</a>]</span>'
        out.append(f'  <tr><td>{m}</td><td class="num">{r["before"]}</td><td class="arr">→</td><td class="num hi">{r["after"]}</td></tr>')
    return "\n".join(out)


MD = {"engines": md_engines, "baseline": md_baseline, "ladder": md_ladder, "peaks": md_peaks, "quality": md_quality}
HTML = {"engines": html_engines, "baseline": html_baseline_small, "ladder": html_ladder, "peaks": html_peaks, "quality": html_quality}


def render(path, renderers, check):
    text = open(path).read()
    new = text
    for name, fn in renderers.items():
        pat = re.compile(rf"(<!-- table:{name} -->\n)(.*?)(<!-- /table:{name} -->)", re.S)
        if not pat.search(new):
            continue
        new = pat.sub(lambda m: m.group(1) + fn() + "\n" + m.group(3), new)
    if new == text:
        print(f"{path}: up to date")
        return True
    if check:
        print(f"{path}: OUT OF DATE (run tools/tables.py)")
        return False
    open(path, "w").write(new)
    print(f"{path}: rewritten")
    return True


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--html", default=None, help="the write-up's index.html")
    ap.add_argument("--check", action="store_true")
    a = ap.parse_args()
    ok = render(os.path.join(ROOT, "README.md"), MD, a.check)
    if a.html:
        ok = render(a.html, HTML, a.check) and ok
    sys.exit(0 if ok else 1)
