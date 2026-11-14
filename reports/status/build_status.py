# Usage: .venv/Scripts/python.exe reports/status/build_status.py
"""Builds reports/status/status.html, the live status board for the hard-mode work.
Edit status_data.json, rerun, republish. Text fields accept `code` and **bold**."""
import html
import json
import re
from pathlib import Path

HERE = Path(__file__).parent

DATA = json.loads((HERE / "status_data.json").read_text(encoding="utf-8"))


def fmt(s):
    s = html.escape(str(s))
    s = re.sub(r"`([^`]+)`", r"<code>\1</code>", s)
    return re.sub(r"\*\*([^*]+)\*\*", r"<strong>\1</strong>", s)


STATE = {"running": ("●", "Running"), "done": ("✓", "Done"), "queued": ("○", "Queued"), "blocked": ("!", "Blocked")}

CSS = """
/* Layout: one reading column; a status header with a charge gauge, then tables that scroll inside themselves. */
:root{--bg:#f4f6f8;--panel:#ffffff;--ink:#101820;--ink2:#465362;--muted:#7a8794;--line:#dde3e9;--accent:#2a78d6;--accent-soft:#e3eefb;
--good:#0a7d0a;--good-bg:#e4f5e4;--idle:#5d6975;--idle-bg:#e9edf1;--warn:#9a5b00;--warn-bg:#fdf0d5;--paper:#fcfcfb;
--display:"Space Grotesk",system-ui,sans-serif;--body:"Inter",system-ui,-apple-system,"Segoe UI",sans-serif;--mono:"IBM Plex Mono",ui-monospace,Consolas,monospace}
@media (prefers-color-scheme: dark){:root:not([data-theme="light"]){--bg:#0e1318;--panel:#161d24;--ink:#e9eef3;--ink2:#b3bec9;--muted:#8794a1;--line:#27313b;--accent:#5a9df0;--accent-soft:#17283d;
--good:#4cc24c;--good-bg:#132a16;--idle:#a3afba;--idle-bg:#222b34;--warn:#f0b445;--warn-bg:#33270f;--paper:#fcfcfb;color-scheme:dark}}
:root[data-theme="dark"]{--bg:#0e1318;--panel:#161d24;--ink:#e9eef3;--ink2:#b3bec9;--muted:#8794a1;--line:#27313b;--accent:#5a9df0;--accent-soft:#17283d;
--good:#4cc24c;--good-bg:#132a16;--idle:#a3afba;--idle-bg:#222b34;--warn:#f0b445;--warn-bg:#33270f;--paper:#fcfcfb;color-scheme:dark}
body{background:var(--bg);color:var(--ink);font-family:var(--body);font-size:15px;line-height:1.5}
.wrap{max-width:980px;margin:0 auto;padding-block:28px 56px;padding-left:max(16px,env(safe-area-inset-left,0px));padding-right:max(16px,env(safe-area-inset-right,0px));display:flex;flex-direction:column;gap:28px}
h1{font-family:var(--display);font-size:1.75rem;line-height:1.15;margin:0;text-wrap:balance}
h2{font-family:var(--display);font-size:1.1rem;margin:0 0 10px;text-wrap:balance}
.eyebrow{font-family:var(--mono);font-size:.75rem;letter-spacing:.08em;text-transform:uppercase;color:var(--muted);margin:0 0 6px}
.stamp{font-family:var(--mono);font-size:.82rem;color:var(--ink2);margin:8px 0 0}
code{font-family:var(--mono);font-size:.86em;background:var(--accent-soft);padding:1px 5px;border-radius:4px;overflow-wrap:anywhere}
.top{display:grid;grid-template-columns:minmax(0,1.5fr) minmax(0,1fr);gap:20px;align-items:start}
@media (max-width:720px){.top{grid-template-columns:minmax(0,1fr)}}
.now{margin:0;padding-left:1.15em;display:flex;flex-direction:column;gap:8px;max-width:65ch}
.gauge{background:var(--panel);border:1px solid var(--line);border-radius:10px;padding:16px 18px;display:flex;flex-direction:column;gap:10px}
.gauge .big{font-family:var(--display);font-size:2.4rem;line-height:1;font-weight:600}
.gauge .big small{font-size:1rem;font-weight:500;color:var(--ink2)}
.track{height:14px;border-radius:7px;background:var(--accent-soft);overflow:hidden}
.fill{height:100%;background:var(--accent);border-radius:7px 0 0 7px}
.gauge p{margin:0;color:var(--ink2);font-size:.86rem}
.lb{margin:4px 0 0;padding:10px 0 0;border-top:1px solid var(--line);display:grid;grid-template-columns:minmax(0,1fr) auto;gap:4px 12px;font-size:.88rem}
.lb dt{color:var(--ink2)} .lb dd{margin:0;font-family:var(--mono);text-align:right}
.scroll{overflow-x:auto;background:var(--panel);border:1px solid var(--line);border-radius:10px}
table{border-collapse:collapse;width:100%;font-size:.9rem}
th,td{padding:9px 12px;text-align:left;border-bottom:1px solid var(--line);vertical-align:top}
tr:last-child td{border-bottom:0}
th{font-family:var(--mono);font-size:.72rem;letter-spacing:.06em;text-transform:uppercase;color:var(--muted);font-weight:500;white-space:nowrap}
td.num,th.num{text-align:right;font-variant-numeric:tabular-nums;font-family:var(--mono);white-space:nowrap}
tr.ref td{color:var(--ink2)} tr.lead td{font-weight:600}
.pill{display:inline-flex;align-items:center;gap:6px;font-family:var(--mono);font-size:.74rem;padding:2px 9px;border-radius:999px;white-space:nowrap}
.pill.running{color:var(--accent);background:var(--accent-soft)} .pill.done{color:var(--good);background:var(--good-bg)}
.pill.queued{color:var(--idle);background:var(--idle-bg)} .pill.blocked{color:var(--warn);background:var(--warn-bg)}
figure{margin:0;display:flex;flex-direction:column;gap:8px}
figure img{background:var(--paper);border:1px solid var(--line);border-radius:10px;display:block;width:100%;height:auto}
figcaption{color:var(--ink2);font-size:.88rem;max-width:75ch}
figcaption strong{color:var(--ink)}
.figs{display:flex;flex-direction:column;gap:22px}
.log{margin:0;padding:0;list-style:none;display:flex;flex-direction:column;gap:7px}
.log li{display:grid;grid-template-columns:3.6em minmax(0,1fr);gap:10px}
.log time{font-family:var(--mono);font-size:.82rem;color:var(--muted);padding-top:2px}
.files{margin:0;padding:0;list-style:none;display:flex;flex-direction:column;gap:5px;font-size:.9rem;color:var(--ink2)}
"""


def build():
    d = DATA
    pct = max(0.0, min(100.0, 100.0 * d["best"]["score"] / d["ceiling"]))
    o = ['<meta name="claude:viewport-fit-sides" content="cover">', "<title>Hard Mode Status Board</title>",
         '<link rel="stylesheet" href="https://fonts.googleapis.com/css2?family=IBM+Plex+Mono:wght@400;500&family=Inter:wght@400;600&family=Space+Grotesk:wght@500;600&display=swap">',
         f"<style>{CSS}</style>", '<div class="wrap">']
    o.append('<header><p class="eyebrow">FME Datathon 2026 · UPC Sostenible · battery challenge</p><h1>Hard mode status board</h1>'
             f'<p class="stamp">Last updated {fmt(d["updated"])}</p></header>')
    o.append('<section class="top"><div><h2>Where we are</h2><ul class="now">' + "".join(f"<li>{fmt(x)}</li>" for x in d["now"]) + "</ul></div>")
    o.append('<div class="gauge"><p class="eyebrow" style="margin:0">Best validated agent vs ceiling</p>'
             f'<div class="big">{d["best"]["score"]:.2f} <small>of {d["ceiling"]:.2f} points</small></div>'
             f'<div class="track" role="img" aria-label="{pct:.0f} percent of the ceiling"><div class="fill" style="width:{pct:.1f}%"></div></div>'
             f'<p>{fmt(d["best"]["name"])}: {pct:.0f} % of the perfect-foresight ceiling ({fmt(d["best"]["note"])}).</p>'
             '<dl class="lb">' + "".join(f"<dt>{fmt(n)}</dt><dd>{fmt(v)}</dd>" for n, v in d["leaderboard"]) + "</dl>"
             f'<p>Hard-mode leaderboard at {fmt(d["leaderboard_time"])}.</p></div></section>')
    o.append('<section><h2>Scoreboard</h2><div class="scroll"><table><thead><tr><th>Agent</th><th class="num">Fold A</th><th class="num">Fold B</th>'
             '<th class="num">Fold C</th><th class="num">Mean</th><th>Status</th></tr></thead><tbody>')
    for i, (n, a, b, c, m, s) in enumerate(d["scoreboard"]):
        cls = "ref" if s == "reference" else ("lead" if i == 1 else "")
        o.append(f'<tr class="{cls}"><td>{fmt(n)}</td><td class="num">{a}</td><td class="num">{b}</td><td class="num">{c}</td><td class="num">{m}</td><td>{fmt(s)}</td></tr>')
    o.append("</tbody></table></div></section>")
    o.append('<section><h2>Running now</h2><div class="scroll"><table><thead><tr><th>Who</th><th>Task</th><th>State</th><th>Detail</th></tr></thead><tbody>')
    for who, task, state, det in d["running"]:
        ic, lab = STATE[state]
        o.append(f'<tr><td><strong>{fmt(who)}</strong></td><td>{fmt(task)}</td><td><span class="pill {state}"><span aria-hidden="true">{ic}</span>{lab}</span></td><td>{fmt(det)}</td></tr>')
    o.append("</tbody></table></div></section>")
    o.append('<section><h2>Tried so far</h2><div class="scroll"><table><thead><tr><th>What</th><th>Result (score points)</th><th>Verdict</th></tr></thead><tbody>')
    for what, res, ver in d["tried"]:
        o.append(f"<tr><td>{fmt(what)}</td><td>{fmt(res)}</td><td>{fmt(ver)}</td></tr>")
    o.append("</tbody></table></div></section>")
    o.append('<section><h2>Figures</h2><div class="figs">')
    for f, title, cap in d["figures"]:
        o.append(f'<figure><img src="figs/{html.escape(f)}" alt="{html.escape(title)}"><figcaption><strong>{fmt(title)}.</strong> {fmt(cap)}</figcaption></figure>')
    o.append("</div></section>")
    o.append('<section><h2>Log</h2><ul class="log">' + "".join(f"<li><time>{fmt(t)}</time><span>{fmt(x)}</span></li>" for t, x in d["log"]) + "</ul></section>")
    o.append('<section><h2>In the repo</h2><ul class="files">' + "".join(f"<li><code>{html.escape(p)}</code> {fmt(x)}</li>" for p, x in d["files"]) + "</ul></section>")
    o.append("</div>")
    (HERE / "status.html").write_text("\n".join(o), encoding="utf-8")
    print("wrote", HERE / "status.html")


if __name__ == "__main__":
    build()
