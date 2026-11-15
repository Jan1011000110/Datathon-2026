"""Export the deck in presentation/deck/ to files anyone can open without the artifact link.

    .venv/Scripts/python.exe presentation/build_export.py

Writes, next to this script:
    slides.pdf          one slide per page (GitHub shows it in the browser)
    slides.html         every slide on one page, with its speaker notes underneath
    speaker_notes.md    the notes alone

Needs Chrome for the PDF step. The slide sources stay in deck/project/ (deck.json, deck.css, slides/*.html).
"""
import base64
import html
import json
import re
import subprocess
import tempfile
from pathlib import Path

HERE = Path(__file__).resolve().parent
DECK = HERE / "deck"
CHROME = r"C:\Program Files\Google\Chrome\Application\chrome.exe"
ASSETS = {"/_blob/fce1805c60e33f64c1ae9aaa02235231": DECK / "assets" / "decision_rule.png"}

BASE_CSS = """
* { margin: 0 }
body { font: 32px/1.2 sans-serif }
section { width: 1920px; height: 1080px; overflow: hidden; box-sizing: border-box; position: relative; isolation: isolate }
p, li, h1, h2, h3, td, th { overflow-wrap: break-word; text-wrap: pretty }
aside { display: none }
"""


def load():
    index = json.loads((DECK / "project" / "deck.json").read_text(encoding="utf-8"))
    css = (DECK / "project" / "deck.css").read_text(encoding="utf-8")
    fonts = "\n".join(f'<link rel="stylesheet" href="{f["href"]}">' for f in index["faces"].values() if "href" in f)
    slides = []
    for sid in index["order"]:
        sec = (DECK / "project" / "slides" / f"{sid}.html").read_text(encoding="utf-8").strip()
        for ref, path in ASSETS.items():
            if ref in sec:
                sec = sec.replace(ref, "data:image/png;base64," + base64.b64encode(path.read_bytes()).decode())
        m = re.search(r"<aside>(.*?)</aside>", sec, re.S)
        notes = html.unescape(m.group(1).strip()) if m else ""
        t = re.search(r"<h[12][^>]*>(.*?)</h[12]>", sec, re.S)
        title = html.unescape(re.sub(r"<[^>]+>", "", t.group(1))).strip() if t else sid
        slides.append((sid, title, notes, sec))
    return index, css, fonts, slides


def main():
    index, css, fonts, slides = load()
    head = f"<meta charset='utf-8'><title>{html.escape(index['title'])}</title>{fonts}<style>{BASE_CSS}</style><style>{css}</style>"

    # ---- PDF: one 1920x1080 page per slide
    print_css = "@page { size: 1920px 1080px; margin: 0 } section { break-after: page }"
    with tempfile.TemporaryDirectory() as tmp:
        src = Path(tmp) / "print.html"
        src.write_text(f"<!doctype html><html><head>{head}<style>{print_css}</style></head><body>"
                       + "\n".join(s[3] for s in slides) + "</body></html>", encoding="utf-8")
        pdf = HERE / "slides.pdf"
        subprocess.run([CHROME, "--headless=new", "--disable-gpu", "--no-pdf-header-footer", "--virtual-time-budget=10000",
                        f"--user-data-dir={Path(tmp) / 'profile'}", f"--print-to-pdf={pdf}", src.as_uri()],
                       capture_output=True, timeout=180, check=True)

    # ---- HTML: all slides stacked and scaled to the window, notes under each
    view_css = """
    body { background: #E9E9E6; padding: 24px 0 }
    .wrap { width: calc(1920px * var(--s)); height: calc(1080px * var(--s)); margin: 0 auto; box-shadow: 0 1px 6px #0003 }
    .wrap > section { transform: scale(var(--s)); transform-origin: top left }
    .notes { width: calc(1920px * var(--s)); margin: 10px auto 36px; font: 16px/1.5 Inter, Arial, sans-serif; color: #333; box-sizing: border-box; padding: 0 4px }
    .notes b { color: #1A1A1A }
    """
    body = []
    for n, (sid, title, notes, sec) in enumerate(slides, 1):
        body.append(f'<div class="wrap" id="slide-{n}">{sec}</div>')
        body.append(f'<div class="notes"><b>{n}. {html.escape(title)}.</b> {html.escape(notes)}</div>')
    script = ("<script>const fit=()=>document.documentElement.style.setProperty('--s',"
              "Math.min(1,(document.documentElement.clientWidth-48)/1920));fit();addEventListener('resize',fit);</script>")
    (HERE / "slides.html").write_text(f"<!doctype html><html style='--s:0.5'><head>{head}<style>{view_css}</style></head><body>"
                                      + "\n".join(body) + script + "</body></html>", encoding="utf-8", newline="\n")

    # ---- notes
    md = [f"# {index['title']}: speaker notes", "", "Slides: [slides.pdf](slides.pdf) (or open `slides.html` locally to see slides and notes together).", ""]
    for n, (sid, title, notes, _) in enumerate(slides, 1):
        md += [f"## {n}. {title}", "", notes, ""]
    (HERE / "speaker_notes.md").write_text("\n".join(md), encoding="utf-8", newline="\n")
    print(f"{len(slides)} slides -> slides.pdf ({(HERE / 'slides.pdf').stat().st_size // 1024} KB), slides.html, speaker_notes.md")


if __name__ == "__main__":
    main()
