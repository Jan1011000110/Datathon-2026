# Usage: <venv>/Scripts/python.exe -I experiments/05_leaderboard/scrape.py [--out-dir reports/05_leaderboard] [--work-dir <temp dir>]   (needs `websocket-client`; read-only, never touches the submission tab)
"""Read-only snapshot of the public leaderboard at https://repteupc.datathon.cat/.

The site is a Streamlit app, so plain HTTP only returns an empty shell. This drives
headless Chrome over the DevTools Protocol, opens the Leaderboard tab, prints the
tab's text and table, and saves the site's own CSV export as
<out-dir>/leaderboard_<YYYYMMDD_HHMM>.csv.

It only clicks two things: the Leaderboard tab and the CSV download button inside
that tab. It never opens the submission tab and never types anything.
"""
import argparse
import csv
import datetime as dt
import io
import json
import shutil
import socket
import subprocess
import sys
import tempfile
import time
import urllib.request
from pathlib import Path

import websocket  # pip: websocket-client

URL = "https://repteupc.datathon.cat/"
CHROME = r"C:\Program Files\Google\Chrome\Application\chrome.exe"
REPO = Path(__file__).resolve().parents[2]
SCORE_COL = "Puntuacio"
TEAM_COL = "Equip"


class Cdp:
    def __init__(self, ws_url):
        self.ws = websocket.create_connection(ws_url, suppress_origin=True, timeout=60)
        self.mid = 0

    def call(self, method, **params):
        self.mid += 1
        self.ws.send(json.dumps({"id": self.mid, "method": method, "params": params}))
        while True:
            msg = json.loads(self.ws.recv())
            if msg.get("id") == self.mid:
                if "error" in msg:
                    raise RuntimeError(f"{method}: {msg['error']}")
                return msg["result"]

    def js(self, expr):
        r = self.call("Runtime.evaluate", expression=expr, returnByValue=True, awaitPromise=True)
        if "exceptionDetails" in r:
            raise RuntimeError(json.dumps(r["exceptionDetails"])[:2000])
        return r["result"].get("value")


def free_port():
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def wait_for(fn, timeout, what, step=0.5):
    end = time.time() + timeout
    while time.time() < end:
        v = fn()
        if v:
            return v
        time.sleep(step)
    raise TimeoutError(f"timed out after {timeout}s waiting for {what}")


# JS that resolves the Leaderboard tab's panel element (`panel`) or throws.
PANEL_JS = """
const tab = Array.from(document.querySelectorAll('[role="tab"]'))
  .find(t => /leaderboard|classificaci/i.test(t.innerText));
if (!tab) throw new Error('leaderboard tab not found');
const panel = document.getElementById(tab.getAttribute('aria-controls'));
"""


def read_grid(cdp):
    """Read the on-screen (virtualised) data grid by scrolling it. Returns (header, rows)."""
    grab = "(() => {" + PANEL_JS + """
      const sc = panel.querySelector('.dvn-scroller');
      const out = {cells: {}, head: [], top: sc ? sc.scrollTop : 0,
                   max: sc ? sc.scrollHeight - sc.clientHeight : 0};
      panel.querySelectorAll('[data-testid^="glide-cell-"]').forEach(c => {
        out.cells[c.getAttribute('data-testid').slice(11)] = c.innerText;
      });
      out.head = Array.from(panel.querySelectorAll('[role="columnheader"]')).map(h => h.innerText);
      return out;
    })()"""
    cells, head, top = {}, [], 0
    for _ in range(500):
        got = cdp.js(grab)
        cells.update(got["cells"])
        head = got["head"] or head
        if top >= got["max"]:
            break
        top = min(top + 150, got["max"])
        cdp.js("(() => {" + PANEL_JS + f"panel.querySelector('.dvn-scroller').scrollTop = {top}; }})()")
        time.sleep(0.4)
    by_row = {}
    for key, text in cells.items():
        col, row = (int(x) for x in key.split("-"))
        by_row.setdefault(row, {})[col] = text
    ncol = 1 + max((c for r in by_row.values() for c in r), default=-1)
    rows = [[by_row[r].get(c, "") for c in range(ncol)] for r in sorted(by_row)]
    return head, rows


def fmt_table(header, rows):
    widths = [max(len(str(x)) for x in col) for col in zip(header, *rows)] if rows else [len(h) for h in header]
    line = lambda r: " | ".join(str(x).ljust(w) for x, w in zip(r, widths))
    return "\n".join([line(header), "-+-".join("-" * w for w in widths)] + [line(r) for r in rows])


def to_float(s):
    try:
        return float(str(s).replace(",", "."))
    except ValueError:
        return float("-inf")


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--out-dir", default=str(REPO / "reports" / "05_leaderboard"))
    ap.add_argument("--work-dir", default=None, help="where to put the temporary Chrome profile and download (default: system temp)")
    ap.add_argument("--chrome", default=CHROME)
    args = ap.parse_args()
    for stream in (sys.stdout, sys.stderr):
        stream.reconfigure(encoding="utf-8", errors="replace")

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    if args.work_dir:
        Path(args.work_dir).mkdir(parents=True, exist_ok=True)
    work = Path(tempfile.mkdtemp(prefix="lb_scrape_", dir=args.work_dir))
    profile, dl_dir = work / "chrome_profile", work / "download"
    dl_dir.mkdir()

    port = free_port()
    proc = subprocess.Popen(
        [args.chrome, "--headless=new", f"--remote-debugging-port={port}", f"--user-data-dir={profile}",
         "--no-first-run", "--no-default-browser-check", "--window-size=1600,2400", "about:blank"],
        stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    cdp = None
    try:
        def targets():
            try:
                return json.load(urllib.request.urlopen(f"http://127.0.0.1:{port}/json", timeout=2))
            except OSError:
                return None
        page = next(t for t in wait_for(targets, 30, "Chrome's debugging port") if t["type"] == "page")
        cdp = Cdp(page["webSocketDebuggerUrl"])
        cdp.call("Page.enable")
        cdp.call("Browser.setDownloadBehavior", behavior="allow", downloadPath=str(dl_dir))
        cdp.call("Page.navigate", url=URL)
        wait_for(lambda: (cdp.js("document.querySelectorAll('[role=\"tab\"]').length") or 0) >= 5, 90, "the app's tabs to render")
        tabs = cdp.js("Array.from(document.querySelectorAll('[role=\"tab\"]')).map(t => t.innerText)")

        cdp.js("(() => {" + PANEL_JS + "tab.click(); })()")
        wait_for(lambda: cdp.js("(() => {" + PANEL_JS + "return !!panel && !!panel.querySelector('[data-testid=\"stDownloadButton\"] button') && panel.innerText.length > 50; })()"),
                 60, "the leaderboard tab to render")
        time.sleep(3)  # let metrics / grid / chart settle
        snapshot = dt.datetime.now()
        panel_text = cdp.js("(() => {" + PANEL_JS + "return panel.innerText; })()")
        grid_head, grid_rows = read_grid(cdp)

        # Click the CSV export button that lives inside the leaderboard panel (and only that one).
        label = cdp.js("(() => {" + PANEL_JS + """
          const btns = panel.querySelectorAll('[data-testid="stDownloadButton"] button');
          if (btns.length !== 1) throw new Error('expected exactly 1 download button in the leaderboard tab, found ' + btns.length);
          btns[0].click(); return btns[0].innerText; })()""")

        def downloaded():
            files = [f for f in dl_dir.iterdir() if f.is_file() and f.suffix not in (".crdownload", ".tmp")]
            return files[0] if files and files[0].stat().st_size > 0 else None
        src = wait_for(downloaded, 60, "the CSV download")
        time.sleep(0.5)
        raw = src.read_bytes()
    finally:
        try:
            if cdp:
                cdp.ws.settimeout(5)
                cdp.call("Browser.close")
        except Exception:
            pass
        try:
            proc.wait(timeout=10)
        except subprocess.TimeoutExpired:
            subprocess.run(["taskkill", "/F", "/T", "/PID", str(proc.pid)], capture_output=True)
            proc.kill()
        time.sleep(0.5)
        shutil.rmtree(work, ignore_errors=True)

    dest = out_dir / f"leaderboard_{snapshot:%Y%m%d_%H%M}.csv"
    dest.write_bytes(raw)

    text = raw.decode("utf-8-sig", errors="replace")
    rows = list(csv.reader(io.StringIO(text)))
    header, body = rows[0], [r for r in rows[1:] if any(c.strip() for c in r)]

    print(f"Snapshot: {snapshot:%Y-%m-%d %H:%M:%S} (local time)   URL: {URL}")
    print(f"Tabs: {tabs}")
    print(f"Download button: {label!r}  (server file name: {src.name})")
    print(f"Saved raw CSV: {dest}  ({len(raw)} bytes, {len(body)} rows)")
    print("\n===== Leaderboard tab text (verbatim) =====")
    print(panel_text)
    print("\n===== On-screen grid (as displayed) =====")
    print(fmt_table(grid_head or [f"c{i}" for i in range(len(grid_rows[0]) if grid_rows else 0)], grid_rows))

    if SCORE_COL in header:
        si = header.index(SCORE_COL)
        body.sort(key=lambda r: -to_float(r[si]))  # stable: keeps file order among ties
    print("\n===== CSV: every row, sorted by score (ties keep file order) =====")
    print(fmt_table(["#"] + header, [[i + 1] + r for i, r in enumerate(body)]))

    if SCORE_COL in header and TEAM_COL in header:
        si, ti = header.index(SCORE_COL), header.index(TEAM_COL)
        best, count = {}, {}
        for r in body:
            count[r[ti]] = count.get(r[ti], 0) + 1
            if r[ti] not in best or to_float(r[si]) > to_float(best[r[ti]][si]):
                best[r[ti]] = r
        ranked = sorted(best.values(), key=lambda r: -to_float(r[si]))
        print(f"\n===== Best per team ({len(ranked)} distinct teams) =====")
        print(fmt_table(["#", TEAM_COL, SCORE_COL, "Lliuraments", "Data (millor)"],
                        [[i + 1, r[ti], r[si], count[r[ti]], r[header.index("Data")] if "Data" in header else ""]
                         for i, r in enumerate(ranked)]))
    else:
        print(f"\n(CSV has no {SCORE_COL!r}/{TEAM_COL!r} columns; header is {header})")


if __name__ == "__main__":
    main()
