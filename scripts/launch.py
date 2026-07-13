"""Build both Lilac web apps and serve them locally in one step.

    python scripts/launch.py              # build both, serve, open a browser
    python scripts/launch.py --build-only # just regenerate the HTML, don't serve
    python scripts/launch.py --port 9000  # serve on a specific port

The two apps are self-contained HTML files written to outputs/. This script
regenerates them (so they reflect the current data/sensors) and starts a small
local web server rooted at outputs/, printing links to each app.
"""

from __future__ import annotations

import argparse
import subprocess
import sys
import webbrowser
from functools import partial
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "outputs"

APPS = [
    ("build_app.py", "lilac_pairings.html", "🌸 Pairing explorer"),
    ("build_molecule_widget.py", "lilac_molecules.html", "🔬 Molecule inspector"),
]


def build() -> None:
    for script, _, label in APPS:
        print(f"building {label} …")
        subprocess.run([sys.executable, str(ROOT / "scripts" / script)], check=True)


def serve(port: int, open_browser: bool) -> None:
    handler = partial(SimpleHTTPRequestHandler, directory=str(OUT))
    with ThreadingHTTPServer(("127.0.0.1", port), handler) as httpd:
        base = f"http://127.0.0.1:{port}"
        print("\nServing Lilac apps — press Ctrl+C to stop:")
        for _, filename, label in APPS:
            print(f"  {label:24} {base}/{filename}")
        if open_browser:
            try:
                webbrowser.open(f"{base}/{APPS[0][1]}")
            except Exception:
                pass  # headless environments have no browser; the URLs above still work
        try:
            httpd.serve_forever()
        except KeyboardInterrupt:
            print("\nstopped.")


def main() -> None:
    ap = argparse.ArgumentParser(description="Build and serve the Lilac web apps.")
    ap.add_argument("--build-only", action="store_true", help="regenerate HTML, don't serve")
    ap.add_argument("--port", type=int, default=8000)
    ap.add_argument("--no-open", action="store_true", help="don't try to open a browser")
    args = ap.parse_args()

    build()
    if args.build_only:
        print(f"\nDone. Open these files in a browser:")
        for _, filename, label in APPS:
            print(f"  {label:24} {OUT / filename}")
        return
    serve(args.port, open_browser=not args.no_open)


if __name__ == "__main__":
    main()
