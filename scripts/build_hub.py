"""Generate the Lilac hub — a small landing page linking the three web apps.

    python scripts/build_hub.py     # -> outputs/index.html

Self-contained, no external requests. Served at the site root by launch.py.
"""

from __future__ import annotations

from pathlib import Path

CARDS = [
    ("lilac_compose.html", "🍽", "Composition studio",
     "Grow a whole dish from one base — partners chosen to bridge and complement it "
     "across the sensor palette, each with its role, bridge note, and a Harmonious↔"
     "Adventurous dial.", "compose"),
    ("lilac_pairings.html", "🌸", "Pairing explorer",
     "For any ingredient, what the model says reinforces, bridges, or contrasts it — "
     "ranked by IDF-weighted overlap, each pair labelled with the distinctive sensor "
     "that connects them.", "pair"),
    ("lilac_molecules.html", "🔬", "Molecule inspector",
     "Open an ingredient and read the molecules inside it — each rendered as its sensor "
     "signature, plus the ingredient's superimposed 'smell number' and a plain-English "
     "index of every bit.", "inspect"),
]


def main() -> None:
    cards = "\n".join(
        f'''    <a class="card {slug}" href="{href}">
      <div class="ico">{ico}</div>
      <div class="body"><h2>{title}</h2><p>{blurb}</p></div>
      <div class="go" aria-hidden="true">→</div>
    </a>''' for href, ico, title, blurb, slug in CARDS)
    html = TEMPLATE.replace("<!--__CARDS__-->", cards)
    out = Path("outputs/index.html")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(html, encoding="utf-8")
    print(f"Wrote {out}")


TEMPLATE = r"""<style>
:root{
  --bg:#FAF9FE; --surface:#FFFFFF; --surface-2:#F4F1FC; --border:#E7E2F4;
  --text:#1B1726; --muted:#6C6482; --accent:#7C5CF0; --accent-soft:#EEE9FE;
  --compose:#7C5CF0; --pair:#2E9E6B; --inspect:#C0851C;
  --mono:ui-monospace,"SF Mono",Menlo,Consolas,monospace;
  --sans:-apple-system,BlinkMacSystemFont,"Segoe UI",Roboto,system-ui,sans-serif;
}
@media (prefers-color-scheme:dark){:root{
  --bg:#121019; --surface:#1A1624; --surface-2:#221C30; --border:#2E2740;
  --text:#ECE7F7; --muted:#9A93AE; --accent:#A78BFF; --accent-soft:#241C3A;
  --compose:#A78BFF; --pair:#4FBD86; --inspect:#E0A63C;
}}
:root[data-theme="light"]{
  --bg:#FAF9FE; --surface:#FFFFFF; --surface-2:#F4F1FC; --border:#E7E2F4;
  --text:#1B1726; --muted:#6C6482; --accent:#7C5CF0; --accent-soft:#EEE9FE;
  --compose:#7C5CF0; --pair:#2E9E6B; --inspect:#C0851C;
}
:root[data-theme="dark"]{
  --bg:#121019; --surface:#1A1624; --surface-2:#221C30; --border:#2E2740;
  --text:#ECE7F7; --muted:#9A93AE; --accent:#A78BFF; --accent-soft:#241C3A;
  --compose:#A78BFF; --pair:#4FBD86; --inspect:#E0A63C;
}
*{box-sizing:border-box}
body{background:var(--bg)}
.wrap{max-width:760px;margin:0 auto;padding:56px 20px 72px;color:var(--text);
  font-family:var(--sans);line-height:1.5}
.top{display:flex;align-items:baseline;gap:14px;flex-wrap:wrap}
.logo{font-family:var(--mono);font-weight:700;font-size:30px;letter-spacing:-.02em;color:var(--accent)}
.logo b{color:var(--text)}
.theme{margin-left:auto;font-family:var(--sans);font-size:13px;font-weight:600;color:var(--muted);
  background:var(--surface);border:1px solid var(--border);border-radius:9px;padding:8px 12px;cursor:pointer}
.theme:hover{color:var(--accent);border-color:var(--accent)}
.lede{color:var(--muted);font-size:15.5px;max-width:60ch;margin:14px 0 34px}
.lede b{color:var(--text);font-weight:600}
.cards{display:flex;flex-direction:column;gap:14px}
.card{display:flex;gap:18px;align-items:center;text-decoration:none;color:inherit;
  background:var(--surface);border:1px solid var(--border);border-radius:16px;padding:20px 22px;
  transition:border-color .15s ease,transform .15s ease}
.card:hover{transform:translateY(-2px)}
.card.compose:hover{border-color:var(--compose)}
.card.pair:hover{border-color:var(--pair)}
.card.inspect:hover{border-color:var(--inspect)}
.ico{font-size:30px;width:56px;height:56px;flex:none;display:grid;place-items:center;
  background:var(--surface-2);border-radius:14px}
.body h2{margin:0 0 4px;font-size:18px;letter-spacing:-.01em}
.card.compose h2{color:var(--compose)} .card.pair h2{color:var(--pair)} .card.inspect h2{color:var(--inspect)}
.body p{margin:0;color:var(--muted);font-size:13.5px;max-width:56ch}
.go{margin-left:auto;font-size:22px;color:var(--muted);flex:none}
.card:hover .go{color:var(--text)}
.foot{margin-top:34px;color:var(--muted);font-size:12.5px;max-width:64ch}
.foot code{font-family:var(--mono);background:var(--surface-2);padding:1px 5px;border-radius:5px}
</style>

<div class="wrap">
  <div class="top">
    <div class="logo"><b>lilac</b> · olfactory studio</div>
    <button class="theme" id="theme" type="button">◑ Theme</button>
  </div>
  <p class="lede">An interpretable map from <b>molecules → a named sensor "nose" → flavor</b>.
    Three ways in: compose a dish, explore a pairing, or inspect the chemistry.</p>

  <div class="cards">
<!--__CARDS__-->
  </div>

  <p class="foot">All three pages are self-contained and precomputed from
    <code>lilac.sensors</code> over ~595 ingredients (Ahn <em>et al.</em> Flavor Network).
    Aroma-only hypotheses — read them as leads, not verdicts.</p>
</div>

<script>
document.getElementById("theme").onclick = () => {
  const r = document.documentElement;
  const c = r.getAttribute("data-theme")
    || (matchMedia("(prefers-color-scheme:dark)").matches ? "dark":"light");
  r.setAttribute("data-theme", c==="dark"?"light":"dark");
};
</script>
"""


if __name__ == "__main__":
    main()
