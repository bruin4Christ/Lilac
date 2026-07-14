"""Generate the self-contained Lilac pairing explorer (single HTML file).

    python scripts/build_app.py

Precomputes, for every ingredient, its reinforce / bridge / contrast partner
lists (IDF-weighted cosine over the sensor panel) plus the distinctive shared
"bridge note" for each pair, and embeds it all in one static HTML page with no
external requests. Writes outputs/lilac_pairings.html.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import numpy as np  # noqa: E402

from lilac.data import load_flavor_network  # noqa: E402
from lilac.ingredients import build_ingredient_signatures, idf_weights  # noqa: E402
from lilac.sensors import BIT_NAMES  # noqa: E402

TOP = 10  # partners per list


def build_data() -> dict:
    df = load_flavor_network(min_compounds=5)
    sigs = build_ingredient_signatures(df=df)
    names = list(sigs)
    cats = dict(zip(df["ingredient"], df["category"]))
    idf = idf_weights(sigs)

    soft = np.array([sigs[n].soft for n in names])
    W = soft * idf
    Wn = W / (np.linalg.norm(W, axis=1, keepdims=True) + 1e-9)
    S = Wn @ Wn.T                      # cosine similarity matrix
    np.fill_diagonal(S, -1.0)          # exclude self from "most alike"

    methyl = BIT_NAMES.index("methyl")

    def note(i: int, j: int) -> int:
        shared = np.minimum(W[i], W[j]).copy()
        shared[methyl] = 0.0           # ubiquitous, uninformative
        b = int(np.argmax(shared))
        return b if shared[b] > 0 else -1

    def top_sensors(i: int, k: int = 5) -> list[int]:
        order = np.argsort(-soft[i])
        return [int(b) for b in order if b != methyl and soft[i][b] > 0][:k]

    records = []
    for i, name in enumerate(names):
        sims = S[i]
        reinforce = np.argsort(-sims)[:TOP]
        contrast = np.argsort(sims)
        contrast = [j for j in contrast if sims[j] >= 0][:TOP]  # skip the -1 self
        valid = sims[sims >= 0]
        # Bridge centre = a percentile of THIS ingredient's own partner
        # similarities, so the "middle overlap" self-calibrates per ingredient
        # (garlic is far from everything, blueberry close to everything).
        target = float(np.percentile(valid, 65)) if valid.size else 0.0
        bridge = np.argsort(np.abs(sims - target))
        bridge = [j for j in bridge if sims[j] >= 0][:TOP]

        def pack(idxs):
            return [[int(j), round(float(sims[j]), 3), note(i, j)] for j in idxs]

        records.append({
            "p": top_sensors(i),
            "r": pack(reinforce),
            "b": pack(bridge),
            "c": pack(contrast),
        })

    return {
        "names": names,
        "cats": [cats[n] for n in names],
        "ncomp": [int(sigs[n].n_molecules) for n in names],
        "sensors": BIT_NAMES,
        "recs": records,
    }


def main() -> None:
    data = build_data()
    html = TEMPLATE.replace("/*__DATA__*/", json.dumps(data, separators=(",", ":")))
    out = Path("outputs/lilac_pairings.html")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(html, encoding="utf-8")
    kb = out.stat().st_size / 1024
    print(f"Wrote {out}  ({len(data['names'])} ingredients, {kb:.0f} KB)")


TEMPLATE = r"""<style>
:root{
  --bg:#FAF9FE; --surface:#FFFFFF; --surface-2:#F4F1FC; --border:#E7E2F4;
  --text:#1B1726; --muted:#6C6482; --accent:#7C5CF0; --accent-soft:#EEE9FE;
  --reinforce:#2E9E6B; --bridge:#C0851C; --contrast:#5670D6;
  --track:#EDE9F8;
  --mono:ui-monospace,"SF Mono",Menlo,Consolas,monospace;
  --sans:-apple-system,BlinkMacSystemFont,"Segoe UI",Roboto,system-ui,sans-serif;
}
@media (prefers-color-scheme:dark){
  :root{
    --bg:#121019; --surface:#1A1624; --surface-2:#221C30; --border:#2E2740;
    --text:#ECE7F7; --muted:#9A93AE; --accent:#A78BFF; --accent-soft:#241C3A;
    --reinforce:#4FBD86; --bridge:#E0A63C; --contrast:#8AA0FF; --track:#26203440;
  }
}
:root[data-theme="light"]{
  --bg:#FAF9FE; --surface:#FFFFFF; --surface-2:#F4F1FC; --border:#E7E2F4;
  --text:#1B1726; --muted:#6C6482; --accent:#7C5CF0; --accent-soft:#EEE9FE;
  --reinforce:#2E9E6B; --bridge:#C0851C; --contrast:#5670D6; --track:#EDE9F8;
}
:root[data-theme="dark"]{
  --bg:#121019; --surface:#1A1624; --surface-2:#221C30; --border:#2E2740;
  --text:#ECE7F7; --muted:#9A93AE; --accent:#A78BFF; --accent-soft:#241C3A;
  --reinforce:#4FBD86; --bridge:#E0A63C; --contrast:#8AA0FF; --track:#26203440;
}
*{box-sizing:border-box}
.wrap{max-width:1140px;margin:0 auto;padding:28px 20px 72px;color:var(--text);
  font-family:var(--sans);line-height:1.5;background:var(--bg)}
body{background:var(--bg)}
.masthead{display:flex;flex-wrap:wrap;align-items:baseline;gap:12px;
  border-bottom:1px solid var(--border);padding-bottom:16px}
.logo{font-family:var(--mono);font-weight:700;font-size:24px;letter-spacing:-.02em;
  color:var(--accent)}
.logo b{color:var(--text)}
.tagline{color:var(--muted);font-size:14px;max-width:52ch}
.pickrow{display:flex;flex-wrap:wrap;gap:10px;align-items:center;margin:22px 0 8px}
.combo{position:relative;flex:1;min-width:240px;max-width:420px}
.combo input{width:100%;font-family:var(--mono);font-size:15px;color:var(--text);
  background:var(--surface);border:1px solid var(--border);border-radius:10px;
  padding:11px 13px}
.combo input:focus{outline:2px solid var(--accent);outline-offset:1px;border-color:transparent}
.menu{position:absolute;z-index:20;top:calc(100% + 4px);left:0;right:0;max-height:280px;
  overflow:auto;background:var(--surface);border:1px solid var(--border);border-radius:10px;
  box-shadow:0 12px 34px -18px #0009;display:none}
.menu.open{display:block}
.opt{display:flex;justify-content:space-between;gap:10px;padding:8px 12px;cursor:pointer;
  font-family:var(--mono);font-size:14px}
.opt:hover,.opt.active{background:var(--accent-soft)}
.opt .cat{color:var(--muted);font-size:12px}
.btn{font-family:var(--sans);font-size:14px;font-weight:600;color:var(--text);
  background:var(--surface);border:1px solid var(--border);border-radius:10px;
  padding:10px 14px;cursor:pointer}
.btn:hover{border-color:var(--accent);color:var(--accent)}
.btn:focus-visible{outline:2px solid var(--accent);outline-offset:1px}

.base{background:var(--surface-2);border:1px solid var(--border);border-radius:14px;
  padding:18px 20px;margin:14px 0 22px;display:flex;flex-wrap:wrap;gap:18px 26px;align-items:center}
.base .name{font-size:26px;font-weight:700;letter-spacing:-.01em;text-transform:capitalize;
  text-wrap:balance}
.base .meta{font-family:var(--mono);font-size:13px;color:var(--muted)}
.base .meta b{color:var(--text)}
.chips{display:flex;flex-wrap:wrap;gap:6px;margin-top:4px}
.chip{font-family:var(--mono);font-size:12px;color:var(--accent);background:var(--accent-soft);
  border:1px solid var(--border);border-radius:999px;padding:3px 9px}

.cols{display:grid;grid-template-columns:repeat(3,1fr);gap:16px}
@media (max-width:820px){.cols{grid-template-columns:1fr}}
.card{background:var(--surface);border:1px solid var(--border);border-radius:14px;
  overflow:hidden;display:flex;flex-direction:column}
.card h2{font-size:15px;margin:0;padding:14px 16px 4px;display:flex;align-items:center;gap:8px}
.dot{width:9px;height:9px;border-radius:50%}
.card .rule{padding:0 16px 12px;color:var(--muted);font-size:12.5px;border-bottom:1px solid var(--border)}
.reinforce .accentbar{background:var(--reinforce)}
.bridge .accentbar{background:var(--bridge)}
.contrast .accentbar{background:var(--contrast)}
.row{display:grid;grid-template-columns:18px 1fr auto;gap:8px 10px;align-items:center;
  padding:9px 16px;border-bottom:1px solid var(--border)}
.row:last-child{border-bottom:0}
.rank{font-family:var(--mono);font-size:12px;color:var(--muted);text-align:right}
.pname{font-family:var(--mono);font-size:14px;color:var(--text);background:none;border:0;
  padding:0;text-align:left;cursor:pointer;text-transform:capitalize;text-decoration:underline;
  text-decoration-color:var(--border);text-underline-offset:2px}
.pname:hover{color:var(--accent);text-decoration-color:var(--accent)}
.pname:focus-visible{outline:2px solid var(--accent);outline-offset:2px;border-radius:3px}
.sub{grid-column:2;display:flex;flex-wrap:wrap;gap:6px;align-items:center;margin-top:2px}
.tag{font-size:11px;color:var(--muted);font-family:var(--mono)}
.via{font-size:11px;color:var(--muted);font-family:var(--mono)}
.via em{color:var(--accent);font-style:normal}
.sim{font-family:var(--mono);font-size:12.5px;color:var(--text);
  font-variant-numeric:tabular-nums;text-align:right}
.meter{grid-column:2 / span 2;height:4px;border-radius:3px;background:var(--track);margin-top:6px;overflow:hidden}
.meter i{display:block;height:100%;border-radius:3px}
.reinforce .meter i{background:var(--reinforce)}
.bridge .meter i{background:var(--bridge)}
.contrast .meter i{background:var(--contrast)}
.foot{margin-top:26px;color:var(--muted);font-size:12.5px;max-width:78ch}
.foot code{font-family:var(--mono);background:var(--surface-2);padding:1px 5px;border-radius:5px}
@media (prefers-reduced-motion:no-preference){
  .card,.base{transition:opacity .18s ease}
  .fade{opacity:0}
}
</style>

<div class="wrap">
  <header class="masthead">
    <div class="logo"><b>lilac</b> · pairing explorer</div>
    <div class="tagline">Each ingredient is its aroma molecules superimposed on a 55-sensor
      panel. Pick a base and see what the model says harmonizes, bridges, or contrasts —
      ranked by IDF-weighted overlap.</div>
  </header>

  <div class="pickrow">
    <div class="combo">
      <input id="q" type="text" placeholder="Search an ingredient…" autocomplete="off"
             role="combobox" aria-expanded="false" aria-controls="menu" aria-label="Base ingredient">
      <div class="menu" id="menu" role="listbox"></div>
    </div>
    <button class="btn" id="rand" type="button">🎲 Surprise me</button>
    <button class="btn" id="theme" type="button" aria-label="Toggle theme">◑ Theme</button>
  </div>

  <section class="base" id="base" aria-live="polite"></section>

  <div class="cols">
    <div class="card reinforce"><h2><span class="dot accentbar"></span>Reinforce</h2>
      <div class="rule">Most shared aroma — pair by similarity (the “shared-compound” idea)</div>
      <div id="list-r"></div></div>
    <div class="card bridge"><h2><span class="dot accentbar"></span>Bridge</h2>
      <div class="rule">Mid overlap — some notes shared, some new. Where surprising pairings live</div>
      <div id="list-b"></div></div>
    <div class="card contrast"><h2><span class="dot accentbar"></span>Contrast</h2>
      <div class="rule">Least overlap — pairing by opposition</div>
      <div id="list-c"></div></div>
  </div>

  <p class="foot">Aroma-only hypotheses from ~595 ingredients (Ahn <em>et al.</em> Flavor
    Network × Lilac sensors). Compounds are weighted equally (no concentrations), so read
    these as leads, not verdicts — taste, texture and culture decide the plate. Click any
    ingredient to re-center; <code>🎲</code> jumps somewhere random.</p>
</div>

<script>
const DATA = /*__DATA__*/;
const {names, cats, ncomp, sensors, recs} = DATA;
const el = id => document.getElementById(id);
const cap = s => s.replace(/_/g," ");

// ---- theme toggle ----
el("theme").onclick = () => {
  const root = document.documentElement;
  const cur = root.getAttribute("data-theme")
    || (matchMedia("(prefers-color-scheme:dark)").matches ? "dark" : "light");
  root.setAttribute("data-theme", cur === "dark" ? "light" : "dark");
};

// ---- rows ----
function rowsHTML(list, mode){
  const max = Math.max(...list.map(p=>p[1]), 0.001);
  return list.map((p,k)=>{
    const [j,sim,note] = p;
    const via = note>=0 ? `<span class="via">via <em>${sensors[note]}</em></span>` : "";
    const w = Math.max(4, Math.round(100*sim/max));
    return `<div class="row">
      <span class="rank">${k+1}</span>
      <button class="pname" data-go="${j}">${cap(names[j])}</button>
      <span class="sim">${sim.toFixed(2)}</span>
      <span class="sub"><span class="tag">${cats[j]}</span>${via}</span>
      <span class="meter"><i style="width:${w}%"></i></span>
    </div>`;
  }).join("");
}

function render(i){
  const r = recs[i];
  el("base").innerHTML = `
    <div>
      <div class="name">${cap(names[i])}</div>
      <div class="chips">${r.p.map(b=>`<span class="chip">${sensors[b]}</span>`).join("")||
        '<span class="tag">no distinctive sensors above baseline</span>'}</div>
    </div>
    <div class="meta"><b>${cats[i]}</b> · <b>${ncomp[i]}</b> aroma compounds ·
      base signature superimposed from all of them</div>`;
  el("list-r").innerHTML = rowsHTML(r.r,"r");
  el("list-b").innerHTML = rowsHTML(r.b,"b");
  el("list-c").innerHTML = rowsHTML(r.c,"c");
  el("q").value = cap(names[i]);
  history.replaceState(null,"", "#"+encodeURIComponent(names[i]));
  document.querySelectorAll(".pname").forEach(b=>{
    b.onclick = () => go(+b.dataset.go);
  });
}
function go(i){
  for(const box of ["base"]) el(box).classList.add("fade");
  render(i);
  requestAnimationFrame(()=>el("base").classList.remove("fade"));
  window.scrollTo({top:0,behavior:"smooth"});
}

// ---- combobox ----
const menu = el("menu"), q = el("q");
let filtered = [], active = -1;
function openMenu(term){
  term = term.toLowerCase().trim();
  filtered = names.map((n,i)=>[n,i])
    .filter(([n])=>n.includes(term))
    .slice(0,60);
  menu.innerHTML = filtered.map(([n,i],k)=>
    `<div class="opt${k===active?' active':''}" role="option" data-i="${i}">
       <span>${cap(n)}</span><span class="cat">${cats[i]}</span></div>`).join("")
    || `<div class="opt"><span>no match</span></div>`;
  menu.classList.add("open"); q.setAttribute("aria-expanded","true");
  menu.querySelectorAll(".opt[data-i]").forEach(o=>{
    o.onmousedown = e => { e.preventDefault(); go(+o.dataset.i); closeMenu(); };
  });
}
function closeMenu(){menu.classList.remove("open");q.setAttribute("aria-expanded","false");active=-1;}
q.addEventListener("input", ()=>{active=-1;openMenu(q.value);});
q.addEventListener("focus", ()=>openMenu(q.value));
q.addEventListener("blur", ()=>setTimeout(closeMenu,120));
q.addEventListener("keydown", e=>{
  if(!menu.classList.contains("open"))return;
  if(e.key==="ArrowDown"){active=Math.min(active+1,filtered.length-1);openMenu(q.value);e.preventDefault();}
  else if(e.key==="ArrowUp"){active=Math.max(active-1,0);openMenu(q.value);e.preventDefault();}
  else if(e.key==="Enter"&&active>=0){go(filtered[active][1]);closeMenu();e.preventDefault();}
  else if(e.key==="Escape"){closeMenu();}
});
el("rand").onclick = ()=>go(Math.floor(Math.random()*names.length));

// ---- boot ----
function start(){
  const h = decodeURIComponent(location.hash.slice(1));
  let i = names.indexOf(h);
  if(i<0) i = names.indexOf("coffee");
  if(i<0) i = 0;
  render(i);
}
start();
</script>
"""


if __name__ == "__main__":
    main()
