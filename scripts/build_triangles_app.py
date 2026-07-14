"""Generate the self-contained Lilac triangle explorer (single HTML file).

    python scripts/build_triangles_app.py

For every ingredient, precomputes its best closed **A–B–C bridge triangles** at both
levels -- **bit** (each edge a shared sensor / note family) and **molecular** (each
edge an actual shared compound) -- and embeds them, each drawn as a little triangle
diagram. Click a vertex to re-anchor. Writes outputs/lilac_triangles.html.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from lilac.data import load_flavor_network, load_odorant_library  # noqa: E402
from lilac.ingredients import build_ingredient_signatures, idf_weights  # noqa: E402
from lilac.triangles import _bit_edges, _molecular_edges, _search  # noqa: E402

TOP = 6          # triangles per ingredient per level
FAMILIES = ["terpene", "sulfur", "roasted/animalic", "fruity/creamy",
            "phenolic/balsamic", "oxygenated", "other"]


def build_data() -> dict:
    df = load_flavor_network(min_compounds=5)
    sigs = build_ingredient_signatures(df=df)
    idf = idf_weights(sigs)
    names = list(sigs)
    idx = {n: i for i, n in enumerate(names)}
    cats = dict(zip(df["ingredient"], df["category"]))

    cname = dict(zip(load_odorant_library()["smiles"],
                     load_odorant_library()["name"].astype(str)))

    providers = {
        "bit": _bit_edges(sigs, idf, 0.88),
        "mol": _molecular_edges(df, 0.9, cname),
    }
    fam_idx = {f: i for i, f in enumerate(FAMILIES)}
    vocab: dict[str, int] = {}

    def via_idx(s: str) -> int:
        if s not in vocab:
            vocab[s] = len(vocab)
        return vocab[s]

    def pack(t) -> list:
        members = [idx[m] for m in t.members]
        edges = []
        for e in t.edges:
            fam = e.group if e.group in fam_idx else "other"      # bit: family; mol: "other"
            edges.append([via_idx(e.via), fam_idx[fam], round(e.strength, 2)])
        magical = len(t.families) >= 3 and len(t.categories) >= 3
        return [members, edges, int(magical)]

    levels: dict[str, list] = {"bit": [], "mol": []}
    for lvl, (lnames, strength, too_sim, edge_of) in providers.items():
        for name in names:
            tris = _search(lnames, strength, too_sim, edge_of, cats, name, TOP,
                           80.0, None, 1, 1)
            levels[lvl].append([pack(t) for t in tris])

    compounds_via = [None] * len(vocab)
    for s, i in vocab.items():
        compounds_via[i] = s

    return {
        "names": names,
        "cats": [cats[n] for n in names],
        "ncomp": [int(sigs[n].n_molecules) for n in names],
        "families": FAMILIES,
        "vocab": compounds_via,
        "levels": levels,
    }


def main() -> None:
    data = build_data()
    html = TEMPLATE.replace("/*__DATA__*/", json.dumps(data, separators=(",", ":")))
    out = Path("outputs/lilac_triangles.html")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(html, encoding="utf-8")
    kb = out.stat().st_size / 1024
    print(f"Wrote {out}  ({len(data['names'])} ingredients, {kb:.0f} KB)")


TEMPLATE = r"""<style>
:root{
  --bg:#FAF9FE; --surface:#FFFFFF; --surface-2:#F4F1FC; --border:#E7E2F4;
  --text:#1B1726; --muted:#6C6482; --accent:#7C5CF0; --accent-soft:#EEE9FE;
  --node:#221C30; --line:#B9B0D0;
  --f0:#3FA46A; --f1:#C98A1C; --f2:#B5561F; --f3:#C2568F; --f4:#7C5CF0; --f5:#2E86C1; --f6:#6C6482;
  --mono:ui-monospace,"SF Mono",Menlo,Consolas,monospace;
  --sans:-apple-system,BlinkMacSystemFont,"Segoe UI",Roboto,system-ui,sans-serif;
}
@media (prefers-color-scheme:dark){:root{
  --bg:#121019; --surface:#1A1624; --surface-2:#221C30; --border:#2E2740;
  --text:#ECE7F7; --muted:#9A93AE; --accent:#A78BFF; --accent-soft:#241C3A;
  --node:#ECE7F7; --line:#4A4266;
  --f0:#5BC489; --f1:#E0A63C; --f2:#E0764A; --f3:#E58BB8; --f4:#A78BFF; --f5:#5FA8DE; --f6:#9A93AE;
}}
:root[data-theme="light"]{ --bg:#FAF9FE; --surface:#FFFFFF; --surface-2:#F4F1FC; --border:#E7E2F4;
  --text:#1B1726; --muted:#6C6482; --accent:#7C5CF0; --accent-soft:#EEE9FE; --node:#221C30; --line:#B9B0D0;
  --f0:#3FA46A; --f1:#C98A1C; --f2:#B5561F; --f3:#C2568F; --f4:#7C5CF0; --f5:#2E86C1; --f6:#6C6482; }
:root[data-theme="dark"]{ --bg:#121019; --surface:#1A1624; --surface-2:#221C30; --border:#2E2740;
  --text:#ECE7F7; --muted:#9A93AE; --accent:#A78BFF; --accent-soft:#241C3A; --node:#ECE7F7; --line:#4A4266;
  --f0:#5BC489; --f1:#E0A63C; --f2:#E0764A; --f3:#E58BB8; --f4:#A78BFF; --f5:#5FA8DE; --f6:#9A93AE; }
*{box-sizing:border-box}
body{background:var(--bg)}
.wrap{max-width:1000px;margin:0 auto;padding:26px 20px 72px;color:var(--text);font-family:var(--sans);line-height:1.5}
.masthead{display:flex;flex-wrap:wrap;align-items:baseline;gap:12px;border-bottom:1px solid var(--border);padding-bottom:16px}
.logo{font-family:var(--mono);font-weight:700;font-size:23px;letter-spacing:-.02em;color:var(--accent)}
.logo b{color:var(--text)}
.tagline{color:var(--muted);font-size:14px;max-width:58ch}
.nav{margin-left:auto;display:flex;gap:14px;font-size:13px}
.nav a{color:var(--muted);text-decoration:none;border-bottom:1px solid transparent}
.nav a:hover{color:var(--accent);border-bottom-color:var(--accent)}
.pickrow{display:flex;flex-wrap:wrap;gap:10px;align-items:center;margin:20px 0 6px}
.combo{position:relative;flex:1;min-width:220px;max-width:360px}
.combo input{width:100%;font-family:var(--mono);font-size:15px;color:var(--text);background:var(--surface);
  border:1px solid var(--border);border-radius:10px;padding:11px 13px}
.combo input:focus{outline:2px solid var(--accent);outline-offset:1px;border-color:transparent}
.menu{position:absolute;z-index:20;top:calc(100% + 4px);left:0;right:0;max-height:280px;overflow:auto;
  background:var(--surface);border:1px solid var(--border);border-radius:10px;box-shadow:0 12px 34px -18px #0009;display:none}
.menu.open{display:block}
.opt{display:flex;justify-content:space-between;gap:10px;padding:8px 12px;cursor:pointer;font-family:var(--mono);font-size:14px}
.opt:hover,.opt.active{background:var(--accent-soft)}
.opt .cat{color:var(--muted);font-size:12px}
.btn{font-family:var(--sans);font-size:14px;font-weight:600;color:var(--text);background:var(--surface);
  border:1px solid var(--border);border-radius:10px;padding:10px 14px;cursor:pointer}
.btn:hover{border-color:var(--accent);color:var(--accent)}
.seg{display:inline-flex;border:1px solid var(--border);border-radius:10px;overflow:hidden}
.seg button{font-family:var(--sans);font-size:13px;font-weight:600;color:var(--muted);background:var(--surface);
  border:0;padding:10px 14px;cursor:pointer}
.seg button.on{background:var(--accent-soft);color:var(--accent)}
.hint{color:var(--muted);font-size:13px;margin:8px 0 14px}
.grid{display:grid;grid-template-columns:repeat(auto-fill,minmax(280px,1fr));gap:16px}
.tri{background:var(--surface);border:1px solid var(--border);border-radius:14px;padding:12px 12px 6px;position:relative}
.tri.magic{border-color:var(--accent)}
.badge{position:absolute;top:10px;right:10px;font-size:11px;font-family:var(--mono);color:var(--accent);
  background:var(--accent-soft);border-radius:999px;padding:2px 8px}
.tnode{fill:var(--node);font-family:var(--sans);font-size:12.5px;font-weight:600;cursor:pointer}
.tnode:hover{fill:var(--accent);text-decoration:underline}
.elabel{font-family:var(--mono);font-size:10.5px}
.edesc{padding:2px 6px 8px;font-size:12px;color:var(--muted)}
.edesc b{color:var(--text);font-weight:600}
.empty{color:var(--muted);font-size:14px;padding:20px}
.foot{margin-top:26px;color:var(--muted);font-size:12.5px;max-width:80ch}
.foot code{font-family:var(--mono);background:var(--surface-2);padding:1px 5px;border-radius:5px}
</style>

<div class="wrap">
  <header class="masthead">
    <div class="logo"><b>lilac</b> · triangle explorer</div>
    <div class="tagline">Three ingredients in a closed loop — every pair bridges, and the best
      loops bridge on three <em>different</em> notes.</div>
    <nav class="nav">
      <a href="lilac_compose.html">compose ↗</a>
      <a href="lilac_pairings.html">pairings ↗</a>
      <a href="lilac_molecules.html">molecules ↗</a>
    </nav>
  </header>

  <div class="pickrow">
    <div class="combo">
      <input id="q" type="text" placeholder="Anchor on an ingredient…" autocomplete="off"
             role="combobox" aria-expanded="false" aria-controls="menu" aria-label="Anchor ingredient">
      <div class="menu" id="menu" role="listbox"></div>
    </div>
    <div class="seg" role="tablist" aria-label="Level">
      <button id="lv-bit" class="on" type="button" title="Edges = shared sensor / note family">Sensor bits</button>
      <button id="lv-mol" type="button" title="Edges = an actual shared molecule">Shared molecules</button>
    </div>
    <button class="btn" id="rand" type="button">🎲</button>
    <button class="btn" id="theme" type="button" aria-label="Toggle theme">◑</button>
  </div>

  <div class="hint" id="hint"></div>
  <section class="grid" id="grid" aria-live="polite"></section>

  <p class="foot"><b>Sensor bits</b>: each edge is the distinctive sensor two foods most share
    (an abstraction of structure) — ✨ when the three edges are three different note families.
    <b>Shared molecules</b>: each edge is an actual compound the two have in common — ✨ when
    all three edges are different molecules. Edge thickness = bridge strength. Click any vertex
    to re-anchor. Aroma-only leads from ~595 ingredients.</p>
</div>

<script>
const DATA = /*__DATA__*/;
const {names, cats, ncomp, families, vocab, levels} = DATA;
const el = id => document.getElementById(id);
const cap = s => s.replace(/_/g," ");
let level = "bit", cur = 0;

el("theme").onclick = () => {
  const r = document.documentElement;
  const c = r.getAttribute("data-theme") || (matchMedia("(prefers-color-scheme:dark)").matches?"dark":"light");
  r.setAttribute("data-theme", c==="dark"?"light":"dark");
};

const TRIAD = [3,5,0];   // positional family-colors for molecular edges (3 distinct hues)
function edgeColor(fam, k){ return `var(--f${level==="mol"?TRIAD[k]:fam})`; }

// SVG triangle: A top, B bottom-left, C bottom-right; edges AB, BC, AC.
function triSVG(members, edges){
  const P = [[140,34],[34,196],[246,196]];       // A, B, C
  const mid = (a,b)=>[(P[a][0]+P[b][0])/2,(P[a][1]+P[b][1])/2];
  const EP = [[0,1],[1,2],[0,2]];                // edge -> vertex pair (AB,BC,AC)
  let s = `<svg viewBox="0 0 280 220" width="100%" height="200" role="img">`;
  edges.forEach((e,k)=>{
    const [va,vb]=EP[k], w=1.5+7*e[2];
    s += `<line x1="${P[va][0]}" y1="${P[va][1]}" x2="${P[vb][0]}" y2="${P[vb][1]}"
      stroke="${edgeColor(e[1],k)}" stroke-width="${w.toFixed(1)}" stroke-linecap="round" opacity="0.9"/>`;
  });
  edges.forEach((e,k)=>{
    const [va,vb]=EP[k], m=mid(va,vb);
    const label = cap(vocab[e[0]]);
    s += `<text class="elabel" x="${m[0]}" y="${m[1]-4}" text-anchor="middle"
      fill="${edgeColor(e[1],k)}">${label.length>18?label.slice(0,17)+"…":label}</text>`;
  });
  const anchor=["middle","end","start"];
  members.forEach((mi,v)=>{
    const dy = v===0? -6 : 16;
    s += `<text class="tnode" x="${P[v][0]}" y="${P[v][1]+dy}" text-anchor="${anchor[v]}"
      data-go="${mi}">${cap(names[mi])}</text>`;
  });
  return s+`</svg>`;
}

function render(i){
  cur = i;
  const list = levels[level][i];
  el("q").value = cap(names[i]);
  history.replaceState(null,"","#"+encodeURIComponent(names[i]));
  el("hint").innerHTML = list.length
    ? `<b>${cap(names[i])}</b> — ${list.length} triangle${list.length>1?"s":""} `
      + `(${level==="bit"?"bridging on shared sensors":"bridging on shared molecules"})`
    : "";
  if(!list.length){
    el("grid").innerHTML = `<div class="empty">No strong ${level==="bit"?"sensor":"molecular"} triangles for
      <b>${cap(names[i])}</b> — its bridges may be too weak or all one kind. Try the other level, or 🎲.</div>`;
  } else {
    el("grid").innerHTML = list.map(t=>{
      const [members, edges, magic] = t;
      const desc = edges.map((e,k)=>{
        const [va,vb]=[[0,1],[1,2],[0,2]][k];
        return `${cap(names[members[va]])}–${cap(names[members[vb]])} <b>${cap(vocab[e[0]])}</b>`;
      }).join(" · ");
      return `<div class="tri${magic?' magic':''}">${magic?'<span class="badge">✨ magical</span>':''}
        ${triSVG(members,edges)}<div class="edesc">${desc}</div></div>`;
    }).join("");
  }
  el("grid").querySelectorAll(".tnode[data-go]").forEach(t=>{
    t.onclick = ()=>go(+t.dataset.go);
  });
}
function go(i){ render(i); window.scrollTo({top:0,behavior:"smooth"}); }
function setLevel(l){
  level=l; el("lv-bit").classList.toggle("on",l==="bit"); el("lv-mol").classList.toggle("on",l==="mol");
  render(cur);
}
el("lv-bit").onclick=()=>setLevel("bit");
el("lv-mol").onclick=()=>setLevel("mol");

// combobox
const menu=el("menu"), q=el("q"); let filtered=[], active=-1;
function openMenu(term){
  term=term.toLowerCase().trim();
  filtered=names.map((n,i)=>[n,i]).filter(([n])=>n.includes(term)).slice(0,60);
  menu.innerHTML=filtered.map(([n,i],k)=>`<div class="opt${k===active?' active':''}" role="option" data-i="${i}">
    <span>${cap(n)}</span><span class="cat">${cats[i]}</span></div>`).join("")||`<div class="opt"><span>no match</span></div>`;
  menu.classList.add("open"); q.setAttribute("aria-expanded","true");
  menu.querySelectorAll(".opt[data-i]").forEach(o=>{o.onmousedown=e=>{e.preventDefault();go(+o.dataset.i);closeMenu();};});
}
function closeMenu(){menu.classList.remove("open");q.setAttribute("aria-expanded","false");active=-1;}
q.addEventListener("input",()=>{active=-1;openMenu(q.value);});
q.addEventListener("focus",()=>openMenu(q.value));
q.addEventListener("blur",()=>setTimeout(closeMenu,120));
q.addEventListener("keydown",e=>{
  if(!menu.classList.contains("open"))return;
  if(e.key==="ArrowDown"){active=Math.min(active+1,filtered.length-1);openMenu(q.value);e.preventDefault();}
  else if(e.key==="ArrowUp"){active=Math.max(active-1,0);openMenu(q.value);e.preventDefault();}
  else if(e.key==="Enter"&&active>=0){go(filtered[active][1]);closeMenu();e.preventDefault();}
  else if(e.key==="Escape"){closeMenu();}
});
el("rand").onclick=()=>go(Math.floor(Math.random()*names.length));

function start(){
  const h=decodeURIComponent(location.hash.slice(1));
  let i=names.indexOf(h); if(i<0) i=names.indexOf("tarragon"); if(i<0) i=0;
  render(i);
}
start();
</script>
"""


if __name__ == "__main__":
    main()
