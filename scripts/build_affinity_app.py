"""Generate the self-contained Lilac affinity explorer (single HTML file).

    python scripts/build_affinity_app.py

The canon-derived pairing model made clickable. For every ingredient it precomputes
its best **deepeners** (partners that share its character) and **lifters** (partners
that add a distinctive, consonant note in a new register), plus a suggested dish
(one of each), and embeds it all. Writes outputs/lilac_affinity.html.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import numpy as np  # noqa: E402

from lilac.affinity import FAMILIES, rank, register, suggest_dish  # noqa: E402
from lilac.compose import _CHAR_MASK  # noqa: E402
from lilac.data import load_flavor_network  # noqa: E402
from lilac.ingredients import build_ingredient_signatures, idf_weights  # noqa: E402
from lilac.sensors import BIT_NAMES  # noqa: E402
from lilac.triangles import note_family  # noqa: E402

TOP = 12


def build_data() -> dict:
    df = load_flavor_network(min_compounds=5)
    sigs = build_ingredient_signatures(df=df)
    idf = idf_weights(sigs)
    names = list(sigs)
    idx = {n: i for i, n in enumerate(names)}
    cats = dict(zip(df["ingredient"], df["category"]))
    cw = idf * _CHAR_MASK
    fam_idx = {f: i for i, f in enumerate(FAMILIES)}
    sensor_fam = [fam_idx.get(note_family(s), fam_idx["other"]) for s in BIT_NAMES]

    deep, lift, dish, char = [], [], [], []
    for n in names:
        dr = rank(n, sigs, idf, mode="deepener", top=TOP)
        lr = rank(n, sigs, idf, mode="lifter", top=TOP)
        deep.append([[idx[f.partner], BIT_NAMES.index(f.anchor_sensor),
                      round(f.anchor, 2)] for f in dr])
        lift.append([[idx[f.partner], BIT_NAMES.index(f.lift_sensor), round(f.lift, 2),
                      round(f.consonance, 2), int(f.crosses_register)] for f in lr])
        d = suggest_dish(n, sigs, idf, cats)
        dish.append([idx[d["deepener"].partner] if d["deepener"] else -1,
                     idx[d["lifter"].partner] if d["lifter"] else -1])
        bfam = note_family(BIT_NAMES[int(np.argmax(sigs[n].soft * cw))])
        char.append(register(bfam))

    return {
        "names": names,
        "cats": [cats[n] for n in names],
        "sensors": BIT_NAMES,
        "sensorFam": sensor_fam,
        "families": FAMILIES,
        "char": char,
        "deep": deep,
        "lift": lift,
        "dish": dish,
    }


def main() -> None:
    data = build_data()
    html = TEMPLATE.replace("/*__DATA__*/", json.dumps(data, separators=(",", ":")))
    out = Path("outputs/lilac_affinity.html")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(html, encoding="utf-8")
    kb = out.stat().st_size / 1024
    print(f"Wrote {out}  ({len(data['names'])} ingredients, {kb:.0f} KB)")


TEMPLATE = r"""<style>
:root{
  --bg:#FAF9FE; --surface:#FFFFFF; --surface-2:#F4F1FC; --border:#E7E2F4;
  --text:#1B1726; --muted:#6C6482; --accent:#7C5CF0; --accent-soft:#EEE9FE;
  --deep:#3557B8; --deep-soft:#E7ECFA; --lift:#1E9E6B; --lift-soft:#E2F3EC;
  --warn:#C2410C; --track:#EDE9F8;
  --f0:#3FA46A; --f1:#C98A1C; --f2:#B5561F; --f3:#C2568F; --f4:#7C5CF0; --f5:#2E86C1; --f6:#6C6482;
  --mono:ui-monospace,"SF Mono",Menlo,Consolas,monospace;
  --sans:-apple-system,BlinkMacSystemFont,"Segoe UI",Roboto,system-ui,sans-serif;
}
@media (prefers-color-scheme:dark){:root{
  --bg:#121019; --surface:#1A1624; --surface-2:#221C30; --border:#2E2740;
  --text:#ECE7F7; --muted:#9A93AE; --accent:#A78BFF; --accent-soft:#241C3A;
  --deep:#7C9BEC; --deep-soft:#20263F; --lift:#4FBD86; --lift-soft:#16281F;
  --warn:#F0956A; --track:#26203440;
  --f0:#5BC489; --f1:#E0A63C; --f2:#E0764A; --f3:#E58BB8; --f4:#A78BFF; --f5:#5FA8DE; --f6:#9A93AE;
}}
:root[data-theme="light"]{ --bg:#FAF9FE;--surface:#FFFFFF;--surface-2:#F4F1FC;--border:#E7E2F4;
  --text:#1B1726;--muted:#6C6482;--accent:#7C5CF0;--accent-soft:#EEE9FE;
  --deep:#3557B8;--deep-soft:#E7ECFA;--lift:#1E9E6B;--lift-soft:#E2F3EC;--warn:#C2410C;--track:#EDE9F8;
  --f0:#3FA46A;--f1:#C98A1C;--f2:#B5561F;--f3:#C2568F;--f4:#7C5CF0;--f5:#2E86C1;--f6:#6C6482; }
:root[data-theme="dark"]{ --bg:#121019;--surface:#1A1624;--surface-2:#221C30;--border:#2E2740;
  --text:#ECE7F7;--muted:#9A93AE;--accent:#A78BFF;--accent-soft:#241C3A;
  --deep:#7C9BEC;--deep-soft:#20263F;--lift:#4FBD86;--lift-soft:#16281F;--warn:#F0956A;--track:#26203440;
  --f0:#5BC489;--f1:#E0A63C;--f2:#E0764A;--f3:#E58BB8;--f4:#A78BFF;--f5:#5FA8DE;--f6:#9A93AE; }
*{box-sizing:border-box}
body{background:var(--bg)}
.wrap{max-width:1000px;margin:0 auto;padding:26px 20px 72px;color:var(--text);font-family:var(--sans);line-height:1.5}
.masthead{display:flex;flex-wrap:wrap;align-items:baseline;gap:12px;border-bottom:1px solid var(--border);padding-bottom:16px}
.logo{font-family:var(--mono);font-weight:700;font-size:23px;letter-spacing:-.02em;color:var(--accent)}
.logo b{color:var(--text)}
.tagline{color:var(--muted);font-size:14px;max-width:60ch}
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

.dish{background:var(--surface-2);border:1px solid var(--border);border-radius:16px;padding:16px 18px;margin:16px 0 20px}
.dish h3{margin:0 0 3px;font-size:14px}
.dish .base{text-transform:capitalize;font-weight:700}
.dish .reg{font-family:var(--mono);font-size:12px;color:var(--muted)}
.dishrow{display:flex;flex-wrap:wrap;gap:12px;margin-top:12px}
.slot{flex:1;min-width:240px;background:var(--surface);border:1px solid var(--border);border-radius:12px;padding:12px 14px}
.slot.deep{border-left:4px solid var(--deep)} .slot.lift{border-left:4px solid var(--lift)}
.slot .tag{font-family:var(--mono);font-size:11px;font-weight:700;letter-spacing:.06em;text-transform:uppercase}
.slot.deep .tag{color:var(--deep)} .slot.lift .tag{color:var(--lift)}
.slot .nm{font-size:19px;font-weight:700;text-transform:capitalize;margin:2px 0}
.slot .nm button{background:none;border:0;color:inherit;font:inherit;cursor:pointer;padding:0;text-align:left}
.slot .nm button:hover{color:var(--accent)}
.slot .why{font-size:13px;color:var(--muted)}
.slot .why b{color:var(--text)}

.cols{display:grid;grid-template-columns:1fr 1fr;gap:16px}
@media (max-width:760px){.cols{grid-template-columns:1fr}}
.card{background:var(--surface);border:1px solid var(--border);border-radius:14px;overflow:hidden;display:flex;flex-direction:column}
.card h2{font-size:15px;margin:0;padding:14px 16px 3px;display:flex;align-items:center;gap:8px}
.dot{width:9px;height:9px;border-radius:50%}
.card.deepc h2 .dot{background:var(--deep)} .card.liftc h2 .dot{background:var(--lift)}
.card .rule{padding:0 16px 12px;color:var(--muted);font-size:12.5px;border-bottom:1px solid var(--border)}
.row{display:grid;grid-template-columns:20px 1fr auto;gap:5px 10px;align-items:center;padding:9px 16px;border-bottom:1px solid var(--border)}
.row:last-child{border-bottom:0}
.rank{font-family:var(--mono);font-size:12px;color:var(--muted);text-align:right}
.pname{grid-row:1;font-family:var(--mono);font-size:14px;color:var(--text);background:none;border:0;padding:0;
  text-align:left;cursor:pointer;text-transform:capitalize;text-decoration:underline;text-decoration-color:var(--border);
  text-underline-offset:2px;white-space:normal;overflow-wrap:anywhere}
.pname:hover{color:var(--accent);text-decoration-color:var(--accent)}
.val{grid-row:1;font-family:var(--mono);font-size:12px;color:var(--muted);text-align:right;white-space:nowrap}
.sub{grid-column:2 / -1;display:flex;flex-wrap:wrap;gap:7px;align-items:center;margin-top:2px}
.note{font-family:var(--mono);font-size:11.5px;padding:1px 7px;border-radius:999px;background:var(--surface-2);border:1px solid var(--border)}
.reg{font-size:11px;font-family:var(--mono);color:var(--lift)}
.reg.deepens{color:var(--muted)}
.cons{grid-column:2 / -1;display:flex;align-items:center;gap:6px;margin-top:5px;font-size:11px;color:var(--muted)}
.cons .bar{flex:1;max-width:120px;height:4px;border-radius:3px;background:var(--track);overflow:hidden}
.cons .bar i{display:block;height:100%}
.cat{font-size:11px;color:var(--muted);font-family:var(--mono)}
.foot{margin-top:26px;color:var(--muted);font-size:12.5px;max-width:82ch}
.foot code{font-family:var(--mono);background:var(--surface-2);padding:1px 5px;border-radius:5px}
</style>

<div class="wrap">
  <header class="masthead">
    <div class="logo"><b>lilac</b> · affinity</div>
    <div class="tagline">Every good pairing is two things at once: a <b>deepener</b> that shares a
      base's character, and a <b>lifter</b> that adds a new note it lacks. Learned from the canon.</div>
    <nav class="nav">
      <a href="lilac_compose.html">compose ↗</a><a href="lilac_pairings.html">pairings ↗</a>
      <a href="lilac_triangles.html">triangles ↗</a>
    </nav>
  </header>

  <div class="pickrow">
    <div class="combo">
      <input id="q" type="text" placeholder="Start from an ingredient…" autocomplete="off"
             role="combobox" aria-expanded="false" aria-controls="menu" aria-label="Base ingredient">
      <div class="menu" id="menu" role="listbox"></div>
    </div>
    <button class="btn" id="rand" type="button">🎲</button>
    <button class="btn" id="theme" type="button" aria-label="Toggle theme">◑</button>
  </div>

  <section class="dish" id="dish" aria-live="polite"></section>

  <div class="cols">
    <div class="card deepc"><h2><span class="dot"></span>Deepeners</h2>
      <div class="rule">Share the base's character — pair for <b>depth</b> (big shared note)</div>
      <div id="list-d"></div></div>
    <div class="card liftc"><h2><span class="dot"></span>Lifters</h2>
      <div class="rule">Add a distinctive, <b>consonant</b> note in a new register — pair for <b>contrast</b></div>
      <div id="list-l"></div></div>
  </div>

  <p class="foot">Reverse-engineered from timeless pairings: overlap alone doesn't predict them, but
    every one is an <b>anchor</b> (shared note) plus a <b>lift</b> (a note one adds). A lift is scored
    for <b>consonance</b> — does it belong with this base (garlic's sulfur completes beef, wrecks a
    custard) — and for crossing to a <b>new register</b> (↗) rather than just deepening. Click any
    partner to re-anchor. Aroma-only leads — taste decides the plate.</p>
</div>

<script>
const DATA = /*__DATA__*/;
const {names, cats, sensors, sensorFam, families, char, deep, lift, dish} = DATA;
const el = id => document.getElementById(id);
const cap = s => s.replace(/_/g," ");
const fcol = s => `var(--f${sensorFam[s]})`;

el("theme").onclick = () => {
  const r=document.documentElement, c=r.getAttribute("data-theme")||(matchMedia("(prefers-color-scheme:dark)").matches?"dark":"light");
  r.setAttribute("data-theme", c==="dark"?"light":"dark");
};

function deepRows(list){
  const max = Math.max(...list.map(p=>p[2]), .001);
  return list.map((p,k)=>{
    const [j,sn,st]=p;
    return `<div class="row">
      <span class="rank">${k+1}</span>
      <button class="pname" data-go="${j}">${cap(names[j])}</button>
      <span class="val">${st.toFixed(2)}</span>
      <span class="sub"><span class="cat">${cats[j]}</span>
        <span class="note" style="color:${fcol(sn)}">shares ${sensors[sn]}</span></span>
    </div>`;
  }).join("");
}
function liftRows(list){
  return list.map((p,k)=>{
    const [j,sn,lv,cons,crosses]=p;
    const reg = crosses ? `<span class="reg">↗ new register</span>` : `<span class="reg deepens">deepens</span>`;
    const cc = cons>=0.66? "var(--lift)" : cons>=0.4? "var(--f1)" : "var(--warn)";
    return `<div class="row">
      <span class="rank">${k+1}</span>
      <button class="pname" data-go="${j}">${cap(names[j])}</button>
      <span class="val">${lv.toFixed(2)}</span>
      <span class="sub"><span class="cat">${cats[j]}</span>
        <span class="note" style="color:${fcol(sn)}">adds ${sensors[sn]}</span>${reg}</span>
      <span class="cons">consonance <span class="bar"><i style="width:${Math.round(cons*100)}%;background:${cc}"></i></span> ${cons.toFixed(2)}</span>
    </div>`;
  }).join("");
}

function slot(kind,j,why){
  if(j<0) return `<div class="slot ${kind}"><span class="tag">${kind==='deep'?'deepener':'lifter'}</span><div class="why">—</div></div>`;
  return `<div class="slot ${kind}"><span class="tag">${kind==='deep'?'deepener':'lifter'}</span>
    <div class="nm"><button data-go="${j}">${cap(names[j])}</button></div>
    <div class="why">${why}</div></div>`;
}

function render(i){
  cur=i;
  const [dj,lj]=dish[i];
  const dWhy = dj>=0 ? `deepens <b>${cap(names[i])}</b> — shares <b>${sensors[deep[i][0]?deep[i][0][1]:0]}</b>` : "";
  // lifter reasoning from the lift list entry for lj
  let lWhy="";
  const le=(lift[i]||[]).find(p=>p[0]===lj);
  if(le) lWhy = `lifts it — adds <b>${sensors[le[1]]}</b>, consonance ${le[3].toFixed(2)}${le[4]?" ↗":""}`;
  el("dish").innerHTML = `
    <h3><span class="base">${cap(names[i])}</span> <span class="reg">· character: ${char[i]}</span></h3>
    <div class="reg">a dish wants one of each — depth + a new dimension</div>
    <div class="dishrow">${slot('deep',dj,dWhy)}${slot('lift',lj,lWhy)}</div>`;
  el("list-d").innerHTML = deepRows(deep[i]);
  el("list-l").innerHTML = liftRows(lift[i]);
  el("q").value = cap(names[i]);
  history.replaceState(null,"","#"+encodeURIComponent(names[i]));
  document.querySelectorAll("[data-go]").forEach(b=>b.onclick=()=>go(+b.dataset.go));
}
let cur=0;
function go(i){ render(i); window.scrollTo({top:0,behavior:"smooth"}); }

const menu=el("menu"), q=el("q"); let filtered=[], active=-1;
function openMenu(t){ t=t.toLowerCase().trim();
  filtered=names.map((n,i)=>[n,i]).filter(([n])=>n.includes(t)).slice(0,60);
  menu.innerHTML=filtered.map(([n,i],k)=>`<div class="opt${k===active?' active':''}" role="option" data-i="${i}">
    <span>${cap(n)}</span><span class="cat">${cats[i]}</span></div>`).join("")||`<div class="opt"><span>no match</span></div>`;
  menu.classList.add("open");q.setAttribute("aria-expanded","true");
  menu.querySelectorAll(".opt[data-i]").forEach(o=>o.onmousedown=e=>{e.preventDefault();go(+o.dataset.i);closeMenu();});
}
function closeMenu(){menu.classList.remove("open");q.setAttribute("aria-expanded","false");active=-1;}
q.addEventListener("input",()=>{active=-1;openMenu(q.value);});
q.addEventListener("focus",()=>openMenu(q.value));
q.addEventListener("blur",()=>setTimeout(closeMenu,120));
q.addEventListener("keydown",e=>{ if(!menu.classList.contains("open"))return;
  if(e.key==="ArrowDown"){active=Math.min(active+1,filtered.length-1);openMenu(q.value);e.preventDefault();}
  else if(e.key==="ArrowUp"){active=Math.max(active-1,0);openMenu(q.value);e.preventDefault();}
  else if(e.key==="Enter"&&active>=0){go(filtered[active][1]);closeMenu();e.preventDefault();}
  else if(e.key==="Escape"){closeMenu();}});
el("rand").onclick=()=>go(Math.floor(Math.random()*names.length));
function start(){ const h=decodeURIComponent(location.hash.slice(1));
  let i=names.indexOf(h); if(i<0)i=names.indexOf("beef"); if(i<0)i=0; render(i); }
start();
</script>
"""


if __name__ == "__main__":
    main()
