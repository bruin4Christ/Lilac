"""Generate the Lilac molecule-signature widget (single self-contained HTML file).

    python scripts/build_molecule_widget.py

Pick a raw ingredient and inspect the molecules inside it: each molecule's 55-bit
sensor signature (as a colour-coded strip + hex), its SMILES, and the ingredient's
own superimposed signature (per sensor, the fraction of its molecules that fire it).
Writes outputs/lilac_molecules.html. No external requests.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import numpy as np  # noqa: E402

from lilac.data import load_flavor_network, load_odorant_library  # noqa: E402
from lilac.sensors import (  # noqa: E402
    _DESCRIPTOR,
    _LARGE_STRUCTURAL,
    _LARGE_TOPO,
    _STRUCTURAL,
    BIT_NAMES,
    encode,
)


def sensor_groups() -> list[int]:
    """Group id per bit: 0 structural, 1 large-scaffold, 2 physicochemical, 3 topology."""
    sizes = [len(_STRUCTURAL), len(_LARGE_STRUCTURAL), len(_DESCRIPTOR), len(_LARGE_TOPO)]
    groups = []
    for g, n in enumerate(sizes):
        groups += [g] * n
    return groups


def build_data() -> dict:
    lib = load_odorant_library()
    smi_to_name = dict(zip(lib["smiles"], lib["name"]))
    df = load_flavor_network(min_compounds=5)

    # Global de-duplicated molecule table.
    mol_index: dict[str, int] = {}
    mols: list[dict] = []
    for smi_list in df["smiles"]:
        for smi in smi_list:
            if smi in mol_index:
                continue
            code = encode(smi)
            if code is None:
                continue
            mol_index[smi] = len(mols)
            mols.append({
                "n": str(smi_to_name.get(smi, smi)),
                "s": smi,
                "b": [int(i) for i, v in enumerate(code) if v],   # set-bit indices
            })

    ingredients = []
    for _, row in df.iterrows():
        idxs = [mol_index[s] for s in row["smiles"] if s in mol_index]
        if not idxs:
            continue
        ingredients.append({
            "name": row["ingredient"],
            "cat": row["category"],
            "mols": idxs,
        })

    return {
        "sensors": BIT_NAMES,
        "groups": sensor_groups(),
        "mols": mols,
        "ings": ingredients,
    }


def main() -> None:
    data = build_data()
    html = TEMPLATE.replace("/*__DATA__*/", json.dumps(data, separators=(",", ":")))
    out = Path("outputs/lilac_molecules.html")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(html, encoding="utf-8")
    kb = out.stat().st_size / 1024
    print(f"Wrote {out}  ({len(data['ings'])} ingredients, "
          f"{len(data['mols'])} molecules, {kb:.0f} KB)")


TEMPLATE = r"""<style>
:root{
  --bg:#FAF9FE; --surface:#FFFFFF; --surface-2:#F4F1FC; --border:#E7E2F4;
  --text:#1B1726; --muted:#6C6482; --accent:#7C5CF0; --accent-soft:#EEE9FE;
  --track:#E9E4F5; --on:#1B1726;
  --g0:#7C5CF0; --g1:#1FA8A0; --g2:#5670D6; --g3:#C0851C;
  --mono:ui-monospace,"SF Mono",Menlo,Consolas,monospace;
  --sans:-apple-system,BlinkMacSystemFont,"Segoe UI",Roboto,system-ui,sans-serif;
}
@media (prefers-color-scheme:dark){:root{
  --bg:#121019; --surface:#1A1624; --surface-2:#221C30; --border:#2E2740;
  --text:#ECE7F7; --muted:#9A93AE; --accent:#A78BFF; --accent-soft:#241C3A;
  --track:#2A2338; --on:#ECE7F7;
  --g0:#A78BFF; --g1:#37C4BB; --g2:#8AA0FF; --g3:#E0A63C;
}}
:root[data-theme="light"]{
  --bg:#FAF9FE; --surface:#FFFFFF; --surface-2:#F4F1FC; --border:#E7E2F4;
  --text:#1B1726; --muted:#6C6482; --accent:#7C5CF0; --accent-soft:#EEE9FE;
  --track:#E9E4F5; --on:#1B1726;
  --g0:#7C5CF0; --g1:#1FA8A0; --g2:#5670D6; --g3:#C0851C;
}
:root[data-theme="dark"]{
  --bg:#121019; --surface:#1A1624; --surface-2:#221C30; --border:#2E2740;
  --text:#ECE7F7; --muted:#9A93AE; --accent:#A78BFF; --accent-soft:#241C3A;
  --track:#2A2338; --on:#ECE7F7;
  --g0:#A78BFF; --g1:#37C4BB; --g2:#8AA0FF; --g3:#E0A63C;
}
*{box-sizing:border-box}
body{background:var(--bg)}
.wrap{max-width:1080px;margin:0 auto;padding:26px 20px 72px;color:var(--text);
  font-family:var(--sans);line-height:1.5}
.masthead{display:flex;flex-wrap:wrap;align-items:baseline;gap:12px;
  border-bottom:1px solid var(--border);padding-bottom:16px}
.logo{font-family:var(--mono);font-weight:700;font-size:23px;letter-spacing:-.02em;color:var(--accent)}
.logo b{color:var(--text)}
.tagline{color:var(--muted);font-size:14px;max-width:56ch}
.pickrow{display:flex;flex-wrap:wrap;gap:10px;align-items:center;margin:20px 0 6px}
.combo{position:relative;flex:1;min-width:230px;max-width:400px}
.combo input{width:100%;font-family:var(--mono);font-size:15px;color:var(--text);
  background:var(--surface);border:1px solid var(--border);border-radius:10px;padding:11px 13px}
.combo input:focus{outline:2px solid var(--accent);outline-offset:1px;border-color:transparent}
.menu{position:absolute;z-index:20;top:calc(100% + 4px);left:0;right:0;max-height:280px;overflow:auto;
  background:var(--surface);border:1px solid var(--border);border-radius:10px;
  box-shadow:0 12px 34px -18px #0009;display:none}
.menu.open{display:block}
.opt{display:flex;justify-content:space-between;gap:10px;padding:8px 12px;cursor:pointer;
  font-family:var(--mono);font-size:14px}
.opt:hover,.opt.active{background:var(--accent-soft)}
.opt .cat{color:var(--muted);font-size:12px}
.btn{font-family:var(--sans);font-size:14px;font-weight:600;color:var(--text);background:var(--surface);
  border:1px solid var(--border);border-radius:10px;padding:10px 14px;cursor:pointer}
.btn:hover{border-color:var(--accent);color:var(--accent)}
.btn:focus-visible{outline:2px solid var(--accent);outline-offset:1px}

.ihead{display:flex;flex-wrap:wrap;gap:8px 22px;align-items:baseline;margin:16px 0 6px}
.ihead .name{font-size:26px;font-weight:700;letter-spacing:-.01em;text-transform:capitalize;text-wrap:balance}
.ihead .meta{font-family:var(--mono);font-size:13px;color:var(--muted)}
.ihead .meta b{color:var(--text)}

.panel{background:var(--surface-2);border:1px solid var(--border);border-radius:14px;padding:16px 18px;margin:12px 0 22px}
.panel h3{margin:0 0 4px;font-size:14px}
.panel .hint{color:var(--muted);font-size:12.5px;margin:0 0 12px;max-width:80ch}
.legend{display:flex;flex-wrap:wrap;gap:14px;margin-top:12px;font-family:var(--mono);font-size:12px;color:var(--muted)}
.legend span{display:inline-flex;align-items:center;gap:6px}
.legend i{width:11px;height:11px;border-radius:3px;display:inline-block}

/* signature strips */
.strip{display:flex;flex-wrap:wrap;gap:3px}
.cell{width:15px;height:15px;border-radius:3px;background:var(--track);border:1px solid transparent}
.strip.big .cell{width:17px;height:17px}
.cell.lit{border-color:transparent}

.chips{display:flex;flex-wrap:wrap;gap:6px;margin-top:12px}
.chip{font-family:var(--mono);font-size:12px;border:1px solid var(--border);border-radius:999px;
  padding:3px 9px;color:var(--text);background:var(--surface)}
.chip .pct{color:var(--muted);margin-left:5px}

.mol{background:var(--surface);border:1px solid var(--border);border-radius:12px;padding:12px 14px;margin-bottom:10px}
.mol .top{display:flex;flex-wrap:wrap;justify-content:space-between;gap:6px 16px;align-items:baseline;cursor:pointer}
.mol .mname{font-size:15px;font-weight:600;text-transform:capitalize}
.mol .smiles{font-family:var(--mono);font-size:12.5px;color:var(--muted);word-break:break-all}
.mol .hex{font-family:var(--mono);font-size:12.5px;color:var(--accent);font-variant-numeric:tabular-nums}
.mol .count{font-family:var(--mono);font-size:11.5px;color:var(--muted)}
.mol .strip{margin-top:9px}
.mol .names{display:none;margin-top:9px}
.mol.open .names{display:flex}
.mol .names .chip{cursor:default}
.listhead{display:flex;justify-content:space-between;align-items:baseline;margin:2px 0 10px}
.listhead h3{margin:0;font-size:15px}
.listhead .sort{font-family:var(--mono);font-size:12.5px;color:var(--muted)}
.foot{margin-top:24px;color:var(--muted);font-size:12.5px;max-width:80ch}
.foot code{font-family:var(--mono);background:var(--surface-2);padding:1px 5px;border-radius:5px}
@media (prefers-reduced-motion:no-preference){.panel,.mol{transition:opacity .16s ease}.fade{opacity:0}}
</style>

<div class="wrap">
  <header class="masthead">
    <div class="logo"><b>lilac</b> · molecule inspector</div>
    <div class="tagline">A raw ingredient is many molecules at once. Pick one to see every
      constituent's 55-bit sensor signature — and how they superimpose into the ingredient's
      own code.</div>
  </header>

  <div class="pickrow">
    <div class="combo">
      <input id="q" type="text" placeholder="Search an ingredient…" autocomplete="off"
             role="combobox" aria-expanded="false" aria-controls="menu" aria-label="Ingredient">
      <div class="menu" id="menu" role="listbox"></div>
    </div>
    <button class="btn" id="rand" type="button">🎲 Surprise me</button>
    <button class="btn" id="theme" type="button" aria-label="Toggle theme">◑ Theme</button>
  </div>

  <div class="ihead" id="ihead"></div>

  <section class="panel" id="superimposed" aria-live="polite">
    <h3>Superimposed signature</h3>
    <p class="hint">Each of the 55 sensors, shaded by the share of this ingredient's molecules
      that trip it. This is the ingredient "smell number" — many molecules stacked on one bitmask.</p>
    <div class="strip big" id="supstrip"></div>
    <div class="chips" id="supchips"></div>
    <div class="legend">
      <span><i style="background:var(--g0)"></i>structural “corner”</span>
      <span><i style="background:var(--g1)"></i>large scaffold</span>
      <span><i style="background:var(--g2)"></i>physicochemical</span>
      <span><i style="background:var(--g3)"></i>whole-molecule topology</span>
    </div>
  </section>

  <div class="listhead">
    <h3 id="listtitle">Molecules present</h3>
    <span class="sort">click a molecule to name its active sensors</span>
  </div>
  <div id="mollist"></div>

  <p class="foot">55-bit signatures from <code>lilac.sensors</code>; molecules per ingredient from
    the Ahn <em>et al.</em> Flavor Network (compounds resolved to structures by name, ~65%
    coverage). A molecule's cell is lit when its sensor fires; the superimposed strip shades each
    sensor by prevalence across the ingredient's molecules.</p>
</div>

<script>
const DATA = /*__DATA__*/;
const {sensors, groups, mols, ings} = DATA;
const NB = sensors.length;
const el = id => document.getElementById(id);
const cap = s => String(s).replace(/_/g," ");
const GVAR = ["--g0","--g1","--g2","--g3"];
const gcol = g => `var(${GVAR[g]})`;

function litSet(bits){const s=new Set(bits);return s;}
function hexOf(bits){let v=0n;for(const b of bits)v|=(1n<<BigInt(NB-1-b));
  return "0x"+v.toString(16).toUpperCase().padStart(Math.ceil(NB/4),"0");}

// a 55-cell strip; `fill` maps bit->intensity[0..1] (superimposed) or a Set (molecule)
function stripHTML(fill){
  let h="";
  for(let i=0;i<NB;i++){
    let style="", lit=false, title=sensors[i];
    if(fill instanceof Set){
      if(fill.has(i)){lit=true;style=`background:${gcol(groups[i])}`;}
    }else{
      const v=fill[i]||0;
      if(v>0){lit=true;title+=` — ${Math.round(v*100)}% of molecules`;
        style=`background:${gcol(groups[i])};opacity:${(0.18+0.82*v).toFixed(2)}`;}
    }
    h+=`<span class="cell${lit?' lit':''}" style="${style}" title="${title}"></span>`;
  }
  return h;
}

function render(i){
  const ing=ings[i];
  const list=ing.mols.map(m=>mols[m]);
  el("ihead").innerHTML=`<span class="name">${cap(ing.name)}</span>
    <span class="meta"><b>${ing.cat}</b> · <b>${list.length}</b> molecules · 55 sensors</span>`;

  // superimposed: fraction of molecules lighting each bit
  const frac=new Array(NB).fill(0);
  for(const mo of list) for(const b of mo.b) frac[b]++;
  for(let k=0;k<NB;k++) frac[k]/=list.length;
  el("supstrip").innerHTML=stripHTML(frac);
  const top=[...frac.keys()].filter(k=>frac[k]>0).sort((a,b)=>frac[b]-frac[a]).slice(0,10);
  el("supchips").innerHTML=top.map(k=>
    `<span class="chip" style="border-color:${gcol(groups[k])}">${sensors[k]}
      <span class="pct">${Math.round(frac[k]*100)}%</span></span>`).join("")
    || `<span class="chip">no sensors fire</span>`;

  // molecules, most-distinctive (fewest-but-present large bits) first? keep source order but
  // sort by active-sensor count desc so rich molecules lead.
  const sorted=[...list].sort((a,b)=>b.b.length-a.b.length);
  el("listtitle").textContent=`Molecules present · ${list.length}`;
  el("mollist").innerHTML=sorted.map((mo,idx)=>{
    const names=mo.b.map(b=>`<span class="chip" style="border-color:${gcol(groups[b])}">${sensors[b]}</span>`).join("");
    return `<div class="mol" data-k="${idx}">
      <div class="top">
        <span class="mname">${cap(mo.n)}</span>
        <span class="hex">${hexOf(mo.b)}</span>
      </div>
      <div class="smiles">${mo.s} <span class="count">· ${mo.b.length} sensors</span></div>
      <div class="strip">${stripHTML(litSet(mo.b))}</div>
      <div class="names">${names}</div>
    </div>`;
  }).join("");
  el("mollist").querySelectorAll(".mol .top").forEach(t=>{
    t.onclick=()=>t.parentElement.classList.toggle("open");
  });

  el("q").value=cap(ing.name);
  history.replaceState(null,"","#"+encodeURIComponent(ing.name));
}
function go(i){el("superimposed").classList.add("fade");render(i);
  requestAnimationFrame(()=>el("superimposed").classList.remove("fade"));
  window.scrollTo({top:0,behavior:"smooth"});}

// combobox
const menu=el("menu"),q=el("q");let filtered=[],active=-1;
function openMenu(term){term=term.toLowerCase().trim();
  filtered=ings.map((g,i)=>[g,i]).filter(([g])=>g.name.includes(term)).slice(0,60);
  menu.innerHTML=filtered.map(([g,i],k)=>`<div class="opt${k===active?' active':''}" role="option" data-i="${i}">
    <span>${cap(g.name)}</span><span class="cat">${g.cat}</span></div>`).join("")||`<div class="opt"><span>no match</span></div>`;
  menu.classList.add("open");q.setAttribute("aria-expanded","true");
  menu.querySelectorAll(".opt[data-i]").forEach(o=>{o.onmousedown=e=>{e.preventDefault();go(+o.dataset.i);closeMenu();};});}
function closeMenu(){menu.classList.remove("open");q.setAttribute("aria-expanded","false");active=-1;}
q.addEventListener("input",()=>{active=-1;openMenu(q.value);});
q.addEventListener("focus",()=>openMenu(q.value));
q.addEventListener("blur",()=>setTimeout(closeMenu,120));
q.addEventListener("keydown",e=>{if(!menu.classList.contains("open"))return;
  if(e.key==="ArrowDown"){active=Math.min(active+1,filtered.length-1);openMenu(q.value);e.preventDefault();}
  else if(e.key==="ArrowUp"){active=Math.max(active-1,0);openMenu(q.value);e.preventDefault();}
  else if(e.key==="Enter"&&active>=0){go(filtered[active][1]);closeMenu();e.preventDefault();}
  else if(e.key==="Escape"){closeMenu();}});
el("rand").onclick=()=>go(Math.floor(Math.random()*ings.length));
el("theme").onclick=()=>{const r=document.documentElement;
  const cur=r.getAttribute("data-theme")||(matchMedia("(prefers-color-scheme:dark)").matches?"dark":"light");
  r.setAttribute("data-theme",cur==="dark"?"light":"dark");};

function start(){const h=decodeURIComponent(location.hash.slice(1));
  let i=ings.findIndex(g=>g.name===h);
  if(i<0)i=ings.findIndex(g=>g.name==="coffee");
  if(i<0)i=0;render(i);}
start();
</script>
"""


if __name__ == "__main__":
    main()
