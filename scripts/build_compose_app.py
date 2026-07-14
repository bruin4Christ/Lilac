"""Generate the self-contained Lilac composition studio (single HTML file).

    python scripts/build_compose_app.py

For every ingredient, precomputes a composed *dish* -- a base plus a handful of
partners chosen to bridge and complement it (see `lilac.compose`) -- at two
"adventurousness" settings (Harmonious / Adventurous), and embeds it all in one
static page. Each pick shows its role, the distinctive sensor it bridges on, the
new notes it brings, any culinary-category leap, and a non-eliminating hedonic
warning. Writes outputs/lilac_compose.html. No external requests.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import numpy as np  # noqa: E402

from lilac.compose import compose  # noqa: E402
from lilac.data import load_flavor_network  # noqa: E402
from lilac.ingredients import build_ingredient_signatures, idf_weights  # noqa: E402
from lilac.sensors import (  # noqa: E402
    _COMPOSITION,
    _DESCRIPTOR,
    _LARGE_STRUCTURAL,
    _LARGE_TOPO,
    _STRUCTURAL,
    BIT_NAMES,
    N_BITS,
)

SIZE = 5  # ingredients per dish (base included)
_ROLE = {"base": 0, "reinforce": 1, "bridge": 2, "accent": 3}

# The two "adventurousness" presets the UI toggles between: (surprise, novelty, diversity).
PRESETS = {
    "h": dict(surprise_weight=0.25, novelty_weight=0.30, diversity_weight=0.50),
    "a": dict(surprise_weight=0.70, novelty_weight=0.60, diversity_weight=0.60),
}

# The two compound-weighting modes: uniform (every compound equal) vs specificity
# (up-weight distinctive compounds), which rescues ingredients whose character lives
# in trace notes -- e.g. coconut's lactones. Keys: "u" and "s".
WEIGHTINGS = {"u": "uniform", "s": "specificity"}


def _sensor_groups() -> list[int]:
    sizes = [len(_STRUCTURAL), len(_LARGE_STRUCTURAL), len(_DESCRIPTOR),
             len(_LARGE_TOPO), len(_COMPOSITION)]
    return [g for g, n in enumerate(sizes) for _ in range(n)]


def _pack_member(m, idx_of) -> list:
    """Member -> compact record: [idx, role, via_idx, adds[], surprise, challenge, warn, thread]."""
    via_idx = BIT_NAMES.index(m.via) if m.via else -1
    adds = [BIT_NAMES.index(a) for a in m.adds]
    return [idx_of[m.ingredient], _ROLE.get(m.role, 3), via_idx, adds,
            int(m.surprise), round(float(m.challenge), 2),
            m.warning or "", round(float(m.thread), 2)]


def build_data() -> dict:
    df = load_flavor_network(min_compounds=5)
    names = list(build_ingredient_signatures(df=df))   # ingredient set is weighting-independent
    idx_of = {n: i for i, n in enumerate(names)}
    cats = dict(zip(df["ingredient"], df["category"]))

    # dishes[weighting][preset] -> per-base {members, palette}
    dishes: dict[str, dict[str, list]] = {}
    ncomp = None
    for wkey, wname in WEIGHTINGS.items():
        sigs = build_ingredient_signatures(df=df, weighting=wname)
        idf = idf_weights(sigs)
        if ncomp is None:
            ncomp = [int(sigs[n].n_molecules) for n in names]
        dishes[wkey] = {}
        for pkey, params in PRESETS.items():
            row = []
            for n in names:
                comp = compose(n, sigs, idf, categories=cats, size=SIZE, **params)
                row.append({
                    "m": [_pack_member(m, idx_of) for m in comp.members],
                    "p": [round(float(v), 2) for v in comp.palette],
                })
            dishes[wkey][pkey] = row

    return {
        "names": names,
        "cats": [cats[n] for n in names],
        "ncomp": ncomp,
        "sensors": BIT_NAMES,
        "groups": _sensor_groups(),
        "dishes": dishes,
    }


def main() -> None:
    data = build_data()
    html = TEMPLATE.replace("/*__DATA__*/", json.dumps(data, separators=(",", ":")))
    html = html.replace("__NBITS__", str(N_BITS))
    out = Path("outputs/lilac_compose.html")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(html, encoding="utf-8")
    kb = out.stat().st_size / 1024
    print(f"Wrote {out}  ({len(data['names'])} ingredients, {kb:.0f} KB)")


TEMPLATE = r"""<style>
:root{
  --bg:#FAF9FE; --surface:#FFFFFF; --surface-2:#F4F1FC; --border:#E7E2F4;
  --text:#1B1726; --muted:#6C6482; --accent:#7C5CF0; --accent-soft:#EEE9FE;
  --base:#7C5CF0; --reinforce:#2E9E6B; --bridge:#C0851C; --accent3:#5670D6;
  --warn:#C2410C; --track:#EDE9F8;
  --g0:#7C5CF0; --g1:#1FA8A0; --g2:#5670D6; --g3:#C0851C; --g4:#C2568F;
  --mono:ui-monospace,"SF Mono",Menlo,Consolas,monospace;
  --sans:-apple-system,BlinkMacSystemFont,"Segoe UI",Roboto,system-ui,sans-serif;
}
@media (prefers-color-scheme:dark){:root{
  --bg:#121019; --surface:#1A1624; --surface-2:#221C30; --border:#2E2740;
  --text:#ECE7F7; --muted:#9A93AE; --accent:#A78BFF; --accent-soft:#241C3A;
  --base:#A78BFF; --reinforce:#4FBD86; --bridge:#E0A63C; --accent3:#8AA0FF;
  --warn:#F0956A; --track:#26203440;
  --g0:#A78BFF; --g1:#37C4BB; --g2:#8AA0FF; --g3:#E0A63C; --g4:#E58BB8;
}}
:root[data-theme="light"]{
  --bg:#FAF9FE; --surface:#FFFFFF; --surface-2:#F4F1FC; --border:#E7E2F4;
  --text:#1B1726; --muted:#6C6482; --accent:#7C5CF0; --accent-soft:#EEE9FE;
  --base:#7C5CF0; --reinforce:#2E9E6B; --bridge:#C0851C; --accent3:#5670D6;
  --warn:#C2410C; --track:#EDE9F8;
  --g0:#7C5CF0; --g1:#1FA8A0; --g2:#5670D6; --g3:#C0851C; --g4:#C2568F;
}
:root[data-theme="dark"]{
  --bg:#121019; --surface:#1A1624; --surface-2:#221C30; --border:#2E2740;
  --text:#ECE7F7; --muted:#9A93AE; --accent:#A78BFF; --accent-soft:#241C3A;
  --base:#A78BFF; --reinforce:#4FBD86; --bridge:#E0A63C; --accent3:#8AA0FF;
  --warn:#F0956A; --track:#26203440;
  --g0:#A78BFF; --g1:#37C4BB; --g2:#8AA0FF; --g3:#E0A63C; --g4:#E58BB8;
}
*{box-sizing:border-box}
body{background:var(--bg)}
.wrap{max-width:900px;margin:0 auto;padding:26px 20px 72px;color:var(--text);
  font-family:var(--sans);line-height:1.5}
.masthead{display:flex;flex-wrap:wrap;align-items:baseline;gap:12px;
  border-bottom:1px solid var(--border);padding-bottom:16px}
.logo{font-family:var(--mono);font-weight:700;font-size:23px;letter-spacing:-.02em;color:var(--accent)}
.logo b{color:var(--text)}
.tagline{color:var(--muted);font-size:14px;max-width:58ch}
.nav{margin-left:auto;display:flex;gap:14px;font-size:13px}
.nav a{color:var(--muted);text-decoration:none;border-bottom:1px solid transparent}
.nav a:hover{color:var(--accent);border-bottom-color:var(--accent)}
.pickrow{display:flex;flex-wrap:wrap;gap:10px;align-items:center;margin:20px 0 6px}
.combo{position:relative;flex:1;min-width:220px;max-width:380px}
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
.btn{font-family:var(--sans);font-size:14px;font-weight:600;color:var(--text);
  background:var(--surface);border:1px solid var(--border);border-radius:10px;padding:10px 14px;cursor:pointer}
.btn:hover{border-color:var(--accent);color:var(--accent)}
.btn:focus-visible{outline:2px solid var(--accent);outline-offset:1px}
.seg{display:inline-flex;border:1px solid var(--border);border-radius:10px;overflow:hidden}
.seg button{font-family:var(--sans);font-size:13px;font-weight:600;color:var(--muted);
  background:var(--surface);border:0;padding:10px 14px;cursor:pointer}
.seg button.on{background:var(--accent-soft);color:var(--accent)}

.dish{background:var(--surface-2);border:1px solid var(--border);border-radius:16px;
  padding:6px 6px 18px;margin:16px 0 20px}
.member{background:var(--surface);border:1px solid var(--border);border-radius:12px;
  margin:8px;padding:13px 15px;display:grid;grid-template-columns:auto 1fr auto;
  gap:4px 12px;align-items:start;position:relative;overflow:hidden}
.member.base{border-color:var(--base)}
.member::before{content:"";position:absolute;left:0;top:0;bottom:0;width:4px}
.member.base::before{background:var(--base)}
.member.reinforce::before{background:var(--reinforce)}
.member.bridge::before{background:var(--bridge)}
.member.accent::before{background:var(--accent3)}
.role{grid-row:1;font-family:var(--mono);font-size:10.5px;font-weight:700;letter-spacing:.06em;
  text-transform:uppercase;padding:3px 8px;border-radius:999px;align-self:center;white-space:nowrap}
.role.base{color:var(--base);background:var(--accent-soft)}
.role.reinforce{color:var(--reinforce);background:color-mix(in srgb,var(--reinforce) 14%,transparent)}
.role.bridge{color:var(--bridge);background:color-mix(in srgb,var(--bridge) 16%,transparent)}
.role.accent{color:var(--accent3);background:color-mix(in srgb,var(--accent3) 16%,transparent)}
.mname{grid-row:1;font-size:19px;font-weight:700;letter-spacing:-.01em;text-transform:capitalize;
  background:none;border:0;color:var(--text);cursor:pointer;text-align:left;padding:0}
.member.base .mname{cursor:default}
.mname:not(.nolink):hover{color:var(--accent)}
.mcat{grid-row:1;justify-self:end;font-family:var(--mono);font-size:12px;color:var(--muted);align-self:center}
.why{grid-column:2 / span 2;grid-row:2;font-size:13.5px;color:var(--muted);margin-top:2px}
.why .via{color:var(--text);font-weight:600}
.why .leap{color:var(--accent)}
.adds{grid-column:2 / span 2;grid-row:3;display:flex;flex-wrap:wrap;gap:5px;margin-top:6px}
.pill{font-family:var(--mono);font-size:11px;color:var(--muted);background:var(--surface-2);
  border:1px solid var(--border);border-radius:999px;padding:2px 8px}
.pill.g0{color:var(--g0)} .pill.g1{color:var(--g1)} .pill.g2{color:var(--g2)} .pill.g3{color:var(--g3)} .pill.g4{color:var(--g4)}
.warn{grid-column:1 / -1;grid-row:4;display:flex;gap:7px;align-items:baseline;margin-top:9px;
  font-size:12.5px;color:var(--warn);background:color-mix(in srgb,var(--warn) 10%,transparent);
  border-radius:8px;padding:6px 10px}
.thread{grid-column:1 / -1;grid-row:5;height:3px;border-radius:2px;background:var(--track);
  margin-top:10px;overflow:hidden}
.thread i{display:block;height:100%;border-radius:2px;background:var(--muted)}

.palette{margin:6px 8px 0;padding:14px 15px;background:var(--surface);
  border:1px solid var(--border);border-radius:12px}
.palette h3{margin:0 0 3px;font-size:13px}
.palette .sub{color:var(--muted);font-size:12px;margin-bottom:10px}
.strip{display:flex;gap:1px;height:26px;border-radius:5px;overflow:hidden}
.cell{flex:1}
.legend{display:flex;flex-wrap:wrap;gap:12px;margin-top:9px;font-size:11.5px;color:var(--muted)}
.legend span{display:inline-flex;align-items:center;gap:5px}
.sw{width:10px;height:10px;border-radius:2px;display:inline-block}
.foot{margin-top:26px;color:var(--muted);font-size:12.5px;max-width:80ch}
.foot code{font-family:var(--mono);background:var(--surface-2);padding:1px 5px;border-radius:5px}
@media (prefers-reduced-motion:no-preference){.dish{transition:opacity .18s ease}.fade{opacity:0}}
@media (max-width:560px){.member{grid-template-columns:1fr}.mcat{justify-self:start}}
</style>

<div class="wrap">
  <header class="masthead">
    <div class="logo"><b>lilac</b> · composition studio</div>
    <div class="tagline">Not "what pairs with X" but "what dish grows from X" — a base plus
      partners chosen to bridge and complement it across the sensor palette.</div>
    <nav class="nav">
      <a href="lilac_pairings.html">pairings ↗</a>
      <a href="lilac_molecules.html">molecules ↗</a>
    </nav>
  </header>

  <div class="pickrow">
    <div class="combo">
      <input id="q" type="text" placeholder="Start from an ingredient…" autocomplete="off"
             role="combobox" aria-expanded="false" aria-controls="menu" aria-label="Base ingredient">
      <div class="menu" id="menu" role="listbox"></div>
    </div>
    <div class="seg" role="tablist" aria-label="Adventurousness">
      <button id="mode-h" class="on" type="button">Harmonious</button>
      <button id="mode-a" type="button">Adventurous</button>
    </div>
    <div class="seg" role="tablist" aria-label="Compound weighting">
      <button id="wt-u" class="on" type="button" title="Every compound weighted equally">Balanced</button>
      <button id="wt-s" type="button" title="Up-weight distinctive compounds — rescues trace-character ingredients like coconut">Character</button>
    </div>
    <button class="btn" id="rand" type="button">🎲</button>
    <button class="btn" id="theme" type="button" aria-label="Toggle theme">◑</button>
  </div>

  <section class="dish" id="dish" aria-live="polite"></section>

  <p class="foot">A dish is grown greedily: each pick maximizes coherence with the palette
    so far, plus a bit of novelty and a bonus for a surprising cross-category bridge, minus a
    redundancy penalty so it spans the palette instead of repeating itself. <b>Harmonious</b>
    stays close; <b>Adventurous</b> reaches further. <b>Balanced</b> weights every compound
    equally; <b>Character</b> up-weights an ingredient's distinctive compounds (rescuing
    trace-character foods like coconut, whose creamy lactones are otherwise drowned out).
    The ⚠ flag is a caution, never a veto —
    bold notes belong in bold dishes. Aroma-only leads from ~595 ingredients; taste, texture
    and culture decide the plate. Click any partner to grow a new dish from it.</p>
</div>

<script>
const DATA = /*__DATA__*/;
const {names, cats, ncomp, sensors, groups, dishes} = DATA;
const el = id => document.getElementById(id);
const cap = s => s.replace(/_/g," ");
const ROLE = ["base","reinforce","bridge","accent"];
let preset = "h", weight = "u", cur = 0;

el("theme").onclick = () => {
  const root = document.documentElement;
  const c = root.getAttribute("data-theme")
    || (matchMedia("(prefers-color-scheme:dark)").matches ? "dark":"light");
  root.setAttribute("data-theme", c==="dark"?"light":"dark");
};

function memberHTML(rec){
  const [idx, role, via, adds, surprise, challenge, warn, thread] = rec;
  const rn = ROLE[role];
  const isBase = role===0;
  const cat = cats[idx];
  let why = "";
  if(!isBase){
    const viaTxt = via>=0 ? `bridges on <span class="via">${sensors[via]}</span>` : "";
    const addTxt = adds.length ? ` · adds ${adds.map(a=>sensors[a]).join(", ")}` : "";
    why = `<div class="why">${viaTxt}${addTxt}</div>`;
  } else {
    why = `<div class="why">the anchor — a ${cat} base</div>`;
  }
  const pills = (!isBase && adds.length)
    ? `<div class="adds">${adds.map(a=>`<span class="pill g${groups[a]}">${sensors[a]}</span>`).join("")}</div>`
    : "";
  const leap = (surprise && !isBase) ? ` <span class="leap">↗ category leap</span>` : "";
  const warnHTML = warn ? `<div class="warn"><span>⚠</span><span>${warn} — a bold note, not a veto</span></div>` : "";
  const nameCls = isBase ? "mname nolink" : "mname";
  const threadHTML = isBase ? "" : `<div class="thread" title="aroma thread to base ${thread}"><i style="width:${Math.round(thread*100)}%"></i></div>`;
  return `<div class="member ${rn}">
    <span class="role ${rn}">${rn}</span>
    <button class="${nameCls}" data-go="${idx}" ${isBase?'disabled':''}>${cap(names[idx])}${leap}</button>
    <span class="mcat">${cat}</span>
    ${why}${pills}${warnHTML}${threadHTML}
  </div>`;
}

const GNAME = ["structural","scaffold","physicochem","topology","composition"];
function paletteHTML(pal){
  const cells = pal.map((v,b)=>{
    const op = v<=0 ? 0.06 : 0.12 + 0.88*v;
    return `<span class="cell" title="${sensors[b]} ${v.toFixed(2)}"
      style="background:var(--g${groups[b]});opacity:${op.toFixed(2)}"></span>`;
  }).join("");
  const legend = GNAME.map((g,i)=>`<span><span class="sw" style="background:var(--g${i})"></span>${g}</span>`).join("");
  return `<div class="palette">
    <h3>Dish palette</h3>
    <div class="sub">the combined signature — each of __NBITS__ sensors shaded by the strongest coverage in the dish</div>
    <div class="strip">${cells}</div>
    <div class="legend">${legend}</div>
  </div>`;
}

function render(i){
  cur = i;
  const d = dishes[weight][preset][i];
  el("dish").innerHTML = d.m.map(memberHTML).join("") + paletteHTML(d.p);
  el("q").value = cap(names[i]);
  history.replaceState(null,"","#"+encodeURIComponent(names[i]));
  el("dish").querySelectorAll(".mname[data-go]:not([disabled])").forEach(b=>{
    b.onclick = () => go(+b.dataset.go);
  });
}
function go(i){
  el("dish").classList.add("fade");
  render(i);
  requestAnimationFrame(()=>el("dish").classList.remove("fade"));
  window.scrollTo({top:0,behavior:"smooth"});
}
function setPreset(m){
  preset = m;
  el("mode-h").classList.toggle("on", m==="h");
  el("mode-a").classList.toggle("on", m==="a");
  render(cur);
}
function setWeight(w){
  weight = w;
  el("wt-u").classList.toggle("on", w==="u");
  el("wt-s").classList.toggle("on", w==="s");
  render(cur);
}
el("mode-h").onclick = ()=>setPreset("h");
el("mode-a").onclick = ()=>setPreset("a");
el("wt-u").onclick = ()=>setWeight("u");
el("wt-s").onclick = ()=>setWeight("s");

// combobox
const menu = el("menu"), q = el("q");
let filtered = [], active = -1;
function openMenu(term){
  term = term.toLowerCase().trim();
  filtered = names.map((n,i)=>[n,i]).filter(([n])=>n.includes(term)).slice(0,60);
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
