"""
Редактор отношений Ауры — страница в браузере.

    python tools\\people_ui.py          (или aura.bat → 7)
    Открыть: http://127.0.0.1:8766

Имя и прозвища, ползунки отношения и доверия (-10…10), её мнение о человеке, факты (можно удалить лишние).
Можно открывать при работающем боте.  Перед каждым сохранением — копия базы в backup_length.
"""
from __future__ import annotations

import json
import shutil
import sqlite3
import sys
import webbrowser
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List

from aiohttp import web

sys.path.insert(0, str(Path(__file__).resolve().parent))
from people_report import ROOT, _db_path  # noqa: E402

HOST = "127.0.0.1"
PORT = int(sys.argv[1]) if len(sys.argv) > 1 and sys.argv[1].isdigit() else 8766


def _connect() -> sqlite3.Connection:
    con = sqlite3.connect(_db_path(), timeout=15)
    con.row_factory = sqlite3.Row
    cols = {r["name"] for r in con.execute("PRAGMA table_info(people)")}
    if "name_locked" not in cols:
        con.execute("ALTER TABLE people ADD COLUMN name_locked INTEGER DEFAULT 0")
    if "real_name" not in cols:
        con.execute("ALTER TABLE people ADD COLUMN real_name TEXT DEFAULT ''")
    con.commit()
    return con


def _clean_aliases(raw: Any, name: str) -> List[str]:
    """Прозвища: без пустых, без повторов и без самого имени."""
    items = raw if isinstance(raw, list) else str(raw or "").split(",")
    out: List[str] = []
    seen = {name.strip().lower()}
    for a in items:
        a = str(a).strip().lstrip("@")[:40]
        if a and a.lower() not in seen:
            seen.add(a.lower())
            out.append(a)
    return out[:30]


def _backup() -> str:
    dst = ROOT / "backup_length" / f"events.db.{datetime.now():%Y%m%d-%H%M%S}"
    dst.parent.mkdir(exist_ok=True)
    shutil.copy2(_db_path(), dst)
    return dst.name


@web.middleware
async def same_origin(request: web.Request, handler):
    """Не даём чужим сайтам в браузере менять базу."""
    if request.method != "GET":
        origin = request.headers.get("Origin", "")
        if origin and origin not in (f"http://{HOST}:{PORT}", f"http://localhost:{PORT}"):
            return web.json_response({"error": "forbidden"}, status=403)
        if request.content_type != "application/json":
            return web.json_response({"error": "json only"}, status=415)
    if request.host.split(":")[0] not in (HOST, "localhost"):
        return web.json_response({"error": "forbidden"}, status=403)
    return await handler(request)


async def index(_: web.Request) -> web.Response:
    return web.Response(text=PAGE, content_type="text/html", charset="utf-8")


async def api_people(_: web.Request) -> web.Response:
    con = _connect()
    try:
        people: List[Dict[str, Any]] = []
        for r in con.execute("SELECT key, name, aliases, affinity, trust, opinion, interactions, last_seen,"
                             " COALESCE(name_locked, 0) AS name_locked, COALESCE(real_name, '') AS real_name"
                             " FROM people ORDER BY affinity DESC, interactions DESC"):
            facts = [dict(id=f["id"], fact=f["fact"], said_by=f["said_by"], gossip=bool(f["is_gossip"]))
                     for f in con.execute("SELECT id, fact, said_by, is_gossip FROM facts WHERE subject_key = ?"
                                          " ORDER BY last_confirmed DESC", (r["key"],))]
            people.append(dict(
                key=r["key"], name=r["name"], aliases=json.loads(r["aliases"] or "[]"),
                name_locked=bool(r["name_locked"]), real_name=r["real_name"],
                affinity=round(float(r["affinity"]) * 10), trust=round(float(r["trust"]) * 10),
                opinion=r["opinion"] or "", interactions=r["interactions"],
                last_seen=str(r["last_seen"])[:16].replace("T", " "), facts=facts))
        return web.json_response({"people": people, "db": _db_path().name})
    finally:
        con.close()


async def api_save(request: web.Request) -> web.Response:
    body = await request.json()
    changes = body.get("changes") or []
    drop_facts = [int(i) for i in body.get("delete_facts") or []]
    if not changes and not drop_facts:
        return web.json_response({"ok": True, "saved": 0})
    backup = _backup()
    con = _connect()
    try:
        now = datetime.now(timezone.utc).isoformat()
        for c in changes:
            aff = max(-1.0, min(1.0, float(c["affinity"]) / 10))
            trust = max(-1.0, min(1.0, float(c["trust"]) / 10))
            con.execute("UPDATE people SET affinity = ?, trust = ?, opinion = ?, updated_at = ? WHERE key = ?",
                        (aff, trust, str(c.get("opinion") or "")[:300], now, str(c["key"])))
            if "name" in c:
                row = con.execute("SELECT name FROM people WHERE key = ?", (str(c["key"]),)).fetchone()
                name = str(c.get("name") or "").strip()[:40] or (row["name"] if row else "")
                locked = 1 if row is not None and name != row["name"] else None
                aliases = _clean_aliases(c.get("aliases"), name)
                if locked and row["name"].lower() not in {a.lower() for a in aliases}:
                    aliases.append(row["name"])  # старый ник из Discord — тоже её прозвище
                con.execute("UPDATE people SET real_name = ? WHERE key = ?",
                            (str(c.get("real_name") or "").strip()[:40], str(c["key"])))
                con.execute(
                    "UPDATE people SET name = ?, aliases = ?,"
                    " name_locked = CASE WHEN ? IS NOT NULL THEN ? ELSE name_locked END WHERE key = ?",
                    (name, json.dumps(aliases, ensure_ascii=False), locked, locked, str(c["key"])),
                )
        for fid in drop_facts:
            con.execute("DELETE FROM facts WHERE id = ?", (fid,))
        con.commit()
    finally:
        con.close()
    return web.json_response({"ok": True, "saved": len(changes), "facts_deleted": len(drop_facts), "backup": backup})


PAGE = r"""<!doctype html><html lang="ru"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>Аура: отношения</title>
<style>
:root{--bg:#f6f5f8;--card:#fff;--text:#1d1b22;--muted:#6b6876;--line:#e3e0ea;--accent:#7a4fd6;--neg:#c2413a;--pos:#1f8a4c}
@media (prefers-color-scheme:dark){:root{--bg:#16151a;--card:#201e26;--text:#ecebf0;--muted:#9a97a6;--line:#33303c;--accent:#a986f0;--neg:#e0716a;--pos:#4cc77f}}
*{box-sizing:border-box}body{margin:0;background:var(--bg);color:var(--text);font:15px/1.45 system-ui,Segoe UI,sans-serif}
header{position:sticky;top:0;z-index:2;background:var(--bg);border-bottom:1px solid var(--line);padding:12px 16px;display:flex;gap:10px;align-items:center;flex-wrap:wrap}
h1{font-size:18px;margin:0 8px 0 0}
input[type=search]{padding:7px 10px;border:1px solid var(--line);border-radius:8px;background:var(--card);color:var(--text);min-width:200px}
button{padding:8px 14px;border-radius:8px;border:1px solid var(--line);background:var(--card);color:var(--text);cursor:pointer;font:inherit}
button.primary{background:var(--accent);border-color:var(--accent);color:#fff}button:disabled{opacity:.5;cursor:default}
main{max-width:980px;margin:0 auto;padding:16px}
.p{background:var(--card);border:1px solid var(--line);border-radius:12px;padding:12px 14px;margin-bottom:10px}
.p.changed{outline:2px solid var(--accent)}
.top{display:flex;gap:10px;align-items:baseline;flex-wrap:wrap}.name{font-weight:600;font-size:16px}
.muted{color:var(--muted);font-size:13px}
.sl{display:grid;grid-template-columns:90px 1fr 120px;gap:10px;align-items:center;margin-top:8px}
.sl input{width:100%;accent-color:var(--accent)}.val{font-variant-numeric:tabular-nums;font-size:14px}
.neg{color:var(--neg)}.pos{color:var(--pos)}
textarea{width:100%;margin-top:8px;padding:7px 9px;border:1px solid var(--line);border-radius:8px;background:var(--bg);color:var(--text);font:inherit;resize:vertical;min-height:38px}
details{margin-top:8px}summary{cursor:pointer;color:var(--muted);font-size:14px}
.f{display:flex;gap:8px;align-items:flex-start;padding:4px 0;border-top:1px solid var(--line);font-size:14px}
.nm{display:grid;grid-template-columns:90px 1fr;gap:6px 10px;align-items:center;margin-top:8px}
.nm input{padding:6px 9px;border:1px solid var(--line);border-radius:8px;background:var(--bg);color:var(--text);font:inherit;width:100%}
.f span{flex:1}.f.del span{text-decoration:line-through;opacity:.5}.f button{padding:2px 8px;font-size:12px}
#toast{position:fixed;bottom:16px;left:50%;transform:translateX(-50%);background:var(--text);color:var(--bg);padding:10px 16px;border-radius:10px;display:none}
@media (max-width:560px){.sl{grid-template-columns:70px 1fr}.sl .val{grid-column:2}}
</style></head><body>
<header><h1>Отношения Ауры</h1><input type="search" id="q" placeholder="поиск по имени">
<span class="muted" id="info"></span><span style="flex:1"></span>
<button id="reload">Обновить</button><button class="primary" id="save" disabled>Сохранить</button></header>
<main id="list"></main><div id="toast"></div>
<script>
function label(v){return v<=-6?"терпеть не может":v<=-2.5?"неприятен":v<1.5?"нейтрально":v<4.5?"скорее симпатичен":v<7.5?"тепло относится":"очень дорог"}
let people=[],orig={},dropFacts=new Set();
const $=s=>document.querySelector(s);
function el(tag,cls,text){const e=document.createElement(tag);if(cls)e.className=cls;if(text!=null)e.textContent=text;return e}
function toast(t){const e=$("#toast");e.textContent=t;e.style.display="block";setTimeout(()=>e.style.display="none",2500)}
function aliasText(p){return p.aliases.join(", ")}
function dirty(p){const o=orig[p.key];return o.affinity!==p.affinity||o.trust!==p.trust||o.opinion!==p.opinion||o.name!==p.name||o.aliases!==p.aliasStr||o.real!==p.real_name}
function refreshSave(){const n=people.filter(dirty).length+dropFacts.size;$("#save").disabled=!n;$("#save").textContent=n?`Сохранить (${n})`:"Сохранить"}
function slider(p,field,title){
  const row=el("div","sl");row.append(el("span","muted",title));
  const inp=document.createElement("input");inp.type="range";inp.min=-10;inp.max=10;inp.step=1;inp.value=p[field];
  const val=el("span","val");
  const show=()=>{const v=p[field];val.textContent=(v>0?"+":"")+v+(field==="affinity"?" · "+label(v):"");val.className="val "+(v<0?"neg":v>0?"pos":"")};
  inp.oninput=()=>{p[field]=+inp.value;show();card(p).classList.toggle("changed",dirty(p));refreshSave()};
  show();row.append(inp,val);return row}
function card(p){return document.getElementById("p_"+p.key)}
function render(){
  const q=$("#q").value.trim().toLowerCase(),list=$("#list");list.innerHTML="";
  for(const p of people){
    if(q&&![p.name,p.real_name,...p.aliases].some(n=>n.toLowerCase().includes(q)))continue;
    const c=el("div","p"+(dirty(p)?" changed":""));c.id="p_"+p.key;
    const top=el("div","top");top.append(el("span","name",p.name));
    if(p.aliases.length)top.append(el("span","muted","ещё: "+p.aliases.join(", ")));
    top.append(el("span","muted",`сообщений ${p.interactions} · был(а) ${p.last_seen}`));c.append(top);
    const nm=el("div","nm");
    const ni=document.createElement("input");ni.value=p.name;ni.maxLength=40;
    ni.title="Как она его(её) называет. Поправленное вручную имя не перезаписывается ником из Discord.";
    const ai=document.createElement("input");ai.value=p.aliasStr;ai.placeholder="через запятую: Паша, Пашка, флэймп";
    ai.title="Другие имена и прозвища — по ним она узнаёт человека, когда о нём говорят или просят написать ему.";
    const ri=document.createElement("input");ri.value=p.real_name;ri.maxLength=40;ri.placeholder="как зовут на самом деле, например Павел";
    ri.title="Настоящее имя: она знает его и узнаёт человека по нему.";
    const upd=()=>{p.name=ni.value.trim()||orig[p.key].name;p.aliasStr=ai.value;p.real_name=ri.value.trim();c.classList.toggle("changed",dirty(p));refreshSave()};
    ni.oninput=upd;ai.oninput=upd;ri.oninput=upd;
    nm.append(el("span","muted","имя"+(p.name_locked?" 🔒":"")),ni,el("span","muted","настоящее имя"),ri,el("span","muted","прозвища"),ai);c.append(nm);
    c.append(slider(p,"affinity","отношение"),slider(p,"trust","доверие"));
    const ta=el("textarea");ta.placeholder="её мнение о человеке (одной фразой, от первого лица)";ta.value=p.opinion;ta.maxLength=300;
    ta.oninput=()=>{p.opinion=ta.value;c.classList.toggle("changed",dirty(p));refreshSave()};c.append(ta);
    if(p.facts.length){const d=el("details");d.append(el("summary",null,`что она знает (${p.facts.length})`));
      for(const f of p.facts){const r=el("div","f"+(dropFacts.has(f.id)?" del":""));
        r.append(el("span",null,f.fact+(f.gossip?` (со слов ${f.said_by})`:"")));
        const b=el("button",null,dropFacts.has(f.id)?"вернуть":"удалить");
        b.onclick=()=>{dropFacts.has(f.id)?dropFacts.delete(f.id):dropFacts.add(f.id);r.classList.toggle("del");b.textContent=dropFacts.has(f.id)?"вернуть":"удалить";refreshSave()};
        r.append(b);d.append(r)}
      c.append(d)}
    list.append(c)}
}
async function load(){
  const r=await fetch("/api/people");const d=await r.json();people=d.people;orig={};dropFacts.clear();
  for(const p of people){p.aliasStr=aliasText(p);orig[p.key]={affinity:p.affinity,trust:p.trust,opinion:p.opinion,name:p.name,aliases:p.aliasStr,real:p.real_name}}
  $("#info").textContent=`${people.length} чел. · база ${d.db}`;render();refreshSave()}
$("#q").oninput=render;
$("#reload").onclick=()=>{if(!$("#save").disabled&&!confirm("Есть несохранённые изменения. Обновить и потерять их?"))return;load()};
$("#save").onclick=async()=>{
  const changes=people.filter(dirty).map(p=>({key:p.key,affinity:p.affinity,trust:p.trust,opinion:p.opinion,name:p.name,aliases:p.aliasStr.split(","),real_name:p.real_name}));
  const r=await fetch("/api/save",{method:"POST",headers:{"Content-Type":"application/json"},body:JSON.stringify({changes,delete_facts:[...dropFacts]})});
  const d=await r.json();if(!r.ok){toast("Ошибка: "+(d.error||r.status));return}
  toast(`Сохранено. Копия базы: ${d.backup}`);load()};
load();
</script></body></html>"""


def make_app() -> web.Application:
    app = web.Application(middlewares=[same_origin])
    app.add_routes([web.get("/", index), web.get("/api/people", api_people), web.post("/api/save", api_save)])
    return app


def main() -> None:
    app = make_app()
    url = f"http://{HOST}:{PORT}"
    print(f"Отношения Ауры: {url}  (закройте окно, чтобы остановить)")
    try:
        webbrowser.open(url)
    except Exception:  # noqa: BLE001
        pass
    web.run_app(app, host=HOST, port=PORT, print=None)


if __name__ == "__main__":
    main()
