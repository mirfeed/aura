"""
settings_ui.py — локальная страница настроек Ауры.

Запуск:  python tools/settings_ui.py   (или aura.bat → 6)
Открыть: http://127.0.0.1:8765

Список настроек, их разделы, подсказки и значения по умолчанию берутся
прямо из config.py, текущие значения — из .env.  Сохранение правит только
изменённые строки .env (комментарии и порядок остаются), перед записью
делается копия .env.bak.  Бот подхватывает .env после перезапуска;
persona.md — сразу, со следующего ответа.
"""

from __future__ import annotations

import json
import re
import shutil
import sys
import webbrowser
from pathlib import Path
from typing import Any, Dict, List, Optional

import aiohttp
from aiohttp import web

ROOT = Path(__file__).resolve().parent.parent
CONFIG_PY = ROOT / "config.py"
ENV_FILE = ROOT / ".env"
PERSONA_FILE = ROOT / "persona.md"

HOST = "127.0.0.1"
PORT = int(sys.argv[1]) if len(sys.argv) > 1 and sys.argv[1].isdigit() else 8765

SECRET_KEYS = {"DISCORD_TOKEN"}
_KEY_RE = re.compile(r"^\s*(?:export\s+)?([A-Za-z_][A-Za-z0-9_]*)\s*=(.*)$")
_GET_RE = re.compile(
    r'(?:_get_(str|int|float|bool|id_list|str_list)|os\.getenv)\(\s*"([A-Z0-9_]+)"'
    r'(?:\s*,\s*("(?:[^"\\]|\\.)*"|[-\d.]+|True|False))?'
)
# Разделы страницы (по началу имени ключа); всё остальное — по разделам из config.py.
GROUPS = [
    ("Discord и доступ", ("DISCORD_", "GUILD_IDS", "CHANNEL_WHITELIST", "OWNER_")),
    ("Характер и поведение в чате", ("BOT_NAME", "PERSONALITY_SEED", "PERSONA_FILE", "BASE_RESPONSE_PROBABILITY",
                                     "REACTION_PROBABILITY", "TYPING_", "BOT_TIMEZONE", "CHAT_HISTORY_LIMIT", "CHAT_SPLIT_", "CHAT_DROP_", "CHAT_TYPING_", "BAIT_", "DISLIKED_")),
    ("Модель (Ollama)", ("OLLAMA_", "MAX_RESPONSE_TOKENS")),
    ("Картинки", ("VISION_",)),
    ("Люди и отношения", ("SPECIAL_", "GRATITUDE_", "AFFINITY_", "RELATIONSHIP_", "SOCIAL_", "PROACTIVE_")),
    ("Память", ("MEMORY_", "CHROMA_", "SQLITE_", "EMBEDDING_", "REFLECT_")),
    ("Голос: поведение", ("VOICE_", "MODELS_DIR")),
    ("Распознавание речи (Whisper)", ("VAD_", "WHISPER_")),
    ("Озвучка (Silero / F5)", ("TTS_", "F5_", "PRONUNCIATIONS_")),
    ("Логи", ("LOG_",)),
]


def group_of(key: str, fallback: str) -> str:
    for title, prefixes in GROUPS:
        if key.startswith(prefixes):
            return title
    return fallback


_SEP_RE = re.compile(r"^#\s*-{5,}\s*$")
_INLINE_SEC_RE = re.compile(r"^#\s*---\s*(.+?)\s*---\s*$")


# ---------------------------------------------------------------------------
# config.py → описание полей
# ---------------------------------------------------------------------------

def parse_config() -> List[Dict[str, Any]]:
    """Найти все настройки в config.py: ключ, тип, умолчание, раздел, подсказку."""
    lines = CONFIG_PY.read_text(encoding="utf-8").splitlines()
    fields: List[Dict[str, Any]] = []
    seen = set()
    section = "Общее"
    for i, line in enumerate(lines):
        st = line.strip()
        if _SEP_RE.match(st):
            if i + 2 < len(lines) and _SEP_RE.match(lines[i + 2].strip()):
                title = lines[i + 1].strip().lstrip("#").strip()
                if title:
                    section = title
            continue
        m = _INLINE_SEC_RE.match(st)
        if m:
            section = m.group(1)
            continue
        if line.lstrip().startswith("#"):
            continue
        # вызов может растянуться на несколько строк
        joined = " ".join([line] + [x.strip() for x in lines[i + 1:i + 4]])
        for gm in _GET_RE.finditer(joined):
            if gm.start() >= len(line):
                break
            kind, key, default = gm.group(1) or "str", gm.group(2), gm.group(3)
            if key in seen:
                continue
            seen.add(key)
            fields.append({
                "key": key,
                "type": {"id_list": "list", "str_list": "list"}.get(kind, kind),
                "default": _literal(default),
                "section": group_of(key, section),
                "hint": _hint(lines, i),
                "secret": key in SECRET_KEYS,
            })
    return fields


def _literal(raw: Optional[str]) -> str:
    if raw is None:
        return ""
    if raw.startswith('"'):
        try:
            return json.loads(raw)
        except ValueError:
            return raw.strip('"')
    return {"True": "true", "False": "false"}.get(raw, raw)


def _hint(lines: List[str], i: int) -> str:
    """Комментарии прямо над строкой + комментарий в конце строки."""
    out: List[str] = []
    j = i - 1
    while j >= 0:
        st = lines[j].strip()
        if not st.startswith("#") or _SEP_RE.match(st) or _INLINE_SEC_RE.match(st) or j < i - 6:
            break
        out.insert(0, st.lstrip("#").strip())
        j -= 1
    m = re.search(r"\)\s*#\s*(.+)$", lines[i])
    if m:
        out.append(m.group(1).strip())
    return " ".join(x for x in out if x)


# ---------------------------------------------------------------------------
# .env
# ---------------------------------------------------------------------------

def _read_env_text() -> str:
    return ENV_FILE.read_text(encoding="utf-8-sig") if ENV_FILE.exists() else ""


def _unquote(v: str) -> str:
    v = v.strip()
    if len(v) >= 2 and v[0] == v[-1] and v[0] in "\"'":
        inner = v[1:-1]
        return inner.replace('\\"', '"').replace("\\\\", "\\") if v[0] == '"' else inner
    # как python-dotenv: « #» в значении без кавычек — начало комментария
    return re.split(r"\s+#", v, maxsplit=1)[0].strip()


def _quote(v: str) -> str:
    v = v.replace("\r", " ").replace("\n", " ").strip()
    if v and (re.search(r"\s#|^#|^[\"']|\\", v) or v != v.strip()):
        return '"' + v.replace("\\", "\\\\").replace('"', '\\"') + '"'
    return v


def read_env() -> Dict[str, str]:
    values: Dict[str, str] = {}
    for line in _read_env_text().splitlines():
        if line.lstrip().startswith("#"):
            continue
        m = _KEY_RE.match(line)
        if m:
            values[m.group(1)] = _unquote(m.group(2))
    return values


def write_env(changes: Dict[str, Optional[str]]) -> None:
    """Обновить ключи в .env.  None — удалить строку (вернуть значение по умолчанию)."""
    text = _read_env_text()
    nl = "\r\n" if "\r\n" in text else "\n"
    lines = text.splitlines()
    done = set()
    out: List[str] = []
    for line in lines:
        m = None if line.lstrip().startswith("#") else _KEY_RE.match(line)
        if m and m.group(1) in changes:
            key = m.group(1)
            if key in done or changes[key] is None:
                continue  # дубликат или сброс
            out.append(f"{key}={_quote(changes[key])}")
            done.add(key)
        else:
            out.append(line)
    new = [k for k, v in changes.items() if k not in done and v is not None]
    if new:
        if out and out[-1].strip():
            out.append("")
        out.append("# --- изменено через settings_ui ---")
        out.extend(f"{k}={_quote(changes[k])}" for k in new)
    if ENV_FILE.exists():
        shutil.copy2(ENV_FILE, ENV_FILE.with_name(".env.bak"))
    ENV_FILE.write_text(nl.join(out) + nl, encoding="utf-8", newline="")


# ---------------------------------------------------------------------------
# Ollama: список моделей и умеет ли модель картинки
# ---------------------------------------------------------------------------

async def ollama_info(base: str, model: str) -> Dict[str, Any]:
    info: Dict[str, Any] = {"models": [], "vision": None, "error": ""}
    tmo = aiohttp.ClientTimeout(total=5)
    try:
        async with aiohttp.ClientSession(timeout=tmo) as s:
            async with s.get(f"{base}/api/tags") as r:
                info["models"] = sorted(m["name"] for m in (await r.json()).get("models", []))
            if model:
                async with s.post(f"{base}/api/show", json={"model": model}) as r:
                    data = await r.json()
                caps = data.get("capabilities")
                if caps is not None:
                    info["vision"] = "vision" in caps
                elif "projector_info" in data:
                    info["vision"] = True
    except Exception as exc:  # noqa: BLE001
        info["error"] = f"Ollama недоступна ({base}): {exc.__class__.__name__}"
    return info


# ---------------------------------------------------------------------------
# HTTP
# ---------------------------------------------------------------------------

@web.middleware
async def same_origin(request: web.Request, handler):
    """Не даём чужим сайтам в браузере менять настройки."""
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


async def api_state(_: web.Request) -> web.Response:
    fields = parse_config()
    env = read_env()
    known = {f["key"] for f in fields}
    for k in env:
        if k not in known:
            fields.append({"key": k, "type": "str", "default": "", "section": "Прочее (есть только в .env)",
                           "hint": "", "secret": k in SECRET_KEYS})
    order = {t: n for n, (t, _) in enumerate(GROUPS)}
    fields.sort(key=lambda f: order.get(f["section"], len(GROUPS)))
    for f in fields:
        f["set"] = f["key"] in env
        f["value"] = env.get(f["key"], f["default"])
        if f["secret"]:
            f["value"] = "•" * 8 if f["value"] else ""
    persona = PERSONA_FILE.read_text(encoding="utf-8") if PERSONA_FILE.exists() else ""
    return web.json_response({"fields": fields, "persona": persona})


async def api_ollama(_: web.Request) -> web.Response:
    env = read_env()
    fields = {f["key"]: f["default"] for f in parse_config()}
    base = (env.get("OLLAMA_BASE_URL") or fields.get("OLLAMA_BASE_URL") or "http://localhost:11434").rstrip("/")
    model = env.get("VISION_MODEL") or env.get("OLLAMA_MODEL") or fields.get("OLLAMA_MODEL", "")
    info = await ollama_info(base, model)
    info["model"] = model
    return web.json_response(info)


async def api_save(request: web.Request) -> web.Response:
    body = await request.json()
    changes = body.get("changes") or {}
    clean: Dict[str, Optional[str]] = {}
    for k, v in changes.items():
        if not re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*", k):
            return web.json_response({"error": f"bad key {k}"}, status=400)
        clean[k] = None if v is None else str(v)
    if clean:
        write_env(clean)
    if "persona" in body and body["persona"] is not None:
        if PERSONA_FILE.exists():
            shutil.copy2(PERSONA_FILE, PERSONA_FILE.with_name("persona.md.bak"))
        PERSONA_FILE.write_text(body["persona"], encoding="utf-8")
    return web.json_response({"ok": True, "saved": len(clean)})


PAGE = r"""<!doctype html>
<html lang="ru"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>Настройки Ауры</title>
<style>
:root{--bg:#f6f5f8;--card:#fff;--text:#1d1b22;--muted:#6b6876;--line:#e3e0ea;--accent:#7a4fd6;--ok:#1f8a4c;--warn:#b5651d}
@media (prefers-color-scheme:dark){:root{--bg:#16151a;--card:#201e26;--text:#ecebf0;--muted:#9a97a6;--line:#33303c;--accent:#a986f0;--ok:#4cc77f;--warn:#e39a52}}
*{box-sizing:border-box}body{margin:0;background:var(--bg);color:var(--text);font:15px/1.45 system-ui,Segoe UI,sans-serif}
header{position:sticky;top:0;z-index:2;background:var(--bg);border-bottom:1px solid var(--line);padding:12px 16px;display:flex;gap:10px;align-items:center;flex-wrap:wrap}
h1{font-size:18px;margin:0 auto 0 0}
input[type=search]{padding:7px 10px;border:1px solid var(--line);border-radius:8px;background:var(--card);color:var(--text);min-width:200px}
button{padding:8px 14px;border-radius:8px;border:1px solid var(--line);background:var(--card);color:var(--text);cursor:pointer;font:inherit}
button.primary{background:var(--accent);border-color:var(--accent);color:#fff}button:disabled{opacity:.5;cursor:default}
main{max-width:960px;margin:0 auto;padding:16px}
.note{color:var(--muted);font-size:13px;margin:0 0 12px}
#ollama{padding:10px 14px;border-radius:10px;background:var(--card);border:1px solid var(--line);margin-bottom:14px;font-size:14px}
.ok{color:var(--ok)}.warn{color:var(--warn)}
details{background:var(--card);border:1px solid var(--line);border-radius:12px;margin-bottom:10px}
summary{padding:12px 16px;cursor:pointer;font-weight:600}summary small{color:var(--muted);font-weight:400}
.row{display:grid;grid-template-columns:minmax(180px,260px) 1fr;gap:4px 16px;padding:10px 16px;border-top:1px solid var(--line)}
.row.changed{background:color-mix(in srgb,var(--accent) 8%,transparent)}
.key{font-family:ui-monospace,Consolas,monospace;font-size:13px;word-break:break-all}
.key .def{display:block;color:var(--muted);font-family:system-ui;font-size:12px}
.hint{grid-column:2;color:var(--muted);font-size:13px}
.ctl{display:flex;gap:6px;align-items:center}
.ctl input[type=text],.ctl input[type=password],.ctl input[type=number]{flex:1;width:100%;padding:6px 9px;border:1px solid var(--line);border-radius:7px;background:var(--bg);color:var(--text);font:inherit}
.ctl button{padding:4px 8px;font-size:12px}
textarea{width:100%;min-height:380px;padding:12px;border:0;border-top:1px solid var(--line);background:var(--card);color:var(--text);font:13px/1.5 ui-monospace,Consolas,monospace;resize:vertical;border-radius:0 0 12px 12px}
#toast{position:fixed;bottom:16px;left:50%;transform:translateX(-50%);background:var(--text);color:var(--bg);padding:10px 16px;border-radius:10px;display:none}
@media (max-width:640px){.row{grid-template-columns:1fr}.hint{grid-column:1}}
</style></head><body>
<header><h1>Настройки Ауры</h1>
<input type="search" id="q" placeholder="Поиск настройки…">
<button id="save" class="primary" disabled>Сохранить</button></header>
<main>
<p class="note">Значения пишутся в .env (копия — .env.bak). Чтобы бот их подхватил, перезапустите его. Характер (persona.md) применяется сразу.</p>
<div id="ollama">Проверяю Ollama…</div>
<div id="list"></div>
<details id="persona-box"><summary>Характер (persona.md)</summary><textarea id="persona" spellcheck="false"></textarea></details>
</main><div id="toast"></div>
<script>
const $=s=>document.querySelector(s);let fields=[],changes={},personaOrig="";
const esc=s=>String(s).replace(/[&<>"]/g,c=>({"&":"&amp;","<":"&lt;",">":"&gt;",'"':"&quot;"}[c]));
function toast(t){const e=$("#toast");e.textContent=t;e.style.display="block";setTimeout(()=>e.style.display="none",2500)}
function dirty(){$("#save").disabled=!Object.keys(changes).length&&$("#persona").value===personaOrig}
function control(f){
  if(f.type==="bool"){const on=["1","true","yes","on","да"].includes(String(f.value).toLowerCase());
    return `<input type="checkbox" data-k="${f.key}" ${on?"checked":""}>`}
  const t=f.secret?"password":(f.type==="int"||f.type==="float")?"number":"text";
  const step=f.type==="float"?' step="any"':"";
  const dl=(f.key.endsWith("_MODEL")&&f.key.includes("OLLAMA")||f.key==="VISION_MODEL")?' list="models"':"";
  return `<input type="${t}"${step}${dl} data-k="${f.key}" value="${esc(f.value)}" ${f.secret?'placeholder="не задан"':""}>`+
    (f.secret?`<button data-show="${f.key}" title="Ввести новое значение">изменить</button>`:"")+
    (f.set?`<button data-reset="${f.key}" title="Убрать из .env, будет значение по умолчанию">сброс</button>`:"")}
function render(){
  const q=$("#q").value.trim().toLowerCase(),groups={};
  for(const f of fields){if(q&&!(f.key+" "+f.hint).toLowerCase().includes(q))continue;(groups[f.section]??=[]).push(f)}
  $("#list").innerHTML=Object.entries(groups).map(([s,fs])=>`<details ${q?"open":""}><summary>${esc(s)} <small>· ${fs.length}</small></summary>`+
    fs.map(f=>`<div class="row ${f.key in changes?"changed":""}"><div class="key">${f.key}<span class="def">${f.default!==""?"по умолчанию: "+esc(f.default):""}${f.set?"":" (не задано в .env)"}</span></div>
    <div class="ctl">${control(f)}</div>${f.hint?`<div class="hint">${esc(f.hint)}</div>`:""}</div>`).join("")+`</details>`).join("");
}
$("#list").addEventListener("input",e=>{const k=e.target.dataset.k;if(!k)return;const f=fields.find(x=>x.key===k);
  const v=e.target.type==="checkbox"?(e.target.checked?"true":"false"):e.target.value;
  if(f.secret&&v.startsWith("•"))return;changes[k]=v;e.target.closest(".row").classList.add("changed");dirty()});
$("#list").addEventListener("click",e=>{const r=e.target.dataset.reset,s=e.target.dataset.show;
  if(r){changes[r]=null;const f=fields.find(x=>x.key===r);f.value=f.default;f.set=false;render();dirty()}
  if(s){const i=document.querySelector(`input[data-k="${s}"]`);i.type="text";i.value="";i.focus()}});
$("#q").addEventListener("input",render);$("#persona").addEventListener("input",dirty);
$("#save").onclick=async()=>{const body={changes};if($("#persona").value!==personaOrig)body.persona=$("#persona").value;
  const r=await fetch("/api/save",{method:"POST",headers:{"Content-Type":"application/json"},body:JSON.stringify(body)});
  if(r.ok){toast("Сохранено. Перезапустите бота, чтобы применить .env");changes={};await load();checkOllama()}else toast("Ошибка: "+r.status)};
async function load(){const d=await (await fetch("/api/state")).json();fields=d.fields;personaOrig=d.persona;$("#persona").value=d.persona;render();dirty()}
async function checkOllama(){const d=await (await fetch("/api/ollama")).json();
  let h=d.error?`<span class="warn">${esc(d.error)}</span>`:`Ollama: ${d.models.length} моделей. Модель для картинок <b>${esc(d.model)}</b>: `+
    (d.vision===true?'<span class="ok">видит картинки</span>':d.vision===false?'<span class="warn">не умеет картинки, задайте VISION_MODEL (например qwen2.5vl:7b)</span>':"неизвестно");
  $("#ollama").innerHTML=h+`<datalist id="models">${(d.models||[]).map(m=>`<option value="${esc(m)}">`).join("")}</datalist>`}
window.addEventListener("beforeunload",e=>{if(!$("#save").disabled){e.preventDefault();e.returnValue=""}});
load();checkOllama();
</script></body></html>"""


def make_app() -> web.Application:
    app = web.Application(middlewares=[same_origin])
    app.add_routes([
        web.get("/", index),
        web.get("/api/state", api_state),
        web.get("/api/ollama", api_ollama),
        web.post("/api/save", api_save),
    ])
    return app


def main() -> None:
    app = make_app()
    url = f"http://{HOST}:{PORT}"
    print(f"Настройки Ауры: {url}  (Ctrl+C — остановить)")
    try:
        webbrowser.open(url)
    except Exception:  # noqa: BLE001
        pass
    web.run_app(app, host=HOST, port=PORT, print=None)


if __name__ == "__main__":
    main()
