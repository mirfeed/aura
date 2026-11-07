"""
tools/panel.py — пульт Ауры: одно окно вместо консолей.

    pythonw tools\\panel.py               — открыть пульт (если уже работает — просто показать окно)
    pythonw tools\\panel.py --background  — запустить в трее без окна (так он стартует вместе с Windows)

Что делает:
  * держит бота (main.py) скрытым процессом без окна: запуск, аккуратная остановка, перезапуск,
    сам поднимает её после падения (не больше 5 раз подряд);
  * поднимает Ollama без окна, выгружает модель, включает/выключает голос F5 и игровой режим;
  * управляет ею на лету через core/control.py: разбудить, уложить спать, отвлечь от книги,
    переключатели поведения;
  * страница http://127.0.0.1:8770 (в отдельном окне через WebView2 или в браузере), значок в трее;
  * внутри же — страницы настроек (8765) и отношений (8766), без отдельных окон.

Окно можно закрыть — пульт и Аура продолжат работать в трее.
"""

from __future__ import annotations

import asyncio
import ctypes
import json
import os
import subprocess
import sys
import threading
import time
import webbrowser
from collections import deque
from ctypes import wintypes
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Deque, Dict, List, Optional

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "tools"))

import aiohttp  # noqa: E402
from aiohttp import web  # noqa: E402

HOST = "127.0.0.1"
PORT = 8770
URL = f"http://{HOST}:{PORT}"

DATA = ROOT / "data"
LOGS = ROOT / "logs"
STOP_FLAG = DATA / "stop.flag"
GAME_FLAG = DATA / "game_mode.flag"
PID_FILE = DATA / "bot.pid"
TOKEN_FILE = DATA / "control.token"
PANEL_CFG = DATA / "panel.json"
BOT_LOG = LOGS / "bot.log"
NO_WINDOW = getattr(subprocess, "CREATE_NO_WINDOW", 0)
PY = Path(sys.executable).with_name("python.exe")
if not PY.exists():
    PY = Path(sys.executable)
PYW = Path(sys.executable).with_name("pythonw.exe")
if not PYW.exists():
    PYW = PY
STARTUP_LNK = Path(os.environ.get("APPDATA", "")) / "Microsoft/Windows/Start Menu/Programs/Startup/Aura.lnk"

# Переключатели на главной: (ключ, подпись, раздел).  Значение по умолчанию — из config.py.
TOGGLES = [
    ("SCHEDULE_ENABLED", "Спит по ночам (распорядок дня)", "Её день"),
    ("READING_ENABLED", "Читает книги", "Её день"),
    ("READING_SHARE_ENABLED", "Рассказывает о прочитанном", "Её день"),
    ("LIFE_EVENTS_ENABLED", "Случаи из её жизни", "Её день"),
    ("BIRTHDAYS_ENABLED", "Поздравляет с днём рождения", "Её день"),
    ("REMINDERS_ENABLED", "Напоминания", "Её день"),
    ("PROACTIVE_FRIENDS", "Сама пишет знакомым", "Пишет первой"),
    ("PROACTIVE_TRUSTED", "Делится с близкими", "Пишет первой"),
    ("DM_CATCHUP", "Отвечает на личку, пришедшую без неё", "Пишет первой"),
    ("WEB_SEARCH_ENABLED", "Ищет в интернете", "Чат"),
    ("VISION_ENABLED", "Видит картинки", "Чат"),
    ("VISION_OBSERVE", "Запоминает картинки, даже когда молчит", "Чат"),
    ("MEDIA_SEND_ENABLED", "Кидает картинки из любимых", "Чат"),
    ("FAVORITES_ENABLED", "Сохраняет любимые картинки", "Чат"),
    ("CHAT_SPLIT_MESSAGES", "Иногда разбивает ответ на два сообщения", "Чат"),
    ("CHAT_EMOJI", "Эмодзи в чате", "Чат"),
    ("VOICE_ENABLED", "Голосовые каналы", "Голос"),
    ("VOICE_PROACTIVE", "Сама вмешивается в разговор в войсе", "Голос"),
    ("F5_AUTO", "Голос F5 поднимается сам, когда она в войсе", "Голос"),
    ("TEST_MODE", "Тестовый режим: только тестовые серверы и личка", "Доступ"),
]
_TRUE = ("1", "true", "yes", "on", "да")


# ---------------------------------------------------------------------------
# .env и настройки пульта
# ---------------------------------------------------------------------------

def _settings():
    import settings_ui  # noqa: WPS433 — тот же разбор .env, что и у страницы настроек

    return settings_ui


def env_get(key: str, default: str) -> str:
    try:
        return _settings().read_env().get(key, default) or default
    except Exception:  # noqa: BLE001
        return default


_defaults_cache: Dict[str, str] = {}


def toggle_values() -> Dict[str, bool]:
    """Переключатели из .env (бот выключен) — с умолчаниями из config.py."""
    st = _settings()
    if not _defaults_cache:
        for f in st.parse_config():
            _defaults_cache[f["key"]] = str(f["default"])
    env = st.read_env()
    return {k: str(env.get(k, _defaults_cache.get(k, "false"))).strip().lower() in _TRUE for k, _l, _g in TOGGLES}


def panel_cfg() -> Dict[str, Any]:
    cfg = {"autostart_bot": True}
    try:
        cfg.update(json.loads(PANEL_CFG.read_text(encoding="utf-8")))
    except Exception:  # noqa: BLE001
        pass
    return cfg


def save_panel_cfg(cfg: Dict[str, Any]) -> None:
    DATA.mkdir(exist_ok=True)
    PANEL_CFG.write_text(json.dumps(cfg, ensure_ascii=False, indent=1), encoding="utf-8")


# ---------------------------------------------------------------------------
# Процессы Windows (без psutil)
# ---------------------------------------------------------------------------

_k32 = ctypes.WinDLL("kernel32", use_last_error=True)
_k32.CreateToolhelp32Snapshot.restype = wintypes.HANDLE
_k32.CreateToolhelp32Snapshot.argtypes = [wintypes.DWORD, wintypes.DWORD]
_k32.OpenProcess.restype = wintypes.HANDLE
_k32.OpenProcess.argtypes = [wintypes.DWORD, wintypes.BOOL, wintypes.DWORD]
_k32.CloseHandle.argtypes = [wintypes.HANDLE]
_k32.GetExitCodeProcess.argtypes = [wintypes.HANDLE, ctypes.POINTER(wintypes.DWORD)]
_k32.QueryFullProcessImageNameW.argtypes = [wintypes.HANDLE, wintypes.DWORD, wintypes.LPWSTR, ctypes.POINTER(wintypes.DWORD)]
_k32.Process32FirstW.argtypes = [wintypes.HANDLE, ctypes.c_void_p]
_k32.Process32NextW.argtypes = [wintypes.HANDLE, ctypes.c_void_p]
_INVALID_HANDLE = ctypes.c_void_p(-1).value


class _PE32(ctypes.Structure):
    _fields_ = [("dwSize", wintypes.DWORD), ("cntUsage", wintypes.DWORD), ("th32ProcessID", wintypes.DWORD),
                ("th32DefaultHeapID", ctypes.c_void_p), ("th32ModuleID", wintypes.DWORD),
                ("cntThreads", wintypes.DWORD), ("th32ParentProcessID", wintypes.DWORD),
                ("pcPriClassBase", ctypes.c_long), ("dwFlags", wintypes.DWORD), ("szExeFile", ctypes.c_wchar * 260)]


def processes() -> List[Dict[str, Any]]:
    """Все процессы: pid, ppid, exe (имя файла)."""
    snap = _k32.CreateToolhelp32Snapshot(0x2, 0)
    out: List[Dict[str, Any]] = []
    if not snap or snap == _INVALID_HANDLE:
        return out
    try:
        e = _PE32()
        e.dwSize = ctypes.sizeof(_PE32)
        ok = _k32.Process32FirstW(snap, ctypes.byref(e))
        while ok:
            out.append({"pid": e.th32ProcessID, "ppid": e.th32ParentProcessID, "exe": e.szExeFile.lower()})
            ok = _k32.Process32NextW(snap, ctypes.byref(e))
    finally:
        _k32.CloseHandle(snap)
    return out


def proc_image(pid: int) -> str:
    """Путь exe живого процесса ('' — процесса нет)."""
    if not pid:
        return ""
    h = _k32.OpenProcess(0x1000, False, int(pid))  # PROCESS_QUERY_LIMITED_INFORMATION
    if not h:
        return ""
    try:
        code = wintypes.DWORD()
        if not _k32.GetExitCodeProcess(h, ctypes.byref(code)) or code.value != 259:  # STILL_ACTIVE
            return ""
        buf = ctypes.create_unicode_buffer(1024)
        size = wintypes.DWORD(1024)
        if _k32.QueryFullProcessImageNameW(h, 0, buf, ctypes.byref(size)):
            return buf.value
        return "?"
    finally:
        _k32.CloseHandle(h)


def is_python(pid: int) -> bool:
    return Path(proc_image(pid)).name.lower() in ("python.exe", "pythonw.exe")


def kill(pid: int, tree: bool = False) -> None:
    subprocess.run(["taskkill", "/PID", str(pid), "/F"] + (["/T"] if tree else []),
                   capture_output=True, creationflags=NO_WINDOW)


def find_bot_by_cmdline() -> int:
    """Бот, запущенный не пультом (старый run_bot.bat): python …main.py.  Медленно — только при старте/запуске."""
    try:
        r = subprocess.run(
            ["powershell", "-NoProfile", "-Command",
             "Get-CimInstance Win32_Process -Filter \"Name='python.exe'\" | "
             "Where-Object { $_.CommandLine -match 'main\\.py' } | Select-Object -First 1 -ExpandProperty ProcessId"],
            capture_output=True, text=True, timeout=20, creationflags=NO_WINDOW)
        return int(r.stdout.strip() or 0)
    except Exception:  # noqa: BLE001
        return 0


def run_hidden(args: List[str], log_name: str) -> subprocess.Popen:  # type: ignore[type-arg]
    LOGS.mkdir(exist_ok=True)
    log = open(LOGS / log_name, "ab")
    try:
        return subprocess.Popen(args, cwd=str(ROOT), stdin=subprocess.DEVNULL, stdout=log, stderr=subprocess.STDOUT,
                                creationflags=NO_WINDOW)
    finally:
        log.close()


def wsl_path(path: Path) -> str:
    posix = path.as_posix()
    return f"/mnt/{path.drive.rstrip(':').lower()}{posix[len(path.drive):]}"


# ---------------------------------------------------------------------------
# Надзор за ботом и сервисами
# ---------------------------------------------------------------------------

class Supervisor:
    def __init__(self) -> None:
        self.proc: Optional[subprocess.Popen] = None  # type: ignore[type-arg]
        self.adopted = 0  # PID бота, которого запустил не пульт
        self.want = False  # должна работать (падение → перезапуск)
        self.crashes = 0
        self.started = 0.0
        self.busy = ""  # что пульт сейчас делает с ботом: starting / stopping / restarting
        self.next_restart = 0.0
        self.events: Deque[Dict[str, Any]] = deque(maxlen=40)
        self.lock = asyncio.Lock()
        self.session: Optional[aiohttp.ClientSession] = None
        self.snap: Dict[str, Any] = {}
        self._f5_at = 0.0
        self._gpu_at = 0.0

    def note(self, text: str, level: str = "info") -> None:
        self.events.appendleft({"t": datetime.now(timezone.utc).isoformat(), "level": level, "text": text})

    # --- бот -----------------------------------------------------------

    def bot_pid(self) -> int:
        if self.proc is not None and self.proc.poll() is None:
            return self.proc.pid
        if self.adopted and is_python(self.adopted):
            return self.adopted
        self.adopted = 0
        try:
            pid = int(PID_FILE.read_text().strip())
        except Exception:  # noqa: BLE001
            pid = 0
        if pid and is_python(pid):
            self.adopted = pid
            return pid
        return 0

    def adopt_running(self) -> None:
        if self.bot_pid():
            return
        pid = find_bot_by_cmdline()
        if pid:
            self.adopted = pid
            self.note("Аура уже работала — пульт следит за ней")

    async def start(self) -> None:
        async with self.lock:
            await self._start()

    async def _start(self) -> None:
        if self.bot_pid():
            self.want = True
            return
        if await asyncio.to_thread(find_bot_by_cmdline):
            self.adopt_running()
            self.want = True
            return
        self.busy = "starting"
        try:
            await self.ensure_ollama()
            STOP_FLAG.unlink(missing_ok=True)
            with open(LOGS / "console.log", "ab") as log:
                log.write(f"=== start {datetime.now():%Y-%m-%d %H:%M:%S} (пульт)\n".encode("utf-8"))
            self.proc = run_hidden([str(PY), "main.py"], "console.log")
            self.adopted = 0
            self.want = True
            self.started = time.monotonic()
            self.note("Запускаю Ауру")
            for _ in range(90):  # ждём «Bot ready» (статус от самого бота)
                await asyncio.sleep(1)
                st = await self.ctl("GET", "/status")
                if st and st.get("ready"):
                    self.note("Аура в сети", "ok")
                    break
                if self.proc.poll() is not None:
                    break
        finally:
            self.busy = ""

    async def stop(self) -> None:
        async with self.lock:
            await self._stop()

    async def _stop(self) -> None:
        self.want = False
        pid = self.bot_pid()
        if not pid:
            return
        self.busy = "stopping"
        try:
            DATA.mkdir(exist_ok=True)
            STOP_FLAG.touch()  # бот сам выйдет из войса и закроет базу
            self.note("Останавливаю аккуратно")
            for _ in range(30):
                await asyncio.sleep(1)
                if not proc_image(pid):
                    self.note("Аура остановлена")
                    break
            else:
                self.note("Не ответила за 30 с — выключаю принудительно", "warn")
                parent = next((p for p in processes() if p["pid"] == pid), None)
                if parent:
                    pp = next((p for p in processes() if p["pid"] == parent["ppid"]), None)
                    if pp and pp["exe"] == "cmd.exe":
                        kill(pp["pid"])  # старый run_bot.bat — иначе он перезапустит её
                kill(pid, tree=True)
            STOP_FLAG.unlink(missing_ok=True)
            self.proc, self.adopted = None, 0
        finally:
            self.busy = ""

    async def restart(self) -> None:
        async with self.lock:
            self.busy = "restarting"
            await self._stop()
            await asyncio.sleep(2)
            self.crashes = 0
            await self._start()

    async def ctl(self, method: str, path: str, body: Optional[Dict[str, Any]] = None) -> Optional[Dict[str, Any]]:
        """Запрос к самому боту (core/control.py).  None — не отвечает."""
        try:
            token = TOKEN_FILE.read_text(encoding="utf-8").strip()
        except OSError:
            return None
        port = env_get("CONTROL_PORT", "8771")
        try:
            async with self.session.request(
                method, f"http://127.0.0.1:{port}{path}", json=body, headers={"X-Aura-Token": token},
                timeout=aiohttp.ClientTimeout(total=4 if method == "GET" else 30),
            ) as r:
                return await r.json()
        except Exception:  # noqa: BLE001
            return None

    # --- Ollama / F5 / видеокарта ---------------------------------------

    def ollama_url(self) -> str:
        return env_get("OLLAMA_BASE_URL", "http://localhost:11434").rstrip("/").replace("localhost", "127.0.0.1")

    async def _get_json(self, url: str, timeout: float = 2.0) -> Optional[Any]:
        try:
            async with self.session.get(url, timeout=aiohttp.ClientTimeout(total=timeout)) as r:
                if r.status != 200:
                    return None
                ct = r.headers.get("Content-Type", "")
                return await r.json() if "json" in ct else {}
        except Exception:  # noqa: BLE001
            return None

    async def ensure_ollama(self) -> bool:
        if await self._get_json(self.ollama_url() + "/api/version") is not None:
            return True
        exe = Path(os.environ.get("LOCALAPPDATA", "")) / "Programs/Ollama/ollama.exe"
        if not exe.exists():
            self.note("Ollama не найдена — бот запустится, но отвечать не сможет", "warn")
            return False
        run_hidden([str(exe), "serve"], "ollama.log")
        self.note("Запускаю Ollama")
        for _ in range(30):
            await asyncio.sleep(1)
            if await self._get_json(self.ollama_url() + "/api/version") is not None:
                return True
        self.note("Ollama не поднялась за 30 секунд", "warn")
        return False

    async def unload(self) -> List[str]:
        done: List[str] = []
        procs = processes()
        alive = {p["pid"] for p in procs if p["exe"] == "ollama.exe"}
        for p in procs:
            if p["exe"] == "llama-server.exe" and p["ppid"] not in alive:
                kill(p["pid"])
                done.append(f"брошенный llama-server (PID {p['pid']})")
        ps = await self._get_json(self.ollama_url() + "/api/ps", 5)
        for m in (ps or {}).get("models", []):
            try:
                async with self.session.post(self.ollama_url() + "/api/generate",
                                             json={"model": m["name"], "keep_alive": 0},
                                             timeout=aiohttp.ClientTimeout(total=30)) as r:
                    await r.read()
                done.append(m["name"])
            except Exception:  # noqa: BLE001
                pass
        self.note("Выгружено: " + (", ".join(done) if done else "в видеопамяти ничего не было"))
        return done

    def f5_url(self) -> str:
        return env_get("F5_API_URL", "http://127.0.0.1:8081").rstrip("/")

    async def f5_start(self) -> None:
        run_hidden(["wsl.exe", "--", "bash", wsl_path(ROOT / "f5" / "run_f5.sh")], "f5.log")
        self.note("Запускаю голос F5 (около минуты)")

    async def f5_stop(self) -> None:
        await asyncio.to_thread(subprocess.run, ["wsl.exe", "-e", "pkill", "-f", "f5_server.py"],
                                capture_output=True, creationflags=NO_WINDOW)
        self.note("Голос F5 выключен")

    async def set_game(self, on: bool) -> None:
        DATA.mkdir(exist_ok=True)
        if on:
            GAME_FLAG.touch()
            self.note("Игровой режим: только чат, видеопамять отдана игре")
            if self.snap.get("f5"):
                await self.f5_stop()
            if not self.bot_pid():
                await self.unload()
        else:
            GAME_FLAG.unlink(missing_ok=True)
            self.note("Игровой режим выключен")

    @staticmethod
    def _gpu() -> Optional[Dict[str, float]]:
        try:
            r = subprocess.run(["nvidia-smi", "--query-gpu=memory.used,memory.total,utilization.gpu",
                                "--format=csv,noheader,nounits"], capture_output=True, text=True, timeout=5,
                               creationflags=NO_WINDOW)
            used, total, util = (float(x) for x in r.stdout.strip().splitlines()[0].split(","))
            return {"used_gb": round(used / 1024, 1), "total_gb": round(total / 1024, 1), "util": util}
        except Exception:  # noqa: BLE001
            return None

    # --- общий цикл ------------------------------------------------------

    async def refresh(self) -> None:
        pid = self.bot_pid()
        bot = await self.ctl("GET", "/status") if pid else None
        ollama = await self._get_json(self.ollama_url() + "/api/ps")
        now = time.monotonic()
        if now - self._f5_at > 8:
            self._f5_at = now
            self.snap["f5"] = await self._get_json(self.f5_url() + "/docs") is not None
        if now - self._gpu_at > 5:
            self._gpu_at = now
            self.snap["gpu"] = await asyncio.to_thread(self._gpu)
        self.snap.update({
            "pid": pid,
            "managed": self.proc is not None and self.proc.poll() is None,
            "busy": self.busy,
            "bot": bot,
            "ollama": None if ollama is None else [
                {"name": m.get("name", ""), "vram_gb": round(m.get("size_vram", 0) / 2**30, 1)}
                for m in ollama.get("models", [])],
            "game": GAME_FLAG.exists(),
            "crashes": self.crashes,
            "restart_in": max(0, round(self.next_restart - now)) if self.next_restart else 0,
        })

    async def watch(self) -> None:
        """Каждые 2 с: статус для страницы и трея; упала — поднять снова."""
        while True:
            try:
                if self.proc is not None and self.proc.poll() is not None and not self.busy:
                    code = self.proc.returncode
                    ran = time.monotonic() - self.started
                    self.proc = None
                    with open(LOGS / "console.log", "ab") as log:
                        log.write(f"=== exit code {code} {datetime.now():%Y-%m-%d %H:%M:%S}\n".encode("utf-8"))
                    if code == 0 or not self.want:
                        self.want = False
                        self.note("Аура остановилась")
                    else:
                        self.crashes = 1 if ran > 600 else self.crashes + 1
                        if self.crashes >= 5:
                            self.want = False
                            self.note(f"Аура упала {self.crashes} раз подряд — не перезапускаю, смотри лог", "error")
                        else:
                            self.next_restart = time.monotonic() + 15
                            self.note(f"Аура упала (код {code}) — перезапуск через 15 с ({self.crashes}/5)", "error")
                if self.next_restart and time.monotonic() >= self.next_restart:
                    self.next_restart = 0.0
                    if self.want:
                        asyncio.create_task(self.start())
                await self.refresh()
                tray_update(self.snap)
            except asyncio.CancelledError:
                raise
            except Exception as exc:  # noqa: BLE001
                self.note(f"Ошибка пульта: {exc}", "error")
            await asyncio.sleep(2)


SUP: Optional[Supervisor] = None
LOOP: Optional[asyncio.AbstractEventLoop] = None


# ---------------------------------------------------------------------------
# Лог
# ---------------------------------------------------------------------------

_LOG_EXTRA = ("text", "reply", "thought", "error", "reason", "query", "why", "book", "key", "title", "summary")


def tail_log(n: int = 120, only: str = "all") -> List[Dict[str, Any]]:
    try:
        with open(BOT_LOG, "rb") as f:
            f.seek(0, 2)
            size = f.tell()
            f.seek(max(0, size - 400_000))
            lines = f.read().decode("utf-8", "replace").splitlines()[1:]
    except OSError:
        return []
    out: List[Dict[str, Any]] = []
    for line in reversed(lines):
        try:
            j = json.loads(line)
        except ValueError:
            continue
        lvl = j.get("level", "INFO")
        ev = j.get("event", "")
        if only == "errors" and lvl not in ("ERROR", "WARNING", "CRITICAL"):
            continue
        if only == "main" and (ev in ("Typing seen", "Background tick") or lvl == "DEBUG"):
            continue
        d = j.get("data") or {}
        extra = next((str(d[k]) for k in _LOG_EXTRA if d.get(k)), "")
        if ev == "Day phase changed":
            extra = f"{d.get('from')} → {d.get('to')}"
        elif ev == "Reading session" and d.get("pos"):
            extra = f"{d.get('book')} ({d.get('pos')}): {d.get('thought', '')}"
        out.append({"t": j.get("timestamp", ""), "level": lvl, "event": ev, "text": extra[:400]})
        if len(out) >= n:
            break
    return out


# ---------------------------------------------------------------------------
# HTTP
# ---------------------------------------------------------------------------

@web.middleware
async def guard(request: web.Request, handler):
    """Только с этого компьютера и только со своей страницы (чужие сайты в браузере — мимо)."""
    if request.host.split(":")[0] not in (HOST, "localhost"):
        return web.json_response({"error": "forbidden"}, status=403)
    if request.method != "GET":
        origin = request.headers.get("Origin", "")
        if origin and origin not in (URL, f"http://localhost:{PORT}"):
            return web.json_response({"error": "forbidden"}, status=403)
        if request.content_type != "application/json":
            return web.json_response({"error": "json only"}, status=415)
    return await handler(request)


async def h_index(_: web.Request) -> web.Response:
    return web.Response(text=(Path(__file__).with_name("panel.html")).read_text(encoding="utf-8"),
                        content_type="text/html", charset="utf-8", headers={"Cache-Control": "no-store"})


async def h_ping(_: web.Request) -> web.Response:
    return web.json_response({"ok": True, "app": "aura-panel"})


async def h_status(_: web.Request) -> web.Response:
    snap = dict(SUP.snap)
    bot = snap.get("bot") or {}
    values = bot.get("toggles") or toggle_values()
    snap["toggles"] = [{"key": k, "label": label, "group": g, "value": bool(values.get(k))} for k, label, g in TOGGLES]
    snap["events"] = list(SUP.events)[:15]
    cfg = panel_cfg()
    snap["panel"] = {"autostart_bot": bool(cfg.get("autostart_bot")), "with_windows": STARTUP_LNK.exists()}
    return web.json_response(snap)


async def h_log(request: web.Request) -> web.Response:
    n = min(500, int(request.query.get("n", "150")))
    return web.json_response({"lines": await asyncio.to_thread(tail_log, n, request.query.get("only", "main"))})


async def _body(request: web.Request) -> Dict[str, Any]:
    try:
        b = await request.json()
        return b if isinstance(b, dict) else {}
    except Exception:  # noqa: BLE001
        return {}


async def h_action(request: web.Request) -> web.Response:
    """Всё, что делается кнопками: {"do": "...", ...}."""
    b = await _body(request)
    do = b.get("do", "")
    sup = SUP
    if do in ("start", "stop", "restart"):
        asyncio.create_task(getattr(sup, do)())  # долго — страница следит по статусу
        return web.json_response({"ok": True})
    if do in ("wake", "sleep", "auto", "distract", "read"):
        if not sup.bot_pid():
            return web.json_response({"ok": False, "error": "Аура выключена"})
        body = {k: b[k] for k in ("minutes", "quiet") if k in b}
        r = await sup.ctl("POST", "/" + do, body)
        if r is None:
            return web.json_response({"ok": False, "error": "бот не отвечает (если она запущена старым способом — перезапусти)"})
        await sup.refresh()
        return web.json_response(r)
    if do == "toggle":
        key, value = b.get("key"), b.get("value")
        if key not in {k for k, _l, _g in TOGGLES} or not isinstance(value, bool):
            return web.json_response({"ok": False, "error": "неизвестный переключатель"}, status=400)
        _settings().write_env({key: "true" if value else "false"})
        live: Optional[bool] = None
        if sup.bot_pid():
            r = await sup.ctl("POST", "/set", {"key": key, "value": value})
            live = bool(r and r.get("live"))
        await sup.refresh()
        return web.json_response({"ok": True, "live": live})
    if do == "game":
        await sup.set_game(bool(b.get("value")))
        await sup.refresh()
        return web.json_response({"ok": True})
    if do == "unload":
        return web.json_response({"ok": True, "done": await sup.unload()})
    if do == "ollama":
        return web.json_response({"ok": await sup.ensure_ollama()})
    if do == "f5":
        await (sup.f5_start() if b.get("value") else sup.f5_stop())
        return web.json_response({"ok": True})
    if do == "panel_cfg":
        cfg = panel_cfg()
        if "autostart_bot" in b:
            cfg["autostart_bot"] = bool(b["autostart_bot"])
        save_panel_cfg(cfg)
        if "with_windows" in b:
            await asyncio.to_thread(set_autostart, bool(b["with_windows"]))
        return web.json_response({"ok": True})
    if do == "quit":
        if b.get("stop_bot"):
            await sup.stop()
        LOOP.call_later(0.3, quit_panel)
        return web.json_response({"ok": True})
    return web.json_response({"ok": False, "error": f"неизвестное действие {do}"}, status=400)


def make_app() -> web.Application:
    app = web.Application(middlewares=[guard])
    app.add_routes([
        web.get("/", h_index),
        web.get("/api/ping", h_ping),
        web.get("/api/status", h_status),
        web.get("/api/log", h_log),
        web.post("/api/action", h_action),
    ])
    return app


# ---------------------------------------------------------------------------
# Ярлыки, окно, трей
# ---------------------------------------------------------------------------

def make_shortcut(lnk: Path, args: str) -> None:
    ps = (
        "$s=(New-Object -ComObject WScript.Shell).CreateShortcut($env:AURA_LNK);"
        "$s.TargetPath=$env:AURA_PYW;$s.Arguments=$env:AURA_ARGS;$s.WorkingDirectory=$env:AURA_ROOT;"
        "$s.Description='Пульт Ауры';$s.Save()"
    )
    env = dict(os.environ, AURA_LNK=str(lnk), AURA_PYW=str(PYW), AURA_ARGS=args, AURA_ROOT=str(ROOT))
    subprocess.run(["powershell", "-NoProfile", "-Command", ps], env=env, capture_output=True, timeout=30,
                   creationflags=NO_WINDOW)


def set_autostart(on: bool) -> None:
    if on:
        make_shortcut(STARTUP_LNK, f'"{ROOT / "tools" / "panel.py"}" --background')
        SUP.note("Пульт будет запускаться вместе с Windows")
    else:
        STARTUP_LNK.unlink(missing_ok=True)
        SUP.note("Автозапуск с Windows выключен")


def open_window() -> None:
    """Окно пульта: отдельный процесс с WebView2; не вышло — браузер."""
    try:
        import webview  # noqa: F401
        subprocess.Popen([str(PYW), str(Path(__file__).resolve()), "--window"], cwd=str(ROOT),
                         creationflags=NO_WINDOW)
    except Exception:  # noqa: BLE001
        webbrowser.open(URL)


def window_main() -> None:
    import webview

    webview.create_window("Аура", URL, width=1180, height=880, min_size=(420, 500), background_color="#16151a")
    webview.start(private_mode=False, storage_path=str(DATA / "panel_webview"))


_icon: Any = None
_icon_key = ""


def _draw_icon(color: str, moon: bool):
    from PIL import Image, ImageDraw

    img = Image.new("RGBA", (64, 64), (0, 0, 0, 0))
    d = ImageDraw.Draw(img)
    d.ellipse((4, 4, 60, 60), fill=color)
    if moon:
        d.ellipse((18, 14, 46, 46), fill="#f3eefc")
        d.ellipse((26, 10, 54, 40), fill=color)
    else:
        d.polygon([(32, 14), (48, 50), (40, 50), (32, 30), (24, 50), (16, 50)], fill="#ffffff")
    return img


def tray_update(snap: Dict[str, Any]) -> None:
    global _icon_key
    if _icon is None:
        return
    bot = snap.get("bot") or {}
    if not snap.get("pid"):
        key, color, moon, tip = "off", "#6b6876", False, "Аура выключена"
    elif bot.get("phase") == "asleep":
        key, color, moon, tip = "sleep", "#4b4a8c", True, "Аура спит"
    elif bot:
        reading = (bot.get("reading") or {})
        key, color, moon = "on", "#7a4fd6", False
        tip = "Аура в сети" + (" — читает" if reading.get("now") or reading.get("binge") else "")
    else:
        key, color, moon, tip = "nolink", "#b5651d", False, "Аура запускается или без связи с пультом"
    try:
        if key != _icon_key:
            _icon.icon = _draw_icon(color, moon)
            _icon_key = key
        _icon.title = tip
    except Exception:  # noqa: BLE001
        pass


def _call(coro_fn, *a) -> None:
    if LOOP is not None:
        asyncio.run_coroutine_threadsafe(coro_fn(*a), LOOP)


async def _tray_ctl(path: str, body: Optional[Dict[str, Any]] = None) -> None:
    await SUP.ctl("POST", path, body or {})
    await SUP.refresh()
    tray_update(SUP.snap)


def run_tray() -> None:
    global _icon
    import pystray
    from pystray import Menu, MenuItem as Item

    def bot() -> Dict[str, Any]:
        return (SUP.snap.get("bot") or {}) if SUP else {}

    def asleep(_=None) -> bool:
        return bot().get("phase") == "asleep"

    def reading(_=None) -> bool:
        r = bot().get("reading") or {}
        return bool(r.get("now") or r.get("binge"))

    def running(_=None) -> bool:
        return bool(SUP and SUP.snap.get("pid"))

    _icon = pystray.Icon("aura", _draw_icon("#6b6876", False), "Аура", Menu(
        Item("Открыть пульт", lambda: open_window(), default=True),
        Menu.SEPARATOR,
        Item("Разбудить", lambda: _call(_tray_ctl, "/wake", {"minutes": 60}), visible=asleep),
        Item("Уложить спать", lambda: _call(_tray_ctl, "/sleep", {}), visible=lambda _: running() and not asleep()),
        Item("Отвлечь от чтения", lambda: _call(_tray_ctl, "/distract"), visible=reading),
        Menu.SEPARATOR,
        Item("Запустить Ауру", lambda: _call(SUP.start), visible=lambda _: not running()),
        Item("Перезапустить Ауру", lambda: _call(SUP.restart), visible=running),
        Item("Остановить Ауру", lambda: _call(SUP.stop), visible=running),
        Menu.SEPARATOR,
        Item("Закрыть пульт (Аура продолжит работать)", lambda: quit_panel()),
    ))
    _icon.run()


def quit_panel() -> None:
    if _icon is not None:
        _icon.stop()
    else:
        os._exit(0)


def serve_forever(ready: threading.Event, autostart: bool) -> None:
    global SUP, LOOP
    LOOP = asyncio.new_event_loop()
    asyncio.set_event_loop(LOOP)

    async def boot() -> None:
        global SUP
        SUP = Supervisor()
        SUP.session = aiohttp.ClientSession()
        runners = []
        for mod_name, port in (("panel", PORT), ("settings_ui", 8765), ("people_ui", 8766)):
            try:
                app = make_app() if mod_name == "panel" else __import__(mod_name).make_app()
                runner = web.AppRunner(app, access_log=None)
                await runner.setup()
                await web.TCPSite(runner, HOST, port).start()
                runners.append(runner)
            except OSError:
                if mod_name == "panel":
                    raise
                SUP.note(f"Порт {port} занят (страница уже открыта отдельно?)", "warn")
        await asyncio.to_thread(SUP.adopt_running)
        ready.set()
        if autostart and panel_cfg().get("autostart_bot") and not SUP.bot_pid():
            asyncio.create_task(SUP.start())
        await SUP.watch()

    try:
        LOOP.run_until_complete(boot())
    except Exception:  # noqa: BLE001
        ready.set()
        raise


def already_running() -> bool:
    import urllib.request

    try:
        with urllib.request.urlopen(URL + "/api/ping", timeout=1.5) as r:
            return b"aura-panel" in r.read()
    except Exception:  # noqa: BLE001
        return False


def main() -> None:
    args = sys.argv[1:]
    if "--window" in args:
        window_main()
        return
    if already_running():
        if "--background" not in args:
            open_window()
        return
    os.chdir(ROOT)
    ready = threading.Event()
    th = threading.Thread(target=serve_forever, args=(ready, True), daemon=True)
    th.start()
    ready.wait(30)
    if "--background" not in args:
        open_window()
    try:
        run_tray()
    except Exception:  # noqa: BLE001 — нет трея (pystray/PIL) — просто живём, пока работает сервер
        th.join()
    os._exit(0)


if __name__ == "__main__":
    main()
