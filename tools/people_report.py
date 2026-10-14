"""
Отчёт: как Аура относится ко всем, кого помнит.

    python tools\people_report.py              — все люди, по убыванию симпатии
    python tools\people_report.py --facts      — плюс что она о каждом знает
    python tools\people_report.py -n chrono    — только те, чьё имя содержит «chrono»

База открывается только на чтение, боту не мешает.  Отчёт ещё сохраняется в people_report.txt.
"""
from __future__ import annotations

import argparse
import json
import os
import sqlite3
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent


def _db_path() -> Path:
    rel = "data/events.db"
    env = ROOT / ".env"
    if env.exists():
        for line in env.read_text(encoding="utf-8", errors="ignore").splitlines():
            if line.strip().startswith("SQLITE_DB_PATH="):
                rel = line.split("=", 1)[1].strip().strip('"\'') or rel
    rel = os.environ.get("SQLITE_DB_PATH", rel)
    return ROOT / rel


def _label(a: float) -> str:  # как affinity_label в memory/social.py
    if a <= -0.6:
        return "терпеть не может"
    if a <= -0.25:
        return "неприятен"
    if a < 0.15:
        return "нейтрально"
    if a < 0.45:
        return "скорее симпатичен"
    if a < 0.75:
        return "тепло относится"
    return "очень дорог"


def _bar(a: float) -> str:
    n = round((a + 1) * 5)  # -1..1 -> 0..10
    return "█" * n + "·" * (10 - n)


def main() -> None:
    ap = argparse.ArgumentParser(description="Отношение Ауры к людям")
    ap.add_argument("--facts", action="store_true", help="показать факты о каждом")
    ap.add_argument("-n", "--name", default="", help="фильтр по имени")
    args = ap.parse_args()

    db = _db_path()
    if not db.exists():
        sys.exit(f"Нет базы: {db}")
    con = sqlite3.connect(f"file:{db}?mode=ro", uri=True)
    con.row_factory = sqlite3.Row
    rows = con.execute(
        "SELECT key, name, aliases, affinity, trust, opinion, interactions, last_seen FROM people"
        " WHERE name LIKE ? OR aliases LIKE ? ORDER BY affinity DESC, interactions DESC",
        (f"%{args.name}%", f"%{args.name}%"),
    ).fetchall()

    out = [f"Аура помнит людей: {len(rows)}   (база: {db.name})", ""]
    for r in rows:
        a = float(r["affinity"])
        try:
            aliases = ", ".join(json.loads(r["aliases"] or "[]"))
        except ValueError:
            aliases = ""
        out.append(f"{r['name']}" + (f"  (ещё: {aliases})" if aliases else ""))
        out.append(f"  отношение: {_bar(a)} {a * 10:+.0f}/10 — {_label(a)};  доверие {float(r['trust']) * 10:+.0f}/10;"
                   f"  сообщений: {r['interactions']};  последний раз: {str(r['last_seen'])[:16].replace('T', ' ')}")
        if r["opinion"]:
            out.append(f"  мнение: «{r['opinion']}»")
        if args.facts:
            facts = con.execute(
                "SELECT fact, said_by, is_gossip FROM facts WHERE subject_key = ? ORDER BY last_confirmed DESC LIMIT 15",
                (r["key"],),
            ).fetchall()
            for f in facts:
                src = f" (со слов {f['said_by']})" if f["is_gossip"] else ""
                out.append(f"    • {f['fact']}{src}")
        out.append("")

    text = "\n".join(out)
    try:
        print(text)
    except UnicodeEncodeError:
        sys.stdout.buffer.write(text.encode("utf-8", "replace"))
    (ROOT / "people_report.txt").write_text(text, encoding="utf-8")
    print(f"Сохранено: {ROOT / 'people_report.txt'}")


if __name__ == "__main__":
    main()
