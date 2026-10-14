"""
Вручную поправить отношение Ауры к человеку.  Шкала как в отчёте: от -10 до 10.

    python tools\people_set.py MirFeeD --affinity 10 --trust 10
    python tools\people_set.py Frauder --affinity 0

Имя ищется по нику и прозвищам (без учёта регистра).  Можно запускать при работающем боте.
"""
from __future__ import annotations

import argparse
import json
import shutil
import sqlite3
import sys
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from people_report import ROOT, _db_path  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser(description="Поправить отношение Ауры к человеку")
    ap.add_argument("name")
    ap.add_argument("--affinity", type=float, help="отношение, от -10 до 10")
    ap.add_argument("--trust", type=float, help="доверие, от -10 до 10")
    args = ap.parse_args()
    if args.affinity is None and args.trust is None:
        sys.exit("Укажи --affinity и/или --trust")

    db = _db_path()
    con = sqlite3.connect(db, timeout=15)
    con.row_factory = sqlite3.Row
    want = args.name.lower()
    found = []
    for r in con.execute("SELECT key, name, aliases, affinity, trust FROM people"):
        names = [r["name"]] + json.loads(r["aliases"] or "[]")
        if any(n.lower() == want for n in names):
            found.append(r)
    if not found:
        sys.exit(f"Не нашла «{args.name}». Посмотри имена в people_report.txt")
    if len(found) > 1:
        sys.exit("Подходят несколько: " + ", ".join(r["name"] for r in found) + " — укажи точный ник")
    r = found[0]

    backup = ROOT / "backup_length" / f"events.db.{datetime.now():%Y%m%d-%H%M%S}"
    backup.parent.mkdir(exist_ok=True)
    shutil.copy2(db, backup)

    aff = float(r["affinity"]) if args.affinity is None else max(-1.0, min(1.0, args.affinity / 10))
    trust = float(r["trust"]) if args.trust is None else max(-1.0, min(1.0, args.trust / 10))
    con.execute("UPDATE people SET affinity = ?, trust = ?, updated_at = ? WHERE key = ?",
                (aff, trust, datetime.now(timezone.utc).isoformat(), r["key"]))
    con.commit()
    print(f"{r['name']}: отношение {float(r['affinity']) * 10:+.0f} -> {aff * 10:+.0f}, "
          f"доверие {float(r['trust']) * 10:+.0f} -> {trust * 10:+.0f}   (бэкап: {backup.name})")


if __name__ == "__main__":
    main()
