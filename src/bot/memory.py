from __future__ import annotations

import json
import sqlite3
from pathlib import Path
from typing import Any, Dict, Optional


class BotMemory:
    def __init__(self, db_path: Path) -> None:
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self.conn = sqlite3.connect(self.db_path)
        self._init_schema()

    def _init_schema(self) -> None:
        cur = self.conn.cursor()
        cur.executescript(
            """
            CREATE TABLE IF NOT EXISTS trades (
                id INTEGER PRIMARY KEY,
                symbol TEXT,
                side TEXT,
                amount REAL,
                price REAL,
                timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
                pnl REAL,
                status TEXT
            );

            CREATE TABLE IF NOT EXISTS logs (
                id INTEGER PRIMARY KEY,
                timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
                level TEXT,
                message TEXT,
                context_snapshot TEXT
            );

            CREATE TABLE IF NOT EXISTS config (
                key TEXT PRIMARY KEY,
                value TEXT
            );
            """
        )
        self.conn.commit()

    def log(self, level: str, message: str, context: Optional[Dict[str, Any]] = None) -> None:
        cur = self.conn.cursor()
        cur.execute(
            "INSERT INTO logs (level, message, context_snapshot) VALUES (?, ?, ?)",
            (level, message, json.dumps(context or {})),
        )
        self.conn.commit()

    def log_info(self, message: str, context: Optional[Dict[str, Any]] = None) -> None:
        self.log("INFO", message, context)

    def log_error(self, message: str, context: Optional[Dict[str, Any]] = None) -> None:
        self.log("ERROR", message, context)

    def log_decision(self, message: str, raw_output: str, context_snapshot: Dict[str, Any]) -> None:
        ctx = dict(context_snapshot)
        ctx["_raw_llm_output"] = raw_output
        self.log("DECISION", message, ctx)

    def log_trade_open(self, order: Dict[str, Any], snapshot: Dict[str, Any], action: Dict[str, Any]) -> None:
        self.log("TRADE_OPEN", f"Ouverture trade {order}", {"order": order, "action": action})

    def log_trade_close(self, order: Dict[str, Any], snapshot: Dict[str, Any], action: Dict[str, Any]) -> None:
        self.log("TRADE_CLOSE", f"Fermeture trade {order}", {"order": order, "action": action})

    def close(self) -> None:
        self.conn.close()


__all__ = ["BotMemory"]
