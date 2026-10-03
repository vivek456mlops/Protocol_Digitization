"""SQLite persistence for protocol metadata and source chunks."""

import sqlite3
from pathlib import Path
from typing import Dict, Iterable, Optional

from .rag import Chunk


class ProtocolStore:
    def __init__(self, path: Path) -> None:
        self.path = path
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self._connect() as connection:
            connection.executescript(
                """
                CREATE TABLE IF NOT EXISTS protocols (
                    name TEXT PRIMARY KEY,
                    category TEXT,
                    version TEXT,
                    updated TEXT,
                    pages INTEGER NOT NULL,
                    filename TEXT
                );
                CREATE TABLE IF NOT EXISTS chunks (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    protocol TEXT NOT NULL,
                    page INTEGER NOT NULL,
                    text TEXT NOT NULL,
                    FOREIGN KEY(protocol) REFERENCES protocols(name) ON DELETE CASCADE
                );
                """
            )

    def _connect(self) -> sqlite3.Connection:
        connection = sqlite3.connect(self.path)
        connection.row_factory = sqlite3.Row
        connection.execute("PRAGMA foreign_keys = ON")
        return connection

    def seed(self, protocols: Iterable[dict]) -> None:
        with self._connect() as connection:
            if connection.execute("SELECT 1 FROM protocols LIMIT 1").fetchone():
                return
            for protocol in protocols:
                self._replace_protocol(connection, protocol, protocol.get("filename", "synthetic.json"))

    def replace(self, protocols: Iterable[dict], filename: str) -> None:
        with self._connect() as connection:
            for protocol in protocols:
                self._replace_protocol(connection, protocol, filename)

    def remove(self, names: Iterable[str]) -> None:
        with self._connect() as connection:
            connection.executemany("DELETE FROM protocols WHERE name = ?", [(name,) for name in names])

    def _replace_protocol(self, connection: sqlite3.Connection, protocol: dict, filename: str) -> None:
        name = protocol["name"]
        connection.execute("DELETE FROM chunks WHERE protocol = ?", (name,))
        connection.execute(
            """INSERT INTO protocols(name, category, version, updated, pages, filename)
               VALUES (?, ?, ?, ?, ?, ?)
               ON CONFLICT(name) DO UPDATE SET category=excluded.category,
               version=excluded.version, updated=excluded.updated,
               pages=excluded.pages, filename=excluded.filename""",
            (name, protocol.get("category", "Uncategorized"), protocol.get("version", "1.0"), protocol.get("updated", "Recently"), protocol.get("pages", 0), filename),
        )
        connection.executemany(
            "INSERT INTO chunks(protocol, page, text) VALUES (?, ?, ?)",
            [(name, chunk["page"], chunk["text"]) for chunk in protocol.get("chunks", [])],
        )

    def list_protocols(self) -> list[dict]:
        with self._connect() as connection:
            rows = connection.execute("SELECT name, category, version, updated, pages, filename FROM protocols ORDER BY name").fetchall()
            return [dict(row) for row in rows]

    def chunks(self) -> list[Chunk]:
        with self._connect() as connection:
            rows = connection.execute("SELECT protocol, page, text FROM chunks ORDER BY protocol, page").fetchall()
            return [Chunk(row["protocol"], row["page"], row["text"]) for row in rows]

    def source(self, protocol: str, page: int) -> Optional[Dict]:
        with self._connect() as connection:
            row = connection.execute("SELECT protocol, page, text FROM chunks WHERE protocol = ? AND page = ?", (protocol, page)).fetchone()
            return dict(row) if row else None
