"""SQLite-based monitoring: request + feedback logging."""

from __future__ import annotations

import os
import sqlite3
from datetime import datetime

from src.config import LOGS_DB_PATH


def get_db_connection() -> sqlite3.Connection:
    """Return a SQLite connection. Creates the `requests` table on first use."""
    os.makedirs(os.path.dirname(LOGS_DB_PATH), exist_ok=True)
    conn = sqlite3.connect(LOGS_DB_PATH)
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS requests (
            id TEXT PRIMARY KEY,
            timestamp TEXT,
            query TEXT,
            model TEXT,
            tokens_in INTEGER,
            tokens_out INTEGER,
            latency_ms INTEGER,
            cost_usd REAL,
            tools_used TEXT,
            user_rating INTEGER,
            user_comment TEXT
        )
        """
    )
    conn.commit()
    return conn


def log_request(
    query_id: str,
    query: str,
    model: str,
    tokens_in: int,
    tokens_out: int,
    latency_ms: int,
    cost_usd: float,
    tools_used: str,
) -> None:
    """Insert a request log entry."""
    conn = get_db_connection()
    conn.execute(
        """
        INSERT INTO requests
            (id, timestamp, query, model, tokens_in, tokens_out, latency_ms, cost_usd, tools_used)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
        (
            query_id,
            datetime.now().isoformat(),
            query,
            model,
            tokens_in,
            tokens_out,
            latency_ms,
            cost_usd,
            tools_used,
        ),
    )
    conn.commit()
    conn.close()


def log_feedback(query_id: str, rating: int, comment: str | None = None) -> None:
    """Attach user rating/comment to an existing request row."""
    conn = get_db_connection()
    conn.execute(
        "UPDATE requests SET user_rating = ?, user_comment = ? WHERE id = ?",
        (rating, comment, query_id),
    )
    conn.commit()
    conn.close()


def get_all_logs() -> list[dict]:
    """Return every request log row as a list of dicts, newest first."""
    conn = get_db_connection()
    cursor = conn.execute("SELECT * FROM requests ORDER BY timestamp DESC")
    columns = [d[0] for d in cursor.description]
    rows = cursor.fetchall()
    conn.close()
    return [dict(zip(columns, row)) for row in rows]
