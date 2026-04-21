# src/db/feedback_store.py
"""
SQLite feedback store — logs every query + thumbs up/down.

Schema:
  feedback(
    id          INTEGER PRIMARY KEY,
    session_id  TEXT,
    query       TEXT,
    rewritten   TEXT,
    intent      TEXT,
    response    TEXT,
    rating      INTEGER,   -- 1 = thumbs up, -1 = thumbs down, 0 = no rating
    fallback    INTEGER,   -- 1 if fallback was used
    latency_ms  INTEGER,
    cited_pages TEXT,      -- JSON array string e.g. "[12, 15]"
    timestamp   TEXT       -- ISO format
  )

Why SQLite?
- Zero setup, zero server, built into Python
- Persistent across restarts
- Queryable for the analytics dashboard
- Trivially exportable to CSV for RAGAS evaluation
"""

import json
import sqlite3
from datetime import datetime, timezone
from pathlib import Path
from loguru import logger

DB_PATH = "data/feedback.db"


def _get_connection() -> sqlite3.Connection:
    """Return a connection with row_factory set for dict-like access."""
    Path(DB_PATH).parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn


def init_db() -> None:
    """
    Create the feedback table if it doesn't exist.
    Safe to call multiple times — uses CREATE TABLE IF NOT EXISTS.
    Call once at app startup.
    """
    conn = _get_connection()
    conn.execute("""
        CREATE TABLE IF NOT EXISTS feedback (
            id          INTEGER PRIMARY KEY AUTOINCREMENT,
            session_id  TEXT    NOT NULL,
            query       TEXT    NOT NULL,
            rewritten   TEXT,
            intent      TEXT,
            response    TEXT,
            rating      INTEGER DEFAULT 0,
            fallback    INTEGER DEFAULT 0,
            latency_ms  INTEGER,
            cited_pages TEXT    DEFAULT '[]',
            timestamp   TEXT    NOT NULL
        )
    """)
    conn.commit()
    conn.close()
    logger.info(f"Feedback DB initialised at {DB_PATH}")


def log_query(
    session_id:  str,
    query:       str,
    rewritten:   str,
    intent:      str,
    response:    str,
    fallback:    bool,
    latency_ms:  int,
    cited_pages: list[int],
    rating:      int = 0
) -> int:
    """
    Insert one query+response record. Returns the new row id.
    Rating can be updated later via update_rating().
    """
    conn = _get_connection()
    cursor = conn.execute(
        """
        INSERT INTO feedback
          (session_id, query, rewritten, intent, response,
           rating, fallback, latency_ms, cited_pages, timestamp)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
        (
            session_id,
            query,
            rewritten,
            intent,
            response,
            rating,
            int(fallback),
            latency_ms,
            json.dumps(cited_pages),
            datetime.now(timezone.utc).isoformat(),
        )
    )
    conn.commit()
    row_id = cursor.lastrowid
    conn.close()
    if row_id is None:
        raise RuntimeError("Failed to retrieve row ID from SQLite insert.")

    logger.debug(f"Feedback logged: row_id={row_id} session={session_id}")
    return row_id


def update_rating(row_id: int, rating: int) -> bool:
    """
    Update the thumbs rating for a previously logged row.
    rating: 1 = thumbs up, -1 = thumbs down, 0 = neutral
    Returns True if the row was found and updated.
    """
    if rating not in {-1, 0, 1}:
        logger.warning(f"Invalid rating value: {rating} — must be -1, 0, or 1")
        return False

    conn = _get_connection()
    cursor = conn.execute(
        "UPDATE feedback SET rating = ? WHERE id = ?",
        (rating, row_id)
    )
    conn.commit()
    updated = cursor.rowcount > 0
    conn.close()

    if updated:
        logger.debug(f"Rating updated: row_id={row_id} rating={rating}")
    else:
        logger.warning(f"Rating update failed: row_id={row_id} not found")

    return updated


def get_analytics() -> dict:
    """
    Aggregate stats for the analytics dashboard.

    Returns:
        {
          total_queries:    int,
          thumbs_up:        int,
          thumbs_down:      int,
          fallback_count:   int,
          avg_latency_ms:   float,
          intent_counts:    dict[str, int],
          recent:           list[dict]   ← last 10 queries
        }
    """
    conn = _get_connection()

    total       = conn.execute("SELECT COUNT(*) FROM feedback").fetchone()[0]
    thumbs_up   = conn.execute("SELECT COUNT(*) FROM feedback WHERE rating = 1").fetchone()[0]
    thumbs_down = conn.execute("SELECT COUNT(*) FROM feedback WHERE rating = -1").fetchone()[0]
    fallbacks   = conn.execute("SELECT COUNT(*) FROM feedback WHERE fallback = 1").fetchone()[0]
    avg_latency = conn.execute("SELECT AVG(latency_ms) FROM feedback").fetchone()[0] or 0

    # Intent breakdown
    intent_rows = conn.execute(
        "SELECT intent, COUNT(*) as cnt FROM feedback GROUP BY intent ORDER BY cnt DESC"
    ).fetchall()
    intent_counts = {row["intent"]: row["cnt"] for row in intent_rows}

    # Last 10 queries for the dashboard table
    recent_rows = conn.execute(
        """SELECT id, session_id, query, intent, rating,
                  fallback, latency_ms, timestamp
           FROM feedback
           ORDER BY id DESC LIMIT 10"""
    ).fetchall()
    recent = [dict(row) for row in recent_rows]

    conn.close()

    return {
        "total_queries":  total,
        "thumbs_up":      thumbs_up,
        "thumbs_down":    thumbs_down,
        "fallback_count": fallbacks,
        "avg_latency_ms": round(avg_latency or 0, 1),
        "intent_counts":  intent_counts,
        "recent":         recent,
    }


def export_for_ragas(limit: int = 50) -> list[dict]:
    """
    Export logged queries for RAGAS evaluation.
    Returns rows with non-fallback responses only.
    """
    conn = _get_connection()
    rows = conn.execute(
        """SELECT query, rewritten, response, cited_pages
           FROM feedback
           WHERE fallback = 0
           ORDER BY id DESC
           LIMIT ?""",
        (limit,)
    ).fetchall()
    conn.close()

    return [
        {
            "query":       row["query"],
            "rewritten":   row["rewritten"],
            "response":    row["response"],
            "cited_pages": json.loads(row["cited_pages"]),
        }
        for row in rows
    ]


# ── Quick test ────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    init_db()

    # Insert test row
    row_id = log_query(
        session_id  = "test-session",
        query       = "how do I charge my Ather?",
        rewritten   = "What is the procedure for charging an Ather scooter at home?",
        intent      = "charging",
        response    = "According to your Ather manual (page 39)...",
        fallback    = False,
        latency_ms  = 4200,
        cited_pages = [39, 32],
    )
    print(f"Inserted row_id: {row_id}")

    update_rating(row_id, 1)
    print("Rating updated to thumbs up")

    stats = get_analytics()
    print(f"\nAnalytics: {stats}")