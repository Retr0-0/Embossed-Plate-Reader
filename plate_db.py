import sqlite3
import datetime
import os

DB_PATH = "plates.db"


def _connect():
    conn = sqlite3.connect(DB_PATH)
    conn.execute("""
        CREATE TABLE IF NOT EXISTS plates (
            id              INTEGER PRIMARY KEY AUTOINCREMENT,
            timestamp       TEXT,
            source_image    TEXT,
            header          TEXT,
            row1            TEXT,
            row2            TEXT,
            plate           TEXT,
            full_number     TEXT,
            annotated_image TEXT
        )
    """)
    return conn


def save_reading(source_image, info, annotated_image):
    """info is the dict returned by read_plate: {"header", "number", "plate"}."""
    rows = info.get("number", []) or []
    row1 = rows[0] if len(rows) >= 1 else "-"
    row2 = rows[1] if len(rows) >= 2 else "-"          # '-' when there's no second row
    full_number = "".join(rows)

    conn = _connect()
    conn.execute(
        """INSERT INTO plates
           (timestamp, source_image, header, row1, row2, plate, full_number, annotated_image)
           VALUES (?, ?, ?, ?, ?, ?, ?, ?)""",
        (datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
         source_image, info.get("header", ""), row1, row2,
         info.get("plate", ""), full_number, annotated_image),
    )
    conn.commit()
    conn.close()


def show_all():
    """Print everything saved so far (quick way to view the database)."""
    conn = _connect()
    cur = conn.execute("SELECT id, timestamp, source_image, header, row1, row2, "
                       "full_number, annotated_image FROM plates ORDER BY id")
    for r in cur.fetchall():
        print(r)
    conn.close()