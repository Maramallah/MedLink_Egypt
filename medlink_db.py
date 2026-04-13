"""
MedLink Egypt — Database Layer
===============================
Three jobs:
  1. write_patient(record)       → store one patient from context.json
  2. write_ai_result(result)     → store AI team's result for a patient
  3. get_patient(national_id)    → look up a patient, get record + AI results merged

Storage: SQLite by default (demo-ready, zero setup, one file).
Switch to PostgreSQL for production — just change DB_URL below.

Run standalone to load all patients from context.json:
  python medlink_db.py --load output/medlink_context.json

Simulate an AI result being received:
  python medlink_db.py --ai_result '{"national_id":"3640201001","model_name":"risk_v1","prediction":{"label":"High","action":"refer_cardiologist"},"confidence":0.87}'

Look up a patient:
  python medlink_db.py --lookup 3640201001
"""

import argparse
import json
import sqlite3
from datetime import datetime
from pathlib import Path

# =============================================================================
# CONFIG
# =============================================================================
DEFAULT_DB = "medlink.db"   # SQLite file path — change to full path if needed


# =============================================================================
# 1. SETUP — create tables if they don't exist
# =============================================================================
SCHEMA = """
CREATE TABLE IF NOT EXISTS patient_records (
    national_id     TEXT PRIMARY KEY,
    patient_id      INTEGER,
    demographics    TEXT,       -- JSON blob
    diagnoses       TEXT,       -- JSON blob
    risk            TEXT,       -- JSON blob
    latest_labs     TEXT,       -- JSON blob
    abnormal_flags  TEXT,       -- JSON array
    narrative       TEXT,       -- pre-built clinical text for LLM
    inserted_at     TEXT DEFAULT (datetime('now'))
);

CREATE TABLE IF NOT EXISTS ai_results (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    national_id     TEXT NOT NULL,
    model_name      TEXT NOT NULL,
    prediction      TEXT,       -- JSON blob (flexible — AI team can put anything here)
    confidence      REAL,
    created_at      TEXT DEFAULT (datetime('now')),
    FOREIGN KEY (national_id) REFERENCES patient_records(national_id)
);

CREATE INDEX IF NOT EXISTS idx_ai_national_id ON ai_results(national_id);
CREATE INDEX IF NOT EXISTS idx_ai_model       ON ai_results(model_name);
"""


def get_connection(db_path: str = DEFAULT_DB) -> sqlite3.Connection:
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row          # lets you access columns by name
    conn.execute("PRAGMA journal_mode=WAL") # safe for concurrent reads
    conn.executescript(SCHEMA)
    conn.commit()
    return conn


# =============================================================================
# 2. WRITE PATIENT
# =============================================================================
def write_patient(record: dict, db_path: str = DEFAULT_DB) -> None:
    """
    Store one patient record from context.json.
    If the patient already exists (same national_id), update their data.

    Args:
        record   : one dict from medlink_context.json
        db_path  : path to the SQLite file
    """
    conn = get_connection(db_path)
    conn.execute("""
        INSERT INTO patient_records
            (national_id, patient_id, demographics, diagnoses, risk,
             latest_labs, abnormal_flags, narrative)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?)
        ON CONFLICT(national_id) DO UPDATE SET
            patient_id     = excluded.patient_id,
            demographics   = excluded.demographics,
            diagnoses      = excluded.diagnoses,
            risk           = excluded.risk,
            latest_labs    = excluded.latest_labs,
            abnormal_flags = excluded.abnormal_flags,
            narrative      = excluded.narrative,
            inserted_at    = datetime('now')
    """, (
        str(record["national_id"]),
        record.get("patient_id"),
        json.dumps(record.get("demographics", {})),
        json.dumps(record.get("diagnoses", {})),
        json.dumps(record.get("risk", {})),
        json.dumps(record.get("latest_labs", {})),
        json.dumps(record.get("abnormal_flags", [])),
        record.get("narrative", ""),
    ))
    conn.commit()
    conn.close()


def load_all_patients(context_json_path: str, db_path: str = DEFAULT_DB) -> int:
    """
    Bulk-load all patients from context.json into the database.
    Returns number of patients written.
    """
    path = Path(context_json_path)
    if not path.exists():
        raise FileNotFoundError(f"Not found: {path}")

    with open(path, encoding="utf-8") as f:
        records = json.load(f)

    conn = get_connection(db_path)
    rows = []
    for record in records:
        rows.append((
            str(record["national_id"]),
            record.get("patient_id"),
            json.dumps(record.get("demographics", {})),
            json.dumps(record.get("diagnoses", {})),
            json.dumps(record.get("risk", {})),
            json.dumps(record.get("latest_labs", {})),
            json.dumps(record.get("abnormal_flags", [])),
            record.get("narrative", ""),
        ))

    conn.executemany("""
        INSERT INTO patient_records
            (national_id, patient_id, demographics, diagnoses, risk,
             latest_labs, abnormal_flags, narrative)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?)
        ON CONFLICT(national_id) DO UPDATE SET
            patient_id     = excluded.patient_id,
            demographics   = excluded.demographics,
            diagnoses      = excluded.diagnoses,
            risk           = excluded.risk,
            latest_labs    = excluded.latest_labs,
            abnormal_flags = excluded.abnormal_flags,
            narrative      = excluded.narrative,
            inserted_at    = datetime('now')
    """, rows)
    conn.commit()
    conn.close()

    print(f"  ✔ Loaded {len(rows):,} patients into {db_path}")
    return len(rows)


# =============================================================================
# 3. WRITE AI RESULT
# =============================================================================
def write_ai_result(result: dict, db_path: str = DEFAULT_DB) -> None:
    """
    Store one AI model result for a patient.
    The AI team calls this (or you call it when they send you data).

    Expected result format:
    {
        "national_id":  "3640201001",
        "model_name":   "risk_v1",              # which model produced this
        "prediction":   { ...anything... },     # flexible JSON — AI team decides
        "confidence":   0.87                    # optional float 0-1
    }
    """
    required = ["national_id", "model_name", "prediction"]
    for field in required:
        if field not in result:
            raise ValueError(f"AI result missing required field: '{field}'")

    conn = get_connection(db_path)
    conn.execute("""
        INSERT INTO ai_results (national_id, model_name, prediction, confidence)
        VALUES (?, ?, ?, ?)
    """, (
        str(result["national_id"]),
        result["model_name"],
        json.dumps(result["prediction"]),
        result.get("confidence"),
    ))
    conn.commit()
    conn.close()


def write_ai_results_batch(results: list[dict], db_path: str = DEFAULT_DB) -> int:
    """
    Store a batch of AI results at once (when AI team sends a bulk file).
    Returns number of results written.
    """
    conn = get_connection(db_path)
    rows = []
    for result in results:
        rows.append((
            str(result["national_id"]),
            result["model_name"],
            json.dumps(result["prediction"]),
            result.get("confidence"),
        ))
    conn.executemany("""
        INSERT INTO ai_results (national_id, model_name, prediction, confidence)
        VALUES (?, ?, ?, ?)
    """, rows)
    conn.commit()
    conn.close()
    print(f"  ✔ Stored {len(rows):,} AI results into {db_path}")
    return len(rows)


# =============================================================================
# 4. GET PATIENT  (the main lookup — joins both tables)
# =============================================================================
def get_patient(national_id: str, db_path: str = DEFAULT_DB) -> dict | None:
    """
    Look up a patient by national ID.
    Returns a merged dict with patient record + all AI results.
    Returns None if patient not found.

    Returned shape:
    {
        "national_id":    "...",
        "patient_id":     42,
        "demographics":   { ... },
        "diagnoses":      { ... },
        "risk":           { ... },
        "latest_labs":    { ... },
        "abnormal_flags": [ ... ],
        "narrative":      "...",
        "inserted_at":    "...",
        "ai_results": [
            {
                "model_name":  "risk_v1",
                "prediction":  { ... },
                "confidence":  0.87,
                "created_at":  "..."
            },
            ...
        ]
    }
    """
    conn = get_connection(db_path)

    # ── fetch patient record ──
    row = conn.execute("""
        SELECT * FROM patient_records WHERE national_id = ?
    """, (str(national_id),)).fetchone()

    if row is None:
        conn.close()
        return None

    # ── deserialize JSON fields ──
    patient = {
        "national_id":    row["national_id"],
        "patient_id":     row["patient_id"],
        "demographics":   json.loads(row["demographics"]),
        "diagnoses":      json.loads(row["diagnoses"]),
        "risk":           json.loads(row["risk"]),
        "latest_labs":    json.loads(row["latest_labs"]),
        "abnormal_flags": json.loads(row["abnormal_flags"]),
        "narrative":      row["narrative"],
        "inserted_at":    row["inserted_at"],
    }

    # ── fetch all AI results for this patient ──
    ai_rows = conn.execute("""
        SELECT model_name, prediction, confidence, created_at
        FROM ai_results
        WHERE national_id = ?
        ORDER BY created_at DESC
    """, (str(national_id),)).fetchall()

    patient["ai_results"] = [
        {
            "model_name": r["model_name"],
            "prediction": json.loads(r["prediction"]),
            "confidence": r["confidence"],
            "created_at": r["created_at"],
        }
        for r in ai_rows
    ]

    conn.close()
    return patient


# =============================================================================
# 5. UTILS
# =============================================================================
def count_records(db_path: str = DEFAULT_DB) -> dict:
    """Quick stats on what's in the DB."""
    conn = get_connection(db_path)
    patients = conn.execute("SELECT COUNT(*) FROM patient_records").fetchone()[0]
    ai_total = conn.execute("SELECT COUNT(*) FROM ai_results").fetchone()[0]
    ai_models = conn.execute(
        "SELECT model_name, COUNT(*) as n FROM ai_results GROUP BY model_name"
    ).fetchall()
    conn.close()
    return {
        "patients": patients,
        "ai_results_total": ai_total,
        "ai_results_by_model": {r["model_name"]: r["n"] for r in ai_models},
    }


def patient_exists(national_id: str, db_path: str = DEFAULT_DB) -> bool:
    conn = get_connection(db_path)
    row = conn.execute(
        "SELECT 1 FROM patient_records WHERE national_id = ?", (str(national_id),)
    ).fetchone()
    conn.close()
    return row is not None


# =============================================================================
# 6. CLI  (for quick testing)
# =============================================================================
def main():
    parser = argparse.ArgumentParser(description="MedLink Egypt — DB layer")
    parser.add_argument("--db",        default=DEFAULT_DB,  help="SQLite DB path")
    parser.add_argument("--load",      metavar="PATH",      help="Load all patients from context.json")
    parser.add_argument("--lookup",    metavar="NID",       help="Look up patient by national ID")
    parser.add_argument("--ai_result", metavar="JSON",      help="Store an AI result (JSON string)")
    parser.add_argument("--stats",     action="store_true", help="Show DB stats")
    args = parser.parse_args()

    if args.load:
        print(f"\nLoading patients from {args.load} ...")
        n = load_all_patients(args.load, db_path=args.db)
        print(f"Done. {n:,} patients stored in {args.db}")

    if args.ai_result:
        result = json.loads(args.ai_result)
        write_ai_result(result, db_path=args.db)
        print(f"  ✔ AI result stored for national_id={result['national_id']}")

    if args.lookup:
        patient = get_patient(args.lookup, db_path=args.db)
        if patient is None:
            print(f"  ✘ Patient not found: {args.lookup}")
        else:
            # print cleanly — hide the full narrative for readability
            display = {k: v for k, v in patient.items() if k != "narrative"}
            display["narrative"] = patient["narrative"][:120] + "..."
            print(json.dumps(display, indent=2, ensure_ascii=False))

    if args.stats:
        stats = count_records(db_path=args.db)
        print(json.dumps(stats, indent=2))


if __name__ == "__main__":
    main()
