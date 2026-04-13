"""
MedLink Egypt — Orchestrator
=============================
Runs the full pipeline in the correct order:

  Step 1 → medlink_pipeline.py   (CSV → context.json)
  Step 2 → medlink_db.py         (context.json → SQLite)
  Step 3 → notify AI team        (send them context.json path)
  Step 4 → store AI results      (when they send back results)

Usage:
  # Full run (steps 1 + 2)
  python medlink_orchestrator.py --data_dir ./data

  # Only re-load DB from existing pipeline output (step 2 only)
  python medlink_orchestrator.py --db_only --out_dir ./output

  # Store AI results from a file the AI team sent you
  python medlink_orchestrator.py --ai_results_file ai_team_output.json

  # Full run + immediately store AI results
  python medlink_orchestrator.py --data_dir ./data --ai_results_file ai_team_output.json
"""

import argparse
import json
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path

from medlink_db import load_all_patients, write_ai_results_batch, count_records

# =============================================================================
# CONFIG
# =============================================================================
DB_PATH      = "medlink.db"
DEFAULT_OUT  = "./output"
LOG_FILE     = "orchestrator.log"


# =============================================================================
# LOGGING
# =============================================================================
def log(msg: str, level: str = "INFO"):
    ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    line = f"[{ts}] [{level}] {msg}"
    print(line)
    with open(LOG_FILE, "a") as f:
        f.write(line + "\n")


# =============================================================================
# STEP 1 — run the pipeline
# =============================================================================
def run_pipeline(data_dir: str, out_dir: str, sample: int = 0) -> Path:
    log("=" * 50)
    log("STEP 1 — Running medlink_pipeline.py")
    log(f"  data_dir : {data_dir}")
    log(f"  out_dir  : {out_dir}")

    cmd = [
        sys.executable, "medlink_pipeline.py",
        "--data_dir", data_dir,
        "--out_dir",  out_dir,
        "--no_embed",               # skip embeddings for speed 
    ]
    if sample > 0:
        cmd += ["--sample", str(sample)]

    start = time.time()
    result = subprocess.run(cmd, capture_output=False)   # streams output live

    if result.returncode != 0:
        log("Pipeline FAILED", level="ERROR")
        raise RuntimeError("medlink_pipeline.py failed — check output above")

    elapsed = round(time.time() - start, 1)
    log(f"STEP 1 complete in {elapsed}s")

    context_path = Path(out_dir) / "medlink_context.json"
    if not context_path.exists():
        raise FileNotFoundError(f"Pipeline ran but context.json not found at {context_path}")

    return context_path


# =============================================================================
# STEP 2 — load patients into DB
# =============================================================================
def run_db_load(context_path: Path) -> int:
    log("=" * 50)
    log("STEP 2 — Loading patients into DB")
    log(f"  source : {context_path}")
    log(f"  db     : {DB_PATH}")

    start = time.time()
    n = load_all_patients(str(context_path), db_path=DB_PATH)
    elapsed = round(time.time() - start, 1)

    log(f"STEP 2 complete — {n:,} patients loaded in {elapsed}s")
    return n


# =============================================================================
# STEP 3 — notify AI team (customize this to your actual handoff method)
# =============================================================================
def notify_ai_team(context_path: Path):
    log("=" * 50)
    log("STEP 3 — Notifying AI team")

    # ── Option A: the AI team reads from a shared folder ──────────────────
    # Nothing to do — they already have access to context_path
    log(f"  context.json ready at: {context_path.resolve()}")

    # ── Option B: call their API endpoint ─────────────────────────────────
    # import requests
    # requests.post("http://ai-team-api/ingest", json={"path": str(context_path)})

    # ── Option C: copy to shared S3 bucket ────────────────────────────────
    # import boto3
    # boto3.client('s3').upload_file(str(context_path), 'medlink-bucket', 'context.json')

    log("  AI team notified — waiting for their results")
    log("  Run with --ai_results_file when they send back results")


# =============================================================================
# STEP 4 — store AI results
# =============================================================================
def run_ai_results_load(ai_results_file: str) -> int:
    log("=" * 50)
    log("STEP 4 — Storing AI team results")
    log(f"  source : {ai_results_file}")

    path = Path(ai_results_file)
    if not path.exists():
        raise FileNotFoundError(f"AI results file not found: {path}")

    with open(path, encoding="utf-8") as f:
        results = json.load(f)

    # handle both a list and a single result
    if isinstance(results, dict):
        results = [results]

    start = time.time()
    n = write_ai_results_batch(results, db_path=DB_PATH)
    elapsed = round(time.time() - start, 1)

    log(f"STEP 4 complete — {n:,} AI results stored in {elapsed}s")
    return n


# =============================================================================
# SUMMARY
# =============================================================================
def print_summary():
    log("=" * 50)
    log("SUMMARY — DB state")
    stats = count_records(DB_PATH)
    log(f"  patients        : {stats['patients']:,}")
    log(f"  AI results      : {stats['ai_results_total']:,}")
    for model, count in stats["ai_results_by_model"].items():
        log(f"    {model:30s}: {count:,}")
    log("=" * 50)
    log("Orchestration complete — DB is ready for lookups")


# =============================================================================
# MAIN
# =============================================================================
def main():
    parser = argparse.ArgumentParser(description="MedLink Egypt — Orchestrator")
    parser.add_argument("--data_dir",        default="./data",   help="Dir with the 4 CSV files")
    parser.add_argument("--out_dir",         default=DEFAULT_OUT, help="Pipeline output dir")
    parser.add_argument("--db_only",         action="store_true", help="Skip pipeline, only reload DB")
    parser.add_argument("--ai_results_file", metavar="PATH",      help="JSON file of AI results to store")
    parser.add_argument("--sample",          type=int, default=0, help="Process only N patients (0=all)")
    args = parser.parse_args()

    log("MedLink Egypt — Orchestrator starting")
    total_start = time.time()

    try:
        if args.db_only:
            # ── skip pipeline, reload DB from existing output ──
            context_path = Path(args.out_dir) / "medlink_context.json"
            if not context_path.exists():
                raise FileNotFoundError(
                    f"No context.json found at {context_path}\n"
                    f"Run without --db_only first to generate it."
                )
            run_db_load(context_path)

        else:
            # ── full run ──
            context_path = run_pipeline(args.data_dir, args.out_dir, args.sample)
            run_db_load(context_path)
            notify_ai_team(context_path)

        # ── store AI results if provided ──
        if args.ai_results_file:
            run_ai_results_load(args.ai_results_file)

        print_summary()

    except Exception as e:
        log(f"FAILED: {e}", level="ERROR")
        raise

    total = round(time.time() - total_start, 1)
    log(f"Total time: {total}s")


if __name__ == "__main__":
    main()
