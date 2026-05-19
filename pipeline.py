"""
pipeline.py — PSI DISC DHIS2 Health Data Pipeline
==================================================

Orchestrates all 8 tasks in dependency order with stage-level logging and
DQ-check exit codes.

Usage:
    python pipeline.py --data-dir ./data --output-dir ./output

Exit codes:
    0  — success
    1  — critical DQ check failed (quarantine rate > 10%, or 0 rows in fact table)
    2  — unhandled exception during pipeline execution
"""

import argparse
import logging
import os
import sys
import time
from datetime import datetime

from pyspark.sql import SparkSession

# ── Task imports ──────────────────────────────────────────────────────────────
from models.task01_ingest      import ingest_all
from models.task02_uid_resolve import resolve_uids
from models.task03_hierarchy   import build_facility_hierarchy, enrich_with_hierarchy
from models.task04_dq_flags    import run_dq_pipeline
from models.task05_dim_model   import build_fact_service_delivery, write_dims
from models.task06_analytics   import run_analytics
from models.task07_aggregation import run_aggregation

# ── Bonus imports (optional) ──────────────────────────────────────────────────
try:
    from models.task_b1_contract   import validate_and_raise, ContractViolationError
    from models.task_b2_incremental import load_incremental
    from models.task_b3_anomaly    import detect_anomalies
    _BONUS_AVAILABLE = True
except ImportError:
    _BONUS_AVAILABLE = False


# ── Logging configuration ─────────────────────────────────────────────────────

def configure_logging(output_dir: str) -> None:
    os.makedirs(output_dir, exist_ok=True)
    log_path = os.path.join(output_dir, "pipeline.log")
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s  %(levelname)-8s  %(name)s  %(message)s",
        handlers=[
            logging.StreamHandler(sys.stdout),
            logging.FileHandler(log_path, mode="w", encoding="utf-8"),
        ],
    )


logger = logging.getLogger("pipeline")


# ── SparkSession ──────────────────────────────────────────────────────────────

def create_spark_session() -> SparkSession:
    return (
        SparkSession.builder
        .master("local[*]")
        .appName("PSI-DISC-DHIS2-Pipeline")
        # Reduce shuffle partitions for local mode (default 200 is wasteful)
        .config("spark.sql.shuffle.partitions", "8")
        # Allow schema merging when reading partitioned Parquet
        .config("spark.sql.parquet.mergeSchema", "true")
        # Disable noisy Spark UI in batch mode
        .config("spark.ui.enabled", "false")
        .getOrCreate()
    )


# ── Stage runner ──────────────────────────────────────────────────────────────

def run_stage(name: str, fn, *args, **kwargs):
    """Run a pipeline stage, log timing, and propagate exceptions."""
    logger.info("=" * 60)
    logger.info("STAGE START: %s", name)
    t0 = time.time()
    try:
        result = fn(*args, **kwargs)
        elapsed = time.time() - t0
        logger.info("STAGE DONE:  %s  (%.1f s)", name, elapsed)
        logger.info("=" * 60)
        return result
    except Exception as exc:
        elapsed = time.time() - t0
        logger.error("STAGE FAILED: %s (%.1f s): %s", name, elapsed, exc, exc_info=True)
        raise


# ── CLI ───────────────────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="PSI DISC DHIS2 Health Data Pipeline",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--data-dir",   required=True,        help="Directory containing the four DHIS2 JSON files")
    p.add_argument("--output-dir", required=True,        help="Directory for all pipeline outputs")
    p.add_argument("--skip-bonus", action="store_true",  help="Skip optional bonus tasks (B1–B3)")
    p.add_argument("--incremental", action="store_true", help="Only process periods not already in the output")
    return p.parse_args()


# ── Critical DQ checks ────────────────────────────────────────────────────────

def critical_dq_checks(quarantine_rate: float, fact_row_count: int) -> list[str]:
    """
    Return a list of failed critical DQ check descriptions.
    An empty list means all checks passed.
    """
    failures = []
    if quarantine_rate > 0.10:
        failures.append(
            f"Quarantine rate {quarantine_rate:.1%} exceeds 10% threshold — "
            f"data source may be fundamentally broken"
        )
    if fact_row_count == 0:
        failures.append("Fact table contains 0 rows — pipeline produced no output")
    return failures


# ── Main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    args = parse_args()
    configure_logging(args.output_dir)

    logger.info("PSI DISC · DHIS2 Health Data Pipeline")
    logger.info("Started at: %s", datetime.now().isoformat())
    logger.info("data_dir=%s  output_dir=%s", args.data_dir, args.output_dir)

    spark = create_spark_session()
    os.makedirs(args.output_dir, exist_ok=True)

    pipeline_start = time.time()

    try:
        # ── Task 01: JSON ingestion & schema flattening ────────────────────
        de_df, coc_df, ou_df, prog_df, dv_df, quarantine_rate = run_stage(
            "01-ingest",
            ingest_all,
            spark, args.data_dir, args.output_dir,
        )

        # ── Bonus B2: Incremental load filter ─────────────────────────────
        if args.incremental and _BONUS_AVAILABLE:
            dv_df, is_incremental = run_stage(
                "B2-incremental",
                load_incremental,
                spark, args.output_dir, dv_df,
            )
            if dv_df.rdd.isEmpty():
                logger.info("Incremental: no new periods to process. Exiting.")
                spark.stop()
                sys.exit(0)

        # ── Task 02: Metadata UID resolution ──────────────────────────────
        resolved_df, unresolvable_df = run_stage(
            "02-uid-resolve",
            resolve_uids,
            dv_df, de_df, coc_df, ou_df, args.output_dir,
        )

        # ── Task 03: Org unit hierarchy resolution ────────────────────────
        hierarchy_df = run_stage(
            "03-hierarchy-build",
            build_facility_hierarchy,
            ou_df,
        )
        enriched_df = run_stage(
            "03-hierarchy-enrich",
            enrich_with_hierarchy,
            resolved_df, hierarchy_df,
        )

        # ── Task 04: DQ flags, dedup, completeness ────────────────────────
        flagged_df = run_stage(
            "04-dq-flags",
            run_dq_pipeline,
            enriched_df, prog_df, hierarchy_df,
        )

        # ── Task 05: Dimensional model build ──────────────────────────────
        dims = run_stage(
            "05-dims",
            write_dims,
            de_df, hierarchy_df, flagged_df, prog_df, args.output_dir,
        )

        # ── Bonus B1: Contract validation before fact write ────────────────
        if not args.skip_bonus and _BONUS_AVAILABLE:
            contract_path = os.path.join(
                os.path.dirname(__file__), "contracts", "fact_service_delivery.yaml"
            )
            if os.path.exists(contract_path):
                try:
                    run_stage(
                        "B1-contract-validate",
                        validate_and_raise,
                        flagged_df, contract_path,
                    )
                except ContractViolationError as e:
                    logger.error("B1 contract validation FAILED:\n%s", e)
                    # Contract failure is logged but not fatal — the data
                    # may still be useful for programme review.

        fact_df = run_stage(
            "05-fact",
            build_fact_service_delivery,
            flagged_df, prog_df, args.output_dir,
        )

        # ── Critical DQ gate ──────────────────────────────────────────────
        fact_row_count = fact_df.count()
        failures = critical_dq_checks(quarantine_rate, fact_row_count)
        if failures:
            for msg in failures:
                logger.error("CRITICAL DQ CHECK FAILED: %s", msg)
            logger.error("Pipeline exiting with code 1 due to critical DQ failures.")
            spark.stop()
            sys.exit(1)

        # ── Task 06: Program analytics & window functions ─────────────────
        run_stage(
            "06-analytics",
            run_analytics,
            fact_df, hierarchy_df, args.output_dir,
        )

        # ── Task 07: Cross-country aggregation ────────────────────────────
        run_stage(
            "07-aggregation",
            run_aggregation,
            fact_df, args.output_dir,
        )

        # ── Bonus B3: Anomaly detection ────────────────────────────────────
        if not args.skip_bonus and _BONUS_AVAILABLE:
            run_stage(
                "B3-anomaly-detection",
                detect_anomalies,
                fact_df, args.output_dir,
            )

    except Exception as exc:
        logger.error("Pipeline failed with unhandled exception: %s", exc, exc_info=True)
        spark.stop()
        sys.exit(2)

    # ── Final summary ─────────────────────────────────────────────────────
    elapsed = time.time() - pipeline_start
    logger.info("")
    logger.info("=" * 60)
    logger.info("PIPELINE COMPLETE")
    logger.info("  Total time:     %.1f s (%.1f min)", elapsed, elapsed / 60)
    logger.info("  Output dir:     %s", os.path.abspath(args.output_dir))
    logger.info("  Quarantine rate: %.2f%%", quarantine_rate * 100)
    logger.info("  Fact rows:      %d", fact_row_count)
    logger.info("=" * 60)

    spark.stop()
    sys.exit(0)


if __name__ == "__main__":
    main()
