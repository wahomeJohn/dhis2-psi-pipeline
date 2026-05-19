"""
Bonus B2 — Incremental load logic

Detects which (health_area, year_month) partition combinations already exist
in the Parquet output and filters the incoming data values DataFrame to only
include periods that have not yet been loaded.

This enables safe re-runs: re-running the pipeline will only process and
append new periods, not re-process partitions that are already present.

Usage:
  existing = get_existing_partitions(output_dir)
  dv_df_new = filter_new_periods(dv_df, existing)
  # ... run the rest of the pipeline on dv_df_new ...
"""

import logging
import os

from pyspark.sql import DataFrame, SparkSession
from pyspark.sql import functions as F

logger = logging.getLogger(__name__)


def get_existing_partitions(output_dir: str) -> set[str]:
    """
    Inspect the fact_service_delivery Parquet directory and return the set of
    year_month values already written.

    Reads partition directory names directly from the filesystem — avoids
    loading the full Parquet data just to check what's present.

    Returns a set of year_month strings (e.g. {"202401", "202402"}).
    """
    fact_path = os.path.join(output_dir, "fact_service_delivery")
    if not os.path.exists(fact_path):
        logger.info("Incremental check: no existing output at %s — full load", fact_path)
        return set()

    existing: set[str] = set()
    for entry in os.scandir(fact_path):
        if entry.is_dir() and entry.name.startswith("health_area="):
            for sub in os.scandir(entry.path):
                if sub.is_dir() and sub.name.startswith("year_month="):
                    ym = sub.name.split("=", 1)[1]
                    existing.add(ym)

    logger.info(
        "Incremental check: found %d existing year_month partitions",
        len(existing),
    )
    return existing


def filter_new_periods(
    df: DataFrame,
    existing_partitions: set[str],
) -> DataFrame:
    """
    Filter a data values DataFrame to only include rows whose period
    (year_month) is NOT already in the existing Parquet output.

    If existing_partitions is empty, returns df unchanged (full load).
    """
    if not existing_partitions:
        logger.info("Incremental: no existing partitions — processing all periods")
        return df

    existing_list = sorted(existing_partitions)
    logger.info(
        "Incremental: skipping %d already-loaded periods: %s ... %s",
        len(existing_list),
        existing_list[0],
        existing_list[-1],
    )

    # Filter out rows belonging to already-loaded periods
    filtered = df.filter(~F.col("period").isin(existing_list))

    new_periods = [
        row["period"]
        for row in filtered.select("period").distinct().orderBy("period").collect()
    ]
    logger.info(
        "Incremental: %d new periods to process: %s",
        len(new_periods),
        new_periods,
    )
    return filtered


def load_incremental(
    spark: SparkSession,
    output_dir: str,
    dv_df: DataFrame,
) -> tuple[DataFrame, bool]:
    """
    Convenience wrapper: determine existing partitions and return the filtered
    DataFrame plus a flag indicating whether this is a full or incremental load.

    Returns:
        (filtered_dv_df, is_incremental)
    """
    existing = get_existing_partitions(output_dir)
    is_incremental = len(existing) > 0
    filtered = filter_new_periods(dv_df, existing)
    return filtered, is_incremental
