"""
Task 04 — Data quality & late-reporting flags

Applies the following transformations in order:
  1. Deduplication — exact duplicates dropped; near-duplicates (same
     composite key, corrected value) resolved by keeping the row with the
     latest lastUpdated timestamp.
  2. Value casting — raw string values cast to the correct Spark type based
     on each data element's valueType from metadata.
  3. DQ boolean flags:
       is_late_reported   — lastUpdated > 60 days after period end date
       is_explicit_zero   — raw value == "0" (meaningfully different from NULL
                            in health reporting)
       is_missing_value   — value is NULL (facility submitted row, no value)
       is_orphaned_coc    — categoryOptionCombo UID not in metadata (already
                            attached by Task 02)
  4. Completeness score — per (facility, period): count of distinct data
     elements reported / count of expected data elements from programs.json

Key invariant: 0 and NULL are NEVER collapsed. A facility reporting zero
clients served is a meaningful datum. A NULL means no value was submitted.
"""

import logging

from pyspark.sql import DataFrame
from pyspark.sql import functions as F
from pyspark.sql.types import DoubleType, LongType
from pyspark.sql.window import Window

logger = logging.getLogger(__name__)

# Composite natural key for deduplication
_DEDUP_KEY = ["dataElement", "period", "orgUnit", "categoryOptionCombo"]


def deduplicate(df: DataFrame) -> DataFrame:
    """
    Remove exact and near-duplicate data value rows.

    Exact duplicates: identical on all fields — keep one arbitrarily.
    Near-duplicates: same composite key, different value / lastUpdated —
      keep the row with the latest lastUpdated (represents a correction).

    A single row_number() window handles both cases: for exact duplicates,
    any row is fine; for near-duplicates, desc ordering on lastUpdated
    ensures the correction survives.
    """
    w = (
        Window
        .partitionBy(*_DEDUP_KEY)
        .orderBy(F.col("lastUpdated").desc())
    )
    deduped = (
        df
        .withColumn("_rn", F.row_number().over(w))
        .filter(F.col("_rn") == 1)
        .drop("_rn")
    )
    n_in  = df.count()
    n_out = deduped.count()
    logger.info("Dedup | in=%d | out=%d | removed=%d", n_in, n_out, n_in - n_out)
    return deduped


def cast_values(df: DataFrame) -> DataFrame:
    """
    Cast raw string values to the appropriate numeric type based on valueType.

    valueType mapping:
      INTEGER_ZERO_OR_POSITIVE  → LongType
      INTEGER / INTEGER_POSITIVE → LongType
      NUMBER                    → DoubleType
      PERCENTAGE                → DoubleType  (0–100 range)
      BOOLEAN                   → DoubleType  (true→1.0, false→0.0)
      TEXT / TRUE_ONLY / other  → kept as string (numeric_value stays null)

    NULL raw values produce NULL numeric_value — never imputed.
    The original raw value string is always preserved.
    """
    numeric_value = (
        F.when(
            F.col("value_type").isin(
                "INTEGER_ZERO_OR_POSITIVE", "INTEGER", "INTEGER_POSITIVE"
            ),
            F.col("value").cast(LongType()),
        )
        .when(
            F.col("value_type").isin("NUMBER", "PERCENTAGE"),
            F.col("value").cast(DoubleType()),
        )
        .when(
            F.col("value_type") == "BOOLEAN",
            F.when(F.lower(F.col("value")) == "true", F.lit(1.0))
             .when(F.lower(F.col("value")) == "false", F.lit(0.0))
             .otherwise(F.lit(None).cast(DoubleType())),
        )
        .otherwise(F.lit(None).cast(DoubleType()))
    )

    return df.withColumn("numeric_value", numeric_value)


def add_dq_flags(df: DataFrame) -> DataFrame:
    """
    Attach boolean DQ flag columns to each data value row.

    Flags:
      is_late_reported  — lastUpdated more than 60 days after period end date.
                          Period end = last day of the reported month.
      is_explicit_zero  — raw value string is exactly "0". Preserved separately
                          from NULL because zero-reporting means the service ran
                          and found no cases, while NULL means no report was filed.
      is_missing_value  — value is NULL (row present, value absent).
      is_orphaned_coc   — already computed in Task 02 join; just ensured here.
    """
    # Period end date: last day of the yyyyMM period
    period_start = F.to_date(F.col("period"), "yyyyMM")
    period_end   = F.date_add(F.add_months(period_start, 1), -1)

    last_updated_date = F.to_date(
        F.col("lastUpdated"), "yyyy-MM-dd'T'HH:mm:ss.SSS"
    )

    flagged = (
        df
        .withColumn(
            "is_late_reported",
            F.datediff(last_updated_date, period_end) > 60,
        )
        .withColumn(
            "is_explicit_zero",
            F.col("value") == F.lit("0"),
        )
        .withColumn(
            "is_missing_value",
            F.col("value").isNull(),
        )
        # is_orphaned_coc should already be present from Task 02;
        # default to False if somehow missing
        .withColumn(
            "is_orphaned_coc",
            F.coalesce(F.col("is_orphaned_coc"), F.lit(False)),
        )
        # Convenience columns for downstream partitioning
        .withColumn("year",       F.col("period").substr(1, 4).cast("int"))
        .withColumn("month",      F.col("period").substr(5, 2).cast("int"))
        .withColumn("year_month", F.col("period"))
        .withColumn(
            "quarter",
            F.concat(
                F.col("period").substr(1, 4),
                F.lit("Q"),
                F.ceil(F.col("period").substr(5, 2).cast("int") / 3).cast("string"),
            ),
        )
        .withColumn("period_end_date", period_end)
    )

    # Log DQ summary
    total = flagged.count()
    n_late = flagged.filter(F.col("is_late_reported")).count()
    n_zero = flagged.filter(F.col("is_explicit_zero")).count()
    n_null = flagged.filter(F.col("is_missing_value")).count()
    n_ococ = flagged.filter(F.col("is_orphaned_coc")).count()

    logger.info(
        "Task04 DQ flags | total=%d | late=%.1f%% | zero=%.1f%% | null=%.1f%% | orphaned_coc=%.1f%%",
        total,
        100 * n_late / max(total, 1),
        100 * n_zero / max(total, 1),
        100 * n_null / max(total, 1),
        100 * n_ococ / max(total, 1),
    )
    return flagged


def compute_completeness(
    df: DataFrame,
    prog_df: DataFrame,
    hierarchy_df: DataFrame,
) -> DataFrame:
    """
    Compute a completeness score per (facility, period).

    Completeness = distinct data elements reported / expected data elements.

    Expected data elements come from programs.json: for each country, the
    union of all data element UIDs across all programs active in that country.

    The score is added as completeness_score (0.0 – 1.0) to every row by
    joining back on (orgUnit, period).
    """
    # Expand programs to (country, expected_de_uid)
    expected_per_country = (
        prog_df
        .select(
            F.col("country"),
            F.explode(F.col("dataElements")).alias("expected_de_uid"),
        )
        .groupBy("country")
        .agg(F.countDistinct("expected_de_uid").alias("expected_count"))
    )

    # Facility → country mapping
    fac_country = hierarchy_df.select(
        F.col("facility_uid"),
        F.col("country_name").alias("fac_country"),
    )

    # Actual distinct DEs reported per (facility, period) — exclude null values
    actual = (
        df
        .filter(F.col("value").isNotNull())
        .groupBy("orgUnit", "period")
        .agg(F.countDistinct("dataElement").alias("actual_count"))
    )

    # Join actual with facility country, then with expected
    actual_with_country = actual.join(
        F.broadcast(fac_country),
        actual["orgUnit"] == fac_country["facility_uid"],
        "left",
    )
    completeness = actual_with_country.join(
        F.broadcast(expected_per_country),
        actual_with_country["fac_country"] == expected_per_country["country"],
        "left",
    ).withColumn(
        "completeness_score",
        F.when(
            F.col("expected_count") > 0,
            F.least(
                F.col("actual_count").cast(DoubleType()) / F.col("expected_count").cast(DoubleType()),
                F.lit(1.0),
            ),
        ).otherwise(F.lit(None).cast(DoubleType())),
    ).select("orgUnit", "period", "completeness_score")

    # Join completeness score back onto main DataFrame
    enriched = df.join(
        completeness,
        on=["orgUnit", "period"],
        how="left",
    )

    avg_completeness = enriched.agg(F.avg("completeness_score")).collect()[0][0]
    logger.info(
        "Task04 completeness | avg_completeness=%.2f%%",
        (avg_completeness or 0) * 100,
    )
    return enriched


def run_dq_pipeline(
    df: DataFrame,
    prog_df: DataFrame,
    hierarchy_df: DataFrame,
) -> DataFrame:
    """
    Run all Task 04 steps in order:
      1. Deduplication
      2. Value casting
      3. DQ flags
      4. Completeness scoring
    """
    df = deduplicate(df)
    df = cast_values(df)
    df = add_dq_flags(df)
    df = compute_completeness(df, prog_df, hierarchy_df)
    return df
