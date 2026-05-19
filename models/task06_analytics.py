"""
Task 06 — Program analytics & window functions

Computes four analytical outputs using native PySpark window functions:

  1. mom_pct_change        — month-over-month % change per indicator per district
  2. rolling_avg_3m        — 3-month rolling average per facility per indicator
  3. country_reporting_rate — % of expected facilities that submitted data per period
  4. top5_underreporting   — top-5 facilities per health area with most zero-data periods

All outputs are written as CSV (one file each) to output_dir/analytics/.
No Python UDFs are used — all logic uses native F.* functions and Window specs.
"""

import logging
import os

from pyspark.sql import DataFrame
from pyspark.sql import functions as F
from pyspark.sql.window import Window

logger = logging.getLogger(__name__)


def mom_pct_change(fact_df: DataFrame) -> DataFrame:
    """
    Month-over-month percentage change in total service volume per indicator
    per district.

    Aggregates numeric_value to district level first (sum across facilities),
    then computes MoM % change using lag() over a period-ordered window.

    Null numeric_value rows (missing values) are excluded from the sum to
    avoid suppressing genuine zero-volume months.
    """
    # Aggregate to district / indicator / period level
    agg = (
        fact_df
        .filter(F.col("numeric_value").isNotNull())
        .groupBy("de_uid", "health_area", "district_name", "country_name", "period", "year_month")
        .agg(
            F.sum("numeric_value").alias("total_value"),
            F.first("de_name").alias("de_name"),
        )
    )

    # Window: partition by indicator + district, order by period string
    # yyyyMM strings sort correctly lexicographically
    w = Window.partitionBy("de_uid", "district_name").orderBy("period")

    result = (
        agg
        .withColumn("prev_total", F.lag("total_value", 1).over(w))
        .withColumn(
            "mom_pct_change",
            F.when(
                F.col("prev_total").isNotNull() & (F.col("prev_total") != 0),
                F.round(
                    (F.col("total_value") - F.col("prev_total")) / F.col("prev_total") * 100,
                    2,
                ),
            ).otherwise(F.lit(None).cast("double")),
        )
        .orderBy("de_uid", "district_name", "period")
    )
    logger.info("MoM change computed | rows=%d", result.count())
    return result


def rolling_avg_3m(fact_df: DataFrame) -> DataFrame:
    """
    3-month rolling average of numeric_value per facility per data element.

    Uses rowsBetween(-2, 0): the current row and the 2 preceding rows in
    period order. This is the standard approach for rolling windows in Spark
    when the data may have gaps (missing periods). An alternative using
    rangeBetween would require converting period to integer — rowsBetween
    is simpler and sufficient here.
    """
    w = (
        Window
        .partitionBy("ou_uid", "de_uid")
        .orderBy("period")
        .rowsBetween(-2, 0)
    )

    result = (
        fact_df
        .withColumn(
            "rolling_avg_3m",
            F.round(F.avg("numeric_value").over(w), 4),
        )
        .select(
            "de_uid", "ou_uid", "period", "health_area",
            "country_name", "district_name", "facility_name",
            "numeric_value", "rolling_avg_3m",
        )
    )
    logger.info("Rolling avg computed | rows=%d", result.count())
    return result


def country_reporting_rate(
    fact_df: DataFrame,
    facility_hierarchy_df: DataFrame,
) -> DataFrame:
    """
    Country-level reporting rate: percentage of expected facilities that
    submitted at least one non-null data value for each period.

    Expected facilities = all level-4 facilities in org_units for that country.
    Actual = distinct ou_uid values present in fact_df for that period and country.
    """
    # Total expected facilities per country
    expected = (
        facility_hierarchy_df
        .groupBy("country_name")
        .agg(F.countDistinct("facility_uid").alias("expected_facilities"))
    )

    # Actual reporting facilities per country per period
    actual = (
        fact_df
        .filter(F.col("numeric_value").isNotNull() | F.col("is_explicit_zero"))
        .groupBy("country_name", "period", "year_month", "quarter", "year", "month")
        .agg(F.countDistinct("ou_uid").alias("actual_facilities"))
    )

    result = (
        actual
        .join(F.broadcast(expected), on="country_name", how="left")
        .withColumn(
            "reporting_rate_pct",
            F.round(
                F.col("actual_facilities").cast("double")
                / F.col("expected_facilities").cast("double") * 100,
                2,
            ),
        )
        .orderBy("country_name", "period")
    )
    logger.info("Reporting rate computed | rows=%d", result.count())
    return result


def top5_underreporting_facilities(fact_df: DataFrame) -> DataFrame:
    """
    Top-5 underreporting facilities per health area, ranked by the number of
    reporting periods in which the facility submitted only zero-value or
    missing-value data (i.e., no positive clinical activity was recorded).

    Rank 1 = most underreporting within the health area.
    """
    # Per facility per health area per period: is every value zero or missing?
    per_period = (
        fact_df
        .groupBy("ou_uid", "facility_name", "country_name", "health_area", "period")
        .agg(
            F.sum(
                F.when(
                    F.col("numeric_value").isNull() | (F.col("numeric_value") == 0),
                    F.lit(1),
                ).otherwise(F.lit(0))
            ).alias("zero_or_null_rows"),
            F.count("*").alias("total_rows"),
        )
        .withColumn(
            "all_zero_or_null",
            F.col("zero_or_null_rows") == F.col("total_rows"),
        )
    )

    # Count how many periods each facility had all-zero/null data
    underreporting = (
        per_period
        .filter(F.col("all_zero_or_null"))
        .groupBy("ou_uid", "facility_name", "country_name", "health_area")
        .agg(F.count("*").alias("zero_periods"))
    )

    # Rank within each health area
    w = Window.partitionBy("health_area").orderBy(F.col("zero_periods").desc())
    result = (
        underreporting
        .withColumn("rank_in_health_area", F.rank().over(w))
        .filter(F.col("rank_in_health_area") <= 5)
        .orderBy("health_area", "rank_in_health_area")
    )
    logger.info("Top-5 underreporting computed | rows=%d", result.count())
    return result


def run_analytics(
    fact_df: DataFrame,
    facility_hierarchy_df: DataFrame,
    output_dir: str,
) -> None:
    """
    Run all four analytics outputs and write results to CSV.
    """
    analytics_dir = os.path.join(output_dir, "analytics")

    outputs = {
        "mom_pct_change":    mom_pct_change(fact_df),
        "rolling_avg_3m":    rolling_avg_3m(fact_df),
        "reporting_rate":    country_reporting_rate(fact_df, facility_hierarchy_df),
        "top5_underreporting": top5_underreporting_facilities(fact_df),
    }

    for name, df in outputs.items():
        path = os.path.join(analytics_dir, name)
        df.coalesce(1).write.mode("overwrite").option("header", "true").csv(path)
        logger.info("Analytics '%s' written to %s", name, path)
