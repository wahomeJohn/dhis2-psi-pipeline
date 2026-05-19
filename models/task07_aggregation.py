"""
Task 07 — Cross-country aggregation

Produces four analytical outputs that aggregate across all countries:

  1. volumes_by_quarter        — total service volumes by health area by quarter
  2. completeness_comparison   — completeness % per country per period
  3. coverage_matrix           — which indicators each country reports (pivot)
  4. low_completeness_countries — countries below 80% completeness for 3+
                                  consecutive periods

All outputs are written as CSV to output_dir/cross_country/.
"""

import logging
import os

from pyspark.sql import DataFrame
from pyspark.sql import functions as F
from pyspark.sql.window import Window

logger = logging.getLogger(__name__)


def volumes_by_quarter(fact_df: DataFrame) -> DataFrame:
    """
    Total service volumes by health area and quarter.

    Only numeric (non-null) values are summed. Zero values are included
    because a zero-reported service is a real programme output.
    """
    result = (
        fact_df
        .filter(F.col("numeric_value").isNotNull())
        .groupBy("health_area", "quarter", "year")
        .agg(
            F.sum("numeric_value").alias("total_volume"),
            F.countDistinct("ou_uid").alias("reporting_facilities"),
            F.countDistinct("de_uid").alias("indicators_reported"),
            F.count("*").alias("total_rows"),
        )
        .orderBy("health_area", "year", "quarter")
    )
    logger.info("Volumes by quarter | rows=%d", result.count())
    return result


def completeness_comparison(fact_df: DataFrame) -> DataFrame:
    """
    Cross-country completeness comparison table.

    Per country per period: average completeness score across all facilities.
    Completeness score was computed in Task 04 (count of reported DEs /
    expected DEs for that country).
    """
    result = (
        fact_df
        .groupBy("country_name", "period", "quarter", "year", "year_month")
        .agg(
            F.avg("completeness_score").alias("avg_completeness"),
            F.min("completeness_score").alias("min_completeness"),
            F.max("completeness_score").alias("max_completeness"),
            F.countDistinct("ou_uid").alias("reporting_facilities"),
        )
        .withColumn(
            "avg_completeness_pct",
            F.round(F.col("avg_completeness") * 100, 2),
        )
        .orderBy("country_name", "period")
    )
    logger.info("Completeness comparison | rows=%d", result.count())
    return result


def coverage_matrix(fact_df: DataFrame) -> DataFrame:
    """
    Data element coverage matrix: which indicators each country reports.

    Produces a pivot table where:
      rows   = data element names
      columns = country names
      values = 1 if that country reported this indicator at least once, else 0

    Uses Spark's native pivot() function.
    """
    # Compute indicator × country presence
    presence = (
        fact_df
        .filter(F.col("numeric_value").isNotNull())
        .select("de_uid", "country_name")
        .distinct()
        .withColumn("reported", F.lit(1))
    )

    # Get the list of countries for pivot columns
    countries = [
        row["country_name"]
        for row in presence.select("country_name").distinct().orderBy("country_name").collect()
    ]

    pivoted = (
        presence
        .groupBy("de_uid")
        .pivot("country_name", countries)
        .agg(F.first("reported"))
        .na.fill(0)
    )

    # Join back de_name for readability
    de_names = fact_df.select("de_uid", "health_area").distinct()
    if "de_name" in fact_df.columns:
        de_names = fact_df.select("de_uid", "de_name", "health_area").distinct()
        result = pivoted.join(F.broadcast(de_names), on="de_uid", how="left") \
                        .orderBy("health_area", "de_uid")
    else:
        result = pivoted.join(F.broadcast(de_names), on="de_uid", how="left") \
                        .orderBy("health_area", "de_uid")

    logger.info("Coverage matrix | indicators=%d | countries=%d", result.count(), len(countries))
    return result


def low_completeness_countries(fact_df: DataFrame) -> DataFrame:
    """
    Identify countries consistently below 80% completeness for 3 or more
    consecutive reporting periods.

    Algorithm:
      1. Compute monthly average completeness per country.
      2. Flag periods where completeness < 0.80.
      3. Use a gap-and-island approach: assign a group number to consecutive
         below-threshold sequences using row_number differences.
      4. Count consecutive run lengths and filter for runs >= 3.
    """
    monthly = (
        fact_df
        .groupBy("country_name", "period", "year", "month")
        .agg(F.avg("completeness_score").alias("avg_completeness"))
        .withColumn("below_threshold", F.col("avg_completeness") < 0.80)
        .orderBy("country_name", "period")
    )

    # Assign an island group using the standard row_number gap technique
    w_all   = Window.partitionBy("country_name").orderBy("period")
    w_below = Window.partitionBy("country_name", "below_threshold").orderBy("period")

    islands = (
        monthly
        .withColumn("rn_all",   F.row_number().over(w_all))
        .withColumn("rn_below", F.row_number().over(w_below))
        .withColumn("island_id", F.col("rn_all") - F.col("rn_below"))
    )

    # Count run lengths for below-threshold islands
    run_lengths = (
        islands
        .filter(F.col("below_threshold"))
        .groupBy("country_name", "island_id")
        .agg(
            F.count("*").alias("consecutive_periods"),
            F.min("period").alias("run_start_period"),
            F.max("period").alias("run_end_period"),
            F.avg("avg_completeness").alias("avg_completeness_in_run"),
        )
        .filter(F.col("consecutive_periods") >= 3)
        .orderBy("country_name", "run_start_period")
    )

    logger.info(
        "Low-completeness countries | qualifying_runs=%d",
        run_lengths.count(),
    )
    return run_lengths


def run_aggregation(fact_df: DataFrame, output_dir: str) -> None:
    """
    Run all four cross-country aggregation outputs and write to CSV.
    """
    agg_dir = os.path.join(output_dir, "cross_country")

    outputs = {
        "volumes_by_quarter":        volumes_by_quarter(fact_df),
        "completeness_comparison":   completeness_comparison(fact_df),
        "coverage_matrix":           coverage_matrix(fact_df),
        "low_completeness_countries": low_completeness_countries(fact_df),
    }

    for name, df in outputs.items():
        path = os.path.join(agg_dir, name)
        df.coalesce(1).write.mode("overwrite").option("header", "true").csv(path)
        logger.info("Aggregation '%s' written to %s", name, path)
