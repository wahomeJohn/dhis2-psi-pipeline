"""
Bonus B3 — Anomaly detection

Flags facilities where any indicator value is more than 3 standard deviations
from that facility's own 12-month rolling mean. These are genuine statistical
outliers that warrant programme review — they may indicate data entry errors,
supply chain disruptions, or real epidemiological events.

Output: anomalies.csv written to output_dir/anomalies/

Algorithm:
  1. Compute a 12-month rolling mean and standard deviation per
     (facility, data element) using Spark window functions.
  2. Flag rows where |value - rolling_mean| > 3 * rolling_stddev.
  3. Only non-null, non-zero numeric values are considered.
  4. A minimum window size of 3 periods is required before flagging —
     too few observations makes sigma estimates unreliable.
"""

import logging
import os

from pyspark.sql import DataFrame
from pyspark.sql import functions as F
from pyspark.sql.window import Window

logger = logging.getLogger(__name__)

_SIGMA_THRESHOLD = 3.0
_MIN_WINDOW_ROWS = 3   # minimum periods needed to compute a reliable sigma
_ROLLING_MONTHS  = 12  # look-back window in months


def detect_anomalies(fact_df: DataFrame, output_dir: str) -> DataFrame:
    """
    Detect statistical anomalies in the fact table and write them to CSV.

    A data point is anomalous if:
      numeric_value > (rolling_mean + 3 * rolling_stddev)
      OR
      numeric_value < (rolling_mean - 3 * rolling_stddev)

    where rolling_mean and rolling_stddev are computed over the preceding
    12 months for that (facility, data element) pair.

    Returns the anomalies DataFrame.
    """
    # Work only with meaningful numeric data
    df = fact_df.filter(
        F.col("numeric_value").isNotNull()
        & (F.col("numeric_value") > 0)
        & ~F.col("is_missing_value")
    )

    # 12-month rolling window (preceding 11 rows + current = 12 months)
    # rowsBetween handles gaps in reporting (missing periods don't shift the window)
    w = (
        Window
        .partitionBy("ou_uid", "de_uid")
        .orderBy("period")
        .rowsBetween(-(_ROLLING_MONTHS - 1), 0)
    )

    # Compute rolling statistics
    with_stats = (
        df
        .withColumn("rolling_mean",   F.avg("numeric_value").over(w))
        .withColumn("rolling_stddev", F.stddev_pop("numeric_value").over(w))
        .withColumn("window_count",   F.count("numeric_value").over(w))
    )

    # Flag anomalies — only where we have enough history
    anomalies = (
        with_stats
        .filter(F.col("window_count") >= _MIN_WINDOW_ROWS)
        .filter(F.col("rolling_stddev").isNotNull() & (F.col("rolling_stddev") > 0))
        .withColumn(
            "z_score",
            F.abs(F.col("numeric_value") - F.col("rolling_mean"))
            / F.col("rolling_stddev"),
        )
        .filter(F.col("z_score") > _SIGMA_THRESHOLD)
        .withColumn("z_score_rounded", F.round(F.col("z_score"), 2))
        .withColumn("rolling_mean_rounded", F.round(F.col("rolling_mean"), 4))
        .withColumn("rolling_stddev_rounded", F.round(F.col("rolling_stddev"), 4))
        .select(
            "ou_uid",
            "facility_name",
            "country_name",
            "district_name",
            "de_uid",
            "health_area",
            "period",
            "numeric_value",
            F.col("rolling_mean_rounded").alias("rolling_mean"),
            F.col("rolling_stddev_rounded").alias("rolling_stddev"),
            F.col("z_score_rounded").alias("z_score"),
            "window_count",
        )
        .orderBy(F.col("z_score_rounded").desc())
    )

    n_anomalies = anomalies.count()
    logger.info(
        "Bonus B3: anomaly detection | threshold=%.0f sigma | anomalies=%d",
        _SIGMA_THRESHOLD, n_anomalies,
    )

    # Write output
    out_path = os.path.join(output_dir, "anomalies")
    anomalies.coalesce(1).write.mode("overwrite").option("header", "true").csv(out_path)
    logger.info("Anomalies written to %s", out_path)

    return anomalies
