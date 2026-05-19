"""
Task 05 — Dimensional model build

Populates a star schema consisting of:
  fact_service_delivery  — one row per (data element × period × facility),
                           partitioned by health_area and year_month
  dim_data_element       — data element attributes
  dim_org_unit           — facility with full hierarchy
  dim_period             — calendar attributes for each yyyyMM period
  dim_program            — program metadata

All writes use mode('overwrite') to be idempotent.

Partitioning rationale:
  - health_area  : enables programme managers to query their own area
                   without scanning unrelated health areas.
  - year_month   : enables incremental processing (Task B2) and limits
                   partition count to a manageable number.
"""

import logging
import os

from pyspark.sql import DataFrame
from pyspark.sql import functions as F
from pyspark.sql.types import DoubleType
from pyspark.sql.window import Window

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Dimension builders
# ---------------------------------------------------------------------------

def build_dim_data_element(de_df: DataFrame) -> DataFrame:
    """One row per data element UID."""
    return de_df.select(
        F.col("id").alias("de_uid"),
        F.col("name").alias("de_name"),
        F.col("valueType").alias("value_type"),
        F.col("domainType").alias("domain_type"),
        F.col("aggregationType").alias("aggregation_type"),
        F.col("zeroIsSignificant").alias("zero_is_significant"),
        F.col("categoryCombo.id").alias("category_combo_id"),
        F.col("categoryCombo.name").alias("category_combo_name"),
        F.col("dataElementGroups")[0]["name"].alias("health_area"),
    ).dropDuplicates(["de_uid"])


def build_dim_org_unit(facility_hierarchy_df: DataFrame) -> DataFrame:
    """
    One row per facility (level-4 org unit) with full ancestry.
    """
    cols = [
        "facility_uid", "facility_name", "facility_type", "facility_level",
        "path",
    ]
    # Add ancestor columns dynamically — whatever resolved in Task 03
    for label in ("country", "region", "district"):
        uid_col  = f"{label}_uid"
        name_col = f"{label}_name"
        if uid_col in facility_hierarchy_df.columns:
            cols += [uid_col, name_col]

    return facility_hierarchy_df.select(*[c for c in cols if c in facility_hierarchy_df.columns]) \
                                .dropDuplicates(["facility_uid"])


def build_dim_period(dv_df: DataFrame) -> DataFrame:
    """
    One row per distinct period (yyyyMM) with calendar breakdowns.
    """
    return (
        dv_df
        .select(F.col("period").alias("period_id"))
        .distinct()
        .withColumn("year",       F.col("period_id").substr(1, 4).cast("int"))
        .withColumn("month",      F.col("period_id").substr(5, 2).cast("int"))
        .withColumn("month_name", F.date_format(F.to_date(F.col("period_id"), "yyyyMM"), "MMMM"))
        .withColumn(
            "quarter",
            F.concat(
                F.col("period_id").substr(1, 4),
                F.lit("Q"),
                F.ceil(F.col("period_id").substr(5, 2).cast("int") / 3).cast("string"),
            ),
        )
        .withColumn("period_start_date", F.to_date(F.col("period_id"), "yyyyMM"))
        .withColumn(
            "period_end_date",
            F.date_add(F.add_months(F.to_date(F.col("period_id"), "yyyyMM"), 1), -1),
        )
    )


def build_dim_program(prog_df: DataFrame) -> DataFrame:
    """
    One row per program. dataElements array is kept as-is for reference.
    """
    return prog_df.select(
        F.col("id").alias("program_id"),
        F.col("name").alias("program_name"),
        F.col("shortName").alias("program_short_name"),
        F.col("healthArea").alias("health_area"),
        F.col("country"),
        F.col("reportingFrequency").alias("reporting_frequency"),
        F.col("dataElements").alias("expected_de_uids"),
    ).dropDuplicates(["program_id"])


# ---------------------------------------------------------------------------
# Fact table builder
# ---------------------------------------------------------------------------

def build_fact_service_delivery(
    enriched_df: DataFrame,
    prog_df: DataFrame,
    output_dir: str,
) -> DataFrame:
    """
    Build fact_service_delivery from the enriched, flagged DataFrame produced
    by Tasks 01–04.

    Maps each data value row to its program (using the programs.dataElements
    array), then selects the canonical fact columns and writes the Parquet
    output partitioned by health_area and year_month.

    Returns the fact DataFrame (unpersisted — caller may cache if needed).
    """
    # Resolve program_id: explode programs to (de_uid, program_id) mapping
    # A data element may belong to one program per country; join on both
    # de_uid and country_name to get the right program.
    prog_de_map = (
        prog_df
        .select(
            F.col("id").alias("prog_id"),
            F.col("country").alias("prog_country"),
            F.col("healthArea").alias("prog_health_area"),
            F.explode(F.col("dataElements")).alias("prog_de_uid"),
        )
    )

    fact = enriched_df.join(
        prog_de_map,
        (enriched_df["dataElement"] == prog_de_map["prog_de_uid"]) &
        (enriched_df["country_name"] == prog_de_map["prog_country"]),
        "left",
    )

    # Select canonical fact columns
    fact = fact.select(
        # Keys
        F.col("dataElement").alias("de_uid"),
        F.col("orgUnit").alias("ou_uid"),
        F.col("period"),
        F.col("prog_id").alias("program_id"),
        F.col("categoryOptionCombo").alias("coc_uid"),
        F.coalesce(F.col("coc_name"), F.lit("unknown")).alias("coc_name"),
        F.col("de_name"),
        # Values
        F.col("value").alias("raw_value"),
        F.col("numeric_value"),
        # DQ flags
        F.col("is_late_reported"),
        F.col("is_explicit_zero"),
        F.col("is_missing_value"),
        F.coalesce(F.col("is_orphaned_coc"), F.lit(False)).alias("is_orphaned_coc"),
        F.col("completeness_score"),
        # Audit
        F.col("storedBy").alias("stored_by"),
        F.col("lastUpdated").alias("last_updated"),
        # Partition + convenience columns (denormalized for query efficiency)
        F.coalesce(F.col("health_area"), F.col("prog_health_area")).alias("health_area"),
        F.col("year_month"),
        F.col("year"),
        F.col("month"),
        F.col("quarter"),
        F.col("country_name"),
        F.col("region_name"),
        F.col("district_name"),
        F.col("facility_name"),
    )

    # Deduplicate on the natural key in case program join introduced extra rows
    # (a DE belonging to multiple programs in the same country edge-case)
    natural_key = ["de_uid", "ou_uid", "period", "coc_uid"]
    w = (
        Window
        .partitionBy(*natural_key)
        .orderBy(F.col("program_id").asc_nulls_last())
    )
    fact = (
        fact
        .withColumn("_rn", F.row_number().over(w))
        .filter(F.col("_rn") == 1)
        .drop("_rn")
    )

    # Write Parquet partitioned by health_area / year_month
    fact_path = os.path.join(output_dir, "fact_service_delivery")
    fact.write \
        .partitionBy("health_area", "year_month") \
        .mode("overwrite") \
        .parquet(fact_path)

    logger.info(
        "Task05 fact_service_delivery written to %s | rows=%d",
        fact_path, fact.count(),
    )
    return fact


# ---------------------------------------------------------------------------
# Dimension writers
# ---------------------------------------------------------------------------

def write_dims(
    de_df: DataFrame,
    facility_hierarchy_df: DataFrame,
    dv_df: DataFrame,
    prog_df: DataFrame,
    output_dir: str,
) -> dict[str, DataFrame]:
    """
    Build and write all four dimension tables.
    Returns a dict of {table_name: DataFrame}.
    """
    dims = {
        "dim_data_element": build_dim_data_element(de_df),
        "dim_org_unit":     build_dim_org_unit(facility_hierarchy_df),
        "dim_period":       build_dim_period(dv_df),
        "dim_program":      build_dim_program(prog_df),
    }

    for name, dim_df in dims.items():
        path = os.path.join(output_dir, name)
        dim_df.write.mode("overwrite").parquet(path)
        logger.info("Dimension %s written to %s | rows=%d", name, path, dim_df.count())

    return dims
