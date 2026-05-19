"""
Task 01 — JSON ingestion & schema flattening

Loads all four DHIS2 JSON files with explicit schemas (no inferSchema).
Explodes nested arrays into flat DataFrames.
Quarantines rows that violate structural expectations.
"""

import logging
import os

from pyspark.sql import DataFrame, SparkSession
from pyspark.sql import functions as F
from pyspark.sql.types import (
    ArrayType,
    BooleanType,
    IntegerType,
    StringType,
    StructField,
    StructType,
)

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Schema definitions — all explicit, no inferSchema
# ---------------------------------------------------------------------------

_GROUP_SCHEMA = StructType([
    StructField("id",   StringType(), True),
    StructField("name", StringType(), True),
])

_PARENT_SCHEMA = StructType([
    StructField("id",   StringType(), True),
    StructField("name", StringType(), True),
])

_CATEGORY_COMBO_SCHEMA = StructType([
    StructField("id",   StringType(), True),
    StructField("name", StringType(), True),
])

DATA_ELEMENT_SCHEMA = StructType([
    StructField("id",              StringType(),  False),
    StructField("name",            StringType(),  True),
    StructField("shortName",       StringType(),  True),
    StructField("code",            StringType(),  True),
    StructField("valueType",       StringType(),  True),
    StructField("domainType",      StringType(),  True),
    StructField("aggregationType", StringType(),  True),
    StructField("zeroIsSignificant", BooleanType(), True),
    StructField("categoryCombo",   _CATEGORY_COMBO_SCHEMA, True),
    StructField("dataElementGroups", ArrayType(_GROUP_SCHEMA), True),
    StructField("created",         StringType(),  True),
    StructField("lastUpdated",     StringType(),  True),
])

COC_SCHEMA = StructType([
    StructField("id",          StringType(), False),
    StructField("name",        StringType(), True),
    StructField("created",     StringType(), True),
    StructField("lastUpdated", StringType(), True),
])

ORG_UNIT_SCHEMA = StructType([
    StructField("id",        StringType(),           False),
    StructField("name",      StringType(),           True),
    StructField("shortName", StringType(),           True),
    StructField("code",      StringType(),           True),
    StructField("level",     IntegerType(),          True),
    StructField("path",      StringType(),           True),
    StructField("parent",    _PARENT_SCHEMA,         True),
    StructField("groups",    ArrayType(_GROUP_SCHEMA), True),
    StructField("created",     StringType(),         True),
    StructField("lastUpdated", StringType(),         True),
])

PROGRAM_SCHEMA = StructType([
    StructField("id",                 StringType(),           False),
    StructField("name",               StringType(),           True),
    StructField("shortName",          StringType(),           True),
    StructField("healthArea",         StringType(),           True),
    StructField("country",            StringType(),           True),
    StructField("reportingFrequency", StringType(),           True),
    StructField("dataElements",       ArrayType(StringType()), True),
    StructField("created",            StringType(),           True),
    StructField("lastUpdated",        StringType(),           True),
])

DATA_VALUE_SCHEMA = StructType([
    StructField("dataElement",          StringType(), True),
    StructField("period",               StringType(), True),
    StructField("orgUnit",              StringType(), True),
    StructField("categoryOptionCombo",  StringType(), True),
    StructField("attributeOptionCombo", StringType(), True),
    StructField("value",                StringType(), True),  # intentionally nullable
    StructField("storedBy",             StringType(), True),
    StructField("created",              StringType(), True),
    StructField("lastUpdated",          StringType(), True),
    StructField("followup",             StringType(), True),
])

# Top-level wrapper schemas
_METADATA_TOP = StructType([
    StructField("date",                 StringType(),                    True),
    StructField("version",              StringType(),                    True),
    StructField("dataElements",         ArrayType(DATA_ELEMENT_SCHEMA),  True),
    StructField("categoryOptionCombos", ArrayType(COC_SCHEMA),           True),
])

_OU_TOP = StructType([
    StructField("date",              StringType(),                   True),
    StructField("version",           StringType(),                   True),
    StructField("organisationUnits", ArrayType(ORG_UNIT_SCHEMA),     True),
])

_PROG_TOP = StructType([
    StructField("date",     StringType(),                True),
    StructField("version",  StringType(),                True),
    StructField("programs", ArrayType(PROGRAM_SCHEMA),   True),
])

_DV_TOP = StructType([
    StructField("responseType", StringType(),                    True),
    StructField("version",      StringType(),                    True),
    StructField("exportDate",   StringType(),                    True),
    StructField("dataValues",   ArrayType(DATA_VALUE_SCHEMA),    True),
])


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _read_whole_json(spark: SparkSession, path: str) -> "DataFrame":
    """Read an entire JSON file as a single-row text DataFrame."""
    return spark.read.option("wholetext", "true").text(path)


def _parse_and_explode(
    spark: SparkSession,
    path: str,
    top_schema: StructType,
    array_field: str,
) -> tuple[DataFrame, DataFrame]:
    """
    Parse a whole-file JSON with explicit top_schema, explode the named
    array field, and return (clean_df, quarantine_df).

    Rows in clean_df have each element of the array as one row.
    quarantine_df contains records where critical structural fields are null
    after schema coercion — indicating an unexpected JSON shape.
    """
    raw = _read_whole_json(spark, path)
    parsed = raw.select(
        F.from_json(F.col("value"), top_schema).alias("root")
    )
    flat = (
        parsed
        .select(F.explode(F.col(f"root.{array_field}")).alias("item"))
        .select("item.*")
    )
    return flat


def _split_quarantine(df: DataFrame, critical_fields: list[str]) -> tuple[DataFrame, DataFrame]:
    """
    Split a flat DataFrame into clean rows and quarantine rows.
    Quarantine = any critical field is null AND that null was unexpected
    (i.e., the field has False nullable in the schema — but since we read
    everything as nullable, we flag structural nulls explicitly).
    """
    null_condition = F.lit(False)
    for field in critical_fields:
        null_condition = null_condition | F.col(field).isNull()

    quarantine = df.filter(null_condition).withColumn(
        "quarantine_reason",
        F.lit(f"Null value in critical field(s): {critical_fields}"),
    )
    clean = df.filter(~null_condition)
    return clean, quarantine


# ---------------------------------------------------------------------------
# Public loader functions
# ---------------------------------------------------------------------------

def load_metadata(
    spark: SparkSession,
    data_dir: str,
) -> tuple[DataFrame, DataFrame, DataFrame]:
    """
    Load metadata.json.

    Returns:
        de_df  — data elements (one row per element)
        coc_df — category option combos
        quarantine_df — malformed rows from both arrays
    """
    path = os.path.join(data_dir, "metadata.json")
    logger.info("Loading metadata from %s", path)

    raw = _read_whole_json(spark, path)
    parsed = raw.select(F.from_json(F.col("value"), _METADATA_TOP).alias("root"))

    de_flat = (
        parsed
        .select(F.explode(F.col("root.dataElements")).alias("item"))
        .select("item.*")
    )
    coc_flat = (
        parsed
        .select(F.explode(F.col("root.categoryOptionCombos")).alias("item"))
        .select("item.*")
    )

    de_clean, de_q = _split_quarantine(de_flat, ["id", "name", "valueType"])
    coc_clean, coc_q = _split_quarantine(coc_flat, ["id", "name"])

    quarantine = de_q.select(
        F.lit("dataElement").alias("source"), F.col("quarantine_reason"),
        F.col("id"), F.col("name")
    ).union(
        coc_q.select(
            F.lit("categoryOptionCombo").alias("source"), F.col("quarantine_reason"),
            F.col("id"), F.col("name")
        )
    )

    logger.info(
        "Metadata loaded: %d dataElements, %d COCs, %d quarantined",
        de_clean.count(), coc_clean.count(), quarantine.count(),
    )
    return de_clean, coc_clean, quarantine


def load_org_units(
    spark: SparkSession,
    data_dir: str,
) -> tuple[DataFrame, DataFrame]:
    """
    Load org_units.json.

    Returns:
        ou_df         — org units (all levels)
        quarantine_df — malformed rows
    """
    path = os.path.join(data_dir, "org_units.json")
    logger.info("Loading org units from %s", path)

    flat = _parse_and_explode(spark, path, _OU_TOP, "organisationUnits")
    clean, quarantine = _split_quarantine(flat, ["id", "name", "level"])

    logger.info(
        "Org units loaded: %d rows, %d quarantined",
        clean.count(), quarantine.count(),
    )
    return clean, quarantine


def load_programs(
    spark: SparkSession,
    data_dir: str,
) -> tuple[DataFrame, DataFrame]:
    """
    Load programs.json.

    Returns:
        prog_df       — programs (one row per program)
        quarantine_df — malformed rows
    """
    path = os.path.join(data_dir, "programs.json")
    logger.info("Loading programs from %s", path)

    flat = _parse_and_explode(spark, path, _PROG_TOP, "programs")
    clean, quarantine = _split_quarantine(flat, ["id", "healthArea", "country"])

    logger.info(
        "Programs loaded: %d rows, %d quarantined",
        clean.count(), quarantine.count(),
    )
    return clean, quarantine


def load_data_values(
    spark: SparkSession,
    data_dir: str,
) -> tuple[DataFrame, DataFrame]:
    """
    Load data_values.json.

    Returns:
        dv_df         — data values (one row per reported indicator)
        quarantine_df — structurally malformed rows

    Note: value=None (null) is an intentional data quality pattern (~3%) and
    is NOT treated as malformed here — it is flagged later in Task 04.
    """
    path = os.path.join(data_dir, "data_values.json")
    logger.info("Loading data values from %s", path)

    flat = _parse_and_explode(spark, path, _DV_TOP, "dataValues")

    # Structural quarantine: rows where the composite key is incomplete
    clean, quarantine = _split_quarantine(flat, ["dataElement", "period", "orgUnit"])

    logger.info(
        "Data values loaded: %d rows, %d structurally quarantined",
        clean.count(), quarantine.count(),
    )
    return clean, quarantine


def ingest_all(
    spark: SparkSession,
    data_dir: str,
    output_dir: str,
) -> tuple[DataFrame, DataFrame, DataFrame, DataFrame, DataFrame]:
    """
    Run all ingestion steps. Writes all quarantine outputs to disk.

    Returns:
        de_df, coc_df, ou_df, prog_df, dv_df
    """
    de_df, coc_df, meta_quarantine = load_metadata(spark, data_dir)
    ou_df, ou_quarantine           = load_org_units(spark, data_dir)
    prog_df, prog_quarantine       = load_programs(spark, data_dir)
    dv_df, dv_quarantine           = load_data_values(spark, data_dir)

    # Persist quarantine outputs
    q_path = os.path.join(output_dir, "quarantine", "task01_schema")
    _write_quarantine(meta_quarantine, os.path.join(q_path, "metadata"))
    _write_quarantine(ou_quarantine,   os.path.join(q_path, "org_units"))
    _write_quarantine(prog_quarantine, os.path.join(q_path, "programs"))
    _write_quarantine(dv_quarantine,   os.path.join(q_path, "data_values"))

    dv_count  = dv_df.count()
    dv_q_count = dv_quarantine.count()
    quarantine_rate = dv_q_count / max(dv_count + dv_q_count, 1)
    logger.info(
        "Task01 complete | dv_rows=%d | quarantine_rate=%.2f%%",
        dv_count, quarantine_rate * 100,
    )
    return de_df, coc_df, ou_df, prog_df, dv_df, quarantine_rate


def _write_quarantine(df: DataFrame, path: str) -> None:
    """Write a quarantine DataFrame as CSV for easy inspection."""
    if df.rdd.isEmpty():
        logger.info("No quarantine records for %s", path)
        return
    df.coalesce(1).write.mode("overwrite").option("header", "true").csv(path)
    logger.info("Quarantine written to %s", path)
