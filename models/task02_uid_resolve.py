"""
Task 02 — Metadata UID resolution

Replaces all UID references in data values with human-readable names using
broadcast-optimised joins. Unresolvable UIDs (ghost dataElement or ghost
orgUnit) are isolated, counted, logged, and written separately.

Design note: ghost COC UIDs are a softer quality issue — the row is still
useful for reporting (we know what was reported and where) so those rows
are flagged but kept in the main dataset. Only rows with unresolvable
dataElement or orgUnit UIDs are quarantined, as those are the two dimensions
needed to give any row meaning.
"""

import logging
import os

from pyspark.sql import DataFrame
from pyspark.sql import functions as F

logger = logging.getLogger(__name__)


def resolve_uids(
    dv_df: DataFrame,
    de_df: DataFrame,
    coc_df: DataFrame,
    ou_df: DataFrame,
    output_dir: str,
) -> tuple[DataFrame, DataFrame]:
    """
    Join data values against metadata to replace UID references with names.

    Steps:
      1. Broadcast-join dataElement UIDs → de_name, value_type, health_area
      2. Broadcast-join categoryOptionCombo UIDs → coc_name (soft: flag orphans)
      3. Anti-join orgUnit UIDs against known org units (ghost OU → quarantine)
      4. Anti-join dataElement UIDs against known DEs (ghost DE → quarantine)

    Returns:
        resolved_df    — rows with all critical UIDs resolved
        unresolvable_df — rows quarantined due to ghost DE or ghost OU UIDs
    """
    # --- Prepare lookup tables ------------------------------------------------
    de_lookup = de_df.select(
        F.col("id").alias("de_uid"),
        F.col("name").alias("de_name"),
        F.col("valueType").alias("value_type"),
        F.col("categoryCombo.id").alias("category_combo_id"),
        F.col("categoryCombo.name").alias("category_combo_name"),
        # Health area is stored in the first dataElementGroups entry
        F.col("dataElementGroups")[0]["name"].alias("health_area"),
        F.col("aggregationType").alias("aggregation_type"),
        F.col("zeroIsSignificant").alias("zero_is_significant"),
    )

    coc_lookup = coc_df.select(
        F.col("id").alias("coc_uid"),
        F.col("name").alias("coc_name"),
    )

    ou_lookup = ou_df.select(
        F.col("id").alias("ou_known_uid"),
    )

    # --- Step 1: Join dataElement metadata (broadcast — small table) ----------
    joined = dv_df.join(
        F.broadcast(de_lookup),
        dv_df["dataElement"] == de_lookup["de_uid"],
        "left",
    )

    # --- Step 2: Join COC metadata (broadcast — small table) ------------------
    joined = joined.join(
        F.broadcast(coc_lookup),
        joined["categoryOptionCombo"] == coc_lookup["coc_uid"],
        "left",
    ).withColumn(
        "is_orphaned_coc",
        F.col("coc_uid").isNull(),
    )

    # --- Step 3: Flag ghost orgUnit UIDs with anti-join -----------------------
    # Anti-join: keep rows from joined whose orgUnit is NOT in ou_lookup
    ghost_ou = joined.join(
        F.broadcast(ou_lookup),
        joined["orgUnit"] == ou_lookup["ou_known_uid"],
        "left_anti",
    ).withColumn("quarantine_reason", F.lit("ghost orgUnit UID not in org_units.json"))

    # Resolvable org units: those that DO match
    joined = joined.join(
        F.broadcast(ou_lookup),
        joined["orgUnit"] == ou_lookup["ou_known_uid"],
        "inner",
    )

    # --- Step 4: Flag ghost dataElement UIDs (already joined — null de_name) --
    ghost_de = joined.filter(F.col("de_name").isNull()).withColumn(
        "quarantine_reason", F.lit("ghost dataElement UID not in metadata.json")
    )
    resolved = joined.filter(F.col("de_name").isNotNull())

    # --- Combine quarantine outputs -------------------------------------------
    # Align schemas before union
    shared_cols = [
        "dataElement", "period", "orgUnit", "categoryOptionCombo",
        "value", "storedBy", "lastUpdated", "quarantine_reason",
    ]
    ghost_de_aligned = ghost_de.select(
        *shared_cols
    )
    ghost_ou_aligned = ghost_ou.select(
        *shared_cols
    )
    unresolvable = ghost_de_aligned.union(ghost_ou_aligned)

    # --- Logging & persistence ------------------------------------------------
    n_ghost_de  = ghost_de.count()
    n_ghost_ou  = ghost_ou.count()
    n_resolved  = resolved.count()
    n_orphan_coc = resolved.filter(F.col("is_orphaned_coc")).count()

    logger.info(
        "Task02 UID resolution | resolved=%d | ghost_de=%d | ghost_ou=%d | orphaned_coc=%d",
        n_resolved, n_ghost_de, n_ghost_ou, n_orphan_coc,
    )

    q_path = os.path.join(output_dir, "quarantine", "task02_uid_resolution")
    if not unresolvable.rdd.isEmpty():
        unresolvable.coalesce(1).write.mode("overwrite").option("header", "true").csv(q_path)
        logger.info("Unresolvable UIDs written to %s", q_path)

    # Clean up duplicate join columns
    resolved = resolved.drop("de_uid", "coc_uid", "ou_known_uid")

    return resolved, unresolvable
