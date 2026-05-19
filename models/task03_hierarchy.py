"""
Task 03 — Org unit hierarchy resolution

Resolves each facility's full ancestry (facility → district → region →
country) by splitting the DHIS2 path column dynamically.

Design decisions:
  - The path column format is /l1_uid/l2_uid/.../lN_uid where the UID at
    position i corresponds to level i (1-indexed after the leading slash).
    This is guaranteed by DHIS2 and lets us extract ancestor UIDs without
    hardcoding any UIDs.
  - We perform one broadcast join per ancestor level against the ou_df
    lookup table. For a 4-level hierarchy that is 3 extra joins (levels
    1, 2, 3). If DHIS2 adds a level, only the loop range needs updating.
  - facility_type is extracted from the second element of the groups array
    (the first element is always the generic level label "Facility").
"""

import logging

from pyspark.sql import DataFrame
from pyspark.sql import functions as F

logger = logging.getLogger(__name__)

# The maximum hierarchy depth we handle. Facilities are always at the deepest
# level, so we resolve all ancestor levels from 1 up to (facility_level - 1).
MAX_LEVELS = 6  # handles up to 6-level DHIS2 hierarchies


def build_facility_hierarchy(ou_df: DataFrame) -> DataFrame:
    """
    Construct a flat facility lookup table with full ancestor names resolved.

    Returns a DataFrame with one row per facility (level == 4 in standard
    4-level DHIS2) with columns:
        facility_uid, facility_name, facility_type, path,
        country_uid,  country_name,
        region_uid,   region_name,
        district_uid, district_name,
        facility_level
    """
    # Broadcast-ready lookup: uid -> (name, level)
    ou_lookup = ou_df.select(
        F.col("id").alias("_lookup_uid"),
        F.col("name").alias("_lookup_name"),
        F.col("level").alias("_lookup_level"),
    )

    # Facilities are the leaf nodes — find the deepest level present
    max_level = ou_df.agg(F.max("level")).collect()[0][0]
    facilities = ou_df.filter(F.col("level") == max_level)

    # Split the path into an array. Path format: /l1/l2/l3/l4
    # split() gives ["", l1_uid, l2_uid, l3_uid, l4_uid]
    # So ancestor at level N is path_parts[N].
    fac = facilities.withColumn("_path_parts", F.split(F.col("path"), "/"))

    # Extract facility type from the groups array (second group entry)
    # groups[0].name is the generic label ("Facility"); groups[1].name is the type
    fac = fac.withColumn(
        "facility_type",
        F.when(
            F.size(F.col("groups")) > 1,
            F.col("groups")[1]["name"],
        ).otherwise(F.lit(None).cast("string")),
    )

    # For each ancestor level (1 up to max_level-1), extract the UID from
    # the path and join it against the lookup to get the ancestor name.
    level_labels = {1: "country", 2: "region", 3: "district"}

    for lvl in range(1, max_level):
        label = level_labels.get(lvl, f"level{lvl}")
        uid_col   = f"{label}_uid"
        name_col  = f"{label}_name"

        # Alias the lookup for this specific join to avoid column ambiguity
        lvl_lookup = ou_lookup.filter(F.col("_lookup_level") == lvl).select(
            F.col("_lookup_uid").alias(f"_join_uid_{lvl}"),
            F.col("_lookup_name").alias(name_col),
        )

        fac = fac.withColumn(uid_col, F.col("_path_parts")[lvl])
        fac = fac.join(
            F.broadcast(lvl_lookup),
            fac[uid_col] == lvl_lookup[f"_join_uid_{lvl}"],
            "left",
        ).drop(f"_join_uid_{lvl}")

    # Tidy up: rename facility columns and drop internals
    fac = (
        fac
        .withColumnRenamed("id",    "facility_uid")
        .withColumnRenamed("name",  "facility_name")
        .withColumnRenamed("level", "facility_level")
        .drop("_path_parts", "shortName", "code", "parent",
              "groups", "created", "lastUpdated")
    )

    logger.info(
        "Hierarchy built for %d facilities at level %d",
        fac.count(), max_level,
    )
    return fac


def enrich_with_hierarchy(
    dv_df: DataFrame,
    facility_hierarchy_df: DataFrame,
) -> DataFrame:
    """
    Add country/region/district/facility name columns to every data value row
    by joining on orgUnit (= facility_uid).

    Returns the enriched DataFrame.
    """
    enriched = dv_df.join(
        F.broadcast(facility_hierarchy_df),
        dv_df["orgUnit"] == facility_hierarchy_df["facility_uid"],
        "left",
    )

    n_enriched   = enriched.filter(F.col("facility_name").isNotNull()).count()
    n_unenriched = enriched.filter(F.col("facility_name").isNull()).count()
    logger.info(
        "Task03 hierarchy | enriched=%d | not_matched=%d",
        n_enriched, n_unenriched,
    )
    return enriched
