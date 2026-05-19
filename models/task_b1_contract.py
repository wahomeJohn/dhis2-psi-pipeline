"""
Bonus B1 — Data contract validation

Loads a YAML schema contract and validates the fact_service_delivery
Parquet output against it before committing the final write.

Contract checks performed:
  - Column presence (all declared columns must exist)
  - Nullability constraints (non-nullable columns must have zero nulls)
  - Type compatibility (declared type vs actual Spark dtype)
  - Value range constraints (min/max for numeric/integer columns)
  - Allowed-value sets (enum columns)
  - Custom cross-column checks (e.g. zero/null mutual exclusivity)

Raises ContractViolationError with a full violation report if any check fails.
"""

import logging
import os
from dataclasses import dataclass, field
from typing import Any

import yaml
from pyspark.sql import DataFrame
from pyspark.sql import functions as F

logger = logging.getLogger(__name__)

# Mapping from YAML type names to acceptable Spark dtype name substrings
_TYPE_MAP = {
    "string":  ("string",),
    "integer": ("int", "long", "short", "byte"),
    "double":  ("double", "float", "decimal"),
    "boolean": ("bool",),
}


class ContractViolationError(Exception):
    """Raised when the DataFrame fails one or more contract checks."""


@dataclass
class ViolationReport:
    violations: list[str] = field(default_factory=list)

    def add(self, message: str) -> None:
        logger.warning("Contract violation: %s", message)
        self.violations.append(message)

    @property
    def passed(self) -> bool:
        return len(self.violations) == 0

    def __str__(self) -> str:
        if self.passed:
            return "All contract checks passed."
        return "Contract violations:\n" + "\n".join(f"  - {v}" for v in self.violations)


def load_contract(yaml_path: str) -> dict[str, Any]:
    """Load a YAML contract file and return it as a dict."""
    with open(yaml_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def _spark_type_matches(spark_dtype: str, declared_type: str) -> bool:
    """Return True if the Spark dtype is compatible with the declared YAML type."""
    spark_lower = spark_dtype.lower()
    for substr in _TYPE_MAP.get(declared_type, (declared_type,)):
        if substr in spark_lower:
            return True
    return False


def validate_contract(df: DataFrame, contract: dict[str, Any]) -> ViolationReport:
    """
    Validate a DataFrame against a YAML contract.

    Returns a ViolationReport. Does NOT raise — the caller decides whether
    to treat violations as fatal.
    """
    report   = ViolationReport()
    columns  = contract.get("columns", {})
    df_schema = {f.name: f.dataType.simpleString() for f in df.schema.fields}

    # ------------------------------------------------------------------
    # 1. Column presence
    # ------------------------------------------------------------------
    for col_name in columns:
        if col_name not in df_schema:
            report.add(f"Missing column: '{col_name}'")

    # ------------------------------------------------------------------
    # 2. Per-column checks (type, nullability, range, allowed values)
    # ------------------------------------------------------------------
    for col_name, spec in columns.items():
        if col_name not in df_schema:
            continue  # already flagged above

        spark_type = df_schema[col_name]

        # Type check
        declared_type = spec.get("type")
        if declared_type and not _spark_type_matches(spark_type, declared_type):
            report.add(
                f"Column '{col_name}': expected type '{declared_type}', "
                f"got '{spark_type}'"
            )

        # Nullability check
        if spec.get("nullable") is False:
            null_count = df.filter(F.col(col_name).isNull()).count()
            if null_count > 0:
                report.add(
                    f"Column '{col_name}' declared non-nullable but has "
                    f"{null_count:,} null rows"
                )

        # Range check
        if "min" in spec or "max" in spec:
            stats = df.filter(F.col(col_name).isNotNull()).agg(
                F.min(col_name).alias("actual_min"),
                F.max(col_name).alias("actual_max"),
            ).collect()[0]
            if stats["actual_min"] is not None:
                if "min" in spec and stats["actual_min"] < spec["min"]:
                    report.add(
                        f"Column '{col_name}': min value {stats['actual_min']} "
                        f"< contract min {spec['min']}"
                    )
                if "max" in spec and stats["actual_max"] > spec["max"]:
                    report.add(
                        f"Column '{col_name}': max value {stats['actual_max']} "
                        f"> contract max {spec['max']}"
                    )

        # Allowed values check
        if "allowed_values" in spec:
            allowed = set(spec["allowed_values"])
            bad = (
                df
                .filter(F.col(col_name).isNotNull())
                .filter(~F.col(col_name).isin(list(allowed)))
                .count()
            )
            if bad > 0:
                report.add(
                    f"Column '{col_name}': {bad:,} rows have values outside "
                    f"allowed set {sorted(allowed)}"
                )

        # Regex pattern check (spot-check on first 10k rows for performance)
        if "pattern" in spec:
            pattern = spec["pattern"]
            bad = (
                df
                .limit(10_000)
                .filter(F.col(col_name).isNotNull())
                .filter(~F.col(col_name).rlike(pattern))
                .count()
            )
            if bad > 0:
                report.add(
                    f"Column '{col_name}': {bad} rows (sample) do not match "
                    f"pattern '{pattern}'"
                )

    # ------------------------------------------------------------------
    # 3. Custom cross-column checks
    # ------------------------------------------------------------------
    # zero/null mutual exclusivity
    if "is_explicit_zero" in df_schema and "is_missing_value" in df_schema:
        both_true = df.filter(
            F.col("is_explicit_zero") & F.col("is_missing_value")
        ).count()
        if both_true > 0:
            report.add(
                f"{both_true:,} rows have both is_explicit_zero=True and "
                f"is_missing_value=True — impossible for a single value"
            )

    logger.info("Contract validation complete: %s", str(report))
    return report


def validate_and_raise(df: DataFrame, contract_path: str) -> None:
    """
    Validate the DataFrame against the YAML contract at contract_path.
    Raises ContractViolationError if any check fails.
    """
    contract = load_contract(contract_path)
    report   = validate_contract(df, contract)
    if not report.passed:
        raise ContractViolationError(str(report))
    logger.info("Bonus B1: contract validation passed for '%s'", contract.get("table"))
