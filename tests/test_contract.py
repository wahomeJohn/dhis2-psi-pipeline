"""
Tests for Bonus B1 — Data contract validation.

Uses a local SparkSession (local[1]) and synthetic DataFrames.
Run with:  pytest tests/test_contract.py -v
"""

import os

import pytest
from pyspark.sql import SparkSession
from pyspark.sql import functions as F
from pyspark.sql.types import (
    BooleanType,
    DoubleType,
    IntegerType,
    StringType,
    StructField,
    StructType,
)

from models.task_b1_contract import (
    ContractViolationError,
    ViolationReport,
    load_contract,
    validate_and_raise,
    validate_contract,
)

CONTRACT_PATH = os.path.join(
    os.path.dirname(__file__), "..", "contracts", "fact_service_delivery.yaml"
)


@pytest.fixture(scope="session")
def spark():
    return (
        SparkSession.builder
        .master("local[1]")
        .appName("psi-dhis2-tests")
        .config("spark.sql.shuffle.partitions", "2")
        .config("spark.ui.enabled", "false")
        .getOrCreate()
    )


@pytest.fixture(scope="session")
def contract():
    return load_contract(CONTRACT_PATH)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_FACT_SCHEMA = StructType([
    StructField("de_uid",             StringType(),  False),
    StructField("ou_uid",             StringType(),  False),
    StructField("period",             StringType(),  False),
    StructField("program_id",         StringType(),  True),
    StructField("coc_uid",            StringType(),  False),
    StructField("coc_name",           StringType(),  False),
    StructField("raw_value",          StringType(),  True),
    StructField("numeric_value",      DoubleType(),  True),
    StructField("is_late_reported",   BooleanType(), False),
    StructField("is_explicit_zero",   BooleanType(), False),
    StructField("is_missing_value",   BooleanType(), False),
    StructField("is_orphaned_coc",    BooleanType(), False),
    StructField("completeness_score", DoubleType(),  True),
    StructField("stored_by",          StringType(),  True),
    StructField("last_updated",       StringType(),  True),
    StructField("health_area",        StringType(),  False),
    StructField("year_month",         StringType(),  False),
    StructField("year",               IntegerType(), False),
    StructField("month",              IntegerType(), False),
    StructField("quarter",            StringType(),  False),
    StructField("country_name",       StringType(),  True),
    StructField("region_name",        StringType(),  True),
    StructField("district_name",      StringType(),  True),
    StructField("facility_name",      StringType(),  True),
])

_VALID_ROW = (
    "ABC1234567X", "DEF1234567Y", "202401", "PROG001",
    "COC001", "default", "42", 42.0,
    False, False, False, False,
    0.85, "reporter1", "2024-02-10T10:00:00.000",
    "Malaria", "202401", 2024, 1, "2024Q1",
    "Kenya", "Nairobi Region", "Westlands District", "City Health Centre",
)


def _make_df(spark, rows, schema=None):
    return spark.createDataFrame(rows, schema=schema or _FACT_SCHEMA)


# ---------------------------------------------------------------------------
# Contract loading
# ---------------------------------------------------------------------------

def test_load_contract(contract):
    assert contract["table"] == "fact_service_delivery"
    assert "columns" in contract
    assert "checks" in contract


# ---------------------------------------------------------------------------
# Passing cases
# ---------------------------------------------------------------------------

def test_valid_row_passes(spark, contract):
    df = _make_df(spark, [_VALID_ROW])
    report = validate_contract(df, contract)
    assert report.passed, str(report)


def test_valid_multiple_rows_pass(spark, contract):
    row2 = list(_VALID_ROW)
    row2[18] = 3      # month = 3
    row2[19] = "2024Q1"
    rows = [_VALID_ROW, tuple(row2)]
    df = _make_df(spark, rows)
    report = validate_contract(df, contract)
    assert report.passed, str(report)


# ---------------------------------------------------------------------------
# Nullability violations
# ---------------------------------------------------------------------------

def test_null_de_uid_fails(spark, contract):
    row = list(_VALID_ROW)
    row[0] = None  # de_uid
    df = _make_df(spark, [tuple(row)])
    report = validate_contract(df, contract)
    assert not report.passed
    assert any("de_uid" in v for v in report.violations)


def test_null_ou_uid_fails(spark, contract):
    row = list(_VALID_ROW)
    row[1] = None  # ou_uid
    df = _make_df(spark, [tuple(row)])
    report = validate_contract(df, contract)
    assert not report.passed
    assert any("ou_uid" in v for v in report.violations)


def test_null_period_fails(spark, contract):
    row = list(_VALID_ROW)
    row[2] = None  # period
    df = _make_df(spark, [tuple(row)])
    report = validate_contract(df, contract)
    assert not report.passed
    assert any("period" in v for v in report.violations)


# ---------------------------------------------------------------------------
# Range violations
# ---------------------------------------------------------------------------

def test_completeness_above_1_fails(spark, contract):
    row = list(_VALID_ROW)
    row[12] = 1.5  # completeness_score > 1.0
    df = _make_df(spark, [tuple(row)])
    report = validate_contract(df, contract)
    assert not report.passed
    assert any("completeness_score" in v for v in report.violations)


def test_month_zero_fails(spark, contract):
    row = list(_VALID_ROW)
    row[18] = 0  # month = 0 (invalid)
    df = _make_df(spark, [tuple(row)])
    report = validate_contract(df, contract)
    assert not report.passed
    assert any("month" in v for v in report.violations)


def test_year_out_of_range_fails(spark, contract):
    row = list(_VALID_ROW)
    row[17] = 1990  # year < 2000
    df = _make_df(spark, [tuple(row)])
    report = validate_contract(df, contract)
    assert not report.passed
    assert any("year" in v for v in report.violations)


# ---------------------------------------------------------------------------
# Allowed-value violations
# ---------------------------------------------------------------------------

def test_invalid_health_area_fails(spark, contract):
    row = list(_VALID_ROW)
    row[15] = "Diabetes"  # not a valid PSI health area
    df = _make_df(spark, [tuple(row)])
    report = validate_contract(df, contract)
    assert not report.passed
    assert any("health_area" in v for v in report.violations)


# ---------------------------------------------------------------------------
# Custom cross-column check
# ---------------------------------------------------------------------------

def test_zero_and_null_both_true_fails(spark, contract):
    row = list(_VALID_ROW)
    row[9]  = True   # is_explicit_zero = True
    row[10] = True   # is_missing_value = True  — impossible combination
    df = _make_df(spark, [tuple(row)])
    report = validate_contract(df, contract)
    assert not report.passed
    assert any("is_explicit_zero" in v or "is_missing_value" in v for v in report.violations)


# ---------------------------------------------------------------------------
# validate_and_raise
# ---------------------------------------------------------------------------

def test_validate_and_raise_on_valid(spark):
    df = _make_df(spark, [_VALID_ROW])
    validate_and_raise(df, CONTRACT_PATH)  # should not raise


def test_validate_and_raise_on_invalid(spark):
    row = list(_VALID_ROW)
    row[0] = None  # null de_uid
    df = _make_df(spark, [tuple(row)])
    with pytest.raises(ContractViolationError):
        validate_and_raise(df, CONTRACT_PATH)
