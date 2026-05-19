# PSI DISC · DHIS2 Health Data Pipeline

A PySpark ELT pipeline that transforms four raw DHIS2 JSON exports into a
clean dimensional warehouse with DQ flagging, programme analytics, and
cross-country aggregation.

---

## Quick Start

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Generate synthetic data (default: 5 countries, 12 months)
python generate_data.py

# Or with custom parameters:
python generate_data.py --countries 8 --periods 18 --seed 42

# 3. Run the full pipeline
python pipeline.py --data-dir ./data --output-dir ./output

# Run with incremental load (skips already-processed periods)
python pipeline.py --data-dir ./data --output-dir ./output --incremental

# Run without bonus tasks
python pipeline.py --data-dir ./data --output-dir ./output --skip-bonus

# 4. Run data contract tests (Bonus B1)
pytest tests/ -v
```

---

## Project Structure

```
dhis2-pipeline/
├── generate_data.py          # Synthetic DHIS2 data generator (provided)
├── pipeline.py               # Orchestration entry point — run this
├── requirements.txt
│
├── models/
│   ├── task01_ingest.py      # JSON ingestion with explicit schemas
│   ├── task02_uid_resolve.py # Broadcast-join UID resolution
│   ├── task03_hierarchy.py   # Org unit path traversal
│   ├── task04_dq_flags.py    # Dedup, casting, DQ flags, completeness
│   ├── task05_dim_model.py   # Star schema build + Parquet write
│   ├── task06_analytics.py   # Window function analytics
│   ├── task07_aggregation.py # Cross-country aggregation + pivot
│   ├── task_b1_contract.py   # [Bonus] YAML data contract validator
│   ├── task_b2_incremental.py# [Bonus] Incremental load logic
│   └── task_b3_anomaly.py    # [Bonus] 3-sigma anomaly detection
│
├── contracts/
│   └── fact_service_delivery.yaml  # Schema contract for the fact table
│
├── tests/
│   └── test_contract.py      # pytest suite for B1 contract validation
│
├── data/                     # Generated JSON inputs (git-ignored)
└── output/                   # Pipeline outputs (git-ignored)
    ├── fact_service_delivery/     # Parquet, partitioned by health_area/year_month
    ├── dim_data_element/          # Parquet
    ├── dim_org_unit/              # Parquet
    ├── dim_period/                # Parquet
    ├── dim_program/               # Parquet
    ├── analytics/                 # CSV: MoM change, rolling avg, reporting rate
    ├── cross_country/             # CSV: volumes, completeness, coverage matrix
    ├── anomalies/                 # CSV: statistical outlier flags (B3)
    └── quarantine/                # CSV: rejected rows per stage
```

---

## Star Schema

```
                          ┌─────────────────────┐
                          │   dim_data_element   │
                          │─────────────────────│
                          │ de_uid (PK)          │
                          │ de_name              │
                          │ value_type           │
                          │ health_area          │
                          │ aggregation_type     │
                          │ category_combo_id    │
                          └──────────┬──────────┘
                                     │
┌─────────────────┐       ┌──────────▼──────────────────────────────────┐       ┌──────────────────┐
│   dim_org_unit  │       │           fact_service_delivery              │       │   dim_period     │
│─────────────────│       │─────────────────────────────────────────────│       │──────────────────│
│ facility_uid(PK)│◄──────│ de_uid          (FK → dim_data_element)     │──────►│ period_id (PK)   │
│ facility_name   │       │ ou_uid          (FK → dim_org_unit)         │       │ year             │
│ facility_type   │       │ period          (FK → dim_period)           │       │ month            │
│ district_uid    │       │ program_id      (FK → dim_program)          │       │ quarter          │
│ district_name   │       │ coc_uid                                     │       │ period_start_date│
│ region_uid      │       │ coc_name                                    │       │ period_end_date  │
│ region_name     │       │ raw_value                                   │       └──────────────────┘
│ country_uid     │       │ numeric_value                               │
│ country_name    │       │ is_late_reported                            │       ┌──────────────────┐
│ path            │       │ is_explicit_zero                            │       │   dim_program    │
└─────────────────┘       │ is_missing_value                            │       │──────────────────│
                          │ is_orphaned_coc                             │──────►│ program_id (PK)  │
                          │ completeness_score                          │       │ program_name     │
                          │ stored_by                                   │       │ health_area      │
                          │ last_updated                                │       │ country          │
                          │ health_area     ◄── partition key           │       │ reporting_freq   │
                          │ year_month      ◄── partition key           │       └──────────────────┘
                          └─────────────────────────────────────────────┘
```

Partitioning by `health_area` and `year_month` enables:
- Programme managers to query their own health area without full table scans
- Incremental processing (Task B2) — new months land in new partitions

---

## Data Quality Handling

| Issue | Volume | Handling |
|-------|--------|----------|
| Exact duplicate rows | ~8% | Deduplicated in Task 04 (row_number + lastUpdated desc) |
| Near-duplicates (corrected value) | ~2% | Same dedup window — latest lastUpdated wins |
| Ghost dataElement UIDs | ~5% | Quarantined in Task 02 (anti-join) |
| Ghost orgUnit UIDs | ~4% | Quarantined in Task 02 (anti-join) |
| Orphaned COC UIDs | ~3% | Flagged (is_orphaned_coc) but kept — row is still useful |
| Late-reported (>60 days) | ~12% | Flagged (is_late_reported) — not quarantined |
| Explicit zero values | ~6% | Flagged (is_explicit_zero) — preserved, never collapsed to NULL |
| NULL values | ~3% | Flagged (is_missing_value) — preserved, never collapsed to zero |
| Non-reporting facilities | ~8% | Detectable via completeness_score |

**Zero vs NULL**: A facility reporting `value="0"` means it ran the service
and found zero cases — a valid epidemiological reading. A `value=NULL` means
no report was filed at all. These are fundamentally different states in health
programme monitoring and are never collapsed.

---

## Design Decisions

### 1. `from_json` over `spark.read.json`
DHIS2 exports are single JSON objects (not JSON Lines). Reading with
`wholetext=True` + `from_json` with an explicit schema gives full control
over the shape of every field. `inferSchema=True` is never used anywhere.

### 2. Broadcast joins for metadata
The metadata tables (dataElements, COCs, programs) are small (<10k rows).
Broadcasting them avoids shuffle entirely for the large data_values join.

### 3. Dynamic hierarchy traversal
The `path` column (`/l1_uid/l2_uid/l3_uid/l4_uid`) encodes the full ancestry.
The pipeline splits this column and joins each position against the org unit
lookup. No UIDs are hardcoded — the depth is inferred from `max(level)` at
runtime, making it forward-compatible with future hierarchy changes.

### 4. Deduplication strategy
A single `row_number()` window partitioned by the composite natural key
(`dataElement, period, orgUnit, categoryOptionCombo`) ordered by
`lastUpdated DESC` handles both exact duplicates and near-duplicates (data
corrections) in one pass, without double-scanning the data.

### 5. Completeness scoring
Expected indicators come from `programs.json` per country. The score is
`distinct_reported_DEs / expected_DEs_for_country`. Null values are excluded
from the "reported" count because a submitted-but-null row does not represent
actual programme data delivery.

### 6. Partition strategy
Partitioning by `health_area` first (low cardinality, ~5 values) and
`year_month` second yields bounded, predictable partition counts. A 12-month
dataset across 5 health areas produces exactly 60 leaf directories — small
enough for `local[*]` execution, and scalable to a data lake.

### 7. Contract validation (Bonus B1)
The YAML contract is validated before the fact write, not after. This catches
issues while the DataFrame is still in memory, rather than discovering them
after a slow Parquet write. Violations are logged with full detail; the
pipeline treats contract failure as non-fatal (logged, not exit-1) to
preserve data for programme review.

---

## Assumptions

1. All DHIS2 data is AGGREGATE domain (as stated in the assessment).
2. Facilities are always at `level == max(level)` in the org unit table.
3. A data element may belong to at most one program per country. Where
   multiple programs claim the same DE (edge case), the first program_id
   encountered is assigned.
4. The `storedBy` casing inconsistency is preserved as-is — normalisation
   is not applied because it could obscure the original reporter identity.
5. Incremental load (B2) uses Parquet partition directories as the source
   of truth for "what has been loaded". This is fast but assumes partitions
   are written atomically (which Spark's `mode=overwrite` guarantees).

---

## Known Limitations

- **Single-node only**: Configured for `local[*]`. For production scale,
  change the Spark master and adjust shuffle partitions.
- **COC resolution**: Orphaned COC UIDs are flagged but their rows are kept.
  If COC identity is required for disaggregated analysis, those rows should
  be quarantined instead.
- **Program assignment**: A data element reported by a facility but not
  listed in any program for that country will have `program_id = NULL`.
  This happens legitimately when facilities report opportunistic data.
- **Rolling statistics**: The 3-month rolling average and 12-month anomaly
  window use `rowsBetween` not `rangeBetween`, so gaps in reporting
  (skipped periods) will cause the window to span more than N calendar months.
  For gap-aware windows, periods would need to be converted to integer offsets.

---

## Requirements

```
pyspark==3.4.3
faker==24.0.0
numpy==1.26.4
pyyaml==6.0.1
pytest==7.4.4
```

Tested on Python 3.10, PySpark 3.4, macOS (Apple Silicon + Intel).
No Airflow, Databricks, or cloud-provider packages required.
