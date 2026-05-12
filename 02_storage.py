"""
================================================================================
SCRIPT 02: Data Storage
Project:   Predicting Number of Hospitals by Level (L1 / L2 / L3)
           per Municipality/City based on Socioeconomic &
           Infrastructural Factors
================================================================================

STORAGE FORMAT: SQLite
-----------------------
All cleaned data are persisted in a single SQLite database:

    data/processed/hospital_data.db

WHY SQLITE?
-----------
The clean data has natural relational structure across 5 source files
(facilities, population, poverty, births, final merged). SQLite is the
ideal fit because:

  * One file    — the entire dataset lives in a single portable .db file
                  that can be committed to git or shared with teammates
  * Relational  — the 5 source tables are stored as separate, normalized
                  tables linked by lgu_id (surrogate PK), matching how
                  Script 01 actually produced them
  * Schema-safe — column types are enforced; poverty % cannot silently
                  round-trip as an object string the way it can in CSV
  * Zero setup  — SQLite is built into Python's stdlib (sqlite3); no
                  server, no credentials, no extra conda package needed
  * Queryable   — teammates can inspect or cross-query tables directly
                  using DB Browser for SQLite (free GUI) or via pandas
  * Fast reads  — pd.read_sql() with a WHERE clause is faster than
                  loading an entire XLSX then filtering in memory

COMPARED TO ALTERNATIVES
--------------------------
  CSV / XLSX    — no type enforcement, slow for repeated loads, one file
                  per table vs. one unified .db
  Parquet       — excellent for flat analytics tables but requires
                  pyarrow (extra dependency) and has no relational
                  structure between tables
  PostgreSQL    — requires a running server; breaks reproducibility
                  across different machines and OS environments

DATABASE SCHEMA (7 normalised + 1 denormalised table)
------------------------------------------------------
  lgu_master          — geographic identifiers; PK = lgu_id (INTEGER)
  lgu_population      — PSA Census 2020/2024 + growth rate
  lgu_poverty         — poverty incidence 2018/2021/2023 + source flag
  lgu_births          — PSA registered live births 2023
  lgu_infrastructure  — OSM amenity counts (20 types)
  lgu_facilities      — DOH NHFR aggregated supply metrics
  lgu_targets         — hospital counts by DOH level (L1 / L2 / L3)
  lgu_merged          — full denormalised feature + target matrix;
                        mirrors final_dataset_clean.xlsx exactly;
                        used as the single source of truth for Script 03

INPUTS  (data/clean/)
  final_dataset_clean.xlsx   — primary source (merged + OSM)

OUTPUT
  data/processed/hospital_data.db

================================================================================
"""

import os
import sqlite3

import pandas as pd

# ── Paths ──────────────────────────────────────────────────────────────────
BASE_DIR      = os.path.dirname(os.path.abspath(__file__))
CLEAN_DIR     = os.path.join(BASE_DIR, "data", "clean")
PROCESSED_DIR = os.path.join(BASE_DIR, "data", "processed")

DB_PATH    = os.path.join(PROCESSED_DIR, "hospital_data.db")
FINAL_XLSX = os.path.join(CLEAN_DIR, "final_dataset_clean.xlsx")

# ── Column groupings ───────────────────────────────────────────────────────
ID_COLS = ["city_municipality", "province", "region"]

POPULATION_COLS = [
    "population_2020",
    "population_2024",
    "pop_growth_rate_pct",
]

POVERTY_COLS = [
    "poverty_incidence_2018_pct",
    "poverty_incidence_2021_pct",
    "poverty_incidence_2023_pct",
    "poverty_source",
]

BIRTH_COLS = [
    "births_occurrence_both",
    "births_occurrence_male",
    "births_occurrence_female",
    "births_residence_both",
    "births_residence_male",
    "births_residence_female",
]

# 18 model features + 2 extra OSM cols present in final_dataset_clean.xlsx
INFRASTRUCTURE_COLS = [
    "atm", "bank", "bar", "bus_station", "cafe", "clinic",
    "community_centre", "fast_food", "fuel", "hospital", "parking",
    "pharmacy", "place_of_worship", "police", "post_office",
    "restaurant", "school", "shelter", "toilets", "townhall",
]

FACILITY_COLS = [
    "total_facilities", "total_hospitals", "total_bed_capacity",
    "weighted_facility_score", "private_count", "gov_count",
    "private_ownership_pct", "facility_density_per10k",
    "hospital_density_per10k", "beds_per_1000",
    "weighted_score_per10k", "level3_per100k",
]

TARGET_COLS = [
    "hospital_count_level1",
    "hospital_count_level2",
    "hospital_count_level3",
]


# ═══════════════════════════════════════════════════════════════════════════
# SCHEMA DDL
# ═══════════════════════════════════════════════════════════════════════════

DDL_STATEMENTS = [
    """
    CREATE TABLE IF NOT EXISTS lgu_master (
        lgu_id            INTEGER PRIMARY KEY,
        city_municipality TEXT    NOT NULL,
        province          TEXT,
        region            TEXT    NOT NULL
    )
    """,
    """
    CREATE TABLE IF NOT EXISTS lgu_population (
        lgu_id              INTEGER PRIMARY KEY
                            REFERENCES lgu_master(lgu_id),
        population_2020     INTEGER,
        population_2024     INTEGER,
        pop_growth_rate_pct REAL
    )
    """,
    """
    CREATE TABLE IF NOT EXISTS lgu_poverty (
        lgu_id                     INTEGER PRIMARY KEY
                                   REFERENCES lgu_master(lgu_id),
        poverty_incidence_2018_pct REAL,
        poverty_incidence_2021_pct REAL,
        poverty_incidence_2023_pct REAL,
        poverty_source             TEXT
    )
    """,
    """
    CREATE TABLE IF NOT EXISTS lgu_births (
        lgu_id                   INTEGER PRIMARY KEY
                                 REFERENCES lgu_master(lgu_id),
        births_occurrence_both   INTEGER,
        births_occurrence_male   INTEGER,
        births_occurrence_female INTEGER,
        births_residence_both    INTEGER,
        births_residence_male    INTEGER,
        births_residence_female  INTEGER
    )
    """,
    """
    CREATE TABLE IF NOT EXISTS lgu_infrastructure (
        lgu_id           INTEGER PRIMARY KEY
                         REFERENCES lgu_master(lgu_id),
        atm              INTEGER DEFAULT 0,
        bank             INTEGER DEFAULT 0,
        bar              INTEGER DEFAULT 0,
        bus_station      INTEGER DEFAULT 0,
        cafe             INTEGER DEFAULT 0,
        clinic           INTEGER DEFAULT 0,
        community_centre INTEGER DEFAULT 0,
        fast_food        INTEGER DEFAULT 0,
        fuel             INTEGER DEFAULT 0,
        hospital         INTEGER DEFAULT 0,
        parking          INTEGER DEFAULT 0,
        pharmacy         INTEGER DEFAULT 0,
        place_of_worship INTEGER DEFAULT 0,
        police           INTEGER DEFAULT 0,
        post_office      INTEGER DEFAULT 0,
        restaurant       INTEGER DEFAULT 0,
        school           INTEGER DEFAULT 0,
        shelter          INTEGER DEFAULT 0,
        toilets          INTEGER DEFAULT 0,
        townhall         INTEGER DEFAULT 0
    )
    """,
    """
    CREATE TABLE IF NOT EXISTS lgu_facilities (
        lgu_id                  INTEGER PRIMARY KEY
                                REFERENCES lgu_master(lgu_id),
        total_facilities        INTEGER DEFAULT 0,
        total_hospitals         INTEGER DEFAULT 0,
        total_bed_capacity      INTEGER DEFAULT 0,
        weighted_facility_score REAL    DEFAULT 0,
        private_count           INTEGER DEFAULT 0,
        gov_count               INTEGER DEFAULT 0,
        private_ownership_pct   REAL,
        facility_density_per10k REAL,
        hospital_density_per10k REAL,
        beds_per_1000           REAL,
        weighted_score_per10k   REAL,
        level3_per100k          REAL
    )
    """,
    """
    CREATE TABLE IF NOT EXISTS lgu_targets (
        lgu_id                INTEGER PRIMARY KEY
                              REFERENCES lgu_master(lgu_id),
        hospital_count_level1 INTEGER DEFAULT 0,
        hospital_count_level2 INTEGER DEFAULT 0,
        hospital_count_level3 INTEGER DEFAULT 0
    )
    """,
    # Denormalised table — Script 03 reads this directly (no JOIN needed)
    """
    CREATE TABLE IF NOT EXISTS lgu_merged (
        lgu_id                     INTEGER PRIMARY KEY
                                   REFERENCES lgu_master(lgu_id),
        city_municipality           TEXT,
        province                    TEXT,
        region                      TEXT,
        population_2020             INTEGER,
        population_2024             INTEGER,
        pop_growth_rate_pct         REAL,
        poverty_incidence_2018_pct  REAL,
        poverty_incidence_2021_pct  REAL,
        poverty_incidence_2023_pct  REAL,
        poverty_source              TEXT,
        births_occurrence_both      INTEGER,
        births_occurrence_male      INTEGER,
        births_occurrence_female    INTEGER,
        births_residence_both       INTEGER,
        births_residence_male       INTEGER,
        births_residence_female     INTEGER,
        hospital_count_level1       INTEGER DEFAULT 0,
        hospital_count_level2       INTEGER DEFAULT 0,
        hospital_count_level3       INTEGER DEFAULT 0,
        total_facilities            INTEGER DEFAULT 0,
        total_hospitals             INTEGER DEFAULT 0,
        total_bed_capacity          INTEGER DEFAULT 0,
        weighted_facility_score     REAL    DEFAULT 0,
        private_count               INTEGER DEFAULT 0,
        gov_count                   INTEGER DEFAULT 0,
        private_ownership_pct       REAL,
        facility_density_per10k     REAL,
        hospital_density_per10k     REAL,
        beds_per_1000               REAL,
        weighted_score_per10k       REAL,
        level3_per100k              REAL,
        atm              INTEGER DEFAULT 0,
        bank             INTEGER DEFAULT 0,
        bar              INTEGER DEFAULT 0,
        bus_station      INTEGER DEFAULT 0,
        cafe             INTEGER DEFAULT 0,
        clinic           INTEGER DEFAULT 0,
        community_centre INTEGER DEFAULT 0,
        fast_food        INTEGER DEFAULT 0,
        fuel             INTEGER DEFAULT 0,
        hospital         INTEGER DEFAULT 0,
        parking          INTEGER DEFAULT 0,
        pharmacy         INTEGER DEFAULT 0,
        place_of_worship INTEGER DEFAULT 0,
        police           INTEGER DEFAULT 0,
        post_office      INTEGER DEFAULT 0,
        restaurant       INTEGER DEFAULT 0,
        school           INTEGER DEFAULT 0,
        shelter          INTEGER DEFAULT 0,
        toilets          INTEGER DEFAULT 0,
        townhall         INTEGER DEFAULT 0
    )
    """,
]

ALL_TABLES = [
    "lgu_master", "lgu_population", "lgu_poverty", "lgu_births",
    "lgu_infrastructure", "lgu_facilities", "lgu_targets", "lgu_merged",
]


# ═══════════════════════════════════════════════════════════════════════════
# HELPERS
# ═══════════════════════════════════════════════════════════════════════════

def _load_final(path: str) -> pd.DataFrame:
    """
    Load final_dataset_clean.xlsx, normalise the province column name,
    assign a surrogate integer key, and coerce numeric columns.
    """
    print(f"  Loading {os.path.basename(path)} ...")
    df = pd.read_excel(path)

    # Script 01 sometimes produces 'province_x' due to pandas join suffixing
    if "province_x" in df.columns and "province" not in df.columns:
        df = df.rename(columns={"province_x": "province"})

    # Coerce numeric columns
    numeric = (POPULATION_COLS + POVERTY_COLS[:-1] + BIRTH_COLS
               + INFRASTRUCTURE_COLS + FACILITY_COLS + TARGET_COLS)
    for col in numeric:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    # Assign stable surrogate PK
    df.insert(0, "lgu_id", range(1, len(df) + 1))

    print(f"  Loaded {len(df)} LGUs x {len(df.columns)} columns")
    return df


def _write_table(conn: sqlite3.Connection,
                 df: pd.DataFrame,
                 table: str,
                 cols: list) -> None:
    """Write selected columns of df into a SQLite table (replace if exists)."""
    available = [c for c in cols if c in df.columns]
    subset = df[available].copy()
    subset.to_sql(table, conn, if_exists="replace", index=False)
    print(f"  OK  {table:<25}  {len(subset):>6} rows x {len(subset.columns):>3} cols")


def _verify(conn: sqlite3.Connection) -> None:
    """Row-count check + orphan FK check + sample JOIN query."""
    print("\n  Verification:")
    for tbl in ALL_TABLES:
        cur = conn.execute(f"SELECT COUNT(*) FROM {tbl}")
        print(f"    {tbl:<25}  {cur.fetchone()[0]:>6} rows")

    # FK orphan check
    for tbl in ALL_TABLES:
        if tbl == "lgu_master":
            continue
        orphans = conn.execute(
            f"SELECT COUNT(*) FROM {tbl} t "
            f"LEFT JOIN lgu_master m ON t.lgu_id = m.lgu_id "
            f"WHERE m.lgu_id IS NULL"
        ).fetchone()[0]
        if orphans:
            print(f"  WARNING  {tbl}: {orphans} rows with no matching lgu_master entry")

    # Demonstrate cross-table SQL query
    print("\n  Sample JOIN — top 5 LGUs with Level 3 hospitals:")
    q = """
        SELECT m.city_municipality,
               m.province,
               t.hospital_count_level3,
               p.population_2024,
               pov.poverty_incidence_2023_pct
        FROM   lgu_master      m
        JOIN   lgu_targets     t   ON m.lgu_id = t.lgu_id
        JOIN   lgu_population  p   ON m.lgu_id = p.lgu_id
        JOIN   lgu_poverty     pov ON m.lgu_id = pov.lgu_id
        WHERE  t.hospital_count_level3 > 0
        ORDER  BY t.hospital_count_level3 DESC
        LIMIT  5
    """
    print(pd.read_sql(q, conn).to_string(index=False))


# ═══════════════════════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════════════════════

def store() -> None:
    """Load final_dataset_clean.xlsx and persist to SQLite."""
    os.makedirs(PROCESSED_DIR, exist_ok=True)

    df = _load_final(FINAL_XLSX)

    print(f"\n  Connecting to {DB_PATH} ...")
    conn = sqlite3.connect(DB_PATH)
    conn.execute("PRAGMA foreign_keys = ON")
    conn.execute("PRAGMA journal_mode = WAL")

    for stmt in DDL_STATEMENTS:
        conn.execute(stmt)
    conn.commit()
    print("  Schema initialised (8 tables)")

    print("\n  Writing tables ...")
    _write_table(conn, df, "lgu_master",         ["lgu_id"] + ID_COLS)
    _write_table(conn, df, "lgu_population",     ["lgu_id"] + POPULATION_COLS)
    _write_table(conn, df, "lgu_poverty",        ["lgu_id"] + POVERTY_COLS)
    _write_table(conn, df, "lgu_births",         ["lgu_id"] + BIRTH_COLS)
    _write_table(conn, df, "lgu_infrastructure", ["lgu_id"] + INFRASTRUCTURE_COLS)
    _write_table(conn, df, "lgu_facilities",     ["lgu_id"] + FACILITY_COLS)
    _write_table(conn, df, "lgu_targets",        ["lgu_id"] + TARGET_COLS)

    all_data = (ID_COLS + POPULATION_COLS + POVERTY_COLS + BIRTH_COLS
                + TARGET_COLS + FACILITY_COLS + INFRASTRUCTURE_COLS)
    _write_table(conn, df, "lgu_merged", ["lgu_id"] + all_data)

    conn.commit()
    _verify(conn)
    conn.close()

    size_kb = os.path.getsize(DB_PATH) / 1024
    print(f"\n  DB size on disk: {size_kb:.1f} KB")


if __name__ == "__main__":
    print("=" * 70)
    print("PREDICTING NUMBER OF HOSPITALS — SCRIPT 02: DATA STORAGE")
    print("Format: SQLite  ->  data/processed/hospital_data.db")
    print("=" * 70)

    if not os.path.exists(FINAL_XLSX):
        print(f"\nERROR: {FINAL_XLSX} not found.")
        print("Ensure data/clean/final_dataset_clean.xlsx exists.")
        raise SystemExit(1)

    store()

    print("\n" + "=" * 70)
    print("DONE. SQLite database -> data/processed/hospital_data.db")
    print()
    print("  Tables:")
    print("    lgu_master          geographic identifiers (PK: lgu_id)")
    print("    lgu_population      PSA census 2020 / 2024")
    print("    lgu_poverty         PSA poverty incidence 2018/21/23")
    print("    lgu_births          PSA registered births 2023")
    print("    lgu_infrastructure  OSM amenity counts (20 types)")
    print("    lgu_facilities      DOH NHFR aggregated supply metrics")
    print("    lgu_targets         hospital counts L1 / L2 / L3")
    print("    lgu_merged          full denormalised feature matrix")
    print()
    print("  Inspect: DB Browser for SQLite (https://sqlitebrowser.org)")
    print("  Load in Python:")
    print("    import sqlite3, pandas as pd")
    print("    conn = sqlite3.connect('data/processed/hospital_data.db')")
    print("    df   = pd.read_sql('SELECT * FROM lgu_merged', conn)")
    print()
    print("Next step: run 03_preprocessing.py")
    print("=" * 70)