"""
================================================================================
SCRIPT 01: Data Cleaning & Feature Engineering
Project:   Predicting Number of Hospitals by Level (L1 / L2 / L3)
           per Municipality/City based on Socioeconomic Factors
================================================================================

RESEARCH QUESTION
-----------------
How many Level 1, Level 2, and Level 3 hospitals should a municipality or city
have, given its population size, poverty rate, and registered live births?

TARGET VARIABLES
----------------
  hospital_count_level1  — Number of DOH Level 1 hospitals in the LGU
  hospital_count_level2  — Number of DOH Level 2 hospitals in the LGU
  hospital_count_level3  — Number of DOH Level 3 hospitals in the LGU

  HOW THESE ARE COMPUTED:
  ----------------------------------------------------------
  Source:  DOH National Health Facilities Registry (NHFR), facilities_raw.xlsx
  Step 1.  All 44,313 raw NHFR records are deduplicated on
           (facility_name, city_municipality): large hospitals appear 2–4 times
           in the NHFR because each licensed sub-service (drug testing unit,
           dialysis bay, etc.) gets its own row.  We keep only the single row
           with the HIGHEST DOH Service Capability level per hospital name.
           Ties broken by bed_capacity (descending), then facility_code.
  Step 2.  doh_level is parsed from the "Service Capability" field:
             "Level 3" → 3  (tertiary: ICU, specialty depts, surgical suites)
             "Level 2" → 2  (secondary: general surgery + internal medicine)
             "Level 1" → 1  (primary hospital: basic inpatient, minor surgery)
             anything else → 0  (clinic, RHU, laboratory, pharmacy, etc.)
  Step 3.  Only rows with facility_category == "hospital" or "infirmary" AND
           doh_level ∈ {1, 2, 3} are counted toward the targets.
           Rows with doh_level == 0 but category == "hospital" exist (hospitals
           with a missing or unlabelled Service Capability field).  They
           contribute to total_hospitals but NOT to the level counts.
  Step 4.  Facilities are grouped by (city_municipality, province) and
           aggregated into hospital_count_level1/2/3.

  KNOWN DATA LIMITATION:
  The NHFR "Service Capability" field is sometimes blank for older registrations.
  A hospital with no capability tag gets doh_level = 0 even if it operates
  at Level 2 capacity.  These are in total_hospitals for reference but excluded
  from level-specific counts to avoid misclassification.

FEATURE DIMENSIONS
------------------
  DEMAND (population)
    population_2020, population_2024, pop_growth_rate_pct

  BARRIER / ECONOMIC (poverty)
    poverty_incidence_2018_pct   — % of families below poverty line, 2018
    poverty_incidence_2021_pct   — % of families below poverty line, 2021
    poverty_incidence_2023_pct   — % of families below poverty line, 2023
    poverty_source               — "municipal" or "ncr_huc" (data provenance)

  BIRTHS (PSA 2023)
    births_occurrence_both    — Live births by place of occurrence (both sexes)
    births_occurrence_male
    births_occurrence_female
    births_residence_both     — Live births by mother's usual residence
    births_residence_male
    births_residence_female

DATA SOURCES
------------
  facilities_raw.xlsx        — DOH NHFR, ~44,000 licensed facilities, PHL-wide
  population_raw.xlsx        — PSA Census (2020 + 2024, 18 regional sheets)
  poverty_raw.xlsx           — PSA Poverty Incidence tab1a; used ONLY for the
                               16 NCR HUC cities (all NCR except Manila)
  poverty_municipal_raw.xlsx — PSA municipal-level poverty 2018/2021/2023
                               (used for all other LGUs + Manila's districts)
  birth_raw.xlsx             — PSA registered live births 2023 per LGU

POVERTY JOIN LOGIC:
-------------------
  Source A (poverty_raw.xlsx, tab1a):
    The 16 NCR cities excluding Manila (Caloocan, Makati, Quezon City, etc.)
    appear in this file with HUC-level poverty incidence.
    Manila is excluded here because its sub-district data is in Source B.

  Source B (poverty_municipal_raw.xlsx):
    All ~1,612 municipalities/cities nationwide, including Manila's districts
    (Tondo, Binondo, Sampaloc, etc.).
    Regional/provincial header rows are excluded (no numeric PSGC ID).

  The two sources are stacked into a single poverty_clean.xlsx table.
  In merged_clean.xlsx, each LGU is matched by city_municipality name.
  Duplicate names across provinces are resolved by keeping first match
  (acceptable since poverty data is primarily used as a feature, not
  a precise municipality-specific identifier across all 1,600+ LGUs).

FACILITY JOIN KEY (NCR HUC fix):
---------------------------------
  In the NHFR, NCR HUCs carry province = "CITY OF CALOOCAN (HUC)" etc.
  In the population table, NCR cities have province = None (NCR has no
  provinces).  To match, we strip the "(HUC)" tag from the NHFR province
  field, then match on (city_municipality, province_stripped).
  For NCR HUCs, province_stripped == city_municipality, so the match key
  becomes (city, city).

OUTPUTS (written to data/clean/)
---------------------------------
  facilities_clean.xlsx      — Deduplicated facility list (nationwide)
  population_clean.xlsx      — All LGUs: 2020/2024 population + growth rate
  poverty_clean.xlsx         — Municipal + NCR HUC poverty (2018/2021/2023)
  births_clean.xlsx          — Live births 2023 per LGU (occurrence + residence)
  merged_clean.xlsx          — Full LGU feature + target matrix
================================================================================
"""

import os
import re
import warnings

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore", category=UserWarning)

# ── Paths ──────────────────────────────────────────────────────────────────
BASE_DIR  = os.path.dirname(os.path.abspath(__file__))
RAW_DIR   = os.path.join(BASE_DIR, "data", "raw")
CLEAN_DIR = os.path.join(BASE_DIR, "data", "clean")

NHFR_FILE        = "facilities_raw.xlsx"
POPULATION_FILE  = "population_raw.xlsx"
POVERTY_FILE     = "poverty_raw.xlsx"
POVERTY_MUN_FILE = "poverty_municipal_raw.xlsx"
BIRTH_FILE       = "birth_raw.xlsx"

OUT_FACILITIES = "facilities_clean.xlsx"
OUT_POPULATION = "population_clean.xlsx"
OUT_POVERTY    = "poverty_clean.xlsx"
OUT_BIRTHS     = "births_clean.xlsx"
OUT_MERGED     = "merged_clean.xlsx"

# ── Columns to drop from raw NHFR ─────────────────────────────────────────
DROP_COLS = [
    "Health Facility Code Short",
    "Ownership Sub-Classification for Government facilities",
    "Ownership Sub-Classification for private facilities",
    "Old Health Facility Name 1",
    "Old Health Facility Name 2",
    "Old Health Facility Name 3",
    "Street Name and #",
    "Building name and #",
    "Region PSGC",
    "Province PSGC",
    "City/Municipality PSGC",
    "Email Address",
    "Alternate Email Address",
    "Barangay PSGC",
    "Zip Code",
    "Landline Number",
    "Landline Number 2",
    "Fax Number",
    "Official Website",
]

# ── Population sheet names ─────────────────────────────────────────────────
POPULATION_SHEETS = [
    "NCR", "CAR", "R01", "R02", "R03", "R04A", "MIMAROPA",
    "R05", "NIR", "R06", "R07", "R08", "R09", "R10",
    "R11", "R12", "Caraga", "BARMM",
]

# ── Known PSA province names (used to skip province total rows) ────────────
KNOWN_PROVINCES = {
    "ABRA", "AGUSAN DEL NORTE", "AGUSAN DEL SUR", "AKLAN", "ALBAY",
    "ANTIQUE", "APAYAO", "AURORA", "BASILAN", "BATAAN", "BATANES",
    "BATANGAS", "BENGUET", "BILIRAN", "BOHOL", "BUKIDNON", "BULACAN",
    "CAGAYAN", "CAMARINES NORTE", "CAMARINES SUR", "CAMIGUIN", "CAPIZ",
    "CATANDUANES", "CAVITE", "CEBU", "COTABATO", "DAVAO DE ORO",
    "DAVAO DEL NORTE", "DAVAO DEL SUR", "DAVAO OCCIDENTAL", "DAVAO ORIENTAL",
    "DINAGAT ISLANDS", "EASTERN SAMAR", "GUIMARAS", "IFUGAO", "ILOCOS NORTE",
    "ILOCOS SUR", "ILOILO", "ISABELA", "KALINGA", "LA UNION", "LAGUNA",
    "LANAO DEL NORTE", "LANAO DEL SUR", "LEYTE", "MAGUINDANAO DEL NORTE",
    "MAGUINDANAO DEL SUR", "MARINDUQUE", "MASBATE", "MISAMIS OCCIDENTAL",
    "MISAMIS ORIENTAL", "MOUNTAIN PROVINCE", "MT. PROVINCE",
    "NEGROS OCCIDENTAL", "NEGROS ORIENTAL", "NORTHERN SAMAR", "NUEVA ECIJA",
    "NUEVA VIZCAYA", "OCCIDENTAL MINDORO", "ORIENTAL MINDORO", "PALAWAN",
    "PAMPANGA", "PANGASINAN", "QUEZON", "QUIRINO", "RIZAL", "ROMBLON",
    "SAMAR", "SARANGANI", "SIQUIJOR", "SORSOGON", "SOUTH COTABATO",
    "SOUTHERN LEYTE", "SULTAN KUDARAT", "SULU", "SURIGAO DEL NORTE",
    "SURIGAO DEL SUR", "TARLAC", "TAWI-TAWI", "ZAMBALES",
    "ZAMBOANGA DEL NORTE", "ZAMBOANGA DEL SUR", "ZAMBOANGA SIBUGAY",
}

# ── NCR HUC cities that have city-level poverty data in poverty_raw.xlsx ──
# Manila is excluded — its district-level data comes from poverty_municipal_raw.
NCR_HUC_CITIES = {
    "CITY OF MANDALUYONG", "CITY OF MARIKINA", "CITY OF PASIG",
    "QUEZON CITY", "CITY OF SAN JUAN", "CITY OF CALOOCAN",
    "CITY OF MALABON", "CITY OF NAVOTAS", "CITY OF VALENZUELA",
    "CITY OF LAS PIÑAS", "CITY OF MAKATI", "CITY OF MUNTINLUPA",
    "CITY OF PARAÑAQUE", "PASAY CITY", "PATEROS", "CITY OF TAGUIG",
}


# ═══════════════════════════════════════════════════════════════════════════
# HELPERS
# ═══════════════════════════════════════════════════════════════════════════

def normalize_text(s: str) -> str:
    return re.sub(r"\s+", " ", str(s).strip().upper())


def clean_bed_capacity(series: pd.Series) -> pd.Series:
    return (
        series.astype(str)
        .str.replace(",", "", regex=False)
        .str.strip()
        .pipe(pd.to_numeric, errors="coerce")
        .fillna(0)
        .astype(int)
    )


def parse_doh_level(svc: str) -> int:
    s = str(svc).upper()
    if "LEVEL 3" in s: return 3
    if "LEVEL 2" in s: return 2
    if "LEVEL 1" in s: return 1
    return 0


def assign_service_level_weight(row: pd.Series) -> float:
    svc  = str(row.get("service_capability", "")).upper()
    ftyp = str(row.get("facility_type",       "")).upper()
    try:
        beds = float(row.get("bed_capacity", 0) or 0)
    except (ValueError, TypeError):
        beds = 0.0

    if "LEVEL 3" in svc:  return 4.0
    if "LEVEL 2" in svc:  return 3.0
    if "LEVEL 1" in svc:  return 2.5
    if "HOSPITAL" in ftyp or "INFIRMARY" in ftyp:
        if beds >= 100: return 4.0
        elif beds >= 50: return 3.0
        else: return 2.5
    if "DIALYSIS"        in ftyp: return 2.0
    if "CANCER"          in ftyp: return 2.0
    if "KIDNEY TRANSPLANT" in ftyp: return 2.0
    if "PSYCHIATRIC"     in ftyp or "CUSTODIAL" in svc: return 2.0
    if "AMBULATORY SURGICAL" in ftyp: return 1.5
    if "BIRTHING"        in ftyp or "LYING-IN" in ftyp: return 1.5
    if "RURAL HEALTH"    in ftyp or "HEALTH CENTER" in ftyp: return 1.5
    if "BARANGAY HEALTH" in ftyp: return 1.5
    if "CLINIC"          in ftyp: return 1.5
    if "LABORATORY"      in ftyp or "DIAGNOSTIC" in ftyp: return 1.0
    if "PHARMACY"        in ftyp or "DRUGSTORE" in ftyp: return 1.0
    if "BLOOD"           in ftyp: return 1.0
    if "DRUG TESTING"    in ftyp: return 0.5
    if "AMBULANCE"       in ftyp: return 0.5
    return 1.0


def categorise_facility(row: pd.Series) -> str:
    ftyp = str(row.get("facility_type", "")).upper()
    if "HOSPITAL"        in ftyp: return "hospital"
    if "INFIRMARY"       in ftyp: return "infirmary"
    if "RURAL HEALTH"    in ftyp: return "rhu"
    if "BARANGAY HEALTH" in ftyp: return "bhs"
    if "BIRTHING" in ftyp or "LYING" in ftyp: return "birthing"
    if "DIALYSIS"        in ftyp: return "dialysis"
    if "CLINIC"          in ftyp: return "clinic"
    if "LABORATORY"      in ftyp: return "laboratory"
    if "PHARMACY" in ftyp or "DRUGSTORE" in ftyp: return "pharmacy"
    if "DRUG TESTING"    in ftyp: return "drug_testing"
    if "AMBULANCE"       in ftyp: return "ambulance"
    return "other"


def svc_priority(svc: str) -> int:
    s = str(svc).upper()
    if "LEVEL 3" in s: return 0
    if "LEVEL 2" in s: return 1
    if "LEVEL 1" in s: return 2
    return 9


def safe_float(val, fallback=np.nan):
    try:
        v = float(str(val).replace(",", "").strip())
        return v if not (isinstance(v, float) and np.isnan(v)) else fallback
    except (ValueError, TypeError):
        return fallback


def strip_huc_tag(s: str) -> str:
    """'CITY OF CALOOCAN (HUC)' → 'CITY OF CALOOCAN'"""
    return re.sub(r"\s*\(HUC\)\s*", "", str(s)).strip()


# ═══════════════════════════════════════════════════════════════════════════
# STEP 1 — Clean NHFR facilities
# ═══════════════════════════════════════════════════════════════════════════

def clean_facilities() -> pd.DataFrame:
    print("\n[1/5] Cleaning NHFR facilities data (nationwide)...")
    path = os.path.join(RAW_DIR, NHFR_FILE)
    df = pd.read_excel(path)
    print(f"  Raw shape: {df.shape}")

    df = df.drop(columns=DROP_COLS, errors="ignore")
    df = df.rename(columns={
        "Health Facility Code":           "facility_code",
        "Facility Name":                  "facility_name",
        "Facility Major Type":            "facility_major_type",
        "Health Facility Type":           "facility_type",
        "Ownership Major Classification": "ownership",
        "Region Name":                    "region",
        "Province Name":                  "province",
        "City/Municipality Name":         "city_municipality",
        "Barangay Name":                  "barangay",
        "Service Capability":             "service_capability",
        "Bed Capacity":                   "bed_capacity",
        "Licensing Status":               "license_status",
        "License Validity Date":          "license_validity",
    })

    for col in ["city_municipality", "province", "region"]:
        df[col] = df[col].apply(lambda x: normalize_text(x) if pd.notna(x) else x)

    df["bed_capacity"] = clean_bed_capacity(df["bed_capacity"])

    # Deduplication: keep best row per (facility_name, city_municipality)
    df["_svc_priority"] = df["service_capability"].apply(svc_priority)
    df = (df
          .sort_values(
              ["facility_name", "city_municipality", "_svc_priority", "bed_capacity"],
              ascending=[True, True, True, False]
          )
          .drop_duplicates(subset=["facility_name", "city_municipality"], keep="first")
          .drop(columns=["_svc_priority"])
          .reset_index(drop=True)
    )
    print(f"  After deduplication: {len(df)} unique facility-city rows")

    df["doh_level"]            = df["service_capability"].apply(parse_doh_level)
    df["service_level_weight"] = df.apply(assign_service_level_weight, axis=1)
    df["facility_category"]    = df.apply(categorise_facility, axis=1)
    df["is_private"] = (
        df["ownership"].astype(str).str.upper().str.contains("PRIVATE", na=False).astype(int)
    )
    df["is_licensed"] = (
        df["license_status"].astype(str).str.upper().str.contains("WITH LICENSE", na=False).astype(int)
    )

    out_path = os.path.join(CLEAN_DIR, OUT_FACILITIES)
    df.to_excel(out_path, index=False)
    print(f"  Saved {len(df)} rows → {OUT_FACILITIES}")
    print(f"\n  Facility category breakdown:")
    print(df["facility_category"].value_counts().to_string())
    print(f"\n  DOH level breakdown (doh_level > 0):")
    print(df[df["doh_level"] > 0]["doh_level"].value_counts().sort_index().to_string())
    return df


# ═══════════════════════════════════════════════════════════════════════════
# STEP 2 — Clean Population data
# ═══════════════════════════════════════════════════════════════════════════

def clean_population() -> pd.DataFrame:
    """
    Parses all 18 PSA census sheets. Outputs ONLY LGU rows — province and
    region aggregate rows are discarded entirely.

    Columns: city_municipality, province, region,
             population_2020, population_2024, pop_growth_rate_pct
    """
    print("\n[2/5] Cleaning population data (PSA Census, all regions)...")
    path = os.path.join(RAW_DIR, POPULATION_FILE)

    REGION_PATTERNS = re.compile(
        r"^(REGION|NATIONAL CAPITAL|CORDILLERA|BANGSAMORO|CARAGA|MIMAROPA|NEGROS ISLAND)",
        re.IGNORECASE
    )

    records = []
    for sheet in POPULATION_SHEETS:
        raw = pd.read_excel(path, sheet_name=sheet, header=None)
        region_title = str(raw.iloc[1, 0]).strip() if pd.notna(raw.iloc[1, 0]) else sheet
        region_title = re.sub(r":\s*\d{4}.*", "", region_title).strip()

        current_province = None

        for _, row in raw.iloc[6:].iterrows():
            name_raw = row.get(0)
            if pd.isna(name_raw):
                continue
            name_str = str(name_raw).strip()
            if not name_str or re.match(r"^(Note|Source|\d+\s|\*)", name_str, re.IGNORECASE):
                continue

            pop_2020 = safe_float(row.get(4))
            pop_2024 = safe_float(row.get(5))
            growth   = safe_float(row.get(9))

            if np.isnan(pop_2020) or pop_2020 <= 0:
                continue

            name_norm = normalize_text(name_str)
            name_norm = re.sub(r"\s+\d+$", "", name_norm).strip()

            is_region   = bool(REGION_PATTERNS.match(name_norm))
            name_bare   = re.sub(r"\s*\(.*?\)", "", name_norm).strip()
            is_province = not is_region and name_bare in KNOWN_PROVINCES

            if is_region or is_province:
                if is_province:
                    current_province = name_bare
                continue  # skip; not an LGU row

            records.append({
                "region":              region_title,
                "province":            current_province,
                "city_municipality":   name_norm,
                "population_2020":     int(pop_2020),
                "population_2024":     int(pop_2024) if not np.isnan(pop_2024) else None,
                "pop_growth_rate_pct": round(growth, 4) if not np.isnan(growth) else None,
            })

    df = pd.DataFrame(records)
    print(f"  LGU rows parsed: {len(df)}")
    df.to_excel(os.path.join(CLEAN_DIR, OUT_POPULATION), index=False)
    print(f"  Saved → {OUT_POPULATION}")
    return df


# ═══════════════════════════════════════════════════════════════════════════
# STEP 3 — Clean Poverty data
# ═══════════════════════════════════════════════════════════════════════════

def clean_poverty() -> pd.DataFrame:
    """
    Combines two PSA poverty sources into one clean table.

    SOURCE A — poverty_raw.xlsx (tab1a):
      Used ONLY for the 16 NCR HUC cities (all except Manila).
      Columns extracted: city_municipality, poverty_incidence_2018/2021/2023_pct.
      Poverty thresholds are dropped. Regional/provincial rows are excluded.
      Manila is excluded because its sub-district data is in SOURCE B.

    SOURCE B — poverty_municipal_raw.xlsx:
      Used for all other LGUs including Manila's districts.
      Detection: only rows with a numeric PSGC ID (col 0) are data rows.
      Regional/provincial header rows have col 0 = NaN → automatically excluded.
      Col mapping: 2=municipality name, 3=inc_2018, 4=inc_2021, 5=inc_2023.

    Final columns: city_municipality, poverty_incidence_2018_pct,
                   poverty_incidence_2021_pct, poverty_incidence_2023_pct,
                   poverty_source ("ncr_huc" | "municipal")
    """
    print("\n[3/5] Cleaning poverty data...")

    # ── SOURCE A: NCR HUC cities ──────────────────────────────────────────
    print("  Parsing NCR HUC cities from poverty_raw.xlsx (tab1a)...")
    raw_a  = pd.read_excel(os.path.join(RAW_DIR, POVERTY_FILE), sheet_name="tab1a", header=None)

    ncr_records = []
    in_ncr = False
    for _, row in raw_a.iloc[7:168].iterrows():
        name_raw = row.get(0)
        if pd.isna(name_raw):
            continue
        name_str = str(name_raw).strip()

        if re.search(r"National Capital Region", name_str, re.IGNORECASE):
            in_ncr = True
            continue
        if in_ncr and re.search(r"^(Cordillera|Region I[^V]|Region II|CAR)", name_str, re.IGNORECASE):
            break
        if not in_ncr:
            continue
        if re.match(r"^\d+(st|nd|rd|th)\s+District", name_str, re.IGNORECASE):
            continue

        name_norm = normalize_text(name_str)

        if name_norm not in NCR_HUC_CITIES:
            continue

        inc_2018 = safe_float(row.get(4))
        inc_2021 = safe_float(row.get(5))
        inc_2023 = safe_float(row.get(6))

        ncr_records.append({
            "city_municipality":          name_norm,
            "poverty_incidence_2018_pct": inc_2018 if not np.isnan(inc_2018) else None,
            "poverty_incidence_2021_pct": inc_2021 if not np.isnan(inc_2021) else None,
            "poverty_incidence_2023_pct": inc_2023 if not np.isnan(inc_2023) else None,
            "poverty_source":             "ncr_huc",
        })

    df_a = pd.DataFrame(ncr_records)
    print(f"    NCR HUC rows extracted: {len(df_a)}")

    # ── SOURCE B: Municipal-level data ────────────────────────────────────
    print("  Parsing municipal data from poverty_municipal_raw.xlsx...")
    raw_b = pd.read_excel(os.path.join(RAW_DIR, POVERTY_MUN_FILE), header=None)

    mun_records = []
    for _, row in raw_b.iterrows():
        psgc = row.get(0)
        if pd.isna(psgc) or not str(psgc).strip().isdigit():
            continue

        name_raw = row.get(2)
        if pd.isna(name_raw):
            continue

        name_norm = normalize_text(str(name_raw).strip())
        inc_2018  = safe_float(row.get(3))
        inc_2021  = safe_float(row.get(4))
        inc_2023  = safe_float(row.get(5))

        mun_records.append({
            "city_municipality":          name_norm,
            "poverty_incidence_2018_pct": inc_2018 if not np.isnan(inc_2018) else None,
            "poverty_incidence_2021_pct": inc_2021 if not np.isnan(inc_2021) else None,
            "poverty_incidence_2023_pct": inc_2023 if not np.isnan(inc_2023) else None,
            "poverty_source":             "municipal",
        })

    df_b = pd.DataFrame(mun_records)
    print(f"    Municipal rows extracted: {len(df_b)}")

    # ── Combine ───────────────────────────────────────────────────────────
    df = pd.concat([df_a, df_b], ignore_index=True)
    print(f"  Combined total: {len(df)} poverty entries")

    print("\n  NCR HUC entries from SOURCE A:")
    print(df[df["poverty_source"] == "ncr_huc"][
        ["city_municipality", "poverty_incidence_2018_pct",
         "poverty_incidence_2021_pct", "poverty_incidence_2023_pct"]
    ].to_string(index=False))

    print("\n  Manila district sample (SOURCE B):")
    manila_districts = ["TONDO", "BINONDO", "QUIAPO", "SAN NICOLAS", "SANTA CRUZ",
                        "SAMPALOC", "SAN MIGUEL", "ERMITA", "INTRAMUROS", "MALATE",
                        "PACO", "PANDACAN", "PORT AREA", "SANTA ANA"]
    print(df_b[df_b["city_municipality"].isin(manila_districts)][
        ["city_municipality", "poverty_incidence_2018_pct",
         "poverty_incidence_2021_pct", "poverty_incidence_2023_pct"]
    ].to_string(index=False))

    df.to_excel(os.path.join(CLEAN_DIR, OUT_POVERTY), index=False)
    print(f"\n  Saved → {OUT_POVERTY}")
    return df


# ═══════════════════════════════════════════════════════════════════════════
# STEP 4 — Clean Births data
# ═══════════════════════════════════════════════════════════════════════════

def clean_births() -> pd.DataFrame:
    """
    Parses PSA registered live births 2023 (birth_raw.xlsx).

    INDENTATION ENCODING in the raw file:
      The PSA uses a special character \x85 followed by dots as a visual
      indent.  The pattern in column 0 is:
        \x85...Name  — city/municipality (3 dots = deepest level → keep these)
        \x85.Name    — province          (1 dot  = intermediate level)
        Name         — region / PHILIPPINES total (no prefix)

      Only \x85... rows are LGU-level.  Province and region rows are dropped.

    The \x85... prefix is stripped after filtering to yield clean names.

    Column mapping (0-indexed after stripping header rows):
      0: name (with \x85... prefix in raw)
      1: occurrence both sexes,  2: occurrence male,  3: occurrence female
      4: residence both sexes,   5: residence male,   6: residence female

    OCCURRENCE vs RESIDENCE:
      Place of Occurrence: the physical location where the birth took place,
        regardless of the mother's home address.  Useful for measuring
        healthcare facility utilisation.
      Usual Residence: counted against the mother's registered home LGU —
        the demographically meaningful measure of demand for maternal and
        child health services in that community.
    """
    print("\n[4/5] Cleaning births data (PSA 2023)...")
    path = os.path.join(RAW_DIR, BIRTH_FILE)
    raw  = pd.read_excel(path, header=None)
    print(f"  Raw shape: {raw.shape}")

    # Filter: keep only rows whose col-0 string starts with \x85 + three dots
    lgu_mask = raw[0].astype(str).str.match(r"^\x85\.\.\.")
    df = raw[lgu_mask].copy().reset_index(drop=True)
    print(f"  LGU rows (\\x85... prefix): {len(df)}")

    # Strip the \x85... prefix and normalise
    df[0] = (df[0].astype(str)
             .str.replace(r"^\x85\.\.\.", "", regex=True)
             .str.strip()
             .apply(normalize_text))

    df = df.rename(columns={
        0: "city_municipality",
        1: "births_occurrence_both",
        2: "births_occurrence_male",
        3: "births_occurrence_female",
        4: "births_residence_both",
        5: "births_residence_male",
        6: "births_residence_female",
    })

    for col in df.columns[1:]:
        df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0).astype(int)

    print(f"\n  Total LGUs: {len(df)}")
    print(f"  Total births occurrence (both): {df['births_occurrence_both'].sum():,}")
    print(f"  Total births residence  (both): {df['births_residence_both'].sum():,}")
    print(f"\n  Sample rows:")
    print(df.head(10).to_string(index=False))

    df.to_excel(os.path.join(CLEAN_DIR, OUT_BIRTHS), index=False)
    print(f"\n  Saved → {OUT_BIRTHS}")
    return df


# ═══════════════════════════════════════════════════════════════════════════
# STEP 5 — Build facility aggregates + merge all tables
# ═══════════════════════════════════════════════════════════════════════════

def build_facility_aggregates(df_fac: pd.DataFrame) -> pd.DataFrame:
    """
    Aggregates cleaned facilities to (city_municipality, province, region) level.

    hospital_count_level1/2/3:
      Count of rows where facility_category IN ("hospital", "infirmary")
      AND doh_level == 1/2/3 respectively.
      Hospitals with doh_level == 0 (missing Service Capability tag) are
      counted in total_hospitals but NOT in the level-specific columns.
    """
    agg = df_fac.groupby(["city_municipality", "province", "region"]).agg(
        total_facilities      = ("facility_name",     "count"),
        total_hospitals       = ("facility_category",
                                 lambda x: x.isin(["hospital", "infirmary"]).sum()),
        hospital_count_level1 = ("doh_level",
                                 lambda x: ((x == 1) &
                                  df_fac.loc[x.index, "facility_category"]
                                        .isin(["hospital", "infirmary"])).sum()),
        hospital_count_level2 = ("doh_level",
                                 lambda x: ((x == 2) &
                                  df_fac.loc[x.index, "facility_category"]
                                        .isin(["hospital", "infirmary"])).sum()),
        hospital_count_level3 = ("doh_level",
                                 lambda x: ((x == 3) &
                                  df_fac.loc[x.index, "facility_category"]
                                        .isin(["hospital", "infirmary"])).sum()),
        total_bed_capacity        = ("bed_capacity",         "sum"),
        weighted_facility_score   = ("service_level_weight", "sum"),
        private_count             = ("is_private",           "sum"),
        gov_count                 = ("is_private",
                                     lambda x: (x == 0).sum()),
    ).reset_index()

    agg["private_ownership_pct"] = (
        agg["private_count"] / agg["total_facilities"].replace(0, np.nan)
    ).round(4)

    return agg


def merge_all(fac_df: pd.DataFrame, pop_df: pd.DataFrame,
              pov_df: pd.DataFrame, birth_df: pd.DataFrame) -> pd.DataFrame:
    print("\n[5/5] Building aggregates and merging all tables...")

    fac_agg = build_facility_aggregates(fac_df)
    print(f"  Facility aggregates: {len(fac_agg)} (city, province) combinations")

    # ── Facility join key: strip (HUC) from NHFR province field ──────────
    # NHFR province for NCR HUCs: "CITY OF CALOOCAN (HUC)" → "CITY OF CALOOCAN"
    # This makes the province field match city_municipality in population for NCR.
    fac_agg["province_join"] = fac_agg["province"].apply(
        lambda x: strip_huc_tag(x) if pd.notna(x) else None
    )

    # Build facility lookup: (city_municipality, province_join) → fac_agg row index
    fac_agg["_key"] = list(zip(fac_agg["city_municipality"], fac_agg["province_join"]))
    fac_dict = {r["_key"]: ri for ri, r in fac_agg.iterrows()}

    # ── Build poverty lookup: city_municipality → pov_df row (as dict) ───
    pov_lookup = {}
    for _, pov_row in pov_df.iterrows():
        key = pov_row["city_municipality"]
        if key not in pov_lookup:
            pov_lookup[key] = pov_row.to_dict()

    # ── Build birth lookup ────────────────────────────────────────────────
    birth_lookup = {}
    for _, b_row in birth_df.iterrows():
        birth_lookup[b_row["city_municipality"]] = b_row.to_dict()

    # ── Start from population table ───────────────────────────────────────
    merged = pop_df.copy()
    print(f"  Population LGU rows: {len(merged)}")

    # ── Initialise all target + supply columns ────────────────────────────
    fac_cols = ["total_facilities", "total_hospitals",
                "hospital_count_level1", "hospital_count_level2", "hospital_count_level3",
                "total_bed_capacity", "weighted_facility_score",
                "private_count", "gov_count", "private_ownership_pct"]
    for col in fac_cols:
        merged[col] = 0.0  # float to avoid dtype conflicts during row assignment

    # ── Join facilities ───────────────────────────────────────────────────
    # For NCR HUCs: province is None in population; in fac_agg province_join
    # equals city_municipality (e.g. "CITY OF CALOOCAN").
    # Key: (city_municipality, province_join).
    # NCR HUC: key = (city, city). Regular LGU: key = (city, province).
    for i, row in merged.iterrows():
        city  = row["city_municipality"]
        prov  = normalize_text(row["province"]) if pd.notna(row["province"]) else None
        key   = (city, city) if prov is None else (city, prov)

        if key in fac_dict:
            fac_row = fac_agg.loc[fac_dict[key]]
            for col in fac_cols:
                merged.at[i, col] = fac_row.get(col, 0)

    # ── Join poverty ──────────────────────────────────────────────────────
    pov_cols = ["poverty_incidence_2018_pct", "poverty_incidence_2021_pct",
                "poverty_incidence_2023_pct", "poverty_source"]
    for col in pov_cols:
        merged[col] = None

    for i, row in merged.iterrows():
        key = row["city_municipality"]
        if key in pov_lookup:
            pov_row = pov_lookup[key]
            for col in pov_cols:
                merged.at[i, col] = pov_row.get(col, None)

    # ── Join births ───────────────────────────────────────────────────────
    birth_cols = ["births_occurrence_both", "births_occurrence_male",
                  "births_occurrence_female", "births_residence_both",
                  "births_residence_male", "births_residence_female"]
    for col in birth_cols:
        merged[col] = None

    for i, row in merged.iterrows():
        key = row["city_municipality"]
        if key in birth_lookup:
            b_row = birth_lookup[key]
            for col in birth_cols:
                merged.at[i, col] = b_row.get(col, None)

    # ── Per-capita derived features ───────────────────────────────────────
    pop = merged["population_2024"].fillna(merged["population_2020"]).replace(0, np.nan)
    merged["facility_density_per10k"] = (merged["total_facilities"]    / pop * 10_000).round(4)
    merged["hospital_density_per10k"] = (merged["total_hospitals"]     / pop * 10_000).round(4)
    merged["beds_per_1000"]           = (merged["total_bed_capacity"]   / pop * 1_000 ).round(4)
    merged["weighted_score_per10k"]   = (merged["weighted_facility_score"] / pop * 10_000).round(4)
    merged["level3_per100k"]          = (merged["hospital_count_level3"]   / pop * 100_000).round(4)

    # ── Reorder columns ───────────────────────────────────────────────────
    id_cols     = ["city_municipality", "province", "region"]
    pop_cols    = ["population_2020", "population_2024", "pop_growth_rate_pct"]
    pov_out     = ["poverty_incidence_2018_pct", "poverty_incidence_2021_pct",
                   "poverty_incidence_2023_pct", "poverty_source"]
    birth_occ   = ["births_occurrence_both", "births_occurrence_male", "births_occurrence_female"]
    birth_res   = ["births_residence_both",  "births_residence_male",  "births_residence_female"]
    target_cols = ["hospital_count_level1", "hospital_count_level2", "hospital_count_level3"]
    supply_cols = ["total_facilities", "total_hospitals", "total_bed_capacity",
                   "weighted_facility_score", "private_count", "gov_count",
                   "private_ownership_pct", "facility_density_per10k",
                   "hospital_density_per10k", "beds_per_1000",
                   "weighted_score_per10k", "level3_per100k"]

    final_cols = [c for c in
                  id_cols + pop_cols + pov_out + birth_occ + birth_res
                  + target_cols + supply_cols
                  if c in merged.columns]
    merged = merged[final_cols]

    # ── Summary ───────────────────────────────────────────────────────────
    print(f"\n  Merged shape: {merged.shape}")

    print(f"\n  Hospital target variable summary:")
    for t in target_cols:
        vals = pd.to_numeric(merged[t], errors="coerce").fillna(0).astype(int)
        print(f"    {t}: mean={vals.mean():.3f}, max={vals.max()}, "
              f"% zero={(vals==0).mean()*100:.1f}%")

    print(f"\n  Poverty coverage: "
          f"{merged['poverty_incidence_2023_pct'].notna().sum()} / {len(merged)} LGUs")
    print(f"  Poverty source breakdown: {merged['poverty_source'].value_counts().to_dict()}")

    print(f"\n  Birth data coverage: "
          f"{merged['births_residence_both'].notna().sum()} / {len(merged)} LGUs")

    print(f"\n  Sample LGUs with Level 3 hospitals:")
    l3 = merged[pd.to_numeric(merged["hospital_count_level3"], errors="coerce") > 0][[
        "city_municipality", "province",
        "hospital_count_level1", "hospital_count_level2", "hospital_count_level3",
        "total_hospitals"
    ]]
    print(l3.head(15).to_string(index=False))

    merged.to_excel(os.path.join(CLEAN_DIR, OUT_MERGED), index=False)
    print(f"\n  Saved full feature matrix → {OUT_MERGED}")
    return merged


# ═══════════════════════════════════════════════════════════════════════════
# ENTRY POINT
# ═══════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    print("=" * 70)
    print("PREDICTING NUMBER OF HOSPITALS — SCRIPT 01: DATA CLEANING")
    print("Targets: hospital_count_level1 / level2 / level3 per LGU")
    print("=" * 70)

    os.makedirs(CLEAN_DIR, exist_ok=True)

    missing = [
        f for f in [NHFR_FILE, POPULATION_FILE, POVERTY_FILE, POVERTY_MUN_FILE, BIRTH_FILE]
        if not os.path.exists(os.path.join(RAW_DIR, f))
    ]
    if missing:
        print("\nERROR: Missing input files in data/raw/:")
        for f in missing:
            print(f"  ✗  {f}")
        raise SystemExit(1)

    fac_df   = clean_facilities()
    pop_df   = clean_population()
    pov_df   = clean_poverty()
    birth_df = clean_births()
    merged   = merge_all(fac_df, pop_df, pov_df, birth_df)

    print("\n" + "=" * 70)
    print("DONE. Cleaned files written to data/clean/")
    print("  facilities_clean.xlsx  — deduplicated facility list (nationwide)")
    print("  population_clean.xlsx  — all LGUs, 2020/2024 pop + growth")
    print("  poverty_clean.xlsx     — municipal + NCR HUC poverty incidence")
    print("  births_clean.xlsx      — registered live births 2023 per LGU")
    print("  merged_clean.xlsx      — full LGU feature + target matrix")
    print("\nNext step: run 02_database.py")
    print("=" * 70)