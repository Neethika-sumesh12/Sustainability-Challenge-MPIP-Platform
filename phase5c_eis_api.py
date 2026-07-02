"""
Phase 5C: Material Intelligence API + Mock IBM EIS Data Feed
Demonstrates the complete data flow:
  MRF Sensor → MPIP Material Intelligence API → IBM EIS Scope 3 Dashboard

IBM EIS (Environmental Intelligence Suite) currently ingests sustainability
data but has no way to verify the quality of PCR material claimed by suppliers.
MPIP's Material Intelligence API fills that verification gap.

No new packages. No network. Pure local Python.
"""

import pandas as pd
import json
from datetime import datetime, timezone
from pathlib import Path

print("=" * 100)
print("PHASE 5C: MATERIAL INTELLIGENCE API + IBM EIS INTEGRATION DEMO")
print("Data flow: MRF Sensor → MPIP API → IBM EIS Scope 3 Dashboard")
print("=" * 100)

# ============================================================================
# STEP 1: Load records
# ============================================================================
df  = pd.read_csv("MPIP_granite_material_records.csv")
print(f"\n[1] Loaded {len(df)} records")

# ============================================================================
# STEP 2: Simulate a MRF batch — group pure polymer samples into a
# hypothetical sorting run at a recycling facility
# ============================================================================
batch_id   = "MRF-BATCH-20260623-001"
facility   = "Veolia MRF Frankfurt — Conveyor Line 3"
operator   = "MPIP v1.0 (IBM Granite 3-8B + NIR Spectroscopy)"
timestamp  = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")

pure = df[df["polymer_type"] != "BLEND"].copy()
blends = df[df["polymer_type"] == "BLEND"].copy()

# ============================================================================
# STEP 3: Build Material Intelligence API response
# This is the JSON payload MPIP's API would return to IBM EIS
# ============================================================================

def build_api_payload(df_batch, batch_id, facility):
    items = []
    for _, row in df_batch.iterrows():
        items.append({
            "item_id":            row["sample_id"],
            "polymer_type":       row["polymer_type"],
            "quality_grade":      row["quality_grade"],
            "recyclability_score": int(row["recyclability_score"]),
            "confidence_pct":     int(row.get("confidence_pct", 80)),
            "dpp_eligible":       bool(row["dpp_eligible"]),
            "contamination_flag": bool(row["contamination_flag"]),
            "recycling_pathway":  row["recycling_pathway"],
            "degradation_level":  row["degradation_level"],
            "carbon_footprint_class": row["carbon_footprint_class"],
            "density_g_cm3":      float(row["Predicted Density"]),
            "crystallinity":      float(row["Predicted Crystallinity"]),
            "scb_ch3_1000c":      float(row["Predicted SCB"]),
        })

    grade_a = sum(1 for i in items if i["quality_grade"] == "A")
    grade_b = sum(1 for i in items if i["quality_grade"] == "B")
    grade_c = sum(1 for i in items if i["quality_grade"] == "C")
    cont    = sum(1 for i in items if i["contamination_flag"])
    dpp_n   = sum(1 for i in items if i["dpp_eligible"])
    avg_rec = round(sum(i["recyclability_score"] for i in items) / len(items), 1) if items else 0

    payload = {
        "api_version":  "MPIP-API v1.0",
        "batch_id":     batch_id,
        "facility":     facility,
        "operator":     operator,
        "timestamp":    timestamp,
        "total_items":  len(items),
        "batch_summary": {
            "grade_A_count":          grade_a,
            "grade_B_count":          grade_b,
            "grade_C_or_reject":      grade_c,
            "contamination_flags":    cont,
            "dpp_eligible_count":     dpp_n,
            "avg_recyclability_score": avg_rec,
            "premium_pcr_yield_pct":  round(grade_a / len(items) * 100, 1) if items else 0,
        },
        "items": items,
    }
    return payload

api_payload = build_api_payload(df, batch_id, facility)

# ============================================================================
# STEP 4: Build IBM EIS Scope 3 verified data entry
# This is what flows into the IBM EIS dashboard for a Fortune 500 client
# using MPIP-verified PCR material in their products
# ============================================================================

# Demo scenario: Fortune 500 client purchases Grade A HDPE + PP from this MRF batch.
# Grades and DPP IDs are real (from NIST data + IBM Granite grading).
# Batch weights use standard industry bale assumptions (replace with weighbridge data in production).
# CO2e rates sourced from PlasticsEurope Eco-profiles 2023 + WRAP UK LCA data.
grade_a_hdpe = df[(df["polymer_type"] == "HDPE") & (df["quality_grade"] == "A")]
grade_a_pp   = df[(df["polymer_type"] == "PP")   & (df["quality_grade"] == "A")]

carbon_savings = {
    "HDPE": 1.93 - 0.45,   # virgin (1.93) - recycled (0.45) kg CO2e/kg — PlasticsEurope 2023
    "LDPE": 2.05 - 0.52,   # virgin (2.05) - recycled (0.52) kg CO2e/kg — PlasticsEurope 2023
    "PP":   1.97 - 0.48,   # virgin (1.97) - recycled (0.48) kg CO2e/kg — PlasticsEurope 2023
}

eis_scope3_entry = {
    "eis_entry_type":        "verified_recycled_content",
    "eis_module":            "Scope 3 — Category 1 (Purchased Goods and Services)",
    "reporting_period":      "2026-Q2",
    "client":                "Fortune 500 Consumer Goods — [Client Name]",
    "supplier_facility":     facility,
    "batch_reference":       batch_id,
    "mpip_api_verified":     True,
    "verification_method":   "NIR spectroscopy + IBM Granite AI — MPIP v1.0",

    "material_purchased": [
        {
            "polymer_type":           "HDPE",
            "quality_grade":          "A",
            "quantity_kg":            len(grade_a_hdpe) * 850,   # 850 kg = standard HDPE bale weight (industry assumption — replace with weighbridge in production)
            "dpp_passport_ids":       [f"MPIP-20260623-{r['sample_id']}" for _, r in grade_a_hdpe.iterrows()],
            "verified_recycled_pct":  100,
            "carbon_footprint_kg_co2e_per_kg": 0.45,
            "carbon_saving_vs_virgin_kg_co2e_per_kg": carbon_savings["HDPE"],
            "food_contact_certified": True,
        },
        {
            "polymer_type":           "PP",
            "quality_grade":          "A",
            "quantity_kg":            len(grade_a_pp) * 720,     # 720 kg = standard PP bale weight (industry assumption — replace with weighbridge in production)
            "dpp_passport_ids":       [f"MPIP-20260623-{r['sample_id']}" for _, r in grade_a_pp.iterrows()],
            "verified_recycled_pct":  100,
            "carbon_footprint_kg_co2e_per_kg": 0.48,
            "carbon_saving_vs_virgin_kg_co2e_per_kg": carbon_savings["PP"],
            "food_contact_certified": True,
        },
    ],

    "scope3_emissions_summary": {
        "total_pcr_kg_purchased":       (len(grade_a_hdpe) * 850) + (len(grade_a_pp) * 720),
        "total_carbon_kg_co2e":         round(
            (len(grade_a_hdpe) * 850 * 0.45) + (len(grade_a_pp) * 720 * 0.48), 1),
        "carbon_saving_vs_all_virgin_kg_co2e": round(
            (len(grade_a_hdpe) * 850 * carbon_savings["HDPE"]) +
            (len(grade_a_pp)   * 720 * carbon_savings["PP"]),  1),
        "verification_status":          "MPIP-verified — DPP-backed",
        "auditable":                    True,
        "prior_verification_method":    "Supplier declaration only (unverified)",
        "improvement_note":             (
            "Previously this Scope 3 data was based solely on supplier declarations. "
            "MPIP provides spectral verification with DPP traceability — "
            "making this data auditable for EU CSRD and SEC climate disclosure requirements."
        ),
    },
}

# ============================================================================
# STEP 5: Save outputs
# ============================================================================
with open("MPIP_api_payload.json", "w") as f:
    json.dump(api_payload, f, indent=2)

with open("MPIP_eis_scope3_feed.json", "w") as f:
    json.dump(eis_scope3_entry, f, indent=2)

print("\n[2] Files saved:")
print("    MPIP_api_payload.json        — Material Intelligence API response (full batch)")
print("    MPIP_eis_scope3_feed.json    — IBM EIS Scope 3 verified data entry")

# ============================================================================
# STEP 6: Print summary
# ============================================================================
print("\n" + "=" * 100)
print("MATERIAL INTELLIGENCE API — BATCH SUMMARY")
print("=" * 100)
bs = api_payload["batch_summary"]
print(f"\n  Batch ID   : {batch_id}")
print(f"  Facility   : {facility}")
print(f"  Items      : {api_payload['total_items']}")
print(f"  Grade A    : {bs['grade_A_count']} items")
print(f"  Grade B    : {bs['grade_B_count']} items")
print(f"  REJECT     : {bs['grade_C_or_reject']} items (blends/contaminated)")
print(f"  DPP issued : {bs['dpp_eligible_count']} items")
print(f"  Premium PCR yield: {bs['premium_pcr_yield_pct']}% of batch")
print(f"  Avg recyclability: {bs['avg_recyclability_score']}/100")

print("\n" + "=" * 100)
print("IBM EIS SCOPE 3 VERIFIED DATA FEED")
print("=" * 100)
s3 = eis_scope3_entry["scope3_emissions_summary"]
print(f"\n  Client      : {eis_scope3_entry['client']}")
print(f"  PCR purchased (verified): {s3['total_pcr_kg_purchased']:,} kg")
print(f"  Carbon footprint        : {s3['total_carbon_kg_co2e']:,} kg CO2e")
print(f"  Carbon saving vs virgin : {s3['carbon_saving_vs_all_virgin_kg_co2e']:,} kg CO2e")
print(f"  Verification            : {s3['verification_status']}")
print(f"  Auditable               : {s3['auditable']}")
print(f"\n  Note: {s3['improvement_note']}")

print("\n" + "=" * 100)
print("KEY OUTCOME STATEMENT:")
print(f"""
  \"MPIP's Material Intelligence API processed a batch of {api_payload['total_items']} items
   at {facility}, yielding {bs['grade_A_count']} Grade A items ({bs['premium_pcr_yield_pct']}% premium PCR yield).
   The verified material data was fed directly into IBM EIS as a Scope 3 emissions entry —
   replacing unverified supplier declarations with spectral-verified, DPP-backed material data.
   This makes Fortune 500 clients' Scope 3 reporting auditable for EU CSRD and SEC climate
   disclosure requirements — a capability IBM Consulting can deploy at recycling operators globally.\"
""")
print("=" * 100)
