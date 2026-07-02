"""
Phase 5A: Confidence Scores and Edge-Case Flags
================================================
Adds a calibrated confidence score and edge-case flags to every Granite record.

Confidence model (v2 — wider spread, weighted average):
  - Base score: Grade A → 88, Grade B → 72, Grade C (blend) → 55
  - Density component:  distance from nearest class boundary, scaled 0–20
  - Crystallinity component: distance from nearest grade boundary, scaled 0–15
  - SCB component:      distance from PP/PE or blend boundary, scaled 0–10
  - Penalty terms:      near-boundary margin violations (−5 to −15 each)
  - Final = base + density_bonus + cryst_bonus + scb_bonus − penalties
  - Capped to [30, 98]

This produces scores ranging from ~30 (borderline blend/edge) to ~97 (clear HDPE/PP
far from boundaries), rather than the previous 45–71% flat cluster.
"""

import pandas as pd
import numpy as np
import json

print("=" * 100)
print("PHASE 5A: CONFIDENCE SCORES + EDGE-CASE FLAGS  (v2 — calibrated wide spread)")
print("=" * 100)

# ============================================================================
# STEP 1: Load Granite records + predicted properties
# ============================================================================
df     = pd.read_csv("MPIP_granite_material_records.csv")
grades = pd.read_csv("MPIP_quality_grades.csv")

# Only merge predicted-property columns if not already present in CSV
if "Predicted Density" not in df.columns:
    df = df.merge(
        grades[["Sample", "Predicted Density", "Predicted Crystallinity", "Predicted SCB"]],
        left_on="sample_id", right_on="Sample", how="left"
    )
    df = df.drop(columns=["Sample"], errors="ignore")
else:
    # Already present — just align with grades for any missing rows (shouldn't happen)
    pass

print(f"\n[1] Loaded {len(df)} records")
print(f"     Columns: {[c for c in df.columns if 'Predicted' in c or c == 'confidence_pct']}")

# ============================================================================
# STEP 2: Decision boundaries
# ============================================================================
B_HDPE_DENS  = 0.948    # density: HDPE/MDPE boundary
B_MDPE_DENS  = 0.930    # density: MDPE/LDPE boundary
B_PP_SCB     = 150.0    # SCB: PP/PE boundary
B_BLEND_SCB  = 35.0     # SCB: blend signal threshold (above here = contaminated PE)

EDGE_DENS    = 0.005    # g/cm3 — within this margin → edge case
EDGE_SCB     = 15.0     # CH3/1000C — within this margin → edge case
EDGE_CRYST   = 0.030    # crystallinity — within this margin → edge case

# ============================================================================
# STEP 3: Confidence computation — weighted average, not minimum
# ============================================================================

def compute_confidence(row):
    """
    Returns (confidence_pct: int, edge_flags: list[str])

    Confidence components:
      base       — grade-dependent anchor (Grade A=88, B=72, C=55, edge grades lower)
      dens_bonus — how far density sits from nearest boundary (max +8)
      cryst_bonus— how far crystallinity sits from nearest grade boundary (max +6)
      scb_bonus  — how far SCB sits from PP/PE boundary (max +5)
      penalties  — subtracted for each near-boundary violation (−5 to −15 each)
    """
    density  = float(row["Predicted Density"])
    cryst    = float(row["Predicted Crystallinity"])
    scb      = float(row["Predicted SCB"])
    ptype    = row["polymer_type"]
    grade    = row["quality_grade"]
    is_blend = (ptype == "BLEND")

    edge_flags = []

    # ── Blends ─────────────────────────────────────────────────────────────
    if is_blend:
        # Base: 55.  Bonus: how far SCB is from the blend boundaries.
        # High SCB (PP-dominant blend): distance from B_PP_SCB upper zone
        # Low SCB (LDPE-HDPE blend): distance from B_BLEND_SCB
        if scb > B_PP_SCB:
            # Clear PP-content blend — SCB far above PE range
            dist = scb - B_PP_SCB
            base = 55
            bonus = min(20, dist * 0.06)       # up to +20 for very clear PP blend
        else:
            # LDPE+HDPE or ambiguous blend — SCB in PE range
            dist = abs(scb - B_BLEND_SCB)
            base = 50
            bonus = min(15, dist * 0.20)       # up to +15 for clear PE-only blend

        conf = base + bonus

        if scb > B_PP_SCB and abs(scb - B_PP_SCB) < EDGE_SCB:
            edge_flags.append(f"SCB {scb:.1f} near PP/PE boundary ({B_PP_SCB:.0f}) — blend composition uncertain")
        if scb <= B_BLEND_SCB + 5 and is_blend:
            edge_flags.append(f"SCB {scb:.1f} near PE blend threshold — LDPE/HDPE blend boundary indistinct")

        conf = max(30, min(82, round(conf)))
        return conf, edge_flags

    # ── Base score by grade ─────────────────────────────────────────────────
    base = {"A": 88, "B": 72, "C": 55}.get(grade, 65)

    # ── Density component ──────────────────────────────────────────────────
    dens_bonus  = 0
    dens_penalty = 0

    if ptype == "HDPE":
        dist = density - B_HDPE_DENS           # positive = clearly above boundary
        if dist >= 0:
            dens_bonus = min(8, dist * 1200)   # full +8 at dist ≥ 0.0067
        else:
            dens_bonus = 0
            dens_penalty += 12                  # below boundary — classification error zone
        if 0 <= dist < EDGE_DENS:
            edge_flags.append(
                f"Density {density:.4f} only {dist:.4f} g/cm³ above HDPE boundary "
                f"({B_HDPE_DENS}) — possible MDPE/LLDPE, flag for secondary confirmation"
            )
            dens_penalty += 5

    elif ptype in ("MDPE", "LLDPE"):
        dist_from_hdpe = B_HDPE_DENS - density  # how far below HDPE boundary
        dist_from_ldpe = density - B_MDPE_DENS  # how far above LDPE boundary
        dist = min(dist_from_hdpe, dist_from_ldpe)
        dens_bonus = min(6, dist * 800)
        if dist_from_hdpe < EDGE_DENS:
            edge_flags.append(
                f"Density {density:.4f} within {dist_from_hdpe:.4f} g/cm³ of HDPE "
                f"boundary — possible HDPE/MDPE overlap"
            )
            dens_penalty += 8
        if dist_from_ldpe < EDGE_DENS:
            edge_flags.append(
                f"Density {density:.4f} within {dist_from_ldpe:.4f} g/cm³ of LDPE "
                f"boundary — possible LDPE/MDPE overlap"
            )
            dens_penalty += 8

    elif ptype == "LDPE":
        dist = B_MDPE_DENS - density
        if dist >= 0:
            dens_bonus = min(8, dist * 500)
        else:
            # Density above MDPE boundary — unusual for LDPE, but happens
            dens_bonus = 0
            dens_penalty += 8
        if abs(density - B_MDPE_DENS) < EDGE_DENS:
            edge_flags.append(
                f"Density {density:.4f} within {abs(density - B_MDPE_DENS):.4f} g/cm³ "
                f"of LDPE/MDPE boundary — possible MDPE content"
            )
            dens_penalty += 5

    elif ptype == "PP":
        # PP density 0.880–0.913; check it's not overlapping PE density zone
        if density > 0.920:
            dens_penalty += 10
            edge_flags.append(
                f"Density {density:.4f} higher than typical PP (0.880–0.913) — "
                f"possible LDPE contamination or measurement error"
            )
        else:
            dens_bonus = min(6, (0.920 - density) * 300)

    # ── Crystallinity (grade boundary) ────────────────────────────────────
    cryst_bonus   = 0
    cryst_penalty = 0

    grade_boundaries = {
        "HDPE":  {"A": 0.55, "B": 0.45},
        "LDPE":  {"A": 0.40, "B": 0.30},
        "LLDPE": {"A": 0.40, "B": 0.30},
        "MDPE":  {"A": 0.45, "B": 0.35},
        "PP":    {"A": 0.38, "B": 0.30},
    }
    gb = grade_boundaries.get(ptype, {})
    boundary = gb.get(grade)
    if boundary is not None:
        cryst_dist = abs(cryst - boundary)
        cryst_bonus = min(6, cryst_dist * 120)
        if cryst_dist < EDGE_CRYST:
            edge_flags.append(
                f"Crystallinity {cryst:.4f} within {cryst_dist:.4f} of Grade {grade} "
                f"boundary ({boundary:.2f}) — grade assignment borderline"
            )
            cryst_penalty = 5

    # ── SCB component ──────────────────────────────────────────────────────
    scb_bonus   = 0
    scb_penalty = 0

    if ptype == "PP":
        dist = scb - B_PP_SCB
        scb_bonus = min(5, dist * 0.015)
        if dist < EDGE_SCB:
            edge_flags.append(
                f"SCB {scb:.1f} only {dist:.1f} above PP boundary ({B_PP_SCB:.0f}) "
                f"— verify PP identification"
            )
            scb_penalty = 8
        # Flag elevated HDPE-range SCB — probably pure PP but cross-check
        if scb > 350:
            edge_flags.append(f"SCB {scb:.1f} unusually high even for PP — verify spectrum quality")

    elif ptype == "HDPE":
        # Pure HDPE: SCB should be 2–12
        if scb > 12:
            over = scb - 12
            scb_penalty = min(12, over * 0.8)
            edge_flags.append(
                f"SCB {scb:.1f} elevated above pure HDPE range (2–12) — "
                f"possible 5–10% LLDPE content, flag before food-contact use"
            )
        else:
            scb_bonus = min(5, (12 - scb) * 0.5)

    elif ptype == "LDPE":
        # LDPE: SCB 18–35
        if scb < 18:
            scb_penalty = min(8, (18 - scb) * 0.8)
            edge_flags.append(
                f"SCB {scb:.1f} below typical LDPE range (18–35) — possible LLDPE content"
            )
        elif scb > 35:
            scb_bonus = 2
        else:
            scb_bonus = min(5, (scb - 18) * 0.15)

    # ── Assemble final score ────────────────────────────────────────────────
    total_bonus   = dens_bonus + cryst_bonus + scb_bonus
    total_penalty = dens_penalty + cryst_penalty + scb_penalty

    conf = base + total_bonus - total_penalty
    conf = max(30, min(98, round(conf)))

    return conf, edge_flags


# ============================================================================
# STEP 4: Apply to all 39 samples
# ============================================================================
print("\n[2] Computing calibrated confidence scores and edge-case flags...\n")

confidences, edge_case_bools, notes_list = [], [], []

for _, row in df.iterrows():
    conf, flags = compute_confidence(row)
    has_edge = len(flags) > 0
    note = (" | ".join(flags)) if flags else None
    confidences.append(conf)
    edge_case_bools.append(has_edge)
    notes_list.append(note)

df["confidence_pct"]  = confidences
df["edge_case_flag"]  = edge_case_bools
df["edge_case_notes"] = notes_list

# ============================================================================
# STEP 5: Print summary table
# ============================================================================
print(f"{'Sample':<22} {'Polymer':<8} {'Grade':<7} {'Conf':>6}  {'Edge?':<8}  Note")
print("─" * 110)
for _, row in df.iterrows():
    note_short = (str(row["edge_case_notes"])[:60] + "…") if row["edge_case_notes"] and len(str(row["edge_case_notes"])) > 60 else (row["edge_case_notes"] or "—")
    edge_str   = "⚠ YES" if row["edge_case_flag"] else "OK"
    print(f"  {row['sample_id']:<20} {row['polymer_type']:<8} {row['quality_grade']:<7} "
          f"{row['confidence_pct']:>4}%   {edge_str:<8}  {note_short}")

# ============================================================================
# STEP 6: Statistics + spread check
# ============================================================================
print("\n" + "=" * 100)
print("CONFIDENCE DISTRIBUTION SUMMARY")
print("=" * 100)

conf_series = df["confidence_pct"]
edge_count  = df["edge_case_flag"].sum()

buckets = [
    ("Very high (≥90%)",  conf_series >= 90),
    ("High     (80–89%)", (conf_series >= 80) & (conf_series < 90)),
    ("Medium   (65–79%)", (conf_series >= 65) & (conf_series < 80)),
    ("Low      (50–64%)", (conf_series >= 50) & (conf_series < 65)),
    ("Very low  (<50%)",  conf_series < 50),
]
print()
for label, mask in buckets:
    count   = mask.sum()
    samples = df[mask]["sample_id"].tolist()
    bar     = "█" * count
    print(f"  {label}: {count:>3}  {bar}")
    if count and count <= 8:
        print(f"         → {', '.join(samples)}")

print(f"\n  Range   : {conf_series.min()}% – {conf_series.max()}%")
print(f"  Mean    : {conf_series.mean():.1f}%")
print(f"  Std dev : {conf_series.std():.1f}%")
print(f"  Spread  : {conf_series.max() - conf_series.min()} pp")
print(f"\n  Edge-case flags : {edge_count} samples")

# Grade-wise breakdown
print(f"\n  Confidence by grade:")
for g in ["A", "B", "C"]:
    sub = df[df["quality_grade"] == g]["confidence_pct"]
    if len(sub):
        print(f"    Grade {g}: n={len(sub)}  mean={sub.mean():.1f}%  "
              f"range={sub.min()}–{sub.max()}%")

# ============================================================================
# STEP 7: Example records
# ============================================================================
print("\n" + "=" * 100)
print("EXAMPLE — HDPE Grade A strong (H0008HDPE)")
print("=" * 100)
for sid in ["H0008HDPE", "H0009PP", "H0007HDPE", "50HDPE_50PP"]:
    row = df[df["sample_id"] == sid]
    if row.empty:
        continue
    r = row.iloc[0]
    rec = {
        "sample_id":          r["sample_id"],
        "polymer_type":       r["polymer_type"],
        "quality_grade":      r["quality_grade"],
        "recyclability_score": int(r["recyclability_score"]),
        "confidence_pct":     int(r["confidence_pct"]),
        "edge_case_flag":     bool(r["edge_case_flag"]),
        "edge_case_notes":    r["edge_case_notes"],
        "dpp_eligible":       bool(r["dpp_eligible"]),
        "reasoning":          r["reasoning"],
        "recommended_action": r["recommended_action"],
    }
    print(f"\n── {sid} ──")
    print(json.dumps(rec, indent=2))

# ============================================================================
# STEP 8: Save updated CSV
# ============================================================================
# Drop helper merge column, keep all others
cols_to_save = [c for c in df.columns if c != "Sample"]
df[cols_to_save].to_csv("MPIP_granite_material_records.csv", index=False)
print(f"\n[3] Saved MPIP_granite_material_records.csv  ({len(df)} rows, "
      f"confidence range {conf_series.min()}–{conf_series.max()}%)")
print("=" * 100)
