"""
Phase 4B: Live IBM Granite Material Intelligence Records
Calls ibm/granite-3-8b-instruct via watsonx.ai to generate
a reasoned quality grade and Material Intelligence Record for
every sample in MPIP_quality_grades.csv.

Credentials are read from environment variables — never hardcode keys.
Before running:
    export WATSONX_API_KEY="your-ibm-cloud-api-key"
    export WATSONX_PROJECT_ID="your-watsonx-project-id"
"""

import os
import json
import time
import urllib.request
import urllib.parse
import urllib.error
import pandas as pd

# ============================================================================
# watsonx.ai credentials — loaded from environment variables
# Set these before running:
#   export WATSONX_API_KEY="your-ibm-cloud-api-key"
#   export WATSONX_PROJECT_ID="your-watsonx-project-id"
# ============================================================================
USE_WATSONX        = True
WATSONX_API_KEY    = os.environ.get("WATSONX_API_KEY", "")
WATSONX_PROJECT_ID = os.environ.get("WATSONX_PROJECT_ID", "c1c2f244-777f-4023-a6c4-189b38521875")
WATSONX_URL        = "https://us-south.ml.cloud.ibm.com"
MODEL_ID           = "ibm/granite-3-8b-instruct"

if USE_WATSONX and not WATSONX_API_KEY:
    raise ValueError(
        "WATSONX_API_KEY environment variable not set.\n"
        "Run: export WATSONX_API_KEY='your-ibm-cloud-api-key'"
    )

# ============================================================================
# IAM token helper (stdlib only — no requests dependency)
# ============================================================================
_iam_token = None

def get_iam_token():
    global _iam_token
    if _iam_token:
        return _iam_token
    data = urllib.parse.urlencode({
        "grant_type": "urn:ibm:params:oauth:grant-type:apikey",
        "apikey": WATSONX_API_KEY
    }).encode()
    req = urllib.request.Request(
        "https://iam.cloud.ibm.com/identity/token",
        data=data,
        headers={"Content-Type": "application/x-www-form-urlencoded"}
    )
    with urllib.request.urlopen(req) as resp:
        _iam_token = json.loads(resp.read())["access_token"]
    return _iam_token

# ============================================================================
# Granite inference call
# ============================================================================
def call_granite(prompt: str) -> str:
    token = get_iam_token()
    payload = json.dumps({
        "model_id": MODEL_ID,
        "input": prompt,
        "parameters": {
            "decoding_method": "greedy",
            "max_new_tokens": 400,
            "temperature": 0.1,
            "repetition_penalty": 1.1
        },
        "project_id": WATSONX_PROJECT_ID
    }).encode()
    req = urllib.request.Request(
        f"{WATSONX_URL}/ml/v1/text/generation?version=2023-05-29",
        data=payload,
        headers={
            "Authorization": f"Bearer {token}",
            "Content-Type": "application/json"
        }
    )
    with urllib.request.urlopen(req) as resp:
        result = json.loads(resp.read())
    return result["results"][0]["generated_text"].strip()

# ============================================================================
# Build prompt for a single sample
# ============================================================================
def build_prompt(row) -> str:
    return f"""You are a polymer materials intelligence system for a plastic recycling facility.

A plastic item has been scanned by NIR spectroscopy. The following physical properties were predicted by a machine learning model trained on NIST data:

Sample ID: {row['Sample']}
Predicted Polymer Type: {row['Predicted Polymer']}
Predicted Density: {row['Predicted Density']} g/cm3
Predicted Crystallinity: {row['Predicted Crystallinity']}
Predicted SCB: {row['Predicted SCB']} CH3/1000C
Quality Grade: {row['Quality Grade']}
Recycling Pathway: {row['Recycling Pathway']}

Write 2-3 sentences explaining why this quality grade was assigned based on the specific property values above. Reference the actual numbers in your explanation."""

# ============================================================================
# Main
# ============================================================================
def main():
    print("=" * 80)
    print("PHASE 4B: IBM GRANITE MATERIAL INTELLIGENCE RECORDS")
    print(f"Model: {MODEL_ID} via watsonx.ai")
    print("=" * 80)

    df = pd.read_csv("MPIP_quality_grades.csv")
    print(f"\nLoaded {len(df)} samples from MPIP_quality_grades.csv")

    records = []
    for i, row in df.iterrows():
        sample = row["Sample"]
        try:
            prompt = build_prompt(row)
            reasoning = call_granite(prompt)
            record = {
                "sample_id":         sample,
                "polymer_type":      row["Predicted Polymer"],
                "quality_grade":     row["Quality Grade"].replace("Grade ", "").replace("CONTAMINATED", "C"),
                "recyclability_score": max(10, 100 - ["Grade A","Grade B","Grade C","CONTAMINATED"].index(row["Quality Grade"]) * 25) if row["Quality Grade"] in ["Grade A","Grade B","Grade C","CONTAMINATED"] else 50,
                "recycling_pathway": row["Recycling Pathway"],
                "contamination_flag": "BLEND" in str(row["Predicted Polymer"]).upper() or "CONTAMINATED" in str(row["Quality Grade"]).upper(),
                "reasoning":         reasoning,
                "reasoning_source":  f"{MODEL_ID} via watsonx.ai",
                "Predicted Density":      row["Predicted Density"],
                "Predicted Crystallinity": row["Predicted Crystallinity"],
                "Predicted SCB":          row["Predicted SCB"],
            }
            records.append(record)
            print(f"  ✓ {sample} → {record['quality_grade']}")
            time.sleep(0.5)
        except Exception as e:
            print(f"  ✗ {sample} failed: {e}")

    out = pd.DataFrame(records)
    out.to_csv("MPIP_granite_material_records.csv", index=False)
    print(f"\nSaved {len(records)} records to MPIP_granite_material_records.csv")
    print("\nGrade distribution:")
    print(out["quality_grade"].value_counts())

if __name__ == "__main__":
    main()
