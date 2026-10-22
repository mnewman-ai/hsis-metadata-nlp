import os
import pandas as pd

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
INPUT_FILE = os.path.join(BASE_DIR, "isolated_grey_zone.csv")

def get_row_classification(field_a, field_b):
    """
    Evaluates field pairs using transportation domain rules 
    and assigns them to Tier 1, Tier 2, Tier 3, or Unrelated.
    """
    fa, fb = str(field_a).upper().strip(), str(field_b).upper().strip()
    
    # --- TIER 1: DIRECT EQUIVALENCE ---
    if ("WEATHER" in fa and "WEATHER" in fb) or \
       ("SPEED LIMIT" in fa and "SPEED" in fb and "DESG" not in fb) or \
       (fa == "TOTAL AADT" and fb == "AADT"):
        return {
            "tier": 1,
            "relationship_type": "Equivalent",
            "mathematical_formula_applicable": "false",
            "transformation_logic": "Direct 1-to-1 schema translation. No calculation required."
        }
        
    # --- TIER 2: FORMULAIC OVERLAP ---
    if ("AADT" in fa or "ADT" in fa) and ("AADT" in fb or "ADT" in fb):
        # Catches TOTAL AADT vs FACTORED AADT, ADT TOTAL NBR vs FACTORED AADT, etc.
        return {
            "tier": 2,
            "relationship_type": "Formulaic_Overlap",
            "mathematical_formula_applicable": "true",
            "transformation_logic": "Apply deterministic ETL script incorporating seasonal factors or segment lengths."
        }
        
    # --- TIER 3: CONCEPTUAL OVERLAP ---
    if ("LIMIT" in fa or "NBR" in fa or "LIM" in fa) and ("DESG SPD" in fb):
        # Catches Posted Speed Limit variations vs Design Speed thresholds
        return {
            "tier": 3,
            "relationship_type": "Conceptual_Overlap",
            "mathematical_formula_applicable": "false",
            "transformation_logic": "Contextually related velocity fields. Map via heuristic matrices; log data discrepancy warning."
        }
        
    # --- FALSE POSITIVES / UNRELATED ---
    return {
        "tier": 0,
        "relationship_type": "Unrelated",
        "mathematical_formula_applicable": "false",
        "transformation_logic": "Spurious text metric collision based on common substrings (e.g., 'TOTAL' or 'CONTROL'). Filter out."
    }

def main():
    print("⏳ Beginning Crosswalk classification routine...")
    if not os.path.exists(INPUT_FILE):
        print(f"❌ Error: Missing required input database '{INPUT_FILE}'.")
        return

    # Ingest the 21 grey-zone records
    df = pd.read_csv(INPUT_FILE)
    
    # Initialize dictionary lists to store the sorted rows
    tier_buckets = {1: [], 2: [], 3: []}
    unrelated_count = 0

    # Process each row through the classification rules engine
    for _, row in df.iterrows():
        field_a = row['field_state_A']
        field_b = row['field_state_B']
        
        meta = get_row_classification(field_a, field_b)
        tier = meta["tier"]
        
        # Build the final schema-compliant record
        record = {
            "state_A": row['state_A'],
            "file_A": row['file_A'],
            "field_state_A": field_a,
            "description_A": row.get('description_A', ''),
            "state_B": row['state_B'],
            "file_B": row['file_B'],
            "field_state_B": field_b,
            "description_B": row.get('description_B', ''),
            "relationship_type": meta["relationship_type"],
            "mathematical_formula_applicable": meta["mathematical_formula_applicable"],
            "transformation_logic": meta["transformation_logic"]
        }
        
        if tier in tier_buckets:
            tier_buckets[tier].append(record)
        else:
            unrelated_count += 1

    # Map output files to their destinations
    output_targets = {
        1: "crosswalk_1_equivalent.csv",
        2: "crosswalk_2_formulaic.csv",
        3: "crosswalk_3_conceptual.csv"
    }

    # Define the expected column schema
    column_schema = [
        "state_A", "file_A", "field_state_A", "description_A",
        "state_B", "file_B", "field_state_B", "description_B",
        "relationship_type", "mathematical_formula_applicable", "transformation_logic"
    ]

    print("\n💾 Writing ground-truth files to local directory:")
    for tier, filename in output_targets.items():
        file_path = os.path.join(BASE_DIR, filename)
        output_df = pd.DataFrame(tier_buckets[tier], columns=column_schema)
        output_df.to_csv(file_path, index=False)
        print(f"   ↳ {filename}: Logged {len(output_df)} seed records.")
    print(f"\n🗑️ Filtered out {unrelated_count} false-positive text collisions.")
    print("🎉 Milestone Accomplished: Ground-truth anchors are completely seeded for next week!")

if __name__ == "__main__":
    main()