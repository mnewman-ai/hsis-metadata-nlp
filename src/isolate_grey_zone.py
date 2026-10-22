import os
import re
import pandas as pd
from rapidfuzz import fuzz

# Define local paths relative to this script's directory
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
INPUT_FILE = os.path.join(BASE_DIR, "hsis_raw_schema.csv")
OUTPUT_FILE = os.path.join(BASE_DIR, "isolated_grey_zone.csv")

def main():
    print("⏳ Reading flat schema metadata fields...")
    if not os.path.exists(INPUT_FILE):
        print(f"Error: Could not find '{INPUT_FILE}'.")
        return

    # Ingest the flat inventory list
    raw_fields = pd.read_csv(INPUT_FILE)
    print(f"Successfully loaded {len(raw_fields)} individual metadata variables.")

    print("Generating cross-examination pairs and calculating similarity scores...")
    
    pairs_list = []
    
    # Loop through the variables to build distinct comparison rows
    for i, row_A in raw_fields.iterrows():
        for j, row_B in raw_fields.iterrows():
            # Avoid self-matching and duplicate pairs (e.g., matching A with B, then B with A)
            if i >= j:
                continue
            
            # Prioritize comparing across different states or different source files
            if row_A['State'] != row_B['State'] or row_A['File'] != row_B['File']:
                
                # Calculate a text similarity score scaled between 0.0 and 1.0
                var_A_name = str(row_A['Variable_Name'])
                var_B_name = str(row_B['Variable_Name'])
                
                # token_sort_ratio handles jumbled styling like WEATHER vs WEATHER_COND_CD brilliantly
                score = fuzz.token_sort_ratio(var_A_name, var_B_name) / 100.0
                
                pairs_list.append({
                    'state_A': row_A['State'],
                    'file_A': row_A['File'],
                    'field_state_A': var_A_name,
                    'description_A': row_A.get('Description', ''),
                    'state_B': row_B['State'],
                    'file_B': row_B['File'],
                    'field_state_B': var_B_name,
                    'description_B': row_B.get('Description', ''),
                    'calculated_similarity': score
                })

    # Create our new pairs matrix DataFrame
    pairs_df = pd.DataFrame(pairs_list)
    print(f"Created {len(pairs_df)} unique cross-variable comparison pairs.")

    # 1. Filter for the Semantic 'Grey Zone' (0.40 to 0.70 similarity)
    print("Filtering for the 0.40 - 0.70 Cosine Similarity Grey Zone...")
    grey_zone_df = pairs_df[(pairs_df['calculated_similarity'] >= 0.40) & (pairs_df['calculated_similarity'] <= 0.70)].copy()

    # 2. Compile transportation stems to catch hidden pairings across states
    transport_stems = r'(_CD|VOL|TYP|AVG|AADT|TOTAL|COUNT|LOG)'
    
    grey_zone_df['transport_stem_flag'] = grey_zone_df.apply(
        lambda row: bool(re.search(transport_stems, str(row['field_state_A']), re.IGNORECASE)) or 
                    bool(re.search(transport_stems, str(row['field_state_B']), re.IGNORECASE)), 
        axis=1
    )

    # Sort results to place high-value transportation terminology matches at the top
    prioritized_audit_list = grey_zone_df.sort_values(
        by=['transport_stem_flag', 'calculated_similarity'], 
        ascending=[False, False]
    )

    # 3. Export to your clean local CSV file
    prioritized_audit_list.to_csv(OUTPUT_FILE, index=False)
    print(f"Success! Extracted {len(prioritized_audit_list)} pairs in the Grey Zone to audit.")
    print(f"Output matrix saved locally to: {OUTPUT_FILE}")

if __name__ == "__main__":
    main()
