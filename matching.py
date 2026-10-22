import pandas as pd
from difflib import SequenceMatcher
import os

# Define paths relative to script location
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
INPUT_FILE = os.path.join(BASE_DIR, "raw_data.csv")
OUTPUT_FILE = os.path.join(BASE_DIR, "step3_candidate_matches.csv")

# Check if input file exists
if not os.path.exists(INPUT_FILE):
    raise FileNotFoundError(f"Could not find '{INPUT_FILE}' in the script directory.")

# Load CSV file
df = pd.read_csv(INPUT_FILE)

# Clean column names
df.columns = (
    df.columns
      .str.strip()
      .str.replace(" ", "_", regex=False)
      .str.replace("/", "_", regex=False)
)

# Verify expected columns exist
required_cols = ["State", "Variable_Name", "Variable_Description"]

for col in required_cols:
    if col not in df.columns:
        raise ValueError(
            f"Column '{col}' not found. Available columns: {df.columns.tolist()}"
        )

# Fill missing values to avoid errors
df["Variable_Name"] = df["Variable_Name"].fillna("").astype(str)
df["Variable_Description"] = df["Variable_Description"].fillna("").astype(str)

def classify_variable(name, description):
    text = f"{name} {description}".lower()

    if "aadt" in text or "traffic" in text:
        return "AADT"

    if "route" in text:
        return "ROUTE"

    if "speed" in text:
        return "SPEED"

    if "lane" in text:
        return "LANE"

    if "median" in text:
        return "MEDIAN"

    if "weather" in text:
        return "WEATHER"

    if "light" in text:
        return "LIGHT"

    if "severity" in text or "injury" in text:
        return "SEVERITY"

    if "date" in text or "time" in text:
        return "DATE"

    if "id" in text or "case number" in text:
        return "ID"

    return "OTHER"

def compute_similarity(str1, str2):
    return SequenceMatcher(
        None,
        str(str1).lower(),
        str(str2).lower()
    ).ratio()

candidate_pairs = []

# Compare each row only once (avoids duplicates)
for i in range(len(df)):
    row_a = df.iloc[i]

    for j in range(i + 1, len(df)):
        row_b = df.iloc[j]

        # Only compare variables from different states
        if row_a["State"] != row_b["State"]:

            name_score = compute_similarity(
                row_a["Variable_Name"],
                row_b["Variable_Name"]
            )

            desc_score = compute_similarity(
                row_a["Variable_Description"],
                row_b["Variable_Description"]
            )

            if name_score > 0.60 or desc_score > 0.70:
                candidate_pairs.append({
                    "Source_State": row_a["State"],
                    "Source_Var": row_a["Variable_Name"],
                    "Target_State": row_b["State"],
                    "Target_Var": row_b["Variable_Name"],
                    "Fuzzy_Name_Score": round(name_score, 2),
                    "Fuzzy_Desc_Score": round(desc_score, 2)
                })

# Create output DataFrame
candidates_df = pd.DataFrame(candidate_pairs)

# Export results
candidates_df.to_csv(OUTPUT_FILE, index=False)

print(f"Step 3 Complete. Generated {len(candidates_df)} candidate pairs.")
print(f"Output saved to: {OUTPUT_FILE}")