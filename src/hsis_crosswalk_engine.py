import argparse
import re
from pathlib import Path

import pandas as pd
from sentence_transformers import SentenceTransformer


MODEL_NAME = "all-MiniLM-L6-v2"


def clean_text(value):
    """Convert missing values to empty strings and normalize whitespace."""
    if pd.isna(value):
        return ""
    return re.sub(r"\s+", " ", str(value)).strip()


def normalize_field_name(value):
    """Normalize field names for rule comparisons."""
    return clean_text(value).upper().replace("_", " ")


def find_column(df, possible_names):
    """
    Find the first column whose normalized name matches one of the options.
    """
    normalized_columns = {
        re.sub(r"[^a-z0-9]", "", column.lower()): column
        for column in df.columns
    }

    for name in possible_names:
        normalized_name = re.sub(r"[^a-z0-9]", "", name.lower())
        if normalized_name in normalized_columns:
            return normalized_columns[normalized_name]

    return None


def load_and_standardize_csv(csv_path):
    """
    Read one CSV and standardize its columns into the project's common schema.
    """
    df = pd.read_csv(csv_path)

    state_col = find_column(df, ["State", "state"])
    file_col = find_column(df, ["File", "file", "File_Name", "filename"])
    variable_col = find_column(
        df,
        ["Variable_Name", "Variable Name", "Field_Name", "Field Name", "variable"]
    )
    description_col = find_column(
        df,
        [
            "Description",
            "Variable_Description",
            "Variable Description",
            "Field_Description",
            "Field Description",
        ],
    )
    field_type_col = find_column(df, ["Field_Type", "Field Type", "type"])

    if variable_col is None or description_col is None:
        raise ValueError(
            f"\n{csv_path.name} is missing required metadata columns.\n"
            f"It must contain a variable-name column and a description column.\n"
            f"Columns found: {list(df.columns)}"
        )

    # If the file has no State column, use the filename as a temporary state/source label.
    source_name = csv_path.stem.upper()

    standardized = pd.DataFrame({
        "state": (
            df[state_col].apply(clean_text)
            if state_col is not None
            else source_name
        ),
        "file": (
            df[file_col].apply(clean_text)
            if file_col is not None
            else csv_path.name
        ),
        "variable_name": df[variable_col].apply(clean_text),
        "description": df[description_col].apply(clean_text),
        "field_type": (
            df[field_type_col].apply(clean_text)
            if field_type_col is not None
            else ""
        ),
        "source_csv": csv_path.name,
    })

    standardized = standardized[
        (standardized["variable_name"] != "")
        | (standardized["description"] != "")
    ].copy()

    standardized["embedding_text"] = standardized.apply(
        lambda row: (
            f"Variable name: {row['variable_name']}. "
            f"Description: {row['description']}. "
            f"Field type: {row['field_type']}."
        ),
        axis=1,
    )

    return standardized


def classify_relationship(field_a, field_b):
    """
    Apply simple deterministic rules after SBERT identifies a candidate pair.
    """
    fa = normalize_field_name(field_a)
    fb = normalize_field_name(field_b)

    if fa == fb:
        return (
            "Equivalent",
            False,
            "The two fields have the same normalized variable name."
        )

    if (
        ("WEATHER" in fa and "WEATHER" in fb)
        or ("SPEED LIMIT" in fa and "SPEED" in fb and "DESG" not in fb)
        or (fa == "TOTAL AADT" and fb == "AADT")
        or (fb == "TOTAL AADT" and fa == "AADT")
    ):
        return (
            "Equivalent",
            False,
            "Direct schema translation; no mathematical transformation is required."
        )

    if (
        ("AADT" in fa or "ADT" in fa)
        and ("AADT" in fb or "ADT" in fb)
    ):
        return (
            "Formulaic_Overlap",
            True,
            "The fields are traffic-volume measures that may require a deterministic transformation such as weighting, aggregation, or seasonal adjustment."
        )

    if (
        (("LIMIT" in fa or "NBR" in fa or "LIM" in fa) and "DESG SPD" in fb)
        or (("LIMIT" in fb or "NBR" in fb or "LIM" in fb) and "DESG SPD" in fa)
    ):
        return (
            "Conceptual_Overlap",
            False,
            "The fields are related speed concepts but may represent different operational definitions."
        )

    return (
        "Candidate_Review",
        False,
        "Semantic similarity exceeded the threshold, but no deterministic crosswalk rule was identified."
    )


def build_crosswalk_records(df, threshold):
    """
    Create SBERT embeddings, compare records from different states, and retain
    pairs at or above the requested cosine-similarity threshold.
    """
    print(f"Loading SBERT model: {MODEL_NAME}")
    model = SentenceTransformer(MODEL_NAME)

    print(f"Creating embeddings for {len(df)} metadata records...")
    embeddings = model.encode(
        df["embedding_text"].tolist(),
        normalize_embeddings=True,
        show_progress_bar=True,
    )

    records = []

    # Because embeddings are normalized, dot product equals cosine similarity.
    for i in range(len(df)):
        for j in range(i + 1, len(df)):
            row_a = df.iloc[i]
            row_b = df.iloc[j]

            # Do not create crosswalks within the same state.
            if row_a["state"] == row_b["state"]:
                continue

            similarity_score = float(embeddings[i] @ embeddings[j])

            if similarity_score < threshold:
                continue

            relationship_type, formula_applicable, transformation_logic = (
                classify_relationship(
                    row_a["variable_name"],
                    row_b["variable_name"],
                )
            )

            records.append({
                "state_A": row_a["state"],
                "file_A": row_a["file"],
                "field_state_A": row_a["variable_name"],
                "description_A": row_a["description"],
                "field_type_A": row_a["field_type"],
                "source_csv_A": row_a["source_csv"],
                "state_B": row_b["state"],
                "file_B": row_b["file"],
                "field_state_B": row_b["variable_name"],
                "description_B": row_b["description"],
                "field_type_B": row_b["field_type"],
                "source_csv_B": row_b["source_csv"],
                "semantic_similarity": round(similarity_score, 4),
                "relationship_type": relationship_type,
                "mathematical_formula_applicable": formula_applicable,
                "transformation_logic": transformation_logic,
                "review_status": "candidate",
            })

    return pd.DataFrame(records)


def main():
    parser = argparse.ArgumentParser(
        description="Generate semantic cross-state metadata crosswalk candidates."
    )

    parser.add_argument(
        "--input_dir",
        required=True,
        help="Folder containing one or more input CSV metadata dictionaries.",
    )

    parser.add_argument(
        "--output",
        required=True,
        help="Path for the generated master crosswalk CSV.",
    )

    parser.add_argument(
        "--threshold",
        type=float,
        default=0.82,
        help="Minimum SBERT cosine-similarity score from 0 to 1. Default: 0.82",
    )

    args = parser.parse_args()

    if not 0 <= args.threshold <= 1:
        raise ValueError("--threshold must be between 0 and 1.")

    input_dir = Path(args.input_dir)
    output_file = Path(args.output)

    if not input_dir.exists() or not input_dir.is_dir():
        raise FileNotFoundError(
            f"Input directory does not exist: {input_dir}"
        )

    csv_files = sorted(input_dir.glob("*.csv"))

    if not csv_files:
        raise FileNotFoundError(
            f"No CSV files were found inside: {input_dir}"
        )

    print(f"Found {len(csv_files)} CSV file(s) in {input_dir}")

    all_data = []

    for csv_path in csv_files:
        print(f"Reading: {csv_path.name}")
        standardized_df = load_and_standardize_csv(csv_path)
        all_data.append(standardized_df)

    combined_df = pd.concat(all_data, ignore_index=True)

    unique_states = combined_df["state"].nunique()

    if unique_states < 2:
        raise ValueError(
            "The engine needs metadata from at least two different states. "
            "Your input files currently contain only one state/source label."
        )

    crosswalk_df = build_crosswalk_records(combined_df, args.threshold)

    output_file.parent.mkdir(parents=True, exist_ok=True)

    crosswalk_df = crosswalk_df.sort_values(
        by=["semantic_similarity"],
        ascending=False,
    )

    crosswalk_df.to_csv(output_file, index=False)

    print("\nPipeline complete.")
    print(f"Input records processed: {len(combined_df)}")
    print(f"Candidate crosswalks found: {len(crosswalk_df)}")
    print(f"Output written to: {output_file}")


if __name__ == "__main__":
    main()
