import argparse
import re
from pathlib import Path
import json
import pandas as pd
from sentence_transformers import SentenceTransformer


MODEL_NAME = "all-MiniLM-L6-v2"


def clean_text(value):
    """Convert missing values to empty strings and normalize whitespace."""
    if pd.isna(value):
        return ""
    return re.sub(r"\s+", " ", str(value)).strip()


def find_column(df, possible_names):
    """Find the first column whose normalized name matches an allowed name."""
    normalized_columns = {
        re.sub(r"[^a-z0-9]", "", column.lower()): column
        for column in df.columns
    }

    for name in possible_names:
        normalized_name = re.sub(r"[^a-z0-9]", "", name.lower())
        if normalized_name in normalized_columns:
            return normalized_columns[normalized_name]

    return None


def load_state_schema(csv_path):
    """Load the combined state metadata CSV."""
    df = pd.read_csv(csv_path)

    required_columns = [
        "State",
        "File",
        "Variable_Name",
        "Description",
        "Field_Type",
    ]

    missing_columns = [column for column in required_columns if column not in df.columns]

    if missing_columns:
        raise ValueError(
            f"State schema is missing columns: {missing_columns}. "
            f"Found: {list(df.columns)}"
        )

    state_df = pd.DataFrame({
        "state": df["State"].apply(clean_text),
        "file": df["File"].apply(clean_text),
        "variable_name": df["Variable_Name"].apply(clean_text),
        "description": df["Description"].apply(clean_text),
        "field_type": df["Field_Type"].apply(clean_text),
    })

    state_df = state_df[
        (state_df["variable_name"] != "") | (state_df["description"] != "")
    ].copy()

    state_df["embedding_text"] = state_df.apply(
        lambda row: (
            "Domain: Highway safety and roadway metadata. "
            f"State field name: {row['variable_name']}. "
            f"Description: {row['description']}. "
            f"Field type: {row['field_type']}."
        ),
        axis=1,
    )

    return state_df


def load_master_schema(csv_path):
    """Load the reconstructed HSIS master target schema."""
    df = pd.read_csv(csv_path)

    required_columns = ["Master_Field", "Description", "Field_Type"]
    missing_columns = [column for column in required_columns if column not in df.columns]

    if missing_columns:
        raise ValueError(
            f"Master schema is missing columns: {missing_columns}. "
            f"Found: {list(df.columns)}"
        )

    master_df = pd.DataFrame({
        "master_field": df["Master_Field"].apply(clean_text),
        "master_description": df["Description"].apply(clean_text),
        "master_field_type": df["Field_Type"].apply(clean_text),
    })

    master_df = master_df[
        (master_df["master_field"] != "") | (master_df["master_description"] != "")
    ].copy()

    master_df["embedding_text"] = master_df.apply(
        lambda row: (
            "Domain: Highway safety and roadway metadata. "
            f"Master target field: {row['master_field']}. "
            f"Description: {row['master_description']}. "
            f"Field type: {row['master_field_type']}."
        ),
        axis=1,
    )

    return master_df


def classify_relationship(state_field, master_field):
    """Assign an initial crosswalk relationship label."""
    state_name = state_field.upper().replace("_", " ")
    master_name = master_field.upper().replace("_", " ")

    if state_name == master_name:
        return "Equivalent", False, "Direct equivalent field-name match."

    if ("AADT" in state_name or "ADT" in state_name) and "AADT" in master_name:
        return (
            "Formulaic_Overlap",
            True,
            "Traffic-volume fields may require aggregation, adjustment, or weighting validation."
        )

    return (
        "Conceptual_Overlap",
        False,
        "Semantic candidate match; confirm field meaning and any required transformation."
    )

def assign_decision_zone(similarity_score):
    """Assign the required action based on semantic similarity."""
    if similarity_score >= 0.82:
        return "Auto_Accept"
    if similarity_score >= 0.70:
        return "HITL_Review"
    return "State_Unique"

def build_crosswalk_records(state_df, master_df):
    """Match every state field to its best master target field."""
    print(f"Loading SBERT model: {MODEL_NAME}")
    model = SentenceTransformer(MODEL_NAME)

    print(f"Creating embeddings for {len(state_df)} state metadata records...")
    state_embeddings = model.encode(
        state_df["embedding_text"].tolist(),
        normalize_embeddings=True,
        show_progress_bar=True,
    )

    print(f"Creating embeddings for {len(master_df)} master target fields...")
    master_embeddings = model.encode(
        master_df["embedding_text"].tolist(),
        normalize_embeddings=True,
        show_progress_bar=True,
    )

    # Normalized embedding dot product equals cosine similarity.
    similarity_matrix = state_embeddings @ master_embeddings.T

    records = []

    for state_index, state_row in state_df.iterrows():
        best_master_index = similarity_matrix[state_index].argmax()
        best_score = float(similarity_matrix[state_index][best_master_index])
        decision_zone = assign_decision_zone(best_score)
        master_row = master_df.iloc[best_master_index]

        relationship_type, formula_applicable, transformation_logic = (
            classify_relationship(
                state_row["variable_name"],
                master_row["master_field"],
            )
        )

        records.append({
            "state": state_row["state"],
            "file": state_row["file"],
            "source_field": state_row["variable_name"],
            "source_description": state_row["description"],
            "source_field_type": state_row["field_type"],
            "master_field": master_row["master_field"],
            "master_description": master_row["master_description"],
            "master_field_type": master_row["master_field_type"],
            "semantic_similarity": round(best_score, 4),
            "decision_zone": decision_zone,
            "HITL_Verified": (
                "Pending" if decision_zone == "HITL_Review" else "Not_Required"
            ),
            "Reviewer_Notes": "",
            "relationship_type": relationship_type,
            "mathematical_formula_applicable": formula_applicable,
            "transformation_logic": transformation_logic,
        })

    return pd.DataFrame(records)


def main():
    parser = argparse.ArgumentParser(
        description="Match state metadata fields to HSIS master target fields."
    )

    parser.add_argument(
        "--state_schema",
        required=True,
        help="Path to the combined state metadata CSV.",
    )

    parser.add_argument(
        "--master_schema",
        required=True,
        help="Path to the HSIS master target schema CSV.",
    )
    parser.add_argument(
        "--output",
        required=True,
        help="Path for the generated crosswalk CSV.",
    )
    
    parser.add_argument(
        "--audit_output",
        default="logs/hitl_audit_queue.json",
        help="Path for HITL review records in JSON format.",
    )

    args = parser.parse_args()

    state_schema_path = Path(args.state_schema)
    master_schema_path = Path(args.master_schema)
    output_path = Path(args.output)
    audit_output_path = Path(args.audit_output)

    if not state_schema_path.exists():
        raise FileNotFoundError(f"State schema file not found: {state_schema_path}")

    if not master_schema_path.exists():
        raise FileNotFoundError(f"Master schema file not found: {master_schema_path}")

    state_df = load_state_schema(state_schema_path)
    master_df = load_master_schema(master_schema_path)

    crosswalk_df = build_crosswalk_records(state_df, master_df)

    output_path.parent.mkdir(parents=True, exist_ok=True)

    crosswalk_df = crosswalk_df.sort_values(
        by=["state", "semantic_similarity"],
        ascending=[True, False],
    )

    crosswalk_df.to_csv(output_path, index=False)
    hitl_queue = crosswalk_df[
        crosswalk_df["decision_zone"] == "HITL_Review"
    ].copy()

    audit_output_path.parent.mkdir(parents=True, exist_ok=True)

    hitl_queue.to_json(
        audit_output_path,
        orient="records",
        indent=2,
    )

    print("\nPipeline complete.")
    print(f"State fields processed: {len(state_df)}")
    print(f"Master target fields: {len(master_df)}")
    print(f"Output written to: {output_path}")
    print(f"HITL audit records: {len(hitl_queue)}")
    print(f"Audit queue written to: {audit_output_path}")


if __name__ == "__main__":
    main()