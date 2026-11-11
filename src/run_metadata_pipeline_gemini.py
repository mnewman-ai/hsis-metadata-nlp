"""Validate a candidate HSIS metadata-field pair with Gemini.

This is a standalone pair-validation prototype. It does not replace the
SBERT crosswalk engine; it returns a structured relationship assessment for
one candidate pair at a time.
"""

from __future__ import annotations

import json
import os
import time
from pathlib import Path
from typing import Literal

import pandas as pd
from google import genai
from google.genai import types
from pydantic import BaseModel, Field


PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_MODEL = "gemini-2.0-flash"
MAX_ATTEMPTS = 3
REQUEST_TIMEOUT_MS = 30_000


class MetadataCrosswalkAssessment(BaseModel):
    """Schema enforced on Gemini's relationship assessment."""

    relationship_type: Literal[
        "Equivalent",
        "Formulaic_Overlap",
        "Conceptual_Overlap",
        "Unrelated",
    ] = Field(
        description=(
            "Crosswalk classification for the candidate variable pair."
        )
    )
    mathematical_formula_applicable: bool = Field(
        description=(
            "True only when a deterministic mathematical or unit "
            "transformation is supported by the supplied metadata."
        )
    )
    transformation_logic: str = Field(
        description=(
            "One or two concise sentences explaining the mapping or why "
            "no mapping is supported."
        )
    )


def load_few_shot_anchors() -> str:
    """Load a small, deterministic sample of audited examples if available."""
    examples: list[str] = [
        "Verified reference examples of cross-state metadata mappings:"
    ]

    target_files = (
        "crosswalk_1_equivalent.csv",
        "crosswalk_2_formulaic.csv",
        "crosswalk_3_conceptual.csv",
    )
    required_columns = {
        "field_state_A",
        "description_A",
        "field_state_B",
        "description_B",
        "relationship_type",
        "mathematical_formula_applicable",
        "transformation_logic",
    }

    for filename in target_files:
        file_path = PROJECT_ROOT / "outputs" / filename
        if not file_path.exists():
            continue

        try:
            dataframe = pd.read_csv(file_path)
        except (OSError, pd.errors.ParserError) as error:
            print(f"⚠️ Skipping unreadable anchor file {filename}: {error}")
            continue

        missing = required_columns - set(dataframe.columns)
        if missing:
            print(
                f"⚠️ Skipping {filename}; missing expected columns: "
                f"{sorted(missing)}"
            )
            continue

        sample_count = min(2, len(dataframe))
        if sample_count == 0:
            continue

        for _, row in dataframe.sample(
            sample_count, random_state=42
        ).iterrows():
            examples.extend(
                [
                    "--- Example Pair ---",
                    (
                        f"Field A: '{row['field_state_A']}' | "
                        f"Description: {row['description_A']}"
                    ),
                    (
                        f"Field B: '{row['field_state_B']}' | "
                        f"Description: {row['description_B']}"
                    ),
                    "Output Assessment:",
                    f"- relationship_type: {row['relationship_type']}",
                    (
                        "- mathematical_formula_applicable: "
                        f"{row['mathematical_formula_applicable']}"
                    ),
                    (
                        "- transformation_logic: "
                        f"{row['transformation_logic']}"
                    ),
                    "",
                ]
            )

    return "\n".join(examples)


def execute_llm_validation(
    field_a: str,
    desc_a: str,
    field_b: str,
    desc_b: str,
    anchors: str,
) -> MetadataCrosswalkAssessment | None:
    """Assess one candidate pair using Gemini with bounded retries."""
    api_key = os.environ.get("GEMINI_API_KEY")
    if not api_key:
        print("⚠️ GEMINI_API_KEY is not set.")
        return None

    model_name = os.environ.get("GEMINI_MODEL", DEFAULT_MODEL)

    # The timeout must be defined before it is passed to the client.
    client = genai.Client(
        api_key=api_key,
        http_options=types.HttpOptions(timeout=REQUEST_TIMEOUT_MS),
    )

    system_instruction = """
You are an expert transportation-safety data analyst and strict metadata
schema-harmonization tool.

Classify the relationship between Field A and Field B using only their supplied
names, descriptions, and verified anchor examples.

Allowed relationship_type values:
- Equivalent: same concept, meaning, unit, and level of aggregation.
- Formulaic_Overlap: same underlying concept but a deterministic mathematical,
  unit, scale, weighting, aggregation, or transformation rule is required.
- Conceptual_Overlap: related concepts, but not directly interchangeable and
  not safely resolved by a deterministic formula alone.
- Unrelated: insufficient evidence of a meaningful harmonization relationship.

Rules:
1. Do not infer facts, units, formulas, or equivalence not supported by the
   supplied metadata.
2. Use Formulaic_Overlap only when a deterministic transformation is supported
   by the descriptions. Otherwise choose Conceptual_Overlap or Unrelated.
3. For Equivalent, mathematical_formula_applicable must be false and
   transformation_logic must say that no transformation is required.
4. For Unrelated, mathematical_formula_applicable must be false and
   transformation_logic must briefly explain why no mapping is supported.
5. Return only JSON conforming to the requested schema.
6. Keep transformation_logic concise: one or two sentences.
""".strip()

    user_query = f"""
Verified reference examples:
{anchors or "No verified examples were available."}

Target pair to assess:
Field A: "{field_a}"
Description A: "{desc_a}"

Field B: "{field_b}"
Description B: "{desc_b}"
""".strip()

    for attempt in range(1, MAX_ATTEMPTS + 1):
        try:
            print(
                f"⏳ Gemini attempt {attempt}/{MAX_ATTEMPTS}: "
                f"'{field_a}' ↔ '{field_b}'"
            )

            response = client.models.generate_content(
                model=model_name,
                contents=f"{system_instruction}\n\n{user_query}",
                config=types.GenerateContentConfig(
                    response_mime_type="application/json",
                    response_schema=MetadataCrosswalkAssessment,
                    temperature=0,
                    max_output_tokens=300,
                ),
            )

            if not response.text:
                raise ValueError("Gemini returned an empty response.")

            return MetadataCrosswalkAssessment.model_validate_json(
                response.text
            )

        except KeyboardInterrupt:
            print("\n⚠️ Gemini request cancelled by user.")
            return None

        except Exception as error:
            if attempt == MAX_ATTEMPTS:
                print(
                    "⚠️ Gemini validation failed after "
                    f"{MAX_ATTEMPTS} attempts: {error}"
                )
                return None

            wait_seconds = 2**attempt
            print(
                f"⚠️ Gemini attempt {attempt}/{MAX_ATTEMPTS} failed: "
                f"{error}\nRetrying in {wait_seconds} seconds..."
            )
            time.sleep(wait_seconds)

    return None


def main() -> None:
    """Run one standalone validation example."""
    print("⏳ Harvesting audited anchor examples...")
    anchors = load_few_shot_anchors()

    test_pair = {
        "field_a": "WEIGHTED_AADT",
        "desc_a": "Weighted average traffic volume by segment length.",
        "field_b": "TOTAL_AADT",
        "desc_b": "Total arithmetic average traffic volume across all segments.",
    }

    print(
        "🧠 Running pair validation on: "
        f"'{test_pair['field_a']}' ↔ '{test_pair['field_b']}'"
    )

    if not os.environ.get("GEMINI_API_KEY"):
        print("\n🔒 Simulation mode: GEMINI_API_KEY was not found.")
        print(
            json.dumps(
                {
                    "relationship_type": "Formulaic_Overlap",
                    "mathematical_formula_applicable": True,
                    "transformation_logic": (
                        "A segment-length weighting rule would be required "
                        "if the source metadata confirms compatible counts."
                    ),
                },
                indent=2,
            )
        )
        return

    result = execute_llm_validation(
        test_pair["field_a"],
        test_pair["desc_a"],
        test_pair["field_b"],
        test_pair["desc_b"],
        anchors,
    )

    if result is not None:
        print("\n🎉 Structured output received:")
        print(f"Type: {result.relationship_type}")
        print(
            "Formula Applicable: "
            f"{result.mathematical_formula_applicable}"
        )
        print(f"Transformation Logic: {result.transformation_logic}")


if __name__ == "__main__":
    main()
