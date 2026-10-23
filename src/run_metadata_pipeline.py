import os
import json
import pandas as pd
from typing import Literal
from pydantic import BaseModel, Field
from openai import OpenAI
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent

# ==========================================
# JULY 10 TASK: SCHEMA-CONSTRAINED GUARDS
# ==========================================
class MetadataCrosswalkAssessment(BaseModel):
    """
    Forces the LLM to return data adhering to a strict schema structure,
    eliminating conversational filler and structural hallucinations.
    """
    relationship_type: Literal["Equivalent", "Formulaic_Overlap", "Conceptual_Overlap", "Unrelated"] = Field(
        description="The multi-tiered crosswalk classification tier for this variable pair."
    )
    mathematical_formula_applicable: bool = Field(
        description="True if a deterministic mathematical calculation or scale modification bridges these fields."
    )
    transformation_logic: str = Field(
        description="A detailed 1-2 sentence explanation or formula tracking the logic required to harmonize the variables."
    )

# ==========================================
# JULY 8-9 TASK: COT & FEW-SHOT ANCHORS
# ==========================================
def load_few_shot_anchors():
    """
    Gathers real, audited examples from your Week 1 files to ground the model.
    """
    examples_text = "Here are verified target examples of cross-state metadata mappings:\n\n"
    target_files = [
        ("crosswalk_1_equivalent.csv"),
        ("crosswalk_2_formulaic.csv"),
        ("crosswalk_3_conceptual.csv")
    ]
    
    for filename in target_files:
        file_path = PROJECT_ROOT / "outputs" / filename
        if os.path.exists(file_path):
            df = pd.read_csv(file_path)
            # Sample up to 2 items from each file if available
            sample_count = min(2, len(df))
            if sample_count > 0:
                sample_df = df.sample(sample_count, random_state=42)
                for _, row in sample_df.iterrows():
                    examples_text += f"--- Example Pair ---\n"
                    examples_text += f"Field A: '{row['field_state_A']}' | Description: {row['description_A']}\n"
                    examples_text += f"Field B: '{row['field_state_B']}' | Description: {row['description_B']}\n"
                    examples_text += f"Output Assessment:\n"
                    examples_text += f"- relationship_type: {row['relationship_type']}\n"
                    examples_text += f"- mathematical_formula_applicable: {row['mathematical_formula_applicable']}\n"
                    examples_text += f"- transformation_logic: {row['transformation_logic']}\n\n"
    return examples_text

def execute_llm_validation(field_a, desc_a, field_b, desc_b, anchors):
    """
    Sends the math-aware prompt layer and candidate pair to the OpenAI API,
    enforcing a Pydantic output grammar schema.
    """
    # Securely retrieve API Key from environment variables
    client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY", "your-key-here"))
    
    system_instruction = (
        "You are an expert transportation safety data analyst acting as a strict schema harmonization tool.\n"
        "Your task is to classify the relationship between two state DOT metadata variables.\n"
        "CRITICAL RULES:\n"
        "1. Evaluate mathematical properties. If Field A and Field B measure the same underlying concept but use "
        "different aggregation metrics, variations, or scales (e.g., weighted vs. total average volume), you MUST "
        "classify them as 'Formulaic_Overlap'.\n"
        "2. Think step-by-step using Chain-of-Thought reasoning, tracing statistical profiles before choosing your class."
    )
    
    user_query = (
        f"{anchors}\n"
        f"Now, carefully analyze this new target pair:\n"
        f"Field A: '{field_a}' | Description: {desc_a}\n"
        f"Field B: '{field_b}' | Description: {desc_b}\n\n"
        "Generate your response matching the requested schema layout."
    )
    
    try:
        # Utilizing beta structured outputs to enforce Pydantic parsing natively
        completion = client.beta.chat.completions.parse(
            model="gpt-4o-mini", # High-speed, high-precision model for schema parsing
            messages=[
                {"role": "system", "content": system_instruction},
                {"role": "user", "content": user_query}
            ],
            response_format=MetadataCrosswalkAssessment,
        )
        return completion.choices[0].message.parsed
    except Exception as e:
        print(f"⚠️ API Execution Error: {e}")
        return None

def main():
    print("⏳ Harvesting Week 1 ground truth anchors...")
    anchors = load_few_shot_anchors()
    
    # Let's test a complex anomaly pair that traditional embeddings fail on!
    test_pair = {
        "field_a": "WEIGHTED_AADT",
        "desc_a": "Weighted Average Traffic Volume by segment length",
        "field_b": "TOTAL_AADT",
        "desc_b": "Total arithmetic average traffic volume across all segments"
    }
    
    print(f"🧠 Running math-aware validation on: '{test_pair['field_a']}' ↔️ '{test_pair['field_b']}'")
    
    # If API key is not configured, preview the payload structure
    if os.environ.get("OPENAI_API_KEY") is None:
        print("\n🔒 [Simulation Mode: No OPENAI_API_KEY found in your environment environment variables]")
        print("Here is how your Pydantic schema will force the output to compile:")
        simulated_json = {
            "relationship_type": "Formulaic_Overlap",
            "mathematical_formula_applicable": True,
            "transformation_logic": "Convert the total arithmetic traffic volume into a segment-weighted average by introducing segment length parameters."
        }
        print(json.dumps(simulated_json, indent=4))
    else:
        # Run production call
        result = execute_llm_validation(test_pair["field_a"], test_pair["desc_a"], test_pair["field_b"], test_pair["desc_b"], anchors)
        if result:
            print("\n🎉 Structured Output Received Successfully:")
            print(f"Type: {result.relationship_type}")
            print(f"Formula Applicable: {result.mathematical_formula_applicable}")
            print(f"Transformation Logic: {result.transformation_logic}")

if __name__ == "__main__":
    main()
    
