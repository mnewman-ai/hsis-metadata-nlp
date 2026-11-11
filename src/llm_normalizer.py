"""Normalize state DOT metadata into a structured representation.

Uses OpenAI when available and caches normalized results locally.
Use `use_llm=False` for an offline, no-API fallback.
"""

import hashlib
import json
import os
import re
from pathlib import Path

from openai import OpenAI
from pydantic import BaseModel


PROJECT_ROOT = Path(__file__).resolve().parent.parent
CACHE_FILE = PROJECT_ROOT / "data" / "cache" / "llm_normalizations.json"
PROMPT_VERSION = "1.0"
DEFAULT_MODEL = "gpt-4o-mini"

ACRONYM_LOOKUP = {
    "AADT": "annual average daily traffic",
    "ADT": "average daily traffic",
    "LNWID": "lane width",
    "LANE_WD": "lane width",
    "LANE_WIDTH_NBR": "lane width number",
    "SPD": "speed",
    "SPEEDLIMIT": "posted speed limit",
    "DESG_SPD": "design speed",
    "SHLD": "shoulder",
    "MED": "median",
}


class NormalizedMetadata(BaseModel):
    """The JSON schema required by the project prompt design."""

    standard_name: str
    domain_category: str
    description: str
    unit_type: str


def clean_text(value: object) -> str:
    """Convert empty/missing values to a clean string."""
    if value is None:
        return ""
    return re.sub(r"\s+", " ", str(value)).strip()


def expand_acronyms(text: str) -> str:
    """Expand known HSIS/state-DOT acronyms without changing unknown tokens."""
    expanded = clean_text(text)

    for acronym, meaning in ACRONYM_LOOKUP.items():
        expanded = re.sub(
            rf"\b{re.escape(acronym)}\b",
            meaning,
            expanded,
            flags=re.IGNORECASE,
        )

    return expanded


def load_cache() -> dict:
    """Load prior normalizations; return an empty cache if none exists."""
    if not CACHE_FILE.exists():
        return {}

    try:
        with CACHE_FILE.open("r", encoding="utf-8") as file:
            return json.load(file)
    except (json.JSONDecodeError, OSError):
        return {}


def save_cache(cache: dict) -> None:
    """Write the normalization cache to disk."""
    CACHE_FILE.parent.mkdir(parents=True, exist_ok=True)

    with CACHE_FILE.open("w", encoding="utf-8") as file:
        json.dump(cache, file, indent=2, sort_keys=True)


def build_cache_key(
    state: str,
    variable_name: str,
    description: str,
    field_type: str,
    use_llm: bool,
) -> str:
    """Create a stable key so only changed fields cause a new API call."""
    source = {
        "prompt_version": PROMPT_VERSION,
        "state": clean_text(state),
        "variable_name": clean_text(variable_name),
        "description": clean_text(description),
        "field_type": clean_text(field_type),
        "normalization_mode": "openai" if use_llm else "offline",
    }

    serialized = json.dumps(source, sort_keys=True)
    return hashlib.sha256(serialized.encode("utf-8")).hexdigest()


def offline_normalization(
    variable_name: str,
    description: str,
    field_type: str,
) -> NormalizedMetadata:
    """Create a deterministic fallback without an API request."""
    expanded_name = expand_acronyms(variable_name)
    expanded_description = expand_acronyms(description)

    return NormalizedMetadata(
        standard_name=expanded_name.lower().replace(" ", "_"),
        domain_category="highway_safety_and_roadway_metadata",
        description=expanded_description or expanded_name,
        unit_type=clean_text(field_type) or "not specified",
    )


def normalize_metadata(
    state: str,
    variable_name: str,
    description: str,
    field_type: str,
    use_llm: bool = True,
) -> tuple[NormalizedMetadata, str, bool]:
    """
    Normalize one state metadata field.

    Returns:
        (normalized_metadata, normalization_source, cache_hit)

    normalization_source is one of:
        - "openai"
        - "offline_fallback"
    """
    cache_key = build_cache_key(
        state,
        variable_name,
        description,
        field_type,
        use_llm,
    )
    cache = load_cache()

    if cache_key in cache:
        cached_record = cache[cache_key]
        return (
            NormalizedMetadata.model_validate(cached_record["normalized"]),
            cached_record["normalization_source"],
            True,
        )

    if not use_llm:
        normalized = offline_normalization(
            variable_name=variable_name,
            description=description,
            field_type=field_type,
        )
        source = "offline_fallback"
    else:
        api_key = os.environ.get("OPENAI_API_KEY")
        if not api_key:
            raise RuntimeError(
                "OPENAI_API_KEY is not set. Add it as a Codespaces secret, "
                "or run the engine with --offline."
            )

        model_name = os.environ.get("ARTN_MODEL")
        if not model_name:
            raise RuntimeError(
                "ARTN_MODEL is not set. Add it as a Codespaces secret."
            )
        base_url = os.environ.get("ARTN_BASE_URL")
        if not base_url:
            raise RuntimeError(
                "ARTN_BASE_URL is not set. Add it as a Codespaces secret."
            )

        client = OpenAI(
            api_key=api_key,
            base_url=base_url,
        )

        system_prompt = """
You are an expert Transportation Data Architecture Engine.

Normalize one raw state DOT data-dictionary field for semantic matching
against a national highway-safety master schema.

Return only JSON with exactly these keys:
- standard_name
- domain_category
- description
- unit_type

Rules:
- Expand abbreviations when their meaning is reasonably supported.
- Do not invent units, meanings, transformations, or source facts.
- Use "not specified" when the unit cannot be determined.
- Keep descriptions concise and factual.
""".strip()

        user_prompt = f"""
State: {clean_text(state)}
Raw variable name: {clean_text(variable_name)}
Known abbreviation expansion: {expand_acronyms(variable_name)}
Source description: {clean_text(description)}
Field type: {clean_text(field_type)}
""".strip()

        response = client.beta.chat.completions.parse(
            model=model_name,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            response_format=NormalizedMetadata,
        )

        normalized = response.choices[0].message.parsed
        if normalized is None:
            raise RuntimeError("OpenAI returned no structured normalization.")

        source = "openai"

    cache[cache_key] = {
        "normalization_source": source,
        "normalized": normalized.model_dump(),
    }
    save_cache(cache)

    return normalized, source, False