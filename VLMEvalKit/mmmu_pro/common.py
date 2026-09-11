"""Shared helpers for the MMMU-Pro evaluator: constants, table I/O,
prompt construction and <image N> token parsing."""

from __future__ import annotations

import ast
import base64
import io
import os
import re
import string
from typing import Any, Dict, List, Optional, Tuple

import pandas as pd

# ---------------------------------------------------------------------------
# Constants – official prompts from mmmu-pro/prompts.yaml (direct mode)
# Source: https://github.com/MMMU-Benchmark/MMMU/blob/main/mmmu-pro/prompts.yaml
# ---------------------------------------------------------------------------

PROMPT_DIRECT_STANDARD = (
    "Answer with the option letter from the given choices directly."
)
PROMPT_DIRECT_VISION = (
    "Answer with the option letter from the given choices directly. "
    "The last line of your response should be of the following format: "
    "'Answer: $LETTER' (without quotes) where LETTER is one of options."
)

# Official domain → subcategory mapping from evaluate.py
DOMAIN_CAT2SUB_CAT = {
    "Art and Design": ["Art", "Art_Theory", "Design", "Music"],
    "Business": ["Accounting", "Economics", "Finance", "Manage", "Marketing"],
    "Science": ["Biology", "Chemistry", "Geography", "Math", "Physics"],
    "Health and Medicine": [
        "Basic_Medical_Science",
        "Clinical_Medicine",
        "Diagnostics_and_Laboratory_Medicine",
        "Pharmacy",
        "Public_Health",
    ],
    "Humanities and Social Science": [
        "History",
        "Literature",
        "Sociology",
        "Psychology",
    ],
    "Tech and Engineering": [
        "Agriculture",
        "Architecture_and_Engineering",
        "Computer_Science",
        "Electronics",
        "Energy_and_Power",
        "Materials",
        "Mechanical_Engineering",
    ],
}

EXPECTED_NUM_SAMPLES = 1730

# ---------------------------------------------------------------------------
# Table I/O helpers
# ---------------------------------------------------------------------------


def read_table(path: str) -> pd.DataFrame:
    ext = os.path.splitext(path)[1].lower()
    if ext in {".tsv", ".txt"}:
        return pd.read_csv(path, sep="\t")
    if ext == ".csv":
        return pd.read_csv(path)
    if ext == ".jsonl":
        return pd.read_json(path, lines=True)
    raise ValueError(f"Unsupported input format: {path}")


def write_table(df: pd.DataFrame, path: str) -> None:
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    ext = os.path.splitext(path)[1].lower()
    if ext in {".tsv", ".txt"}:
        df.to_csv(path, sep="\t", index=False)
    elif ext == ".csv":
        df.to_csv(path, index=False)
    elif ext == ".jsonl":
        df.to_json(path, orient="records", lines=True, force_ascii=False)
    else:
        raise ValueError(f"Unsupported output format: {path}")


# ---------------------------------------------------------------------------
# Image helpers
# ---------------------------------------------------------------------------


def decode_b64_to_pil(s: str):
    """Decode a base64 string to a PIL Image (RGB)."""
    from PIL import Image

    if s.startswith("data:image"):
        s = s.split(",", 1)[1]
    raw = base64.b64decode(s)
    return Image.open(io.BytesIO(raw)).convert("RGB")


def normalize_images_b64(row: pd.Series, dataset: str) -> List[str]:
    """Return a list of base64 strings for the given row."""
    if dataset == "MMMU_Pro_V":
        # MMMU_Pro_V: image field is a plain base64 string (not a list-string)
        return [row["image"]]
    else:
        # MMMU_Pro_10c: image field is a stringified Python list of base64 strings
        return ast.literal_eval(row["image"])


# ---------------------------------------------------------------------------
# Prompt construction (official-compatible)
# ---------------------------------------------------------------------------


def _load_prompts_from_yaml(yaml_path: str, version: str) -> Dict[str, str]:
    """Load prompts for *version* from a YAML file.

    The YAML is expected to have top-level keys (e.g. ``direct``, ``cot``)
    each mapping to ``{"standard": ..., "vision": ...}``.
    """
    import yaml

    with open(yaml_path, "r") as f:
        cfg = yaml.safe_load(f)
    if version not in cfg:
        available = list(cfg.keys())
        raise ValueError(
            f"Prompt version '{version}' not found in {yaml_path}. "
            f"Available: {available}"
        )
    return cfg[version]


def get_prompts(
    work_dir: Optional[str],
    prompt_version: str = "direct",
    prompt_file: Optional[str] = None,
) -> Dict[str, str]:
    """Return prompts for the given *prompt_version*.

    Resolution order:
      1. ``--prompt-file`` (explicit YAML path)  →  load *prompt_version* key
      2. Official YAML under ``work_dir/third_party/MMMU/mmmu-pro/prompts.yaml``
      3. Built-in constants (only available for ``direct``)
    """
    # 1. Explicit YAML file
    if prompt_file:
        prompts = _load_prompts_from_yaml(prompt_file, prompt_version)
        print(f"Loaded prompt version '{prompt_version}' from {prompt_file}")
        return prompts

    # 2. Official YAML in work_dir
    if work_dir:
        yaml_path = os.path.join(
            work_dir, "third_party", "MMMU", "mmmu-pro", "prompts.yaml"
        )
        if os.path.isfile(yaml_path):
            try:
                prompts = _load_prompts_from_yaml(yaml_path, prompt_version)
                print(f"Loaded prompt version '{prompt_version}' from {yaml_path}")
                return prompts
            except Exception as e:
                if prompt_version != "direct":
                    raise
                print(f"Warning: failed to load {yaml_path}: {e}; using built-in prompts")

    # 3. Fallback – built-in constants (direct only)
    if prompt_version != "direct":
        raise ValueError(
            f"Cannot find prompts.yaml for version '{prompt_version}'. "
            f"Provide --prompt-file or ensure third_party/MMMU is available."
        )
    return {"standard": PROMPT_DIRECT_STANDARD, "vision": PROMPT_DIRECT_VISION}


def build_standard_prompt(row: pd.Series, instruction: str) -> str:
    """Build the full text prompt for MMMU_Pro_10c (official format, no prefixes).

    Format (matching official construct_prompt + parse_options):
        {question}
        A. {opt_A}
        B. {opt_B}
        ...
        J. {opt_J}
        {instruction}
    """
    question = str(row["question"])
    option_letters = list(string.ascii_uppercase[:10])  # A-J
    options_parts = []
    for letter in option_letters:
        if letter in row and pd.notna(row[letter]):
            options_parts.append(f"{letter}. {row[letter]}")
    parsed_options = "\n".join(options_parts)
    return f"{question}\n{parsed_options}\n{instruction}"


# ---------------------------------------------------------------------------
# <image N> token parsing
# ---------------------------------------------------------------------------

_IMAGE_TOKEN_RE = re.compile(r"<image\s+(\d+)>")


def replace_image_tokens_full_prompt(
    prompt_text: str,
) -> Tuple[List[str], List[int]]:
    """Split *full* prompt on <image N> tokens.

    Returns:
        text_segments – text pieces between image tokens (len = len(image_order) + 1)
        image_order  – 1-based image indices in the order they appear
    """
    image_order: List[int] = []
    text_segments: List[str] = []
    last_end = 0
    for m in _IMAGE_TOKEN_RE.finditer(prompt_text):
        text_segments.append(prompt_text[last_end : m.start()])
        image_order.append(int(m.group(1)))
        last_end = m.end()
    text_segments.append(prompt_text[last_end:])
    return text_segments, image_order


# ---------------------------------------------------------------------------
# ZipBench subset support
# ---------------------------------------------------------------------------

ZIP_WEIGHT_COL = "zip_weight"


def apply_zip_subset(df: pd.DataFrame, dataset: str, subset: str) -> pd.DataFrame:
    """Filter *df* down to a ZipBench subset and attach the ``zip_weight`` column.

    Row selection is by the value of the ``index`` column (specs live in
    vlmeval/zipbench/subsets/). ``subset in (None, "full")`` returns *df*
    unchanged.
    """
    if subset in (None, "full"):
        return df
    from vlmeval.zipbench.spec import load_subset_spec, select_rows

    spec = load_subset_spec(dataset, subset)
    df = select_rows(df, spec, weight_col=ZIP_WEIGHT_COL, dataset=dataset)
    print(
        f"Applied ZipBench subset '{subset}': {len(df)} rows, "
        f"weight_sum={df[ZIP_WEIGHT_COL].sum():.6f}"
    )
    return df


def load_zip_spec_weights(dataset: str, subset: str) -> Dict[Any, float]:
    """Return ``{index value -> weight}`` for a ZipBench subset."""
    from vlmeval.zipbench.spec import load_subset_spec

    return {rec["index"]: rec["weight"] for rec in load_subset_spec(dataset, subset)}


# ---------------------------------------------------------------------------
# Prediction helpers
# ---------------------------------------------------------------------------


def is_missing_prediction(value: Any) -> bool:
    if value is None:
        return True
    if isinstance(value, float) and pd.isna(value):
        return True
    text = str(value).strip().lower()
    return text in {"", "nan", "none"}


# ---------------------------------------------------------------------------
# Output directory convention
# ---------------------------------------------------------------------------


def resolve_run_dir(
    work_dir: str,
    dataset: str,
    model_name: str,
    prompt_version: str,
    max_tokens: Any,
    subset: str = "full",
    run_tag: str = "",
) -> str:
    """Build run output dir: {work_dir}/outputs/{dataset}/{model_name}_{prompt_version}_{max_tokens}/

    A non-full ZipBench subset appends ``_ZIP_{subset}`` (same tag convention as
    vlmeval.zipbench.naming) so subset runs never clobber the full run. A
    non-empty ``run_tag`` appends ``_{run_tag}`` so repeated runs of the same
    config never clobber each other.
    """
    if not work_dir:
        raise ValueError("--work-dir is required to resolve output directory")
    leaf = f"{model_name}_{prompt_version}_{max_tokens}"
    if subset not in (None, "full"):
        leaf += f"_ZIP_{subset}"
    if run_tag:
        leaf += f"_{run_tag}"
    run_dir = os.path.join(work_dir, "outputs", dataset, leaf)
    os.makedirs(run_dir, exist_ok=True)
    return run_dir
