"""Shared helpers for the OmniDocBench v1.5 evaluator.

Prompt, GT loading, image resolution, output-path conventions, and the
prediction writer shared by both inference backends.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import pandas as pd


REPO_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_WORK_DIR = str(REPO_ROOT)
FAIL_MSG = "Failed to obtain answer via API."

IMAGE_EXTENSIONS = {
    ".jpg",
    ".jpeg",
    ".png",
    ".bmp",
    ".webp",
    ".tif",
    ".tiff",
}

BUILTIN_V15_PROMPT = """You are an AI assistant specialized in converting PDF images to Markdown format. Please follow these instructions for the conversion:

1. Text Processing:
- Accurately recognize all text content in the PDF image without guessing or inferring.
- Convert the recognized text into Markdown format.
- Maintain the original document structure, including headings, paragraphs, lists, etc.

2. Mathematical Formula Processing:
- Convert all mathematical formulas to LaTeX format.
- Enclose inline formulas with \\( \\). For example: This is an inline formula \\( E = mc^2 \\)
- Enclose block formulas with \\[ \\]. For example: \\[ \\frac{-b \\pm \\sqrt{b^2 - 4ac}}{2a} \\]

3. Table Processing:
- Convert tables to HTML format.
- Wrap the entire table with <table> and </table>.

4. Figure Handling:
- Ignore figures content in the PDF image. Do not attempt to describe or convert images.

5. Output Format:
- Ensure the output Markdown document has a clear structure with appropriate line breaks between elements.
- For complex layouts, try to maintain the original document's structure and format as closely as possible.

Please strictly follow these guidelines to ensure accuracy and consistency in the conversion. Your task is to accurately convert the content of the PDF image into Markdown format without adding any extra explanations or comments.
"""


# ---------------------------------------------------------------------------
# Generic file helpers
# ---------------------------------------------------------------------------


def _abspath(path: str | os.PathLike[str]) -> str:
    return str(Path(path).expanduser().resolve())


def _ensure_dir(path: str | os.PathLike[str]) -> str:
    resolved = _abspath(path)
    os.makedirs(resolved, exist_ok=True)
    return resolved


def _read_table(path: str) -> pd.DataFrame:
    ext = os.path.splitext(path)[1].lower()
    if ext in {".tsv", ".txt"}:
        return pd.read_csv(path, sep="\t")
    if ext == ".csv":
        return pd.read_csv(path)
    if ext == ".jsonl":
        return pd.read_json(path, lines=True)
    if ext == ".json":
        data = json.load(open(path, "r", encoding="utf-8"))
        return pd.DataFrame(data)
    raise ValueError(f"Unsupported table format: {path}")


def _write_table(df: pd.DataFrame, path: str) -> None:
    os.makedirs(os.path.dirname(_abspath(path)), exist_ok=True)
    ext = os.path.splitext(path)[1].lower()
    if ext in {".tsv", ".txt"}:
        df.to_csv(path, sep="\t", index=False)
    elif ext == ".csv":
        df.to_csv(path, index=False)
    elif ext == ".jsonl":
        df.to_json(path, orient="records", lines=True, force_ascii=False)
    elif ext == ".json":
        df.to_json(path, orient="records", force_ascii=False, indent=2)
    else:
        raise ValueError(f"Unsupported output format: {path}")


def _load_json(path: str) -> Any:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def _dump_json(data: Any, path: str) -> None:
    os.makedirs(os.path.dirname(_abspath(path)), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(_json_clean(data), f, ensure_ascii=False, indent=2)


def _json_clean(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(k): _json_clean(v) for k, v in value.items()}
    if isinstance(value, list):
        return [_json_clean(v) for v in value]
    if isinstance(value, tuple):
        return [_json_clean(v) for v in value]
    if isinstance(value, float) and (math.isnan(value) or math.isinf(value)):
        return None
    return value


def _write_csv_rows(rows: Sequence[Dict[str, Any]], path: str) -> None:
    os.makedirs(os.path.dirname(_abspath(path)), exist_ok=True)
    fieldnames: List[str] = []
    for row in rows:
        for key in row:
            if key not in fieldnames:
                fieldnames.append(key)
    with open(path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(_json_clean(row))


def _limit_items(items: List[Any], limit: Optional[int]) -> List[Any]:
    if limit is None:
        return items
    if limit < 0:
        raise ValueError("--limit must be >= 0")
    return items[:limit]


# ---------------------------------------------------------------------------
# Prompt handling
# ---------------------------------------------------------------------------


def load_prompt(args: argparse.Namespace) -> str:
    if getattr(args, "prompt_file", None):
        prompt = Path(args.prompt_file).expanduser().read_text(encoding="utf-8").strip()
        if not prompt:
            raise ValueError(f"Prompt file is empty: {args.prompt_file}")
        print(f"Using prompt from {args.prompt_file}")
        return prompt
    print("Using built-in OmniDocBench v1.5 prompt")
    return BUILTIN_V15_PROMPT.strip()


# ---------------------------------------------------------------------------
# OmniDocBench GT and image resolution
# ---------------------------------------------------------------------------


def load_gt_samples(gt_json: str, limit: Optional[int] = None) -> List[Dict[str, Any]]:
    data = _load_json(gt_json)
    if isinstance(data, dict):
        for key in ("data", "samples", "annotations"):
            if isinstance(data.get(key), list):
                data = data[key]
                break
    if not isinstance(data, list):
        raise ValueError(f"Expected a list of samples in GT JSON: {gt_json}")
    samples = _limit_items(data, limit)
    for idx, sample in enumerate(samples):
        if not isinstance(sample, dict):
            raise ValueError(f"GT sample {idx} is not an object")
        page_info = sample.get("page_info")
        if not isinstance(page_info, dict) or not page_info.get("image_path"):
            raise ValueError(f"GT sample {idx} is missing page_info.image_path")
    return samples


def _build_image_index(image_root: str) -> Tuple[Dict[str, str], Dict[str, List[str]]]:
    root = _abspath(image_root)
    if not os.path.isdir(root):
        raise FileNotFoundError(f"Image root not found: {image_root}")
    by_name: Dict[str, str] = {}
    by_stem: Dict[str, List[str]] = {}
    duplicates: Dict[str, List[str]] = {}
    for current_root, _, files in os.walk(root):
        for filename in files:
            ext = os.path.splitext(filename)[1].lower()
            if ext not in IMAGE_EXTENSIONS:
                continue
            path = os.path.join(current_root, filename)
            if filename in by_name:
                duplicates.setdefault(filename, [by_name[filename]]).append(path)
            else:
                by_name[filename] = path
            by_stem.setdefault(os.path.splitext(filename)[0], []).append(path)
    if duplicates:
        shown = "\n".join(
            f"{name}: {paths[:3]}" for name, paths in list(duplicates.items())[:5]
        )
        raise RuntimeError(
            "Duplicate image basenames found under --image-root; cannot match safely:\n"
            f"{shown}"
        )
    return by_name, by_stem


def resolve_sample_images(
    samples: Sequence[Dict[str, Any]],
    image_root: str,
) -> List[Dict[str, Any]]:
    by_name, by_stem = _build_image_index(image_root)
    records: List[Dict[str, Any]] = []
    missing: List[str] = []
    ambiguous: List[str] = []

    for idx, sample in enumerate(samples):
        gt_image_path = str(sample["page_info"]["image_path"])
        basename = os.path.basename(gt_image_path)
        image_path = by_name.get(basename)
        if image_path is None:
            stem = os.path.splitext(basename)[0]
            candidates = by_stem.get(stem, [])
            if len(candidates) == 1:
                image_path = candidates[0]
            elif len(candidates) > 1:
                ambiguous.append(basename)
        if image_path is None:
            missing.append(basename)
            continue
        records.append(
            {
                "index": idx,
                "gt_image_path": gt_image_path,
                "image_basename": basename,
                "image_stem": os.path.splitext(basename)[0],
                "image_path": _abspath(image_path),
            }
        )

    if missing or ambiguous:
        lines = []
        if missing:
            lines.append(f"Missing images ({len(missing)}): {missing[:10]}")
        if ambiguous:
            lines.append(f"Ambiguous images ({len(ambiguous)}): {ambiguous[:10]}")
        raise FileNotFoundError("\n".join(lines))
    return records


def clean_markdown_prediction(prediction: Any) -> str:
    if prediction is None:
        return ""
    if isinstance(prediction, dict):
        prediction = prediction.get("prediction", "")
    text = str(prediction).strip()
    if "</think>" in text:
        text = text.rsplit("</think>", 1)[-1].strip()
    return text


def is_missing_prediction(prediction: Any) -> bool:
    """True if a prediction slot still needs inference (empty / NaN / FAIL_MSG)."""
    if prediction is None:
        return True
    if isinstance(prediction, float) and math.isnan(prediction):
        return True
    text = str(prediction).strip()
    return not text or FAIL_MSG in text


# ---------------------------------------------------------------------------
# Output-path conventions
# ---------------------------------------------------------------------------


def _output_dir(args: argparse.Namespace) -> str:
    if getattr(args, "output_dir", None):
        return _ensure_dir(args.output_dir)
    model_name = getattr(args, "model_name", "unknown_model")
    subset = getattr(args, "subset", "full")
    if subset and subset != "full":
        model_name = f"{model_name}_ZIP_{subset}"
    return _ensure_dir(os.path.join(args.work_dir, "outputs", "OmniDocBench_v1_5", model_name))


def _pred_tsv_path(args: argparse.Namespace) -> str:
    if getattr(args, "output", None):
        return _abspath(args.output)
    return os.path.join(_output_dir(args), "pred.tsv")


def _pred_md_dir(args: argparse.Namespace) -> str:
    if getattr(args, "pred_md_dir", None):
        return _ensure_dir(args.pred_md_dir)
    return _ensure_dir(os.path.join(_output_dir(args), "pred_md"))


def _checkpoint_path(args: argparse.Namespace) -> str:
    if getattr(args, "checkpoint", None):
        return _abspath(args.checkpoint)
    return os.path.join(_output_dir(args), "supp.pkl")


# ---------------------------------------------------------------------------
# Prediction writer shared by both backends
# ---------------------------------------------------------------------------


def write_predictions(
    records: Sequence[Dict[str, Any]],
    pred_map: Dict[str, Any],
    pred_tsv: str,
    pred_md_dir: str,
) -> None:
    """Write pred.tsv plus one pred_md/{stem}.md per page.

    *pred_map* maps image_basename -> raw model output. The markdown files are
    what the official scorer consumes, so both backends must go through here.
    """
    rows = []
    for record in records:
        raw_prediction = pred_map.get(record["image_basename"], "")
        clean_prediction = clean_markdown_prediction(raw_prediction)
        rows.append(
            {
                "index": record["index"],
                "image_basename": record["image_basename"],
                "gt_image_path": record["gt_image_path"],
                "image_path": record["image_path"],
                "prediction": str(raw_prediction),
                "prediction_md": clean_prediction,
            }
        )
        md_path = os.path.join(pred_md_dir, f"{record['image_stem']}.md")
        with open(md_path, "w", encoding="utf-8") as f:
            f.write(clean_prediction)
            if clean_prediction and not clean_prediction.endswith("\n"):
                f.write("\n")

    pred_df = pd.DataFrame(rows)
    _write_table(pred_df, pred_tsv)
    print(f"Saved predictions to {pred_tsv}")
    print(f"Saved markdown predictions to {pred_md_dir}")
