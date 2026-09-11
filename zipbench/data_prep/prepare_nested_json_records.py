#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Non-destructively normalize nested real evaluation JSON records into the directory layout expected by the aggregation script.

The input may be a mixed layout such as:
  root/
    model_A/model_A_MMStar_qwen3-30b_result.json
    model_B_full/model_B/model_B_MMStar_qwen3-30b_result.json

The output is:
  output_dir/
    model_A/MMStar.json              -> symlink/copy of the original JSON
    model_B_full/MMStar.json         -> symlink/copy of the original JSON

Symlinks are used by default; original files are never moved or renamed.
With --fallback-xlsx, a missing JSON is generated as output_dir/model/MMStar.json from the matching xlsx.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import shutil
import sys
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple


def _raise_csv_field_limit() -> None:
    limit = sys.maxsize
    while True:
        try:
            csv.field_size_limit(limit)
            return
        except OverflowError:
            limit //= 10


_raise_csv_field_limit()


def _candidate_suffixes(dataset: str, judge: Optional[str]) -> Sequence[str]:
    if judge:
        return (
            f"_{dataset}_{judge}_result.json",
            f"_{dataset}_{judge}.json",
        )
    return (
        f"_{dataset}.json",
        f"_{dataset}_result.json",
    )


def _split_judges(judge: Optional[str]) -> List[str]:
    if not judge:
        return []
    return [p.strip() for p in str(judge).split(",") if p.strip()]


def _candidate_xlsx_suffixes(dataset: str, judge: Optional[str]) -> Sequence[str]:
    if judge:
        return (
            f"_{dataset}_{judge}_result.xlsx",
            f"_{dataset}_{judge}.xlsx",
        )
    return (
        f"_{dataset}.xlsx",
        f"_{dataset}_result.xlsx",
    )


def _iter_record_roots(root: Path) -> Iterable[Path]:
    for child in sorted(root.iterdir(), key=lambda p: p.name):
        if child.name.startswith(".") or not child.is_dir():
            continue
        yield child


def _model_token_from_result_name(path: Path, dataset: str) -> str:
    marker = f"_{dataset}".lower()
    name = path.name
    idx = name.lower().find(marker)
    return name[:idx].lower() if idx >= 0 else path.stem.lower()


def _match_sort_key(model_dir: Path, dataset: str, path: Path) -> Tuple[int, int, int, str]:
    """Prefer candidates whose file-name prefix / parent directory semantically matches the outer model directory."""
    outer = model_dir.name.lower()
    parent = path.parent.name.lower()
    token = _model_token_from_result_name(path, dataset)
    semantic_match = bool(
        token and (
            token in outer or outer in token or parent in outer or outer in parent
        )
    )
    parent_matches_file = bool(token and parent == token)
    return (
        0 if semantic_match else 1,
        0 if parent_matches_file else 1,
        len(path.relative_to(model_dir).parts),
        str(path).lower(),
    )


def _find_json_in_model_dir(
    model_dir: Path,
    dataset: str,
    judge: Optional[str],
) -> Optional[Path]:
    simplevqa_eval = _find_simplevqa_eval_json_in_model_dir(model_dir, judge) if dataset.strip().lower() == "simplevqa" else None
    if simplevqa_eval is not None:
        return simplevqa_eval

    judge_values = _split_judges(judge)
    judge_keys = {j.lower() for j in judge_values}
    suffix_priority = None

    if judge_keys & {"any", "auto", "*"}:
        dataset_token = f"_{dataset}".lower()

        def is_match(path: Path) -> bool:
            name = path.name.lower()
            return f"{dataset_token}_" in name and name.endswith("_result.json")
    elif judge_values:
        target_suffixes = []
        for one_judge in judge_values:
            target_suffixes.extend([
                f"_{dataset}_{one_judge}_result.json",
                f"_{dataset}_{one_judge}.json",
            ])
        suffixes = tuple(s.lower() for s in target_suffixes)

        def is_match(path: Path) -> bool:
            return path.name.lower().endswith(suffixes)

        def suffix_priority(path: Path) -> int:
            name = path.name.lower()
            for idx, suffix in enumerate(suffixes):
                if name.endswith(suffix):
                    return idx
            return len(suffixes)
    else:
        suffixes = tuple(s.lower() for s in _candidate_suffixes(dataset, judge))

        def is_match(path: Path) -> bool:
            return path.name.lower().endswith(suffixes)

    matches = [
        json_path
        for json_path in model_dir.rglob("*.json")
        if is_match(json_path)
    ]
    if not matches:
        return None
    return sorted(
        matches,
        key=lambda p: (
            *_match_sort_key(model_dir, dataset, p)[:3],
            suffix_priority(p) if suffix_priority else 0,
            _match_sort_key(model_dir, dataset, p)[3],
        ),
    )[0]


def _find_simplevqa_eval_json_in_model_dir(model_dir: Path, judge: Optional[str]) -> Optional[Path]:
    """Find SimpleVQA judged eval JSON inside timestamped model folders."""
    judge_values = _split_judges(judge)
    judge_keys = {j.lower() for j in judge_values}

    def is_match(path: Path) -> bool:
        name = path.name.lower()
        if name == "model_eval.json":
            return True
        if judge_keys & {"any", "auto", "*"}:
            return name.endswith("_eval.json")
        if judge_values:
            return any(name == f"{j.lower()}_eval.json" for j in judge_values)
        return name.endswith("_eval.json")

    matches = [p for p in model_dir.rglob("*.json") if is_match(p)]
    if not matches:
        return None

    def sort_key(path: Path) -> Tuple[int, str, int, str]:
        timestamp = path.parent.name if path.parent.name.startswith("T") else ""
        return (
            0 if timestamp else 1,
            "".join(chr(255 - ord(ch)) for ch in timestamp),
            len(path.relative_to(model_dir).parts),
            str(path).lower(),
        )

    return sorted(
        matches,
        key=sort_key,
    )[0]


def _find_xlsx_in_model_dir(
    model_dir: Path,
    dataset: str,
    judge: Optional[str],
) -> Optional[Path]:
    judge_values = _split_judges(judge)
    judge_keys = {j.lower() for j in judge_values}
    suffix_priority = None

    if judge_keys & {"any", "auto", "*"}:
        dataset_token = f"_{dataset}".lower()

        def is_match(path: Path) -> bool:
            name = path.name.lower()
            return f"{dataset_token}_" in name and name.endswith("_result.xlsx")
    elif judge_values:
        target_suffixes = []
        for one_judge in judge_values:
            target_suffixes.extend([
                f"_{dataset}_{one_judge}_result.xlsx",
                f"_{dataset}_{one_judge}.xlsx",
            ])
        suffixes = tuple(s.lower() for s in target_suffixes)

        def is_match(path: Path) -> bool:
            return path.name.lower().endswith(suffixes)

        def suffix_priority(path: Path) -> int:
            name = path.name.lower()
            for idx, suffix in enumerate(suffixes):
                if name.endswith(suffix):
                    return idx
            return len(suffixes)
    else:
        suffixes = tuple(s.lower() for s in _candidate_xlsx_suffixes(dataset, judge))

        def is_match(path: Path) -> bool:
            return path.name.lower().endswith(suffixes)

    matches = [
        xlsx_path
        for xlsx_path in model_dir.rglob("*.xlsx")
        if is_match(xlsx_path)
    ]
    if not matches:
        tsv_matches = [p for p in model_dir.rglob("detail.tsv") if p.is_file()]
        if not tsv_matches:
            return None
        return sorted(
            tsv_matches,
            key=lambda p: (
                len(p.relative_to(model_dir).parts),
                str(p).lower(),
            ),
        )[0]
    return sorted(
        matches,
        key=lambda p: (
            *_match_sort_key(model_dir, dataset, p)[:3],
            suffix_priority(p) if suffix_priority else 0,
            _match_sort_key(model_dir, dataset, p)[3],
        ),
    )[0]


def _normalize_header(header: Any, index: int) -> str:
    raw = str(header).strip() if header is not None else ""
    if not raw:
        raw = f"col_{index}"
    lower = raw.lower()
    if lower in {"id", "index", "question", "answer", "prediction", "hit", "category", "l2_category", "bench", "log"}:
        return lower
    return raw


def _xlsx_to_records(xlsx_path: Path) -> List[Dict[str, Any]]:
    try:
        import openpyxl
    except ImportError as exc:
        raise RuntimeError("--fallback-xlsx requires openpyxl; please install openpyxl in the current Python environment") from exc

    wb = openpyxl.load_workbook(str(xlsx_path), read_only=True, data_only=True)
    ws = wb.active
    rows = list(ws.iter_rows(values_only=True))
    wb.close()

    if not rows:
        return []

    headers = [_normalize_header(h, i) for i, h in enumerate(rows[0])]
    records: List[Dict[str, Any]] = []
    for row in rows[1:]:
        if not any(cell is not None for cell in row):
            continue
        rec: Dict[str, Any] = {}
        for i, value in enumerate(row):
            if i < len(headers):
                rec[headers[i]] = value
        records.append(rec)
    return records


def _tsv_to_records(tsv_path: Path) -> List[Dict[str, Any]]:
    with tsv_path.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f, delimiter="\t")
        if not reader.fieldnames:
            return []
        headers = [
            _normalize_header(h, i)
            for i, h in enumerate(reader.fieldnames)
        ]
        records: List[Dict[str, Any]] = []
        for row in reader:
            rec: Dict[str, Any] = {}
            for i, raw_header in enumerate(reader.fieldnames):
                if raw_header is None:
                    continue
                rec[headers[i]] = row.get(raw_header)
            records.append(rec)
    return records


def _write_json_from_xlsx(src: Path, dst: Path, overwrite: bool) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    if dst.exists() or dst.is_symlink():
        if not overwrite:
            raise FileExistsError(f"Target file already exists: {dst}")
        _replace_path(dst)

    records = _tsv_to_records(src) if src.suffix.lower() == ".tsv" else _xlsx_to_records(src)
    with dst.open("w", encoding="utf-8") as f:
        json.dump(records, f, ensure_ascii=False, indent=4, default=str)


def _replace_path(path: Path) -> None:
    if path.is_symlink() or path.is_file():
        path.unlink()
    elif path.exists():
        raise IsADirectoryError(f"Target path exists and is neither a file nor a symlink: {path}")


def _link_or_copy(src: Path, dst: Path, mode: str, overwrite: bool) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    if dst.exists() or dst.is_symlink():
        if not overwrite:
            raise FileExistsError(f"Target file already exists: {dst}")
        _replace_path(dst)

    if mode == "symlink":
        os.symlink(src.resolve(), dst)
    elif mode == "copy":
        shutil.copy2(src, dst)
    else:
        raise ValueError(f"Unknown mode: {mode}")


def prepare_records(
    root: Path,
    output_dir: Path,
    dataset: str,
    judge: Optional[str],
    mode: str,
    overwrite: bool,
    dry_run: bool,
    fallback_xlsx: bool,
) -> bool:
    if not root.is_dir():
        print(f"Error: input root does not exist or is not a directory: {root}", file=sys.stderr)
        return False

    found = converted = missing = failed = 0
    target_filename = f"{dataset}.json"

    print(f"Input root: {root}")
    print(f"Output root: {output_dir}")
    print(f"Dataset file name: {target_filename}")
    print(f"judge: {judge or '(not specified)'}")
    print(f"Mode: {mode}; overwrite={overwrite}; dry_run={dry_run}; fallback_xlsx={fallback_xlsx}")
    print("=" * 80)

    for model_dir in _iter_record_roots(root):
        src = _find_json_in_model_dir(model_dir, dataset, judge)
        dst = output_dir / model_dir.name / target_filename

        if src is not None:
            print(f"[JSON] {model_dir.name}: {src} -> {dst}")
            found += 1

            if dry_run:
                continue

            try:
                _link_or_copy(src, dst, mode, overwrite)
            except Exception as exc:
                print(f"[ERROR] Failed to write: {dst} ({exc})", file=sys.stderr)
                failed += 1
            continue

        xlsx_src = _find_xlsx_in_model_dir(model_dir, dataset, judge) if fallback_xlsx else None
        if xlsx_src is None:
            print(f"[WARN] No matching JSON{(' or XLSX' if fallback_xlsx else '')} found: {model_dir.name}")
            missing += 1
            continue

        print(f"[XLSX] {model_dir.name}: {xlsx_src} -> {dst}")
        converted += 1

        if dry_run:
            continue

        try:
            _write_json_from_xlsx(xlsx_src, dst, overwrite)
        except Exception as exc:
            print(f"[ERROR] XLSX to JSON conversion failed: {dst} ({exc})", file=sys.stderr)
            failed += 1

    print("=" * 80)
    print(f"Done: json={found}, converted_from_xlsx={converted}, missing={missing}, failed={failed}")
    return (found + converted) > 0 and failed == 0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Normalize nested JSON records into a model/<dataset>.json layout for aggregate_results_json_multiqa.py."
    )
    parser.add_argument("root", help="Root directory containing the real model record sub-directories")
    parser.add_argument("output_dir", help="Output directory for the normalized layout")
    parser.add_argument("--dataset", default="MMStar", help="Dataset name; default MMStar")
    parser.add_argument("--judge", default=None, help="Judge suffix, e.g. qwen3-30b; comma-separated list allowed, e.g. qwen3-30b,deepseek-v3; any/auto/* accepts any *_result.json")
    parser.add_argument(
        "--mode",
        choices=("symlink", "copy"),
        default="symlink",
        help="Output mode: symlink by default; use copy if symlinks are not supported across file systems",
    )
    parser.add_argument("--overwrite", action="store_true", help="Overwrite existing output files/symlinks")
    parser.add_argument("--dry-run", action="store_true", help="Only print the plan without writing anything")
    parser.add_argument(
        "--fallback-xlsx",
        action="store_true",
        help="When no matching JSON exists, try to generate <dataset>.json from the matching xlsx",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    ok = prepare_records(
        root=Path(args.root).resolve(),
        output_dir=Path(args.output_dir).resolve(),
        dataset=args.dataset,
        judge=args.judge,
        mode=args.mode,
        overwrite=args.overwrite,
        dry_run=args.dry_run,
        fallback_xlsx=args.fallback_xlsx,
    )
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
