"""API inference backend via VLMEvalKit's supported_VLM model configs.

vlmeval is imported lazily inside run_inference so the module itself stays
importable without the full VLMEvalKit dependency stack.
"""

from __future__ import annotations

import argparse
import os
from typing import Any, Dict, List

from .common import (
    FAIL_MSG,
    _checkpoint_path,
    _output_dir,
    _pred_md_dir,
    _pred_tsv_path,
    write_predictions,
)


def build_vlmeval_messages(image_path: str, prompt: str) -> List[Dict[str, str]]:
    return [
        {"type": "text", "value": prompt},
        {"type": "image", "value": image_path},
    ]


def call_model_generate(model_cfg: Any, vlmeval_messages: List[Dict[str, str]]) -> str:
    try:
        result = model_cfg.generate(vlmeval_messages)
        if isinstance(result, dict):
            result = result.get("prediction", result)
        if result is not None and FAIL_MSG not in str(result):
            return str(result)
        last_error = getattr(model_cfg, "last_error", None)
        if last_error:
            print(f"call_model_generate failed: {last_error}")
    except Exception as exc:
        print(f"call_model_generate failed: {exc}")
    return FAIL_MSG


def run_inference(
    args: argparse.Namespace,
    records: List[Dict[str, Any]],
    prompt: str,
) -> str:
    """Run concurrent API inference over *records* and return the pred TSV path."""
    pred_tsv = _pred_tsv_path(args)
    pred_md_dir = _pred_md_dir(args)
    checkpoint = _checkpoint_path(args)
    os.makedirs(os.path.dirname(checkpoint), exist_ok=True)

    all_messages = [
        build_vlmeval_messages(record["image_path"], prompt) for record in records
    ]
    keys = [record["image_basename"] for record in records]

    from vlmeval.config import supported_VLM
    from vlmeval.smp.file import dump, load
    from vlmeval.utils.mp_util import track_progress_rich

    if args.model_name not in supported_VLM:
        raise ValueError(
            f"Model '{args.model_name}' not found in supported_VLM. "
            f"Available examples: {list(supported_VLM.keys())[:20]}"
        )
    model_cfg = supported_VLM[args.model_name]()
    model_label = getattr(model_cfg, "model", args.model_name)
    print(f"Model config: {args.model_name} -> {model_label}")

    res = load(checkpoint) if os.path.exists(checkpoint) else {}
    failed_keys = [
        key for key, value in res.items()
        if isinstance(value, str) and FAIL_MSG in value
    ]
    if failed_keys:
        print(f"Retrying {len(failed_keys)} failed responses from checkpoint")
        for key in failed_keys:
            del res[key]
        dump(res, checkpoint)

    pending = [
        (messages, key)
        for messages, key in zip(all_messages, keys)
        if key not in res
    ]
    print(f"Pending: {len(pending)}/{len(keys)} (checkpoint: {len(res)} done)")

    if pending:
        tasks = [
            {"model_cfg": model_cfg, "vlmeval_messages": messages}
            for messages, _ in pending
        ]
        pending_keys = [key for _, key in pending]
        track_progress_rich(
            call_model_generate,
            tasks,
            nproc=args.nproc,
            save=checkpoint,
            keys=pending_keys,
        )

    res = load(checkpoint) if os.path.exists(checkpoint) else {}
    write_predictions(records, res, pred_tsv, pred_md_dir)
    print(f"Saved checkpoint to {checkpoint}")
    return pred_tsv
