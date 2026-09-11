"""vLLM local batch inference backend (Qwen3-VL style chat template).

All heavyweight dependencies (vllm / torch / transformers / qwen_vl_utils /
PIL) are imported lazily so that API-only users do not need the GPU stack.
"""

from __future__ import annotations

import argparse
import os
import sys
import time
from dataclasses import dataclass
from typing import Any, Dict, List

from tqdm import tqdm

from .common import (
    _pred_md_dir,
    _pred_tsv_path,
    _read_table,
    is_missing_prediction,
    write_predictions,
)


@dataclass
class InferenceConfig:
    model: str
    model_name: str
    tensor_parallel_size: int
    max_model_len: int
    max_tokens: int
    temperature: float
    top_p: float
    top_k: int
    batch_size: int
    trust_remote_code: bool
    save_every: int
    seed: int = 0
    max_num_seqs: int = 8


# ---------------------------------------------------------------------------
# vLLM runner
# ---------------------------------------------------------------------------


def _make_vllm_runner(cfg: InferenceConfig, prompt: str):
    """Create vLLM LLM + processor and return a run_batch callable."""
    from PIL import Image
    from transformers import AutoProcessor
    from vllm import LLM, SamplingParams

    os.environ["VLLM_WORKER_MULTIPROC_METHOD"] = "spawn"

    processor = AutoProcessor.from_pretrained(
        cfg.model, trust_remote_code=cfg.trust_remote_code
    )

    llm = LLM(
        model=cfg.model,
        max_num_seqs=cfg.max_num_seqs,
        max_model_len=cfg.max_model_len,
        limit_mm_per_prompt={"image": 1},
        tensor_parallel_size=cfg.tensor_parallel_size,
        trust_remote_code=cfg.trust_remote_code,
        gpu_memory_utilization=0.9,
        seed=cfg.seed,
    )

    sampling = SamplingParams(
        temperature=cfg.temperature,
        top_p=cfg.top_p,
        top_k=cfg.top_k,
        max_tokens=cfg.max_tokens,
    )

    try:
        from qwen_vl_utils import process_vision_info
    except ImportError:
        print("ERROR: qwen_vl_utils not found. pip install qwen-vl-utils")
        sys.exit(1)

    def _build_request(record: Dict[str, Any]) -> Dict[str, Any]:
        pil_img = Image.open(record["image_path"]).convert("RGB")
        # Same content order as the API backend: prompt text first, page image after.
        content = [
            {"type": "text", "text": prompt},
            {"type": "image", "image": pil_img},
        ]
        messages = [{"role": "user", "content": content}]
        text = processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        image_inputs, video_inputs, video_kwargs = process_vision_info(
            messages,
            image_patch_size=16,
            return_video_kwargs=True,
            return_video_metadata=True,
        )

        req: Dict[str, Any] = {"prompt": text}
        if image_inputs is not None:
            req["multi_modal_data"] = {"image": image_inputs}
        if video_kwargs is not None:
            req["mm_processor_kwargs"] = video_kwargs
        return req

    def run_batch(batch_records: List[Dict[str, Any]]) -> List[str]:
        requests = [_build_request(r) for r in batch_records]
        outputs = llm.generate(requests, sampling_params=sampling)
        return [
            out.outputs[0].text.strip() if out.outputs else ""
            for out in outputs
        ]

    return run_batch


# ---------------------------------------------------------------------------
# Inference main flow
# ---------------------------------------------------------------------------


def run_inference(
    args: argparse.Namespace,
    records: List[Dict[str, Any]],
    prompt: str,
) -> str:
    """Run vLLM batch inference over *records* and return the pred TSV path."""
    # Auto-detect tensor parallel size
    if args.tensor_parallel_size == 0:
        import torch

        gpu_count = torch.cuda.device_count()
        args.tensor_parallel_size = max(gpu_count, 1)
        print(
            f"Auto-detected {gpu_count} GPUs, "
            f"using tensor_parallel_size={args.tensor_parallel_size}"
        )

    cfg = InferenceConfig(
        model=args.model,
        model_name=args.model_name,
        tensor_parallel_size=args.tensor_parallel_size,
        max_model_len=args.max_model_len,
        max_tokens=args.max_tokens,
        temperature=args.temperature,
        top_p=args.top_p,
        top_k=args.top_k,
        batch_size=args.batch_size,
        trust_remote_code=args.trust_remote_code,
        save_every=args.save_every,
        seed=getattr(args, "seed", 0),
        max_num_seqs=getattr(args, "max_num_seqs", 8),
    )

    pred_tsv = _pred_tsv_path(args)
    pred_md_dir = _pred_md_dir(args)

    # Resume from an existing prediction TSV, keyed by image_basename (the
    # same key the API backend's checkpoint uses).
    pred_map: Dict[str, str] = {}
    if args.resume and os.path.exists(pred_tsv):
        old = _read_table(pred_tsv)
        if "prediction" in old.columns and "image_basename" in old.columns:
            restored = 0
            for key, value in zip(old["image_basename"], old["prediction"]):
                if not is_missing_prediction(value):
                    pred_map[str(key)] = str(value)
                    restored += 1
            print(f"Resumed {restored} predictions from {pred_tsv}")

    pending = [
        record for record in records
        if is_missing_prediction(pred_map.get(record["image_basename"]))
    ]
    print(f"Pending: {len(pending)}/{len(records)}")

    run_batch = _make_vllm_runner(cfg, prompt)

    t0 = time.time()
    processed_since_save = 0
    pbar = tqdm(total=len(pending), desc="Inference", unit="sample")
    for start in range(0, len(pending), cfg.batch_size):
        batch_records = pending[start : start + cfg.batch_size]

        try:
            batch_preds = run_batch(batch_records)
        except Exception as e:
            print(f"ERROR in batch starting at {batch_records[0]['image_basename']}: {e}")
            print("Retrying samples one-by-one...")
            batch_preds = []
            for single_record in batch_records:
                try:
                    single_pred = run_batch([single_record])
                    batch_preds.append(single_pred[0])
                except Exception as e2:
                    print(f"  SKIP {single_record['image_basename']}: {e2}")
                    batch_preds.append("")

        for record, pred in zip(batch_records, batch_preds):
            pred_map[record["image_basename"]] = pred

        processed_since_save += len(batch_records)
        pbar.update(len(batch_records))

        if cfg.save_every > 0 and processed_since_save >= cfg.save_every:
            write_predictions(records, pred_map, pred_tsv, pred_md_dir)
            pbar.write(f"Checkpoint: {pred_tsv}")
            processed_since_save = 0

    pbar.close()
    elapsed = time.time() - t0
    write_predictions(records, pred_map, pred_tsv, pred_md_dir)
    print(f"Total inference time: {elapsed:.1f}s ({elapsed/60:.1f}min)")
    return pred_tsv
