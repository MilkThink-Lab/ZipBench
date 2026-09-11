"""vLLM local batch inference backend (Qwen3-VL style chat template).

All heavyweight dependencies (vllm / torch / transformers / qwen_vl_utils)
are imported lazily so that API-only users do not need the GPU stack.
"""

from __future__ import annotations

import argparse
import os
import sys
import time
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

import pandas as pd
from tqdm import tqdm

from .common import (
    build_standard_prompt,
    decode_b64_to_pil,
    is_missing_prediction,
    normalize_images_b64,
    read_table,
    replace_image_tokens_full_prompt,
    resolve_run_dir,
    write_table,
)


@dataclass
class InferenceConfig:
    model: str
    model_name: str
    dataset: str
    tensor_parallel_size: int
    max_model_len: int
    max_tokens: int
    temperature: float
    top_p: float
    top_k: int
    batch_size: int
    trust_remote_code: bool
    save_every: int
    work_dir: Optional[str]
    seed: int = 0
    max_num_seqs: int = 8


# ---------------------------------------------------------------------------
# Interleaved content builders
# ---------------------------------------------------------------------------


def build_interleaved_content(
    prompt_text: str,
    images_b64: List[str],
    sample_id: str = "",
) -> List[Dict[str, Any]]:
    """Build Qwen3-VL content list from full prompt + base64 image list.

    Returns list of dicts: [{type: text, ...}, {type: image, ...}, ...]
    """
    text_segments, image_order = replace_image_tokens_full_prompt(prompt_text)

    if not image_order:
        # No image tokens at all – return plain text
        return [{"type": "text", "text": prompt_text}]

    # Validate indices
    for idx in image_order:
        if idx < 1 or idx > len(images_b64):
            print(
                f"WARNING: sample {sample_id!r} references <image {idx}> "
                f"but only {len(images_b64)} images available. Skipping this image."
            )

    content: List[Dict[str, Any]] = []
    for i, seg in enumerate(text_segments):
        if seg.strip():
            content.append({"type": "text", "text": seg})
        if i < len(image_order):
            img_idx = image_order[i]
            if 1 <= img_idx <= len(images_b64):
                pil_img = decode_b64_to_pil(images_b64[img_idx - 1])
                content.append({"type": "image", "image": pil_img})
    return content


def build_vision_content(
    row: pd.Series,
    instruction: str,
) -> List[Dict[str, Any]]:
    """Build Qwen3-VL content for MMMU_Pro_V (single image, no question text)."""
    pil_img = decode_b64_to_pil(row["image"])
    return [
        {"type": "image", "image": pil_img},
        {"type": "text", "text": instruction},
    ]


# ---------------------------------------------------------------------------
# vLLM runner
# ---------------------------------------------------------------------------


def _make_vllm_runner(cfg: InferenceConfig, prompts: Dict[str, str]):
    """Create vLLM LLM + processor and return a run_batch callable."""
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
        limit_mm_per_prompt={"image": 36},
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

    is_vision = cfg.dataset == "MMMU_Pro_V"
    instruction = prompts["vision"] if is_vision else prompts["standard"]

    try:
        from qwen_vl_utils import process_vision_info
    except ImportError:
        print("ERROR: qwen_vl_utils not found. pip install qwen-vl-utils")
        sys.exit(1)

    def _build_request(row: pd.Series) -> Dict[str, Any]:
        if is_vision:
            content = build_vision_content(row, instruction)
        else:
            prompt_text = build_standard_prompt(row, instruction)
            images_b64 = normalize_images_b64(row, cfg.dataset)
            sample_id = str(row.get("id", row.get("index", "")))
            content = build_interleaved_content(prompt_text, images_b64, sample_id)

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

    def run_batch(rows: List[pd.Series]) -> List[str]:
        requests = [_build_request(r) for r in rows]
        outputs = llm.generate(requests, sampling_params=sampling)
        return [
            out.outputs[0].text.strip() if out.outputs else ""
            for out in outputs
        ]

    return run_batch


# ---------------------------------------------------------------------------
# Inference main flow
# ---------------------------------------------------------------------------


def resolve_output_path(args: argparse.Namespace) -> str:
    """Return the prediction TSV path, preferring --output override."""
    if args.output:
        return args.output
    run_dir = resolve_run_dir(
        args.work_dir,
        args.dataset,
        args.model_name,
        getattr(args, "prompt_version", "direct"),
        getattr(args, "max_tokens", 2048),
        subset=getattr(args, "subset", "full"),
        run_tag=getattr(args, "run_tag", ""),
    )
    return os.path.join(run_dir, "pred.tsv")


def run_inference(
    args: argparse.Namespace,
    df: pd.DataFrame,
    prompts: Dict[str, str],
) -> str:
    """Run vLLM batch inference over *df* and return the prediction TSV path."""
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
        dataset=args.dataset,
        tensor_parallel_size=args.tensor_parallel_size,
        max_model_len=args.max_model_len,
        max_tokens=args.max_tokens,
        temperature=args.temperature,
        top_p=args.top_p,
        top_k=args.top_k,
        batch_size=args.batch_size,
        trust_remote_code=args.trust_remote_code,
        save_every=args.save_every,
        work_dir=args.work_dir,
        seed=getattr(args, "seed", 0),
        max_num_seqs=getattr(args, "max_num_seqs", 8),
    )

    run_batch = _make_vllm_runner(cfg, prompts)

    pred_path = resolve_output_path(args)
    out = df.copy()
    if "prediction" not in out.columns:
        out["prediction"] = ""

    # Resume
    if args.resume and os.path.exists(pred_path):
        old = read_table(pred_path)
        if "prediction" in old.columns and "index" in out.columns and "index" in old.columns:
            old_map = {
                str(k): v for k, v in zip(old["index"], old["prediction"])
            }
            restored = 0
            for i in range(len(out)):
                key = str(out.iloc[i].get("index", ""))
                if key in old_map and not is_missing_prediction(old_map[key]):
                    out.at[i, "prediction"] = old_map[key]
                    restored += 1
            print(f"Resumed {restored} predictions from {pred_path}")

    pending = [
        i
        for i in range(len(out))
        if is_missing_prediction(out.iloc[i].get("prediction", ""))
    ]
    print(f"Pending: {len(pending)}/{len(out)}")

    t0 = time.time()
    processed_since_save = 0
    pbar = tqdm(total=len(pending), desc="Inference", unit="sample")
    for start in range(0, len(pending), cfg.batch_size):
        batch_idx = pending[start : start + cfg.batch_size]
        batch_rows = [out.iloc[i] for i in batch_idx]

        try:
            batch_preds = run_batch(batch_rows)
        except Exception as e:
            print(f"ERROR in batch starting at index {batch_idx[0]}: {e}")
            print("Retrying samples one-by-one...")
            batch_preds = []
            for single_row in batch_rows:
                try:
                    single_pred = run_batch([single_row])
                    batch_preds.append(single_pred[0])
                except Exception as e2:
                    sid = single_row.get("index", "?")
                    print(f"  SKIP index {sid}: {e2}")
                    batch_preds.append("")

        for ridx, pred in zip(batch_idx, batch_preds):
            out.at[ridx, "prediction"] = pred

        processed_since_save += len(batch_idx)
        pbar.update(len(batch_idx))

        if cfg.save_every > 0 and processed_since_save >= cfg.save_every:
            write_table(out, pred_path)
            pbar.write(f"Checkpoint: {pred_path}")
            processed_since_save = 0

    pbar.close()
    elapsed = time.time() - t0
    write_table(out, pred_path)
    print(f"Saved predictions to {pred_path}")
    print(f"Total inference time: {elapsed:.1f}s ({elapsed/60:.1f}min)")
    return pred_path
