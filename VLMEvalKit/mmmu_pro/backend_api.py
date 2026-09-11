"""API inference backend.

Uses models configured in vlmeval/config.py (supported_VLM). vlmeval is
imported lazily so the module can be inspected without it installed.
"""

from __future__ import annotations

import argparse
import ast
import json
import os
import random
import time
from typing import Any, Dict, List

import pandas as pd
import requests

from .common import (
    build_standard_prompt,
    read_table,
    replace_image_tokens_full_prompt,
    resolve_run_dir,
    write_table,
)

FAIL_MSG = "Failed to obtain answer via API."


# ---------------------------------------------------------------------------
# Message builders
# ---------------------------------------------------------------------------


def _is_openai_compatible(model_cfg) -> bool:
    """Check if model uses OpenAI-compatible HTTP API."""
    from vlmeval.api.gpt import OpenAIWrapper
    return isinstance(model_cfg, OpenAIWrapper)


def _b64_to_data_url(b64: str) -> str:
    """Wrap raw base64 string as a data URL so parse_file() in BaseAPI recognises it."""
    return f"data:image/jpeg;base64,{b64}"


def build_vlmeval_messages(row: pd.Series, dataset: str, instruction: str) -> List[Dict]:
    """Build VLMEvalKit standard format messages with base64 images.

    Returns a list of {type, value} dicts compatible with BaseAPI.generate().
    Images are wrapped as data URLs for compatibility with BaseAPI.preproc_content().
    """
    messages: List[Dict[str, Any]] = []

    if dataset == "MMMU_Pro_V":
        b64 = row["image"]
        messages.append({"type": "image", "value": _b64_to_data_url(b64)})
        messages.append({"type": "text", "value": instruction})
    else:
        prompt_text = build_standard_prompt(row, instruction)
        images_b64 = ast.literal_eval(row["image"])
        text_segments, image_order = replace_image_tokens_full_prompt(prompt_text)

        for i, seg in enumerate(text_segments):
            if seg.strip():
                messages.append({"type": "text", "value": seg})
            if i < len(image_order):
                idx = image_order[i] - 1
                if 0 <= idx < len(images_b64):
                    messages.append({"type": "image", "value": _b64_to_data_url(images_b64[idx])})

    return messages


def call_model_generate(model_cfg, vlmeval_messages, retry=10) -> str:
    """Call model using VLMEvalKit's standard generate() interface.

    Works with any model backend (Gemini, etc.) that implements BaseAPI.
    """
    try:
        result = model_cfg.generate(vlmeval_messages)
        if result and FAIL_MSG not in str(result):
            return str(result)
        err_msg = getattr(model_cfg, "last_error", None)
        if err_msg:
            print(f"call_model_generate failed: {err_msg}")
    except Exception as e:
        print(f"call_model_generate failed: {e}")
    return FAIL_MSG


def build_openai_messages(row: pd.Series, dataset: str, instruction: str) -> List[Dict]:
    """Build OpenAI API format messages with base64 images inlined."""
    content: List[Dict[str, Any]] = []

    if dataset == "MMMU_Pro_V":
        # Single image + instruction
        b64 = row["image"]
        content.append({
            "type": "image_url",
            "image_url": {"url": f"data:image/jpeg;base64,{b64}", "detail": "high"},
        })
        content.append({"type": "text", "text": instruction})
    else:
        # MMMU_Pro_10c: parse <image N> tokens, interleave images and text
        prompt_text = build_standard_prompt(row, instruction)
        images_b64 = ast.literal_eval(row["image"])
        text_segments, image_order = replace_image_tokens_full_prompt(prompt_text)

        for i, seg in enumerate(text_segments):
            if seg.strip():
                content.append({"type": "text", "text": seg})
            if i < len(image_order):
                idx = image_order[i] - 1
                if 0 <= idx < len(images_b64):
                    b64 = images_b64[idx]
                    content.append({
                        "type": "image_url",
                        "image_url": {"url": f"data:image/jpeg;base64,{b64}", "detail": "high"},
                    })

    return [{"role": "user", "content": content}]


def build_openai_messages_responses(row: pd.Series, dataset: str, instruction: str) -> List[Dict]:
    """Build OpenAI Responses API format messages with base64 images inlined."""
    content: List[Dict[str, Any]] = []

    if dataset == "MMMU_Pro_V":
        b64 = row["image"]
        content.append({
            "type": "input_image",
            "image_url": f"data:image/jpeg;base64,{b64}",
            "detail": "high",
        })
        content.append({"type": "input_text", "text": instruction})
    else:
        prompt_text = build_standard_prompt(row, instruction)
        images_b64 = ast.literal_eval(row["image"])
        text_segments, image_order = replace_image_tokens_full_prompt(prompt_text)

        for i, seg in enumerate(text_segments):
            if seg.strip():
                content.append({"type": "input_text", "text": seg})
            if i < len(image_order):
                idx = image_order[i] - 1
                if 0 <= idx < len(images_b64):
                    b64 = images_b64[idx]
                    content.append({
                        "type": "input_image",
                        "image_url": f"data:image/jpeg;base64,{b64}",
                        "detail": "high",
                    })

    return [{"role": "user", "content": content}]


# ---------------------------------------------------------------------------
# HTTP API call
# ---------------------------------------------------------------------------


def _build_payload_responses(model_cfg, messages) -> dict:
    """Build request payload for the OpenAI Responses API."""
    payload = {"model": model_cfg.model, "input": messages}
    payload["max_output_tokens"] = model_cfg.max_tokens

    if getattr(model_cfg, "system_prompt", None):
        payload["instructions"] = model_cfg.system_prompt

    if model_cfg.is_reasoning_model():
        payload["reasoning"] = {
            "effort": getattr(model_cfg, "reasoning_effort", "medium"),
            "summary": "auto",
        }
    else:
        payload["temperature"] = model_cfg.temperature

    return payload


def _parse_responses_api(resp: dict, fail_msg: str) -> str:
    """Parse a Responses API JSON response, returning answer text."""
    output_items = resp.get("output", [])
    reasoning_parts = []
    content_parts = []
    for item in output_items:
        if item.get("type") == "reasoning":
            for s in item.get("summary", []):
                if s.get("type") == "summary_text":
                    reasoning_parts.append(s.get("text", ""))
        elif item.get("type") == "message":
            for c in item.get("content", []):
                if c.get("type") == "output_text":
                    content_parts.append(c.get("text", ""))
    reasoning_text = "".join(reasoning_parts)
    answer = "".join(content_parts).strip() if content_parts else fail_msg
    if reasoning_text:
        answer = reasoning_text + "</think>" + answer
    return answer


def call_api(model_cfg, messages, retry=10) -> str:
    """Send HTTP request to OpenAI-compatible API.

    model_cfg: an instantiated supported_VLM model object. Only its config
    attributes are used: api_base, key, model, temperature, max_tokens,
    timeout, is_max_completion_tokens, is_reasoning_model(), etc.
    """
    # Build headers (from gpt.py generate_inner)
    if getattr(model_cfg, 'use_azure', False):
        headers = {'Content-Type': 'application/json', 'api-key': model_cfg.key}
    elif 'internvl2-pro' in model_cfg.model:
        headers = {'Content-Type': 'application/json', 'Authorization': model_cfg.key}
    else:
        headers = {'Content-Type': 'application/json', 'Authorization': f'Bearer {model_cfg.key}'}
    if hasattr(model_cfg, 'baidu_appid'):
        headers['appid'] = model_cfg.baidu_appid

    # Build payload
    use_responses = getattr(model_cfg, 'use_responses_api', False)

    if use_responses:
        payload = _build_payload_responses(model_cfg, messages)
    else:
        payload = dict(
            model=model_cfg.model,
            messages=messages,
            n=1,
            temperature=model_cfg.temperature,
        )

        # Merge default_kwargs (e.g. enable_thinking, thinking_budget) into payload
        default_kw = getattr(model_cfg, "default_kwargs", {})
        if default_kw:
            payload.update({k: v for k, v in default_kw.items() if k != "extra_body"})

        if model_cfg.is_reasoning_model():
            payload['reasoning'] = {'enabled': True}

        if model_cfg.is_max_completion_tokens:
            payload['max_completion_tokens'] = model_cfg.max_tokens
        else:
            payload['max_tokens'] = model_cfg.max_tokens

    # Proxy settings
    proxies = {}
    if os.getenv('http_proxy'):
        proxies['http'] = os.getenv('http_proxy')
    if os.getenv('https_proxy'):
        proxies['https'] = os.getenv('https_proxy')
    proxies = proxies or None

    # Retry loop
    for attempt in range(retry):
        try:
            response = requests.post(
                model_cfg.api_base,
                headers=headers,
                data=json.dumps(payload),
                proxies=proxies,
                timeout=model_cfg.timeout * 1.1,
            )

            if 200 <= response.status_code < 300:
                resp = response.json()
                if use_responses:
                    answer = _parse_responses_api(resp, FAIL_MSG)
                else:
                    msg = resp['choices'][0]['message']
                    reasoning = (msg.get('reasoning') or
                                 msg.get('reasoning_content') or '')
                    answer = (msg.get('content') or '').strip()
                    if reasoning and answer:
                        answer = reasoning + '</think>' + answer
                if answer and FAIL_MSG not in answer:
                    return answer
            else:
                print(f"Attempt {attempt + 1}/{retry}: HTTP {response.status_code} - {response.text[:200]}")
        except Exception as e:
            print(f"Attempt {attempt + 1}/{retry} failed: {e}")

        time.sleep(random.random() * 2)

    return FAIL_MSG


# ---------------------------------------------------------------------------
# Inference main flow
# ---------------------------------------------------------------------------


def run_inference(
    args: argparse.Namespace,
    df: pd.DataFrame,
    prompts: Dict[str, str],
) -> str:
    """Run API inference over *df* and return the prediction TSV path.

    Sets args.max_tokens_resolved (from the model config) so downstream eval
    can locate the run directory.
    """
    from vlmeval.config import supported_VLM
    from vlmeval.smp.file import load, dump
    from vlmeval.utils.mp_util import track_progress_rich

    is_vision = args.dataset == "MMMU_Pro_V"
    instruction = prompts["vision"] if is_vision else prompts["standard"]

    # Instantiate model (only used to read config attributes)
    if args.model_name not in supported_VLM:
        raise ValueError(
            f"Model '{args.model_name}' not found in supported_VLM. "
            f"Available: {list(supported_VLM.keys())[:20]}..."
        )
    model_cfg = supported_VLM[args.model_name]()
    use_openai = _is_openai_compatible(model_cfg)
    print(f"Model config: model={model_cfg.model}, backend={'openai' if use_openai else 'native'}")

    # Build messages for all samples
    all_messages = []
    indices = []
    for i, row in df.iterrows():
        if use_openai:
            if getattr(model_cfg, 'use_responses_api', False):
                msgs = build_openai_messages_responses(row, args.dataset, instruction)
            else:
                msgs = build_openai_messages(row, args.dataset, instruction)
        else:
            msgs = build_vlmeval_messages(row, args.dataset, instruction)
        all_messages.append(msgs)
        indices.append(row.get("index", i))

    # Resolve max_tokens for output path computation
    args.max_tokens_resolved = getattr(model_cfg, "max_tokens", "default")

    run_dir = resolve_run_dir(
        args.work_dir,
        args.dataset,
        args.model_name,
        getattr(args, "prompt_version", "direct"),
        args.max_tokens_resolved,
        subset=getattr(args, "subset", "full"),
        run_tag=getattr(args, "run_tag", ""),
    )
    pred_path = args.output or os.path.join(run_dir, "pred.tsv")
    checkpoint_path = os.path.join(run_dir, "supp.pkl")

    res = load(checkpoint_path) if os.path.exists(checkpoint_path) else {}

    # Filter out failed responses so they will be retried
    failed_keys = [k for k, v in res.items() if isinstance(v, str) and FAIL_MSG in v]
    if failed_keys:
        print(f"Found {len(failed_keys)} failed responses in checkpoint, will retry them")
        for k in failed_keys:
            del res[k]
        dump(res, checkpoint_path)

    pending = [(msgs, idx) for msgs, idx in zip(all_messages, indices) if idx not in res]
    print(f"Pending: {len(pending)}/{len(indices)} (checkpoint: {len(res)} done)")

    if pending:
        pending_keys = [idx for _, idx in pending]
        if use_openai:
            tasks = [dict(model_cfg=model_cfg, messages=msgs) for msgs, _ in pending]
            track_progress_rich(
                call_api, tasks, nproc=args.nproc,
                save=checkpoint_path, keys=pending_keys,
            )
        else:
            tasks = [dict(model_cfg=model_cfg, vlmeval_messages=msgs) for msgs, _ in pending]
            track_progress_rich(
                call_model_generate, tasks, nproc=args.nproc,
                save=checkpoint_path, keys=pending_keys,
            )

    # Assemble final TSV
    res = load(checkpoint_path) if os.path.exists(checkpoint_path) else {}
    out_df = df.copy()
    out_df["prediction"] = [str(res.get(idx, "")) for idx in indices]
    write_table(out_df, pred_path)
    print(f"Saved predictions to {pred_path}")
    return pred_path
