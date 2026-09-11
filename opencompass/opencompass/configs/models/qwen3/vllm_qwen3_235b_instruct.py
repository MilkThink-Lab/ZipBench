import os

from opencompass.models import VLLMwithChatTemplate

# Model root: `export OC_MODEL_ROOT=/path/to/models` points to local weights;
# when unset, the path falls back to the HuggingFace repo id.
models = [
    dict(
        type=VLLMwithChatTemplate,
        abbr='qwen3-235b-a22b-instruct-vllm',
        path=os.path.join(os.environ.get('OC_MODEL_ROOT', 'Qwen'),
                          'Qwen3-235B-A22B-Instruct-2507-FP8'),
        model_kwargs=dict(tensor_parallel_size=2, gpu_memory_utilization=0.95,max_model_len=62914),
        max_out_len=4096,
        batch_size=16,
        generation_kwargs=dict(temperature=0,
                               top_p=0.8,
                               top_k=20),
        run_cfg=dict(num_gpus=2),
    )
]
