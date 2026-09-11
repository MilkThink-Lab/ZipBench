import os

from opencompass.models import VLLMwithChatTemplate

# Model root: `export OC_MODEL_ROOT=/path/to/models` points to local weights;
# when unset, the path falls back to the HuggingFace repo id.
models = [
    dict(
        type=VLLMwithChatTemplate,
        abbr='qwen2.5-3b-instruct-vllm',
        path=os.path.join(os.environ.get('OC_MODEL_ROOT', 'Qwen'),
                          'Qwen2.5-3B-Instruct'),
        model_kwargs=dict(tensor_parallel_size=1, gpu_memory_utilization=0.9, max_model_len=131072),
        max_out_len=4096,
        batch_size=16,
        generation_kwargs=dict(temperature=0),
        run_cfg=dict(num_gpus=1),
    )
]
