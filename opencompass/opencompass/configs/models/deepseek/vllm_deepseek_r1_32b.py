import os

from opencompass.models import VLLMwithChatTemplate

# Model root: `export OC_MODEL_ROOT=/path/to/models` points to local weights;
# when unset, the path falls back to the HuggingFace repo id.
models = [
    dict(
        type=VLLMwithChatTemplate,
        abbr='deepseek-r1-32b-vllm',
        path=os.path.join(os.environ.get('OC_MODEL_ROOT', 'deepseek-ai'),
                          'DeepSeek-R1-Distill-Qwen-32B'),
        max_out_len=32768,
        batch_size=16,
        model_kwargs=dict(tensor_parallel_size=1),
        run_cfg=dict(num_gpus=1),
        generation_kwargs=dict(temperature=0.6,
                               top_p=0.95,
                            #    top_k=20,
                               ),
    )
]