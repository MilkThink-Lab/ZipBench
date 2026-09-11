"""Build a real ``SciCode_datasets_with_background.json`` for OpenCompass.

The ``SciCode_datasets_with_background.json`` shipped in the OpenCompass data
pack (``scicode.zip``) is byte-identical to ``SciCode_datasets.json``: it does
not contain any background text, so the ``with_bg=True`` configs silently
evaluate the *without*-background protocol. This script rebuilds the file from
the official SciCode release (Hugging Face ``SciCode1/SciCode``, test split),
mirroring the official ``with_background`` prompting of
``scicode-bench/SciCode/eval/scripts/gencode.py``:

* every generated step is presented as ``step_description_prompt`` followed by
  its ``step_background`` (verbatim; ~80 of the 291 steps have no background
  in the official data and are left as is);
* the model is no longer asked to write the background itself, because it is
  given (the official ``multistep_template.txt`` semantics).

Everything else (problem order, step split, pre-provided steps, dependencies,
tests, prompt scaffolding) is taken from ``SciCode_datasets.json`` unchanged,
so the ZipBench sub-step ids and the subset fingerprint are unaffected.

Usage (from the ``opencompass/`` folder, after unzipping ``scicode.zip`` into
``data/scicode/``)::

    python zipbench/make_scicode_with_background.py

By default ``problems_test.jsonl`` is downloaded from the Hugging Face
dataset repo; pass ``--hf-jsonl`` to use a local copy instead. An existing
target file is only overwritten when it is a copy of the without-background
file or ``--force`` is given.
"""
import argparse
import filecmp
import json
import os.path as osp

_REPO_ROOT = osp.dirname(osp.dirname(osp.abspath(__file__)))
_DEFAULT_DIR = osp.join(_REPO_ROOT, 'data', 'scicode')
_HF_DATASET = 'SciCode1/SciCode'
_HF_FILE = 'problems_test.jsonl'

# Steps whose solution OpenCompass provides inside the previous prompt
# (mirrors ``opencompass.datasets.scicode.SCICODE_PREPROVIDED_CODE``); they
# have no prompt of their own, so they are skipped when aligning HF steps to
# OpenCompass prompts. (problem_id, 0-based step index).
_PREPROVIDED = {('13', 5), ('62', 0), ('76', 2)}

_GUIDELINES_WITHOUT_BG = (
    'Response Guidelines:\n'
    '1. Start with a comment summarizing the scientific background required '
    'for the subproblem.\n'
    '2. Write the complete and executable Python program for this subproblem '
    'in a single block.\n'
    '3. DO NOT include previous function code, example usage or test code in '
    'your response.\n'
    '4. Ensure your response is formatted as follows:\n'
    '# Background: [Here, insert the necessary scientific knowledge required '
    'for the next step.]\n'
    '```python\n'
    '[Insert the Python code here based on the provided function header and '
    'dependencies.]\n'
    '```')
_GUIDELINES_WITH_BG = (
    'Response Guidelines:\n'
    '1. The scientific background required for the subproblem is given above; '
    'use it.\n'
    '2. Write the complete and executable Python program for this subproblem '
    'in a single block.\n'
    '3. DO NOT include previous function code, example usage or test code in '
    'your response.\n'
    '4. Ensure your response is formatted as follows:\n'
    '```python\n'
    '[Insert the Python code here based on the provided function header and '
    'dependencies.]\n'
    '```')

_INTRO_WITHOUT_BG = (
    'Your task is to develop a Python solution for each step based on the '
    'given descriptions.')
_INTRO_WITH_BG = (
    'Your task is to develop a Python solution for each step based on the '
    'given descriptions and the background knowledge provided with them.')


def load_official(hf_jsonl=None):
    """Return the official test problems as a list of dicts."""
    if not hf_jsonl:
        from huggingface_hub import hf_hub_download
        hf_jsonl = hf_hub_download(_HF_DATASET, _HF_FILE,
                                   repo_type='dataset')
    with open(hf_jsonl, encoding='utf-8') as f:
        return [json.loads(line) for line in f if line.strip()]


def split_steps(problem):
    """Return ``(generated_steps, preprovided_steps)`` of an official problem."""
    pid = problem['problem_id']
    gen, pre = [], []
    for i, step in enumerate(problem['sub_steps']):
        (pre if (pid, i) in _PREPROVIDED else gen).append(step)
    return gen, pre


def _insert_background(prompt, step, required):
    desc = step['step_description_prompt'].strip()
    bg = step['step_background'].strip()
    anchor = f'Subproblem Description:\n{desc}'
    n = prompt.count(anchor)
    if n != 1 and (required or n > 1):
        raise ValueError(f"step {step['step_number']}: description anchor "
                         f'found {n} times')
    if n == 1 and bg:
        # The description line may carry trailing spaces; insert after it.
        end = prompt.index('\n', prompt.index(anchor) + len(anchor)) + 1
        prompt = prompt[:end] + bg + '\n' + prompt[end:]
    return prompt


def add_background(prompt, step, preprovided):
    """Insert ``step_background`` after each step description in a prompt.

    ``step`` is the step this prompt asks the model to solve. A prompt can
    also embed a pre-provided step (announced with its standard answer), so
    those are patched too when their description is present. The response
    guidelines (repeated once per embedded step) stop asking the model to
    write the background itself.
    """
    prompt = _insert_background(prompt, step, required=True)
    for pre in preprovided:
        prompt = _insert_background(prompt, pre, required=False)
    if _GUIDELINES_WITHOUT_BG not in prompt:
        raise ValueError(f"step {step['step_number']}: response guidelines "
                         'block not found')
    prompt = prompt.replace(_GUIDELINES_WITHOUT_BG, _GUIDELINES_WITH_BG)
    prompt = prompt.replace(_INTRO_WITHOUT_BG, _INTRO_WITH_BG, 1)
    return prompt


def build(without_bg, official):
    by_id = {p['problem_id']: p for p in official}
    if set(by_id) != {p['id'] for p in without_bg}:
        raise ValueError('problem ids differ between OpenCompass and official '
                         'SciCode data')
    out, n_bg = [], 0
    for prob in without_bg:
        steps, pre = split_steps(by_id[prob['id']])
        if len(steps) != len(prob['prompt']):
            raise ValueError(f"problem {prob['id']}: {len(prob['prompt'])} "
                             f'prompts vs {len(steps)} official steps')
        new = dict(prob)
        new['prompt'] = [
            add_background(p, s, pre) for p, s in zip(prob['prompt'], steps)
        ]
        n_bg += sum(1 for s in steps if s['step_background'].strip())
        out.append(new)
    return out, n_bg


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument('--scicode-dir', default=_DEFAULT_DIR,
                        help='folder holding SciCode_datasets.json '
                        '(default: data/scicode)')
    parser.add_argument('--hf-jsonl', default=None,
                        help=f'local copy of {_HF_DATASET}/{_HF_FILE} '
                        'instead of downloading it')
    parser.add_argument('--output', default=None,
                        help='output path (default: <scicode-dir>/'
                        'SciCode_datasets_with_background.json)')
    parser.add_argument('--force', action='store_true',
                        help='overwrite an existing, non-duplicate target')
    args = parser.parse_args()

    src = osp.join(args.scicode_dir, 'SciCode_datasets.json')
    dst = args.output or osp.join(args.scicode_dir,
                                  'SciCode_datasets_with_background.json')
    if osp.exists(dst) and not args.force and not filecmp.cmp(
            src, dst, shallow=False):
        raise SystemExit(f'{dst} exists and differs from {src}; '
                         'pass --force to overwrite it')

    with open(src, encoding='utf-8') as f:
        without_bg = json.load(f)
    official = load_official(args.hf_jsonl)
    with_bg, n_bg = build(without_bg, official)

    with open(dst, 'w', encoding='utf-8') as f:
        json.dump(with_bg, f, ensure_ascii=False, indent=2)
    n_steps = sum(len(p['prompt']) for p in with_bg)
    print(f'wrote {dst}: {len(with_bg)} problems, {n_steps} prompts, '
          f'{n_bg} with background text')


if __name__ == '__main__':
    main()
