"""Derive the ArenaHard subset baseline-answer files from the full one.

``SubjectiveEvalTask`` loads the gpt4-0314 baseline answers for a dataset from
``<given_pred path>/<dataset_abbr>.json`` and joins them to the test rows BY
POSITION, so an ArenaHard subset run needs its own baseline file whose entries
are the subset spec's rows re-keyed ``'0'..'n-1'`` in spec order. This script
builds ``arenahard_small.json`` / ``arenahard_tiny.json`` next to the full
``arenahard.json`` (position-keyed ``'0'..'N-1'``; part of the standard
ArenaHard data download).

Usage (from the repo root, after downloading the ArenaHard subjective data)::

    python zipbench/make_arenahard_subset_baselines.py

Re-run it whenever the subset specs change.
"""
import argparse
import json
import os.path as osp

_REPO_ROOT = osp.dirname(osp.dirname(osp.abspath(__file__)))
_DEFAULT_FULL = osp.join(_REPO_ROOT, 'data', 'subjective', 'arena_hard',
                         'arenahard.json')
_SUBSETS_DIR = osp.join(osp.dirname(osp.abspath(__file__)), 'subsets',
                        'ArenaHard')


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument('--full', default=_DEFAULT_FULL,
                        help='full-set baseline json (position-keyed)')
    parser.add_argument('--subsets', nargs='*', default=['small', 'tiny'],
                        help='subset names to build (default: small tiny)')
    args = parser.parse_args()

    with open(args.full, encoding='utf-8') as f:
        full = json.load(f)

    for name in args.subsets:
        spec_path = osp.join(_SUBSETS_DIR, f'{name}.jsonl')
        with open(spec_path, encoding='utf-8') as f:
            rows = [json.loads(line) for line in f if line.strip()]
        subset = {}
        for i, row in enumerate(rows):
            key = str(row['index'])
            if key not in full:
                raise KeyError(
                    f'{spec_path}: index {key} not present in {args.full} '
                    f'(has {len(full)} entries)')
            subset[str(i)] = full[key]
        out_path = osp.join(osp.dirname(args.full), f'arenahard_{name}.json')
        with open(out_path, 'w', encoding='utf-8') as f:
            json.dump(subset, f, ensure_ascii=False, indent=4)
        print(f'wrote {out_path} ({len(subset)} entries)')


if __name__ == '__main__':
    main()
