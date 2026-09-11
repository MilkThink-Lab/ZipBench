"""Pre-dump MMMU-Pro images from TSV (base64) to disk.

This avoids disk IO overhead during evaluation by decoding all images upfront.
Handles both MMMU_Pro_10c and MMMU_Pro_V datasets.
"""
import os
import sys
import os.path as osp
import time

# Add VLMEvalKit to path
VLMEVAL_ROOT = osp.dirname(osp.abspath(__file__))
sys.path.insert(0, VLMEVAL_ROOT)

from vlmeval.smp import load, decode_base64_to_image_file, toliststr, read_ok
import pandas as pd
import numpy as np


def has_image_path(row):
    """Check if a row has a valid image_path field."""
    if 'image_path' not in row.index:
        return False
    val = row['image_path']
    if isinstance(val, (list, np.ndarray)):
        return len(val) > 0
    if isinstance(val, str):
        return len(val) > 0
    try:
        return not pd.isna(val)
    except (ValueError, TypeError):
        return val is not None


def preload_dataset_images(dataset_name, tsv_path, img_root):
    """Decode all base64 images from TSV and save to disk."""
    print(f"\n{'='*60}")
    print(f"Processing: {dataset_name}")
    print(f"TSV: {tsv_path}")
    print(f"Image dir: {img_root}")
    print(f"{'='*60}")

    os.makedirs(img_root, exist_ok=True)

    print("Loading TSV data...")
    t0 = time.time()
    data = load(tsv_path)
    print(f"Loaded {len(data)} rows in {time.time() - t0:.1f}s")

    if 'image' not in data.columns:
        print("No 'image' column found, skipping.")
        return

    # Build image_map: same logic as ImageBaseDataset.__init__
    data['index'] = [str(x) for x in data['index']]
    data['image'] = [str(x) for x in data['image']]
    image_map = {x: y for x, y in zip(data['index'], data['image'])}
    for k in image_map:
        if len(image_map[k]) <= 64:
            idx = image_map[k]
            assert idx in image_map and len(image_map[idx]) > 64
            image_map[k] = image_map[idx]

    # Expand images to list form
    images = [toliststr(image_map[k]) for k in data['index']]
    data['image'] = [x[0] if len(x) == 1 else x for x in images]

    # Resolve image_path
    if 'image_path' in data.columns:
        paths = [toliststr(x) for x in data['image_path']]
        data['image_path'] = [x[0] if len(x) == 1 else x for x in paths]

    total = len(data)
    skipped = 0
    dumped = 0
    errors = 0
    t_start = time.time()

    for i in range(total):
        line = dict(data.iloc[i])

        # Replicate dump_image logic exactly
        if isinstance(line['image'], list):
            img_list = line['image']
            if has_image_path(data.iloc[i]):
                image_path = line['image_path'] if isinstance(line['image_path'], list) else [line['image_path']]
            else:
                image_path = [f"{line['index']}_{j}.png" for j in range(len(img_list))]
            for img, im_name in zip(img_list, image_path):
                path = osp.join(img_root, im_name)
                if read_ok(path):
                    skipped += 1
                else:
                    try:
                        decode_base64_to_image_file(img, path)
                        dumped += 1
                    except Exception as e:
                        errors += 1
                        print(f"  Error on index {line['index']}, img {im_name}: {e}")
        elif isinstance(line['image'], str) and has_image_path(data.iloc[i]) and isinstance(line['image_path'], str):
            path = osp.join(img_root, line['image_path'])
            if read_ok(path):
                skipped += 1
            else:
                try:
                    os.makedirs(osp.dirname(path), exist_ok=True) if osp.dirname(path) != img_root else None
                    decode_base64_to_image_file(line['image'], path)
                    dumped += 1
                except Exception as e:
                    errors += 1
                    print(f"  Error on index {line['index']}: {e}")
        else:
            path = osp.join(img_root, f"{line['index']}.jpg")
            if read_ok(path):
                skipped += 1
            else:
                try:
                    decode_base64_to_image_file(line['image'], path)
                    dumped += 1
                except Exception as e:
                    errors += 1
                    print(f"  Error on index {line['index']}: {e}")

        if (i + 1) % 200 == 0 or (i + 1) == total:
            elapsed = time.time() - t_start
            rate = (i + 1) / elapsed if elapsed > 0 else 0
            print(f"  [{i+1}/{total}] dumped={dumped}, skipped={skipped}, errors={errors}, {rate:.1f} img/s")

    elapsed = time.time() - t_start
    print(f"\nDone: {dumped} dumped, {skipped} skipped, {errors} errors in {elapsed:.1f}s")


def main():
    lmu_root = os.environ.get("LMUData", osp.join(osp.expanduser("~"), "LMUData"))
    img_base = osp.join(lmu_root, "images")

    datasets = [
        ("MMMU_Pro_10c", osp.join(lmu_root, "MMMU_Pro_10c.tsv"), osp.join(img_base, "MMMU_Pro_10c")),
        ("MMMU_Pro_V", osp.join(lmu_root, "MMMU_Pro_V.tsv"), osp.join(img_base, "MMMU_Pro_V")),
    ]

    for name, tsv, img_dir in datasets:
        if not osp.exists(tsv):
            print(f"WARNING: {tsv} not found, skipping {name}")
            continue
        preload_dataset_images(name, tsv, img_dir)

    print("\n" + "="*60)
    print("All MMMU-Pro images pre-dumped successfully!")
    print("="*60)


if __name__ == "__main__":
    main()
