"""export_nucls_hf.py

Export Hugging Face NuCLS dataset to CGT training layout:

data/nucls/
  train/images/*.png
  train/labels_mat/*.mat
  valid/images/*.png
  valid/labels_mat/*.mat

Each .mat contains:
  - inst_map
  - type_map
"""

import argparse
import os
import pathlib

import cv2
import numpy as np
import scipy.io as sio
from datasets import load_dataset


def parse_args():
    parser = argparse.ArgumentParser(description="Export HF NuCLS dataset for CGT")
    parser.add_argument(
        "--dataset_id",
        default="minhanhto09/NuCLS_dataset",
        help="Hugging Face dataset id",
    )
    parser.add_argument(
        "--config_name",
        default="default",
        help="Dataset config name",
    )
    parser.add_argument(
        "--fold",
        type=int,
        default=1,
        help="Fold index (e.g., 1..5 or 999). train_fold_{fold}/test_fold_{fold}",
    )
    parser.add_argument(
        "--output_root",
        default="data/nucls",
        help="Output root directory",
    )
    return parser.parse_args()


def remap_contiguous(inst_map):
    inst_map = inst_map.astype(np.int32)
    uniq_ids = np.unique(inst_map)
    uniq_ids = [v for v in uniq_ids.tolist() if v != 0]
    new_map = np.zeros_like(inst_map, dtype=np.int32)
    for new_id, old_id in enumerate(uniq_ids, start=1):
        new_map[inst_map == old_id] = new_id
    return new_map


def decode_instance_and_type(mask_image):
    """Decode NuCLS mask encoding described in HF card.

    - channel 0: class id
    - channels 1/2: unique instance encoding
    """
    mask = np.array(mask_image)
    if len(mask.shape) != 3 or mask.shape[2] < 3:
        raise RuntimeError("Expected mask_image with 3 channels")

    class_ch = mask[:, :, 0].astype(np.int32)
    inst_raw = (mask[:, :, 1].astype(np.int32) << 8) + mask[:, :, 2].astype(np.int32)
    inst_map = remap_contiguous(inst_raw)
    type_map = np.zeros_like(inst_map, dtype=np.int32)

    for inst_id in np.unique(inst_map):
        if inst_id == 0:
            continue
        values = class_ch[inst_map == inst_id]
        if values.size == 0:
            continue
        uniq, counts = np.unique(values, return_counts=True)
        dominant = int(uniq[np.argmax(counts)])
        type_map[inst_map == inst_id] = dominant

    return inst_map, type_map


def ensure_dirs(root, split_name):
    image_dir = os.path.join(root, split_name, "images")
    label_dir = os.path.join(root, split_name, "labels_mat")
    os.makedirs(image_dir, exist_ok=True)
    os.makedirs(label_dir, exist_ok=True)
    return image_dir, label_dir


def sanitize_stem(file_name, fallback):
    if file_name is None:
        return fallback
    stem = pathlib.Path(str(file_name)).stem
    return stem if stem != "" else fallback


def export_split(dataset_split, out_root, split_name):
    image_dir, label_dir = ensure_dirs(out_root, split_name)
    count = 0

    for idx, sample in enumerate(dataset_split):
        stem = sanitize_stem(sample.get("file_name"), "sample_%06d" % idx)
        rgb = np.array(sample["rgb_image"])
        if len(rgb.shape) != 3 or rgb.shape[2] < 3:
            continue

        inst_map, type_map = decode_instance_and_type(sample["mask_image"])
        save_img = os.path.join(image_dir, stem + ".png")
        save_mat = os.path.join(label_dir, stem + ".mat")

        cv2.imwrite(save_img, cv2.cvtColor(rgb[:, :, :3], cv2.COLOR_RGB2BGR))
        sio.savemat(
            save_mat,
            {
                "inst_map": inst_map.astype(np.int32),
                "type_map": type_map.astype(np.int32),
            },
        )
        count += 1

    return count


def main():
    args = parse_args()
    train_split = "train_fold_%d" % args.fold
    valid_split = "test_fold_%d" % args.fold

    ds = load_dataset(args.dataset_id, name=args.config_name)
    if train_split not in ds or valid_split not in ds:
        raise RuntimeError("Could not find splits %s and %s" % (train_split, valid_split))

    train_count = export_split(ds[train_split], args.output_root, "train")
    valid_count = export_split(ds[valid_split], args.output_root, "valid")
    print("Exported %d train and %d valid samples into %s" % (train_count, valid_count, args.output_root))


if __name__ == "__main__":
    main()
