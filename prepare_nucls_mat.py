"""prepare_nucls_mat.py

Convert NuCLS download layout (rgb/mask/csv) to .mat annotations expected by CGT.

Outputs, per image:
  - inst_map (required)
  - type_map (optional, when class information can be resolved from CSV)
"""

import argparse
import glob
import json
import os
import pathlib

import cv2
import numpy as np
import pandas as pd
import scipy.io as sio
from scipy import ndimage


def parse_args():
    parser = argparse.ArgumentParser(description="Convert NuCLS rgb/mask/csv to .mat labels")
    parser.add_argument("--rgb_dir", required=True, help="Path to rgb images")
    parser.add_argument("--mask_dir", required=True, help="Path to masks")
    parser.add_argument("--csv_dir", required=True, help="Path to per-image CSV metadata")
    parser.add_argument("--output_dir", required=True, help="Where .mat files are written")
    parser.add_argument(
        "--image_exts",
        default=".png,.jpg,.jpeg,.tif,.tiff",
        help="Comma-separated image extensions to scan in rgb_dir",
    )
    parser.add_argument(
        "--mask_exts",
        default=".png,.jpg,.jpeg,.tif,.tiff,.npy",
        help="Comma-separated mask extensions to scan in mask_dir",
    )
    parser.add_argument(
        "--csv_ext",
        default=".csv",
        help="CSV extension",
    )
    parser.add_argument(
        "--save_type_mapping",
        default="",
        help="Optional path to save discovered type-name to type-id mapping JSON",
    )
    return parser.parse_args()


def normalize_exts(raw):
    ext_list = [item.strip().lower() for item in raw.split(",")]
    ext_list = [item if item.startswith(".") else "." + item for item in ext_list if item != ""]
    return ext_list


def find_by_stem(directory, stem, exts):
    for ext in exts:
        candidate = os.path.join(directory, stem + ext)
        if os.path.exists(candidate):
            return candidate
    return None


def remap_contiguous(inst_map):
    inst_map = inst_map.astype(np.int32)
    uniq_ids = np.unique(inst_map)
    uniq_ids = [v for v in uniq_ids.tolist() if v != 0]
    new_map = np.zeros_like(inst_map, dtype=np.int32)
    for new_id, old_id in enumerate(uniq_ids, start=1):
        new_map[inst_map == old_id] = new_id
    return new_map


def mask_rgb_to_instance(mask_rgb):
    if len(mask_rgb.shape) == 2:
        return mask_rgb.astype(np.int32)

    if mask_rgb.shape[2] > 3:
        mask_rgb = mask_rgb[:, :, :3]
    mask_rgb = mask_rgb.astype(np.int32)
    packed = (
        (mask_rgb[:, :, 0] << 16)
        + (mask_rgb[:, :, 1] << 8)
        + mask_rgb[:, :, 2]
    )
    # treat black as background
    packed[packed == 0] = 0
    return remap_contiguous(packed)


def load_inst_map(mask_path):
    ext = pathlib.Path(mask_path).suffix.lower()
    if ext == ".npy":
        mask = np.load(mask_path)
    else:
        # unchanged retains channels; needed for color-coded masks
        mask = cv2.imread(mask_path, cv2.IMREAD_UNCHANGED)
        if mask is None:
            raise RuntimeError("Failed to read mask: %s" % mask_path)
        # OpenCV BGR -> RGB if 3-channel for stable color packing
        if len(mask.shape) == 3 and mask.shape[2] >= 3:
            mask = cv2.cvtColor(mask[:, :, :3], cv2.COLOR_BGR2RGB)

    inst_map = mask_rgb_to_instance(mask)
    if np.unique(inst_map).shape[0] <= 2:
        # likely binary mask, split connected components
        binary = np.array(inst_map > 0, dtype=np.int32)
        inst_map, _ = ndimage.label(binary)
        inst_map = inst_map.astype(np.int32)
    return remap_contiguous(inst_map)


def _pick_column(df, candidates):
    lowered = {col.lower(): col for col in df.columns}
    for name in candidates:
        if name in lowered:
            return lowered[name]
    return None


def resolve_type_map(csv_path, inst_map, type_name_to_id):
    if not os.path.exists(csv_path):
        return None

    df = pd.read_csv(csv_path)
    if df.shape[0] == 0:
        return None

    inst_id_col = _pick_column(
        df,
        [
            "instance_id",
            "inst_id",
            "mask_id",
            "nucleus_id",
            "cell_id",
            "object_id",
            "id",
        ],
    )
    type_col = _pick_column(
        df,
        [
            "type_id",
            "type",
            "class_id",
            "class",
            "classification",
            "nucleus_type",
            "main_classification",
            "label",
        ],
    )
    x_col = _pick_column(df, ["x", "x_centroid", "centroid_x", "center_x", "coord_x"])
    y_col = _pick_column(df, ["y", "y_centroid", "centroid_y", "center_y", "coord_y"])

    if type_col is None:
        return None

    type_map = np.zeros_like(inst_map, dtype=np.int32)
    inst_to_type = {}

    if inst_id_col is not None:
        for _, row in df.iterrows():
            inst_id = row[inst_id_col]
            type_value = row[type_col]
            if pd.isna(inst_id) or pd.isna(type_value):
                continue
            inst_id = int(inst_id)
            if isinstance(type_value, str):
                if type_value not in type_name_to_id:
                    type_name_to_id[type_value] = len(type_name_to_id) + 1
                mapped_type = type_name_to_id[type_value]
            else:
                mapped_type = int(type_value)
            if inst_id > 0:
                inst_to_type[inst_id] = mapped_type
    elif x_col is not None and y_col is not None:
        h, w = inst_map.shape[:2]
        for _, row in df.iterrows():
            xv = row[x_col]
            yv = row[y_col]
            type_value = row[type_col]
            if pd.isna(xv) or pd.isna(yv) or pd.isna(type_value):
                continue
            x = int(round(float(xv)))
            y = int(round(float(yv)))
            if x < 0 or y < 0 or x >= w or y >= h:
                continue
            inst_id = int(inst_map[y, x])
            if inst_id <= 0:
                continue
            if isinstance(type_value, str):
                if type_value not in type_name_to_id:
                    type_name_to_id[type_value] = len(type_name_to_id) + 1
                mapped_type = type_name_to_id[type_value]
            else:
                mapped_type = int(type_value)
            inst_to_type[inst_id] = mapped_type
    else:
        return None

    for inst_id, cls_id in inst_to_type.items():
        type_map[inst_map == inst_id] = int(cls_id)
    return type_map


def main():
    args = parse_args()
    image_exts = normalize_exts(args.image_exts)
    mask_exts = normalize_exts(args.mask_exts)
    csv_ext = args.csv_ext if args.csv_ext.startswith(".") else "." + args.csv_ext
    os.makedirs(args.output_dir, exist_ok=True)

    image_paths = []
    for ext in image_exts:
        image_paths.extend(glob.glob(os.path.join(args.rgb_dir, "*" + ext)))
    image_paths = sorted(set(image_paths))
    if len(image_paths) == 0:
        raise RuntimeError("No images found in %s" % args.rgb_dir)

    type_name_to_id = {}
    converted = 0
    with_type = 0

    for img_path in image_paths:
        stem = pathlib.Path(img_path).stem
        mask_path = find_by_stem(args.mask_dir, stem, mask_exts)
        if mask_path is None:
            continue
        csv_path = os.path.join(args.csv_dir, stem + csv_ext)

        inst_map = load_inst_map(mask_path)
        mat_dict = {"inst_map": inst_map.astype(np.int32)}

        type_map = resolve_type_map(csv_path, inst_map, type_name_to_id)
        if type_map is not None:
            mat_dict["type_map"] = type_map.astype(np.int32)
            with_type += 1

        out_path = os.path.join(args.output_dir, stem + ".mat")
        sio.savemat(out_path, mat_dict)
        converted += 1

    print("Converted %d samples to %s" % (converted, args.output_dir))
    print("Samples with type_map: %d" % with_type)

    if args.save_type_mapping != "":
        with open(args.save_type_mapping, "w") as f:
            json.dump(type_name_to_id, f, indent=2)
        print("Saved type mapping to %s" % args.save_type_mapping)


if __name__ == "__main__":
    main()
