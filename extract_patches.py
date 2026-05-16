"""extract_patches.py

Patch extraction script.
"""

import argparse
import glob
import os
import tqdm
import pathlib

import numpy as np

from misc.patch_extractor import PatchExtractor
from misc.utils import rm_n_mkdir

from dataset import get_dataset


def build_args():
    parser = argparse.ArgumentParser(description="Extract training/validation patches.")
    parser.add_argument("--dataset_name", default="nucls", help="Dataset key from dataset.py (e.g. nucls, fs, kumar)")
    parser.add_argument("--save_root", default="dataset/training_data", help="Root directory for extracted .npy patches")
    parser.add_argument("--train_img_dir", required=True, help="Training images directory")
    parser.add_argument("--train_ann_dir", required=True, help="Training annotations directory")
    parser.add_argument("--valid_img_dir", required=True, help="Validation images directory")
    parser.add_argument("--valid_ann_dir", required=True, help="Validation annotations directory")
    parser.add_argument("--img_ext", default=".png", help="Image extension (e.g. .png, .jpg, .tif)")
    parser.add_argument("--ann_ext", default=".mat", help="Annotation extension")
    parser.add_argument("--win_size", type=int, default=256, help="Patch extraction window size")
    parser.add_argument("--step_size", type=int, default=250, help="Patch extraction step size")
    parser.add_argument(
        "--extract_type",
        choices=["mirror", "valid"],
        default="mirror",
        help="mirror: pad borders, valid: only valid crop regions",
    )
    parser.add_argument(
        "--no_type_classification",
        action="store_true",
        help="Disable type-map extraction for datasets without type labels",
    )
    return parser.parse_args()


def align_image_and_annotation(img, ann):
    if img.shape[0] == ann.shape[0] and img.shape[1] == ann.shape[1]:
        return img, ann

    h = min(img.shape[0], ann.shape[0])
    w = min(img.shape[1], ann.shape[1])
    img = img[:h, :w]
    ann = ann[:h, :w]
    return img, ann


def extract_split(split_name, split_desc, cfg, dataset_parser, xtractor):
    img_ext, img_dir = split_desc["img"]
    ann_ext, ann_dir = split_desc["ann"]
    output_tag = "%dx%d_%dx%d" % (cfg.win_size, cfg.win_size, cfg.step_size, cfg.step_size)
    out_dir = os.path.join(cfg.save_root, cfg.dataset_name, cfg.dataset_name, split_name, output_tag)

    file_list = glob.glob(os.path.join(glob.escape(ann_dir), "*" + ann_ext))
    file_list.sort()
    if len(file_list) == 0:
        raise RuntimeError("No annotation files detected in %s with extension %s" % (ann_dir, ann_ext))

    rm_n_mkdir(out_dir)
    print("Extracting %s split into %s" % (split_name, out_dir))

    pbar_format = "Process File: |{bar}| {n_fmt}/{total_fmt}[{elapsed}<{remaining},{rate_fmt}]"
    pbarx = tqdm.tqdm(total=len(file_list), bar_format=pbar_format, ascii=True, position=0)

    for file_path in file_list:
        base_name = pathlib.Path(file_path).stem
        img_path = "%s/%s%s" % (img_dir, base_name, img_ext)
        ann_path = "%s/%s%s" % (ann_dir, base_name, ann_ext)

        img = dataset_parser.load_img(img_path)
        ann = dataset_parser.load_ann(ann_path, not cfg.no_type_classification)
        img, ann = align_image_and_annotation(img, ann)
        img = np.concatenate([img, ann], axis=-1)
        sub_patches = xtractor.extract(img, cfg.extract_type)

        pbar = tqdm.tqdm(
            total=len(sub_patches),
            leave=False,
            bar_format="Extracting  : |{bar}| {n_fmt}/{total_fmt}[{elapsed}<{remaining},{rate_fmt}]",
            ascii=True,
            position=1,
        )
        for idx, patch in enumerate(sub_patches):
            np.save("{0}/{1}_{2:03d}.npy".format(out_dir, base_name, idx), patch)
            pbar.update()
        pbar.close()
        pbarx.update()
    pbarx.close()


# -------------------------------------------------------------------------------------
if __name__ == "__main__":
    args = build_args()
    win_size = [args.win_size, args.win_size]
    step_size = [args.step_size, args.step_size]

    dataset_info = {
        "train": {"img": (args.img_ext, args.train_img_dir), "ann": (args.ann_ext, args.train_ann_dir)},
        "valid": {"img": (args.img_ext, args.valid_img_dir), "ann": (args.ann_ext, args.valid_ann_dir)},
    }
    dataset_parser = get_dataset(args.dataset_name)
    xtractor = PatchExtractor(win_size, step_size)

    for split_name, split_desc in dataset_info.items():
        extract_split(split_name, split_desc, args, dataset_parser, xtractor)
