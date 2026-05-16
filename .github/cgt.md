# CGT NuCLS experiment (Lightning AI Studio)

This file is the execution checklist for running CGT on **NuCLS** in Lightning AI Studio.

## 1) Environment setup

```bash
conda create -n cgt python=3.8 -y
conda activate cgt
pip install -r requirements.txt
pip install torch==1.13.0+cu116 torchvision==0.14.0+cu116 torchaudio==0.13.0 --extra-index-url https://download.pytorch.org/whl/cu116
pip install torch-geometric torch-scatter torch-sparse
pip install datasets
```

If your Lightning runtime uses a different CUDA/PyTorch stack, install matching Torch wheels first, then install the remaining dependencies.

## 2) Input options

### Option A: Hugging Face dataset (your link)

Use this to export fold-based train/valid directly to CGT format:

```bash
python export_nucls_hf.py \
  --dataset_id minhanhto09/NuCLS_dataset \
  --config_name default \
  --fold 1 \
  --output_root data/nucls
```

This creates:

```text
data/nucls/
  train/images/*.png
  train/labels_mat/*.mat
  valid/images/*.png
  valid/labels_mat/*.mat
```

### Option B: Manual download layout (`rgb/mask/csv`)

Your screenshot matches this layout:

```text
nucls_raw/
  rgb/
  mask/
  csv/
  visualization/
```

CGT cannot train directly on this layout; first convert to `.mat` labels:

```bash
python prepare_nucls_mat.py \
  --rgb_dir nucls_raw/rgb \
  --mask_dir nucls_raw/mask \
  --csv_dir nucls_raw/csv \
  --output_dir data/nucls/all/labels_mat \
  --save_type_mapping data/nucls/type_mapping.json
```

Then split images + labels into train/valid (same basename on both sides):

```text
data/
  nucls/
    train/
      images/
      labels_mat/
    valid/
      images/
      labels_mat/
```

## 3) Patch extraction for training

```bash
python extract_patches.py \
  --dataset_name nucls \
  --save_root dataset/training_data \
  --train_img_dir data/nucls/train/images \
  --train_ann_dir data/nucls/train/labels_mat \
  --valid_img_dir data/nucls/valid/images \
  --valid_ann_dir data/nucls/valid/labels_mat \
  --img_ext .png \
  --ann_ext .mat \
  --win_size 256 \
  --step_size 250 \
  --extract_type mirror
```

If your images are `.jpg`/`.tif`, switch `--img_ext`.

## 4) Runtime configuration through environment variables

Set these before training:

```bash
export CGT_DATASET_NAME=nucls
export CGT_NR_TYPES=6
export CGT_MODEL_MODE=original
export CGT_LOG_DIR=logs/nucls-exp01
export CGT_TRAIN_DIRS="dataset/training_data/nucls/nucls/train/256x256_250x250"
export CGT_VALID_DIRS="dataset/training_data/nucls/nucls/valid/256x256_250x250"

# Optional: phase control
export CGT_ENABLE_PHASE2=1
export CGT_PHASE1_EPOCHS=100
export CGT_PHASE2_EPOCHS=100
export CGT_PHASE1_LR=1e-5
export CGT_PHASE2_LR=1e-4
export CGT_PHASE1_PRETRAINED=""     # set checkpoint path if you have a stage-1 init model

# Optional: dataloader + graph/loss settings
export CGT_TRAIN_WORKERS=16
export CGT_VALID_WORKERS=8
export CGT_EDGE_NUM=4
export CGT_CLASS_WEIGHTS="5,2,4,3,2"   # must have NR_TYPES-1 values
```

`CGT_CLASS_WEIGHTS` length must equal `CGT_NR_TYPES - 1`, otherwise training fails fast with an explicit error.

## 5) Train

```bash
python run_train.py --gpu=0
```

Multi-GPU example:

```bash
python run_train.py --gpu=0,1
```

## 6) Inference (tile mode)

```bash
python run_infer.py \
  --gpu=0 \
  --nr_types 6 \
  --type_info_path type_info.json \
  --model_path logs/nucls-exp01/net_epoch=100.tar \
  --model_mode original \
  tile \
  --input_dir data/nucls/test/images \
  --inst_dir data/nucls/test/inst_map_mat \
  --output_dir outputs/nucls-exp01 \
  --save_raw_map \
  --save_qupath
```

## 7) Evaluate

```bash
python compute_stats.py --mode=instance --pred_dir outputs/nucls-exp01/mat --true_dir data/nucls/test/labels_mat
python compute_stats.py --mode=type --pred_dir outputs/nucls-exp01/mat --true_dir data/nucls/test/labels_mat
```

## 8) Notes specific to this repository

1. `export_nucls_hf.py` exports the Hugging Face NuCLS dataset into this repository's expected train/valid + `.mat` format.
2. `prepare_nucls_mat.py` converts manual NuCLS `rgb/mask/csv` downloads into `.mat` annotations for this codebase.
3. `extract_patches.py` writes patches in the shape expected by `dataloader/train_loader.py`: first 3 channels RGB, then annotation channels.
4. If `csv` columns include instance IDs and type labels, `type_map` is generated directly; if only centroid coordinates are available, types are assigned by centroid lookup on `inst_map`.
5. `CGT_CLASS_WEIGHTS` must match the number of non-background classes (`CGT_NR_TYPES - 1`).
6. Training/inference configuration is controlled by env vars; avoid hardcoding machine-specific paths.
7. `nucls` is a valid dataset key in `dataset.py`.