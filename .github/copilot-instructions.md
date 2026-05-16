# Copilot instructions for CGT

## Build, test, and lint commands

This repository does not define a dedicated lint or unit-test framework (`pytest`, `unittest` suites, `ruff`, `flake8`, etc.). Use the project scripts below as the canonical runnable entrypoints.

### Environment setup

```bash
conda create --name CGT python=3.8
conda activate CGT
pip install -r requirements.txt
pip install torch==1.13.0+cu116 torchvision==0.14.0+cu116 torchaudio==0.13.0 --extra-index-url https://download.pytorch.org/whl/cu116
pip install torch-geometric torch-scatter torch-sparse
```

### Training

```bash
python run_train.py --gpu=0
```

### Inference (tile mode)

```bash
python run_infer.py --gpu=0 --model_path=<checkpoint.tar> --model_mode=fast --nr_types=4 --type_info_path=type_info.json tile --input_dir=<images_dir> --inst_dir=<instance_maps_dir> --output_dir=<output_dir>
```

### Inference (WSI mode)

```bash
python run_infer.py --gpu=0 --model_path=<checkpoint.tar> --model_mode=fast --nr_types=4 --type_info_path=type_info.json wsi --input_dir=<wsi_dir> --output_dir=<output_dir>
```

### Metrics / evaluation

```bash
python compute_stats.py --mode=instance --pred_dir=<pred_mat_dir> --true_dir=<gt_mat_dir>
python compute_stats.py --mode=type --pred_dir=<pred_mat_dir> --true_dir=<gt_mat_dir>
```

### Smallest practical "single-test" run

There is no per-test runner. For a single-case check, place one `.mat` prediction and one matching `.mat` ground-truth file in dedicated directories, then run one of the `compute_stats.py` commands above.

## High-level architecture

### 1. Data preparation and dataset adapters

- `extract_patches.py` creates training patches as `.npy` tensors by stacking image and annotation channels.
- `dataset.py` contains dataset-specific loaders (`kumar`, `cpm17`, `consep`, `fs`) that normalize how image and annotation files are read.
- `dataloader/train_loader.py` reads patch `.npy` files, applies augmentations, and returns `img`, `inst_map`, `tp_map`, plus generated targets.

### 2. Training orchestration

- `run_train.py` is the training entrypoint.
- `config.py` provides run-level config (dataset name, train/valid patch paths, log dir, shape settings).
- `models/CGT/opt.py` provides model/training phase config (`phase_list`, optimizer, LR scheduler, callbacks, pretrained handling).
- `run_utils/engine.py` and `run_utils/callbacks/*` implement the callback-driven training engine (checkpointing, logging, LR scheduling, and triggering validation from train callbacks).

### 3. Model and step logic

- `models/CGT/net_desc.py` defines `CGT` (VAN encoder + decoder + graph/transformer-based classification branch).
- `models/CGT/run_desc.py` defines `train_step`, `valid_step`, and `infer_step`:
  - builds per-instance graph inputs from instance maps (`get_bboxes` / `get_infer_bboxes`),
  - runs forward passes,
  - computes classification losses / outputs.

### 4. Inference and output artifacts

- `run_infer.py` is the CLI entrypoint (tile or WSI mode).
- `infer/base.py` loads the model checkpoint and binds inference/post-processing functions.
- `infer/tile.py` handles patch tiling/reassembly, post-processes predictions, and writes:
  - `output_dir/mat/*.mat`
  - `output_dir/json/*.json`
  - `output_dir/overlay/*.png`
  - optional `output_dir/qupath/*.tsv`

### 5. Metrics

- `compute_stats.py` + `metrics/stats_utils.py` compute instance and type statistics from `.mat` outputs.

## Key conventions for this codebase

1. **Patch tensor format is fixed**  
   Training patches are expected as `.npy` with channels `[RGB, inst, type]` (at minimum: first 3 RGB + instance map channel). Loader logic assumes:
   - `data[..., :3]` is RGB image
   - `ann[..., 0]` is `inst_map`
   - `ann[..., 1]` is `tp_map` when type classification is enabled

2. **Config is code-driven, not config-file-driven**  
   Core training paths and hyperparameters are hard-coded in `config.py` and `models/CGT/opt.py`. Typical workflow is editing these files before running scripts.

3. **DataParallel checkpoint compatibility is expected**  
   Checkpoints are loaded from `["desc"]` and passed through `convert_pytorch_checkpoint()` to strip `module.` prefixes when needed. Keep checkpoint structure consistent with this loader contract.

4. **Type metadata must align with `nr_types`**  
   In inference, if `--nr_types` and `--type_info_path` are set, `type_info.json` must define every type id from `0` to `nr_types-1`.

5. **Validation is callback-triggered from training**  
   Validation runs are not separate top-level loops; `TriggerEngine("valid")` in `models/CGT/opt.py` drives validation from the training engine callback pipeline.
