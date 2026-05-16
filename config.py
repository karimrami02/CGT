import importlib
import random

import cv2
import numpy as np
import os
from dataset import get_dataset


class Config(object):
    """Configuration file."""

    def __init__(self):
        self.seed = 10

        self.logging = True

        # turn on debug flag to trace some parallel processing problems more easily
        self.debug = False

        def _get_env_int(name, default):
            raw = os.environ.get(name)
            if raw is None or raw == "":
                return default
            return int(raw)

        def _get_env_list(name, default):
            raw = os.environ.get(name)
            if raw is None or raw.strip() == "":
                return default
            values = [item.strip() for item in raw.split(";")]
            values = [item for item in values if item != ""]
            return values if len(values) > 0 else default

        model_name = "CGT"
        model_mode = os.environ.get("CGT_MODEL_MODE", "original") # choose either `original` or `fast`

        if model_mode not in ["original", "fast"]:
            raise Exception("Must use either `original` or `fast` as model mode")

        nr_type = _get_env_int("CGT_NR_TYPES", 4) # number of nuclear types (including background)

        # whether to predict the nuclear type, availability depending on dataset!
        self.type_classification = True

        # shape information - 
        # below config is for original mode. 
        # If original model mode is used, use [270,270] and [80,80] for act_shape and out_shape respectively
        # If fast model mode is used, use [256,256] and [164,164] for act_shape and out_shape respectively
        patch_input_shape = _get_env_int("CGT_INPUT_SHAPE", 256)
        patch_output_shape = _get_env_int("CGT_OUTPUT_SHAPE", patch_input_shape)
        patch_step = _get_env_int("CGT_PATCH_STEP", 250)
        aug_shape = [patch_input_shape, patch_input_shape] # patch shape used during augmentation (larger patch may have less border artefacts)
        act_shape = [patch_input_shape, patch_input_shape] # patch shape used as input to network - central crop performed after augmentation
        out_shape = [patch_output_shape, patch_output_shape] # patch shape at output of network


        self.dataset_name = os.environ.get("CGT_DATASET_NAME", "fs") # extracts dataset info from dataset.py
        self.log_dir = os.environ.get("CGT_LOG_DIR", "logs") # where checkpoints will be saved

        # paths to training and validation patches
        default_train_dir = os.path.join(
            "dataset",
            "training_data",
            self.dataset_name,
            self.dataset_name,
            "train",
            "%dx%d_%dx%d"
            % (
                patch_input_shape,
                patch_input_shape,
                patch_step,
                patch_step,
            ),
        )
        default_valid_dir = os.path.join(
            "dataset",
            "training_data",
            self.dataset_name,
            self.dataset_name,
            "valid",
            "%dx%d_%dx%d"
            % (
                patch_input_shape,
                patch_input_shape,
                patch_step,
                patch_step,
            ),
        )
        self.train_dir_list = _get_env_list("CGT_TRAIN_DIRS", [default_train_dir])
        self.valid_dir_list = _get_env_list("CGT_VALID_DIRS", [default_valid_dir])

        self.shape_info = {
            "train": {"input_shape": act_shape, "mask_shape": out_shape,},
            "valid": {"input_shape": act_shape, "mask_shape": out_shape,},
        }

        # * parsing config to the running state and set up associated variables
        self.dataset = get_dataset(self.dataset_name)

        module = importlib.import_module(
            "models.%s.opt" % model_name
        )
        self.model_config = module.get_config(nr_type, model_mode)
