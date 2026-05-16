import torch.optim as optim
import os

from run_utils.callbacks.base import (
    AccumulateRawOutput,
    PeriodicSaver,
    ProcessAccumulatedRawOutput,
    ScalarMovingAverage,
    ScheduleLr,
    TrackLr,
    VisualizeOutput,
    TriggerEngine,
)
from run_utils.callbacks.logging import LoggingEpochOutput, LoggingGradient
from run_utils.engine import Events

from .targets import gen_targets, prep_sample
from .net_desc import create_model
from .run_desc import proc_valid_step_output, train_step, valid_step, viz_step_output


# TODO: training config only ?
# TODO: switch all to function name String for all option
def get_config(nr_type, mode):
    def _get_env_int(name, default):
        raw = os.environ.get(name)
        if raw is None or raw == "":
            return default
        return int(raw)

    def _get_env_float(name, default):
        raw = os.environ.get(name)
        if raw is None or raw == "":
            return default
        return float(raw)

    def _get_env_bool(name, default):
        raw = os.environ.get(name)
        if raw is None or raw == "":
            return default
        return raw.lower() not in ["0", "false", "no", "off"]

    phase1_pretrained = os.environ.get("CGT_PHASE1_PRETRAINED", "").strip()
    if phase1_pretrained == "":
        phase1_pretrained = None

    phase1_cfg = {
        "run_info": {
            "net": {
                "desc": lambda: create_model(
                    input_ch=3, nr_types=nr_type, 
                    freeze=False, mode=mode
                ),
                "optimizer": [
                    optim.Adam,
                    {
                        "lr": _get_env_float("CGT_PHASE1_LR", 1.0e-5),
                        "betas": (0.9, 0.999),
                    },
                ],
                "lr_scheduler": lambda x: optim.lr_scheduler.StepLR(x, 25),
                "extra_info": {
                    "loss": {
                        "np": {"bce": 1, "dice": 1},
                        "hv": {"mse": 1, "msge": 1},
                        "tp": {"bce": 0, "dice": 0},
                    },
                },
                "pretrained": phase1_pretrained,
            },
        },
        "target_info": {"gen": (gen_targets, {}), "viz": (prep_sample, {})},
        "batch_size": {
            "train": _get_env_int("CGT_PHASE1_BATCH_TRAIN", 64),
            "valid": _get_env_int("CGT_PHASE1_BATCH_VALID", 64),
        },
        "nr_epochs": _get_env_int("CGT_PHASE1_EPOCHS", 100),
    }

    phase2_cfg = {
        "run_info": {
            "net": {
                "desc": lambda: create_model(
                    input_ch=3, nr_types=nr_type, 
                    freeze=False, mode=mode
                ),
                "optimizer": [
                    optim.Adam,
                    {
                        "lr": _get_env_float("CGT_PHASE2_LR", 1.0e-4),
                        "betas": (0.9, 0.999),
                    },
                ],
                "lr_scheduler": lambda x: optim.lr_scheduler.StepLR(x, 25),
                "extra_info": {
                    "loss": {
                        "np": {"bce": 1, "dice": 1},
                        "hv": {"mse": 1, "msge": 1},
                        "tp": {"bce": 1, "dice": 1},
                    },
                },
                "pretrained": -1,
            },
        },
        "target_info": {"gen": (gen_targets, {}), "viz": (prep_sample, {})},
        "batch_size": {
            "train": _get_env_int("CGT_PHASE2_BATCH_TRAIN", 16),
            "valid": _get_env_int("CGT_PHASE2_BATCH_VALID", 16),
        },
        "nr_epochs": _get_env_int("CGT_PHASE2_EPOCHS", 100),
    }

    phase_list = [phase1_cfg]
    if _get_env_bool("CGT_ENABLE_PHASE2", True):
        phase_list.append(phase2_cfg)

    dataset_name = os.environ.get("CGT_DATASET_NAME", "fs")
    return {
        # ------------------------------------------------------------------
        # ! All phases have the same number of run engine
        # phases are run sequentially from index 0 to N
        "phase_list": phase_list,
        # ------------------------------------------------------------------
        # TODO: dynamically for dataset plugin selection and processing also?
        # all enclosed engine shares the same neural networks
        # as the on at the outer calling it
        "run_engine": {
            "train": {
                # TODO: align here, file path or what? what about CV?
                "dataset": dataset_name,  # whats about compound dataset ?
                "nr_procs": _get_env_int("CGT_TRAIN_WORKERS", 16),  # number of threads for dataloader
                "run_step": train_step,  # TODO: function name or function variable ?
                "reset_per_run": False,
                # callbacks are run according to the list order of the event
                "callbacks": {
                    Events.STEP_COMPLETED: [
                        # LoggingGradient(), # TODO: very slow, may be due to back forth of tensor/numpy ?
                        ScalarMovingAverage(),
                    ],
                    Events.EPOCH_COMPLETED: [
                        TrackLr(),
                        PeriodicSaver(),
                        #VisualizeOutput(viz_step_output),
                        LoggingEpochOutput(),
                        TriggerEngine("valid"),
                        ScheduleLr(),
                    ],
                },
            },
            "valid": {
                "dataset": dataset_name,  # whats about compound dataset ?
                "nr_procs": _get_env_int("CGT_VALID_WORKERS", 8),  # number of threads for dataloader
                "run_step": valid_step,
                "reset_per_run": True,  # * to stop aggregating output etc. from last run
                # callbacks are run according to the list order of the event
                "callbacks": {
                    Events.STEP_COMPLETED: [AccumulateRawOutput(),],
                    #Events.EPOCH_COMPLETED: [
                        # TODO: is there way to preload these ?
                        #ProcessAccumulatedRawOutput(
                            #lambda a: proc_valid_step_output(a, nr_types=nr_type)
                        #),
                        #LoggingEpochOutput(),
                    #],
                },
            },
        },
    }
