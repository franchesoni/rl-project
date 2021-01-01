import argparse
import json
import random
import os
import shutil

import numpy as np
import torch
from torch.utils.tensorboard import SummaryWriter

import teachers


def config_parser():
    # argument parser
    parser = argparse.ArgumentParser(description="Train on addition.")
    parser.add_argument("--config-file",
        type=str, nargs=1, help="config file name")
    parser.add_argument("--config-folder",
        default="config", type=str, nargs="?", help="config file folder")
    args = parser.parse_args()
    return os.path.join(args.config_folder, args.config_file[0])
CONFIG_FILE = config_parser()

#####################
_CURRICULUMS = {  # example curricula for 4 actions
    "direct": teachers.gen_curriculum_direct,
    "baseline": teachers.gen_curriculum_baseline,
    "incremental": teachers.gen_curriculum_incremental,
    "naive": teachers.gen_curriculum_naive,
    "mixed": teachers.gen_curriculum_mixed,
    "combined": teachers.gen_curriculum_combined,
}

with open(CONFIG_FILE, "r") as f:
    config = json.load(f)

N_INTERACTIONS = config["N_INTERACTIONS"]
MAX_DIGITS = config["MAX_DIGITS"]
CURRICULUM = config["CURRICULUM"]
if type(CURRICULUM) is str:
    if CURRICULUM in _CURRICULUMS.keys():
        CURRICULUM = _CURRICULUMS[CURRICULUM](MAX_DIGITS)
    else:
        raise ValueError("{} is not a curriculum.".format(CURRICULUM))
CURRICULUM_SCHEDULE = config["CURRICULUM_SCHEDULE"]
TRAIN_SIZE = config["TRAIN_SIZE"]
VAL_SIZE = config["VAL_SIZE"]
BATCH_SIZE = config["BATCH_SIZE"]
EPOCHS = config["EPOCHS"]
NUM_CHARS = config["NUM_CHARS"]
LR = config["LR"]
ABSOLUTE = config.get("absolute", False)
SUMMARY_WRITER_PATH = config["SUMMARY_WRITER_PATH"]
SEED = config["SEED"]
TEACHER_NAME = config["TEACHER_NAME"]
SAVE_MODEL = config["SAVE_MODEL"]  # to see if we multitask
SHOW_ADD = config["SHOW_ADD"]
if os.path.isdir(SUMMARY_WRITER_PATH):
    shutil.rmtree(SUMMARY_WRITER_PATH)
WRITER = SummaryWriter(SUMMARY_WRITER_PATH)

with open(os.path.join(SUMMARY_WRITER_PATH, "config.json"), "w") as f:
    json.dump(config, f)

#####################


def seed_everything(seed):
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


seed_everything(SEED)
