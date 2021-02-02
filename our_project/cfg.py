import argparse
import json
import random
import os
import shutil

###############################################################################
import numpy as np
import torch
from torch.utils.tensorboard import SummaryWriter

###############################################################################
import teachers

###############################################################################
def config_parser():
    # argument parser
    parser = argparse.ArgumentParser(description="Train on addition.")
    parser.add_argument("--config-file", type=str, nargs=1, help="config file name")
    parser.add_argument(
        "--config-folder",
        default="config",
        type=str,
        nargs="?",
        help="config file folder",
    )
    args = parser.parse_args()
    return os.path.join(args.config_folder, args.config_file[0])


CONFIG_FILE = config_parser()
###############################################################################
_CURRICULUMS = {  # example curricula for 4 actions
    "direct": teachers.gen_curriculum_direct,
    "baseline": teachers.gen_curriculum_baseline,
    "incremental": teachers.gen_curriculum_incremental,
    "naive": teachers.gen_curriculum_naive,
    "mixed": teachers.gen_curriculum_mixed,
    "combined": teachers.gen_curriculum_combined,
}
###############################################################################
with open(CONFIG_FILE, "r") as f:
    config = json.load(f)
###############################################################################
# problem params
N_INTERACTIONS = config["N_INTERACTIONS"]
MAX_DIGITS = config["MAX_DIGITS"]
NUM_CHARS = config.get("NUM_CHARS", 12)
# bandit params
ABSOLUTE = config.get("absolute", False)
REWARD_FN = config.get("REWARD_FN", "absolute")
OBS_TYPE = config.get("OBS_TYPE", "per_digit_loss")  # 'per_digit_loss' or 'accuracy_per_length' for sequential, 'prediction_gain' for bandit. It's possible to add more!

MODE = config.get("MODE", "sequential")
###############################################################################
# training params
TRAIN_SIZE = config["TRAIN_SIZE"]
OBS_SIZE = config.get("OBS_SIZE", 1000)
BATCH_SIZE = config["BATCH_SIZE"]
EPOCHS = config["EPOCHS"]
LR = config["LR"]
OPTIM = config.get("OPTIM", None)
SEED = config.get("SEED", 42)
###############################################################################
# teacher params
TEACHER_NAME = config["TEACHER_NAME"]
CURRICULUM = config["CURRICULUM"]
if TEACHER_NAME == "curriculum" and type(CURRICULUM) is str:
    if CURRICULUM in _CURRICULUMS.keys():
        CURRICULUM = _CURRICULUMS[CURRICULUM](MAX_DIGITS)
    else:
        raise ValueError("{} is not a curriculum.".format(CURRICULUM))
CURRICULUM_SCHEDULE = config.get(
    "CURRICULUM_SCHEDULE", [N_INTERACTIONS // len(CURRICULUM)] * len(CURRICULUM)
)  # this is number of times at each step
# for teachers using boltzmann (e.g. online)
BOLTZMANN_TEMPERATURE = config.get("BOLTZMANN_TEMPERATURE", 1.0)
# for online teacher
TEACHER_LR = config.get("TEACHER_LR", 0.1)
###############################################################################
# logging params
LOG_DICT = config.get(
    "LOG_DICT", {"activate": False, "size": 32, "freq": 10, "regenerate_data": False}
)

LOG_FREQ = config.get("LOG_FREQ", 10)
SHOW_ADD = config.get("SHOW_ADD", False)
SUMMARY_WRITER_PATH = config["SUMMARY_WRITER_PATH"]
SAVE_MODEL = config["SAVE_MODEL"]  # to see if we multitask
if os.path.isdir(SUMMARY_WRITER_PATH):
    shutil.rmtree(SUMMARY_WRITER_PATH)
WRITER = SummaryWriter(SUMMARY_WRITER_PATH)
with open(os.path.join(SUMMARY_WRITER_PATH, "config.json"), "w") as f:
    json.dump(config, f)
###############################################################################
DEVICE = config.get("DEVICE", "cpu")
if DEVICE != "cuda" or not torch.cuda.is_available():
    DEVICE = "cpu"
###############################################################################
def seed_everything(seed):
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


seed_everything(SEED)
