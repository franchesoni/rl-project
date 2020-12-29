import json
import random
import os

import numpy as np
import torch
from torch.utils.tensorboard import SummaryWriter

#####################

with open("./our_project/config.json", "r") as f:
    config = json.load(f)

N_INTERACTIONS = config["N_INTERACTIONS"]
CURRICULUM = config["CURRICULUM"]
MAX_DIGITS = config["MAX_DIGITS"]
TRAIN_SIZE = config["TRAIN_SIZE"]
VAL_SIZE = config["VAL_SIZE"]
BATCH_SIZE = config["BATCH_SIZE"]
EPOCHS = config["EPOCHS"]
NUM_CHARS = config["NUM_CHARS"]
LR = config["LR"]
SUMMARY_WRITER_PATH = config["SUMMARY_WRITER_PATH"]
WRITER = SummaryWriter(SUMMARY_WRITER_PATH)
SEED = config["SEED"]

#####################

def seed_everything(seed):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
seed_everything(SEED)
