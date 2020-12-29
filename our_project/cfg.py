import random
import os

import numpy as np
import torch
from torch.utils.tensorboard import SummaryWriter

#####################

N_INTERACTIONS = 1
# Addition config
CURRICULUM = [[0, 1]]
MAX_DIGITS = 2
TRAIN_SIZE=10000
VAL_SIZE=1000
BATCH_SIZE=320
EPOCHS=100
NUM_CHARS = 12
LR=0.001

#####################

SEED = 42
def seed_everything(seed):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
seed_everything(SEED)

##################

WRITER = SummaryWriter('./runs/directly_two')

###############