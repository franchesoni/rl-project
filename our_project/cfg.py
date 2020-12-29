N_INTERACTIONS = 3
# Addition config
CURRICULUM = [[0, 0, 1]]
MAX_DIGITS = 3
TRAIN_SIZE=1000
VAL_SIZE=1000
BATCH_SIZE=32
EPOCHS=100
MAX_DIGITS=3
NUM_CHARS = 12
LR=0.001
#####################
import random
import os
import numpy as np
import torch
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
# os.system('rm -rf /home/franchesoni/Documents/mva/mva/rl/project/runs')
from torch.utils.tensorboard import SummaryWriter
WRITER = SummaryWriter()
###############