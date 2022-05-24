import numpy as np
import torch
import random
import ipdb

def fixseed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

SEED = 10
#ipdb.set_trace()
EVALSEED = 0
# Provoc warning: not fully functionnal yet
# torch.set_deterministic(True)
torch.backends.cudnn.benchmark = False

fixseed(SEED)
