import random
import torch
import numpy as np
import os

from utils import parse_args

args = parse_args()

random.seed(args.seed)
np.random.seed(args.seed)
torch.manual_seed(args.seed)
torch.cuda.manual_seed(args.seed)
torch.backends.cudnn.deterministic = True

args.root_model = 'checkpoint'
os.makedirs(args.root_model, exist_ok=True)