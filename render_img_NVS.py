from __future__ import absolute_import, division, print_function
import os
import os.path as osp
import json
import time

import cv2
import torch.distributed as dist
from torch.utils.data import DataLoader
from torch.utils.data import DistributedSampler as _DistributedSampler
from torch.nn.parallel import DistributedDataParallel as DDP

import pickle
import datasets
import networks
from options import MonodepthOptions
from utils.loss_metric import *
from utils.layers import *
from runner import Runer

def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    torch.backends.cudnn.deterministic = True


if __name__ == "__main__":
    options = MonodepthOptions()
    opts = options.parse()
    setup_seed(42)
    runner = Runer(opts)
    runner.val()