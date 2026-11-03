import torch
import numpy as np

from parms_setting import settings
from data_preprocess_test2 import load_data
from instantiation import Create_model
from train_test2 import train_model
import os
import random

os.environ["CUDA_VISIBLE_DEVICES"] = "1"
def set_random_seed(seed, deterministic=False):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # if you are using multi-GPU.
    np.random.seed(seed)  # Numpy module.
    random.seed(seed)  # Python random module.
    torch.manual_seed(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

set_random_seed(1, deterministic=False)

# parameters setting
args = settings()

args.cuda = not args.no_cuda and torch.cuda.is_available()

data_o,data_s, train_loader, val_loader, test_loader,brics_features, brics_dim ,num_classes= load_data(args)

if args.cuda:
    # 把 brics 特征移动到 GPU
    brics_features = brics_features.cuda()


# train and test model
model, optimizer = Create_model(args,brics_dim,num_classes)
train_model(model, optimizer, data_o,data_s ,brics_features,train_loader, val_loader, test_loader, args)

