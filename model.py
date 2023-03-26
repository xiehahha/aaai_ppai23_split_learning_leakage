# define the model 
# 1) non-label--user; 2) label--party

import torch
from torch import nn
from torch.utils.data import DataLoader
from sklearn.datasets import load_boston
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
torch.manual_seed(42)
from itertools import combinations
from itertools import accumulate
import copy, pickle, math
import pandas as pd
import numpy as np

class user_m(nn.Module): # add customerized user side
  def __init__(self, dim_x=16):
      super().__init__()
      self.layers = nn.Sequential(
      nn.Linear(13, 64),
      nn.ReLU(),
      nn.Linear(64, 32),
      nn.ReLU(),
      nn.Linear(32, dim_x),
      nn.ReLU(),
    )
  def forward(self, x):
    return self.layers(x)


class label_m(nn.Module): # add for customerized label party
  def __init__(self, dim_x=16, flag=False):
    super().__init__()
    self.layers = nn.Sequential(
        nn.Linear(dim_x, 8),
        nn.ReLU(),
        nn.Linear(8, 4, bias=flag),
        nn.ReLU(),
        nn.Linear(4, 1, bias=flag),
    )
    
  def forward(self, x):
    return self.layers(x)

class MLP_r(nn.Module):
  def __init__(self, dim_x=16, flag=True):
    super().__init__()
    self.user = user_m(dim_x)
    self.label = label_m(dim_x, flag)

  def forward(self, x):
    self.f_int = self.user(x)
    return self.label(self.f_int)

def weights_init(m):
    if hasattr(m, "weight"):
        m.weight.data.uniform_(-0.5, 0.5)
    if hasattr(m, "bias"):
        m.bias.data.uniform_(-0.5, 0.5)


#====================#

# 2. set up the surrogate model
class label_surrogate(nn.Module): # add for customerized label party
  def __init__(self, dim_x=16, flag=False):
    super().__init__()
    self.layers = nn.Sequential(
        nn.Linear(dim_x, 8, bias=flag),
        nn.Linear(8, 1, bias=flag),
        # nn.Linear(4, 1, bias=flag),
    )
    
  def forward(self, x):
    return self.layers(x)

surrogate = label_surrogate(dim_x=16, flag=True)
