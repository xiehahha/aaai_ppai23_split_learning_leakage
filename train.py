import torch
from torch import nn
from torch.utils.data import DataLoader
from sklearn.datasets import load_boston
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
torch.manual_seed(42)
from itertools import combinations
import copy, pickle, math
import pandas as pd
import numpy as np


'''training and evaluation of a model.'''

'''training and evaluation of a model.'''

def train_model(mlp, trainloader):
  train_loss_function = nn.L1Loss()
  optimizer = torch.optim.Adam(mlp.parameters(), lr=1e-3)

  for epoch in range(0, 10):
    current_loss = 0.0
    for i, data in enumerate(trainloader, 0):
      inputs, targets = data
      inputs, targets = inputs.float(), targets.float()
      targets = targets.reshape((targets.shape[0], 1))
      optimizer.zero_grad()
      outputs = mlp(inputs)
      loss = train_loss_function(outputs, targets)   
      loss.backward()
      optimizer.step()
    print ('training epoch: {epoch}\tLoss: {:0.4f}'.format(loss.item(), epoch=epoch+1))
  print('final training loss: {:0.4f}'.format(loss.item()))
  return mlp


def train_model_save_epoch(mlp, trainloader):
  train_loss_function = nn.L1Loss()
  optimizer = torch.optim.Adam(mlp.parameters(), lr=1e-3)
  PATH = "/content/drive/MyDrive/epoch_m"
  for epoch in range(0, 10):
    current_loss = 0.0
    for i, data in enumerate(trainloader, 0):
      inputs, targets = data
      inputs, targets = inputs.float(), targets.float()
      targets = targets.reshape((targets.shape[0], 1))
      optimizer.zero_grad()
      outputs = mlp(inputs)
      loss = train_loss_function(outputs, targets)   
      loss.backward()
      optimizer.step()
    print ('training epoch: {epoch}\tLoss: {:0.4f}'.format(loss.item(), epoch=epoch+1))

    tmp_path = PATH + "/" + str(epoch)
    torch.save(mlp.state_dict(), tmp_path)

  print('final training loss: {:0.4f}'.format(loss.item()))
  return mlp

def evaluate(model, testloader):
  accumulate_loss = 0
  train_loss_function = nn.L1Loss()
  for i, data in enumerate(testloader, 0):
    inputs, targets = data
    inputs, targets = inputs.float(), targets.float()
    targets = targets.reshape((targets.shape[0], 1))
    outputs = model(inputs)
  
    loss = train_loss_function(outputs, targets)
    accumulate_loss+=loss.item()
  return accumulate_loss/len(testloader)

if False:
  mlp_new = MLP_r(dim_x=16, flag=True)
  mlp_new = train_model(mlp_new, trainloader)