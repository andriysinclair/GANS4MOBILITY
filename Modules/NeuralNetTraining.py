import os
import pandas as pd
import numpy as np
import sys
import re
import logging
from Modules.Loader_wrangler import *
import random
import torch
import torch.nn as nn
from sklearn.preprocessing import OneHotEncoder, MinMaxScaler
import matplotlib.pyplot as plt

from TravelNet import TravelNet

#Load tensors
with open("/home/trapfishscott/Cambridge24.25/D200_ML_econ/ProblemSets/Project/tensors/tensors.pkl", "rb") as f:
    (X, y_cont, y_cat) = pickle.load(f)

X = X.to(torch.float32)
y_cont = y_cont.to(torch.float32)
y_cat = y_cat.to(torch.long)

print(f"Input shape: {X.shape}")
print(f"Cont Output shape: {y_cont.shape}")
print(f"Cat Output shape: {y_cat.shape}")



# Calculating probability weights

unique_vals, counts = torch.unique(y_cat, return_counts=True)

print(unique_vals)

# 17 is missing this refers to "short walk", not relevant for our data, but will be kept for reference

ce_weighting = torch.zeros(24, dtype=torch.float32)
ce_weighting[17] = 0

for val, count in zip(unique_vals, counts):
    ce_weighting[val] = 1/(count/counts.sum())

# Apply log scaling to smooth extreme weight differences
ce_weighting = torch.log1p(ce_weighting)  

# Normalize the weights
ce_weighting /= ce_weighting.sum()

print("Final CE Weights:", ce_weighting)


