import pandas as pd
import numpy as np
import logging
import pickle
import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend
import matplotlib.pyplot as plt
from Transformations import *
from pathlib import Path
import random
from TravNet import train_evaluate_TravNet
import time

# Setting logging

logging.basicConfig(level=logging.INFO, force=True, format='%(levelname)s: %(message)s')

# Setting paths

root_folder = str(  Path(__file__).parent.parent   )
tensors_folder = root_folder + "/tensors"
Plots_folder = root_folder + "/Plots"
Models_folder = root_folder + "/Models"
Results_folder = root_folder + "/Results"

# GPU?

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Training running on: {device}")

# Open Tensors

with open(tensors_folder + "/tensors.pkl", "rb") as f:
    (X, y_cont_raw, y_cat_raw) = pickle.load(f)


def generate_travel_data(N,X=X):

    X_eval_full = X[:,:,:,:].to(device)

    logging.debug(f"X_eval_full shape: {X_eval_full.shape}")

    X_eval = []

    for n in range(N):

        # Getting random individual from the data

        random_index = random.randint(0,X_eval_full.shape[0])

        if random_index == X_eval_full.shape[0]:
            random_index-=1

        logging.debug(f"Selected random index: {random_index}")

        X_n = X_eval_full[random_index,:,:,:].unsqueeze(0)

        logging.debug(f"X_n shape: {X_n.shape}")

        X_eval.append(X_n)

    # Concat across the individual axis

    X_eval = torch.cat(X_eval, dim=0)

    logging.debug(f"Prior to appending with 0s: {X_eval[3,7:,:,:10]}")

    # Appending all target entries with a 0
    X_eval[:,7:,:,:50] = 0

    X_eval.to(device)

    logging.debug(f"Post appending with 0s: {X_eval[3,7:,:,:10]}")

    logging.info(f"Shape of X_eval: {X_eval.shape}")

    gen_start = time.time()
    wide_df, long_df = train_evaluate_TravNet(i_to_loop=X_eval.shape[0],
                                              trained_model_path=Models_folder + "/TravNet6952.pt",
                                              X=X_eval,
                                              evaluation=True)
    gen_end = time.time()

    logging.info(f"Generating {N} individuals took {gen_end - gen_start:.2f}s!")
    
    # Pushing long df to results folder

    long_df.to_pickle(Results_folder + f"/Results_{N}.pkl")

    return wide_df, long_df

wide_df, long_df = generate_travel_data(10000)