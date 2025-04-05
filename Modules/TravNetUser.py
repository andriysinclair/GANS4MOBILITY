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
from pathlib import Path
import random
from .TravNet import train_evaluate_TravNet
from .Transformations import *
import time

# Setting logging

logging.basicConfig(level=logging.INFO, force=True, format='%(levelname)s: %(message)s')

class TravNet:
    def __init__(self):

        # Setting paths

        self.root_folder = str(  Path(__file__).parent.parent   )
        self.tensors_folder = self.root_folder + "/tensors"
        self.Plots_folder = self.root_folder + "/Plots"
        self.Models_folder = self.root_folder + "/Models"
        self.Results_folder = self.root_folder + "/Results"

        # Loading Tensors

        with open(self.tensors_folder + "/tensors.pkl", "rb") as f:
            (self.X, self.y_cont_raw, self.y_cat_raw) = pickle.load(f)

        # GPU?

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # TravNet model -in case of updates

        self.model_extension = "/TravNet6952.pt"

        logging.info(f"Running on: {self.device}")

        self.results = None

        # Loading in true data

        self.true_data_path = 

    def generate_travel_data(self,N, return_df = False):

        X_eval_full = self.X[:,:,:,:].to(self.device)

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

        X_eval.to(self.device)

        logging.debug(f"Post appending with 0s: {X_eval[3,7:,:,:10]}")

        logging.info(f"Shape of X_eval: {X_eval.shape}")

        gen_start = time.time()
        wide_df, long_df = train_evaluate_TravNet(i_to_loop=X_eval.shape[0],
                                                trained_model_path=self.Models_folder + self.model_extension,
                                                X=X_eval,
                                                evaluation=True)
        gen_end = time.time()

        logging.info(f"Generating {N} individuals took {gen_end - gen_start:.2f}s!")
        
        # Pushing long df to results folder

        long_df.to_pickle(self.Results_folder + f"/Results_{N}.pkl")

        self.results = long_df

        if return_df:

            return wide_df, long_df
        
    def get_agg_stats(self, data_real, data_gen):

        data_gen = data_gen.drop(["i_id", "IsTrip"], axis=1)
        data_gen["DoW"]+=1

        data_gen = data_gen.drop("Purpouse", axis=1)
        data_real = data_real.drop("TripPurpose_B01ID", axis=1)
        
        gen_stats = data_gen.agg(["mean", "median", "std"]).round(2)

        data_real = data_real.copy()
        data_real["Duration"] = data_real["TripEnd"] - data_real["TripStart"]

        data_real = data_real.rename(columns={"TripDisExSW": "Distance",
                                            "TripStart": "TripStart",
                                                "Duration": "Duration",
                                                "JourSeq": "TripNum",
                                                "TravelWeekDay_B01ID": "DoW"})
        
        data_real = data_real[data_gen.columns]


        real_stats  = data_real.agg(["mean", "median", "std"]).round(2)

        return gen_stats, real_stats
        
    def evaluate(self, path_to_gen=None):




if __name__ == "__main__":

    print("...")