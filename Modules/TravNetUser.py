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
    """
    A class to load the TravNet model, generate synthetic travel data, evaluate it,
    and visualize aggregate statistics.

    This class handles loading of saved model checkpoints, evaluation data, and real datasets.
    It allows for generating travel diaries, comparing synthetic and real data, and plotting results.

    Attributes:
        root_folder (str): Root directory of the project.
        tensors_folder (str): Path to folder with training tensors.
        Plots_folder (str): Path to folder for saving plots.
        Models_folder (str): Path to folder containing trained models.
        Results_folder (str): Path to folder for saving result files.
        data_folder (str): Path to folder containing real NTS data.
        X (torch.Tensor): Feature tensor loaded from file.
        y_cont_raw (Any): Continuous target values.
        y_cat_raw (Any): Categorical target values.
        device (torch.device): CUDA or CPU device for model execution.
        model_extension (str): File path extension for loading TravNet model.
        results (pd.DataFrame): Generated travel diary results.
        nts_df (pd.DataFrame): Ground truth travel data from NTS survey.
    """
    def __init__(self):

        # Setting paths

        self.root_folder = str(  Path(__file__).parent.parent   )
        self.tensors_folder = self.root_folder + "/tensors"
        self.Plots_folder = self.root_folder + "/Plots"
        self.Models_folder = self.root_folder + "/Models"
        self.Results_folder = self.root_folder + "/Results"
        self.data_folder = self.root_folder + "/data"

        # Loading Tensors

        with open(self.tensors_folder + "/tensors.pkl", "rb") as f:
            (self.X, self.y_cont_raw, self.y_cat_raw) = pickle.load(f)

        # GPU?

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # TravNet model -in case of updates

        self.model_extension = "/TravNet6952.pt"

        logging.info(f"Running on: {self.device}")

        with open(self.Results_folder + "/Results_10000.pkl", "rb") as f:
            self.results = pickle.load(f)

        # Loading in true data

        with open(self.data_folder + "/merged_df2017.pkl", "rb") as f:
            self.nts_df = pickle.load(f)

    def generate_travel_data(self,N, return_df = False):
        """
        Generate synthetic travel diary data for N individuals using the TravNet model.

        Args:
            N (int): Number of synthetic individuals to generate.
            return_df (bool, optional): Whether to return the generated dataframes. Defaults to False.

        Returns:
            Tuple[pd.DataFrame, pd.DataFrame], optional: 
                - wide_df: Wide-format travel diary.
                - long_df: Long-format travel diary.
        """

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
        """
        Compute basic aggregate statistics (mean, median, std) for real and generated data.

        Args:
            data_real (pd.DataFrame): Ground truth travel data.
            data_gen (pd.DataFrame): Generated synthetic travel data.

        Returns:
            Tuple[pd.DataFrame, pd.DataFrame]: Aggregate statistics for generated and real data.
        """

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
    
    def output_aggregate_stats(self):
        """
        Print and compare aggregate statistics between generated and real datasets.

        Outputs:
            - Purpose distribution
            - Mean, median, and standard deviation of trip features overall and per purpose
        """

        # Plotting value counts of purpouse
        print("Purpouse Value counts (Gen)")

        r = self.results["Purpouse"].value_counts(normalize=True).round(2)

        print(f"{'Purpouse':<15} | {'Proportion':>10}")
        print("-" * 30)
        for k, v in r.items():
            print(f"{k:<15} | {v:>10.2f}")

        print("Purpouse Value counts (True)")

        r = self.nts_df["TripPurpose_B01ID"].value_counts(normalize=True).round(2)

        print(f"{'Purpouse':<15} | {'Proportion':>10}")
        print("-" * 30)
        for k, v in r.items():
            print(f"{k:<15} | {v:>10.2f}")


        gen_stats, real_stats = self.get_agg_stats(self.nts_df, self.results)

        print("Overall Aggregate stats")
        print("Generated")
        print(gen_stats)
        print("Real")
        print(real_stats)
        print("")

        for p in self.results["Purpouse"].unique():
            gen_stats, real_stats = self.get_agg_stats(self.nts_df[self.nts_df["TripPurpose_B01ID"]==p], self.results[self.results["Purpouse"]==p])
            print(f"Aggregate stats for purpouse=={p}")
            print("Generated")
            print(gen_stats)
            print("Real")
            print(real_stats)
            print("")
    

    def create_histograms(self, variable_real, variable_gen, purpouse, percentile=None):
        """
        Create and display histograms for a given variable across real and generated data.

        Args:
            variable_real (str): Column name of the variable in the real dataset.
            variable_gen (str): Column name of the variable in the generated dataset.
            purpouse (int): Categorical purpose ID to filter by.
            percentile (float, optional): Maximum percentile threshold to filter outliers.
        """

        data_real = self.nts_df.copy()

        data_real["Duration"] = data_real["TripEnd"] - data_real["TripStart"]
        data_gen = self.results.copy()

        data_real = data_real[data_real[variable_real] >= 0]
        data_gen = data_gen[data_gen[variable_gen] >= 0]

        # filtering for purpouse
        data_real = data_real[data_real["TripPurpose_B01ID"]==purpouse]
        data_gen = data_gen[data_gen["Purpouse"]==purpouse]

        if percentile is not None:
            p = data_real[variable_real].quantile(percentile)

            data_real = data_real[data_real[variable_real]<=p]
            data_gen = data_gen[data_gen[variable_gen]<=p]

        median_real = data_real[variable_real].mean()
        median_gen = data_gen[variable_gen].mean()

        std_real = data_real[variable_real].std()
        std_gen = data_gen[variable_gen].std()

        plt.hist(data_real[variable_real], alpha=0.2, bins= 10, label="Real data")
        plt.hist(data_gen[variable_gen], bins= 10, alpha=0.7, label="Generated data")

        ymin, ymax = plt.ylim()
        xmin, xmax = plt.xlim()


        plt.text(x=xmax/50, y=2*ymax/3, s =f"Median (Real): {median_real:.2f}\nMedian (Gen): {median_gen:.2f}\nStd (Real): {std_real:.2f}\nStd (Gen): {std_gen:.2f}")

        plt.grid()

    def plot_histograms(self):
        rows = len(self.results["Purpouse"].unique())
        plt.figure(figsize=(15,10))

        for i,p in enumerate(self.results["Purpouse"].unique()):

            plt.subplot(rows,3,1 + 3*i)

            plt.title(f"TripStart for purpouse={p}")

            self.create_histograms(variable_real="TripStart", variable_gen="TripStart", purpouse=p)

            plt.tight_layout()

            plt.subplot(rows,3,2 + 3*i)

            plt.title(f"Duration for purpouse={p}")

            self.create_histograms(variable_real="Duration", variable_gen="Duration", purpouse=p, percentile=0.95)

            plt.tight_layout()

            plt.subplot(rows,3,3 + 3*i)

            plt.title(f"Distance for purpouse={p}")

            self.create_histograms(variable_real="TripDisExSW", variable_gen="Distance", purpouse=p, percentile=0.95)

            plt.tight_layout()

            plt.legend()
            

            plt.savefig(self.Plots_folder + "/Results_hist.pdf", format="pdf", bbox_inches="tight")
        

if __name__ == "__main__":

    print("...")