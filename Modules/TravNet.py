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



# Define Max Journey Sequence

max_journey_seq = 10

#Load tensors #TODO Get a better file path to your tensors

with open("/home/trapfishscott/Cambridge24.25/D200_ML_econ/ProblemSets/Project/tensors/tensors.pkl", "rb") as f:
    (X, y_cont_raw, y_cat_raw) = pickle.load(f)

# GPU?

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Training running on: {device}")

# Total input tensor

X = X.to(torch.float32).to(device)

# Targets only

y_ts_te = X[:,7:,:,:2*max_journey_seq].to(torch.float32).to(device)
y_distance = X[:,7:,:,2*max_journey_seq:3*max_journey_seq].to(torch.float32).to(device)
y_purpouse = X[:,7:,:,3*max_journey_seq:4*max_journey_seq].to(torch.long).to(device)
y_istrip = X[:,7:,:,4*max_journey_seq:5*max_journey_seq].to(torch.float32).to(device)

y_ts = y_ts_te[:,:,:,0::2]
y_te = y_ts_te[:,:,:,1::2]

y_duration = y_te - y_ts

# Removing 

# Defining parameters

NUM_CLASSES = 24            # Number of purpouse classes
INPUT_SIZE = X.shape[3] + NUM_CLASSES*max_journey_seq - max_journey_seq   # Input size, *purpouse will be fed back as softmax
HIDDEN_SIZE = 9
NUM_LAYERS = 1
OUTPUT_SIZE_TS = y_ts.shape[3]
OUTPUT_SIZE_DUR = y_duration.shape[3]
OUTPUT_SIZE_DIST = y_distance.shape[3]     # Output Size of distance
OUTPUT_SIZE_PURP = y_purpouse.shape[3]      # 10 outcome vars
OUTPUT_SIZE_ISTRIP = y_istrip.shape[3]      # 10 binary vars vars

normalised_mp_minute = log_transformer(1.1667)  # *normalised minute


class RNNmodel(nn.Module):
    def __init__(self):
        super().__init__()

        # Define RNN layer

        self.rnn = nn.RNN(INPUT_SIZE, HIDDEN_SIZE, num_layers=NUM_LAYERS)

        # Output layers

        self.output_ts = nn.Linear(HIDDEN_SIZE, OUTPUT_SIZE_TS)
        self.output_dur = nn.Linear(HIDDEN_SIZE, OUTPUT_SIZE_DUR)
        self.output_dist = nn.Linear(HIDDEN_SIZE, OUTPUT_SIZE_DIST)
        self.output_purp = nn.Linear(HIDDEN_SIZE, OUTPUT_SIZE_PURP*NUM_CLASSES)
        self.output_bin = nn.Linear(HIDDEN_SIZE, OUTPUT_SIZE_ISTRIP)

        self.T = 1.0  # Can be used to scale categorical logits - useful for class imbalance


    def forward(self, X):

        out, hh = self.rnn(X)
        #print(f"hh_shape: {hh.shape}")

        y_ts_pred = self.output_ts(out[-1]).reshape(1,1,OUTPUT_SIZE_TS)
        y_dur_pred = self.output_dur(out[-1]).reshape(1,1,OUTPUT_SIZE_DUR)
        y_dist_pred = self.output_dist(out[-1]).reshape(1,1,OUTPUT_SIZE_DIST)
        y_purp_pred = self.output_purp(out[-1]).reshape(OUTPUT_SIZE_PURP, NUM_CLASSES)
        y_istrip_pred = self.output_bin(out[-1]).reshape(OUTPUT_SIZE_ISTRIP)

        y_istrip_pred = y_istrip_pred.to(torch.float32)

        # Applying sgmoid to get the loss mask
        y_istrip_pred_sgmoid = F.sigmoid(y_istrip_pred)

        # Making a probabilistic rather than deterministic mask
        y_mask = torch.bernoulli(y_istrip_pred_sgmoid).unsqueeze(1)

        # Adding softplus to ensure monotonicity in start and end time
        y_ts_pred = F.sigmoid(y_ts_pred)
        y_dur_pred = F.sigmoid(y_dur_pred)

        # Softplus to ensure a postive distance
        y_distance_pred = F.softplus(y_dist_pred)
        
        return y_ts_pred, y_dur_pred, y_distance_pred, y_purp_pred, y_istrip_pred, y_mask
### Loading tensors

# Define Max Journey Sequence

max_journey_seq = 10

#Load tensors #TODO Get a better file path to your tensors

with open("/home/trapfishscott/Cambridge24.25/D200_ML_econ/ProblemSets/Project/tensors/tensors.pkl", "rb") as f:
    (X, y_cont_raw, y_cat_raw) = pickle.load(f)

# GPU?

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Training running on: {device}")

# Total input tensor

X = X.to(torch.float32).to(device)

# Targets only

y_ts_te = X[:,7:,:,:2*max_journey_seq].to(torch.float32).to(device)
y_distance = X[:,7:,:,2*max_journey_seq:3*max_journey_seq].to(torch.float32).to(device)
y_purpouse = X[:,7:,:,3*max_journey_seq:4*max_journey_seq].to(torch.long).to(device)
y_istrip = X[:,7:,:,4*max_journey_seq:5*max_journey_seq].to(torch.float32).to(device)

y_ts = y_ts_te[:,:,:,0::2]
y_te = y_ts_te[:,:,:,1::2]

y_duration = y_te - y_ts

### Adding weights for imbalanced categories

# Calculating probability weights for categorical

def return_categorical_weightings(array, num_cats = 24):
    unique_vals, counts = torch.unique(array, return_counts=True)

    # 17 is missing this refers to "short walk", not relevant for our data, but will be kept for reference

    ce_weighting = torch.zeros(num_cats, dtype=torch.float32)
    
    if num_cats == 24:
        ce_weighting[17] = 0

    for val, count in zip(unique_vals, counts):
        ce_weighting[int(val)] = 1/(count/counts.sum())

    # Apply log scaling to smooth extreme weight differences
    ce_weighting = torch.log1p(ce_weighting)  

    # Normalize the weights
    ce_weighting /= ce_weighting.sum()



    ce_weighting = ce_weighting.to(device)

    return ce_weighting

ce_weighting = return_categorical_weightings(y_purpouse[:,:,:])
binary_weightings = return_categorical_weightings(y_istrip[:,:,:], num_cats=2)

# Configure basic logging
logging.basicConfig(level=logging.INFO, force=True, format='%(levelname)s: %(message)s')

def train_TravNet(i_to_loop,
                  rnn_model=RNNmodel(),
                  ce_weighting=ce_weighting,
                  device=device,
                  epochs=1,
                  make_travel_diaries=True,
                  X=X,
                  y_ts=y_ts,
                  y_duration=y_duration,
                  y_distance=y_distance,
                  y_purpouse=y_purpouse,
                  y_istrip = y_istrip):

    rnn_model = rnn_model.to(device)

    seq_length=7

    ce_loss = nn.CrossEntropyLoss(weight=ce_weighting, ignore_index=0).to(device)  #(y_hat, y)
    mse_loss = nn.MSELoss(reduction="none").to(device)
    bce_loss = nn.BCEWithLogitsLoss().to(device)

    optimizer = torch.optim.Adam(rnn_model.parameters(), lr=0.0005)

    ts_loss_weight = 1/800
    dur_loss_weight = 10
    purpouse_loss_weight = 1

    entropy_weight = 2
    temperature = 5
    tau = 2

    complete_travel_diaries = {}
    complete_travel_diaries_dfs = []

    total_losses = []
    ts_losses= []
    dur_losses= []
    distance_losses = []
    purpouse_losses = []
    istrip_losses = []
    entropy_evol = []


    for epochi in range(epochs):

        for individual_i in range(i_to_loop):

            # Setting up prediction matrices for individual's travel week
            prediction_matrix_ts = []
            prediction_matrix_dur = []
            prediction_matrix_distance = []
            prediction_matrix_purpouse = []
            prediction_matrix_istrip = []

            masks = []

            entropies = []

            # Truth values for that individual
            true_matrix_ts = y_ts[individual_i, :, 0, :].to(torch.float32)
            true_matrix_dur = y_duration[individual_i, :, 0, :].to(torch.float32)
            true_matrix_distance = y_distance[individual_i, :, 0, :].to(torch.float32)
            true_matrix_purpouse = y_purpouse[individual_i, :, 0, :].reshape(7*max_journey_seq).to(torch.long)
            true_matrix_istrip = y_istrip[individual_i, :, 0, :].to(torch.float32)

            logging.debug(f"Matrices of true values shape,ts_te, distance, purpouse, istrip")
            logging.debug(f"{true_matrix_ts.shape}")
            logging.debug(f"{true_matrix_dur.shape}")
            logging.debug(f"{true_matrix_distance.shape}")
            logging.debug(f"{true_matrix_purpouse.shape}")
            logging.debug(f"{true_matrix_istrip.shape}")

            travel_diary = X[individual_i, :, 0, :].unsqueeze(1).to(torch.float32)
            logging.debug(travel_diary.shape)

            # 7-day mask containing just the time-independent features
            X_features = X[individual_i, :, 0, 5*max_journey_seq:].unsqueeze(1).to(torch.float32)   # Features with masking and after masking
            ts_feature = torch.zeros(7,1,max_journey_seq).to(torch.float32).to(device)
            dur_feature = torch.zeros(7,1,max_journey_seq).to(torch.float32).to(device)
            distance_feature = torch.zeros(7,1,max_journey_seq).to(torch.float32).to(device)
            purpouse_feature = torch.zeros(7,1,max_journey_seq*24).to(torch.float32).to(device)
            istrip_feature = torch.zeros(7,1,max_journey_seq).to(torch.float32).to(device)
            
            # Input matrix to feed into the model
            input_matrix = torch.concat([ts_feature, dur_feature, distance_feature, purpouse_feature, istrip_feature, X_features[:7,:,:]], dim=2)

            logging.debug(f"Shape of input matrix: {input_matrix.shape}")

            for t in range(0, seq_length):

                # Conditions of violation
        
                y_ts_pred_7, y_dur_pred_7, y_distance_pred_7, y_purpouse_pred_7, y_istrip_pred_7, is_trip_mask = rnn_model.forward(input_matrix)
                categorical_prediction7 = torch.argmax(y_purpouse_pred_7, dim=-1)

                # Reshaping
                y_purpouse_pred_7 = y_purpouse_pred_7.reshape(1,1,max_journey_seq*NUM_CLASSES)

                # Apply temperature-scaled softmax
                y_purpouse_softmax = F.softmax(y_purpouse_pred_7 / temperature, dim=-1)

                # Reshape back to match your input matrix (flattened)
                y_purpouse_softmax = y_purpouse_softmax.reshape(1, 1, max_journey_seq*NUM_CLASSES)

                # Calculating entropy across trip
                entropy_t = -(y_purpouse_softmax.reshape(max_journey_seq,NUM_CLASSES) * torch.log(y_purpouse_softmax + 1e-8).reshape(max_journey_seq,NUM_CLASSES)).sum(dim=1)
                entropies.append(entropy_t)

                if individual_i > i_to_loop-5:
                    logging.debug(y_purpouse_softmax.reshape(max_journey_seq,NUM_CLASSES))

                y_istrip_pred_7 = y_istrip_pred_7.reshape(1,1,-1)

                # New input row

                new_input_row = torch.concat([y_ts_pred_7, 
                                            y_dur_pred_7,
                                            y_distance_pred_7, 
                                            y_purpouse_softmax, 
                                            y_istrip_pred_7,
                                            X_features[seq_length+t,:,:].unsqueeze(1)], dim=2)

                # Appeding to input matrix

                # Slice off the first row of input_matrix

                input_matrix = input_matrix[1:,:,:]

                logging.debug(f"sliced of bottom row of input matrix, shape: {input_matrix.shape}")

                # Concatenate the redictions for t

                input_matrix = torch.cat([new_input_row, input_matrix], dim=0)

                logging.debug(f"New input matrix is input for next day: {input_matrix.shape}")

                logging.debug(input_matrix)

                # Appending to final prediction matrix
                prediction_matrix_ts.append(y_ts_pred_7.reshape(1,-1))
                prediction_matrix_dur.append(y_dur_pred_7.reshape(1,-1))            
                prediction_matrix_distance.append(y_distance_pred_7.reshape(1,-1))
                prediction_matrix_purpouse.append(y_purpouse_pred_7.reshape(max_journey_seq,NUM_CLASSES))
                prediction_matrix_istrip.append(y_istrip_pred_7.reshape(1,-1))

                # Old mask code

                #masks_ts_te.append(is_trip_mask.repeat_interleave(2).view(1,-1))
                masks.append(is_trip_mask.T)

            # Concatenating matrices

            masks = torch.cat(masks, 0)

            purpouse_mask = torch.clamp(true_matrix_purpouse.reshape(7,max_journey_seq),max=1)

            prediction_matrix_ts = torch.cat(prediction_matrix_ts, 0)
            prediction_matrix_dur = torch.cat(prediction_matrix_dur, 0)
            prediction_matrix_distance = torch.cat(prediction_matrix_distance, 0)
            prediction_matrix_purpouse = torch.cat(prediction_matrix_purpouse, 0) 
            prediction_matrix_istrip = torch.cat(prediction_matrix_istrip, 0)

            if make_travel_diaries:
                purpouse_for_show = torch.argmax(prediction_matrix_purpouse, dim=1).reshape(7,max_journey_seq)

                matrix_for_show = torch.cat([prediction_matrix_ts * masks, 
                                            prediction_matrix_dur * masks, 
                                            prediction_matrix_distance * masks, 
                                            purpouse_for_show * masks, 
                                            masks], dim=1)
                
                complete_travel_diaries[individual_i] = matrix_for_show

                df = pd.DataFrame(complete_travel_diaries[individual_i].cpu().detach().numpy())
                df.iloc[:,:2*max_journey_seq] = df.iloc[:,:2*max_journey_seq].apply(lambda x: custom_numerical_scaler(x=x, x_min=0, x_max=60*24, inverse=True))
                df.iloc[:,2*max_journey_seq:3*max_journey_seq] = df.iloc[:,2*max_journey_seq:3*max_journey_seq].apply(lambda x: log_transformer(x, inverse=True))
                df["i_id"] = individual_i

                complete_travel_diaries_dfs.append(df)

            # Calculating loss with masks applied
            logging.debug(f"Matrices prediction/ true in usual order prior to adding to losses")

            logging.debug(f"Prediction of TS")
            logging.debug(prediction_matrix_ts*purpouse_mask)
            logging.debug("")
            logging.debug(true_matrix_ts)

            ts_loss = mse_loss(prediction_matrix_ts,  true_matrix_ts)
            ts_loss = (ts_loss*purpouse_mask).sum() / (purpouse_mask.sum() + 1e-6)

            logging.debug(f"Prediction of DUR")
            logging.debug(prediction_matrix_dur*purpouse_mask)
            logging.debug("")
            logging.debug(true_matrix_dur)

            dur_loss = mse_loss(prediction_matrix_dur,  true_matrix_dur)
            dur_loss = (dur_loss*purpouse_mask).sum() / (purpouse_mask.sum() + 1e-6)

            logging.debug(f"Prediction of Distance")
            logging.debug(prediction_matrix_distance*purpouse_mask)
            logging.debug("")
            logging.debug(true_matrix_distance)

            distance_loss = mse_loss(torch.log(1+prediction_matrix_distance),  torch.log(1+true_matrix_distance) )
            distance_loss = (distance_loss*purpouse_mask).sum() / (purpouse_mask.sum() + 1e-6)

            logging.debug(f"Prediction of Purpouse")
            logging.debug(purpouse_for_show)
            logging.debug("")
            logging.debug(true_matrix_purpouse)

            purpouse_loss = ce_loss(prediction_matrix_purpouse, true_matrix_purpouse)

            logging.debug(f"Prediction of Is trip")
            logging.debug(prediction_matrix_istrip)
            logging.debug("")
            logging.debug(true_matrix_istrip)

            istrip_loss = bce_loss(prediction_matrix_istrip, true_matrix_istrip)

            # RATIONALITY CONDITIONS

            # Encouraging entropy
            purpouse_entropy = torch.stack(entropies, dim=0).mean()

            total_loss = ts_loss*ts_loss_weight + dur_loss*dur_loss_weight + distance_loss + purpouse_loss*purpouse_loss_weight + istrip_loss  + entropy_weight*purpouse_entropy

            logging.debug(f"ts_te Loss requires grad? {ts_loss.requires_grad}")
            logging.debug(f"Distance Loss requires grad? {distance_loss.requires_grad}")
            logging.debug(f"Purpouse Loss requires grad? {purpouse_loss.requires_grad}")
            logging.debug(f"is trip Loss requires grad? {istrip_loss.requires_grad}")

            logging.info(f"epoch: {epochi} | individual: {individual_i+1}")
            logging.info(f"total_loss: {total_loss:.2f}")

            # Appending losses
            total_losses.append(total_loss.cpu().detach().numpy())
            ts_losses.append(  (ts_loss*ts_loss_weight).cpu().detach().numpy())
            dur_losses.append(  (dur_loss*dur_loss_weight).cpu().detach().numpy())
            distance_losses.append(distance_loss.cpu().detach().numpy())
            purpouse_losses.append((purpouse_loss*purpouse_loss_weight).cpu().detach().numpy())
            istrip_losses.append(istrip_loss.cpu().detach().numpy())
            entropy_evol.append(purpouse_entropy.cpu().detach().numpy())

            # BACKPROP
            optimizer.zero_grad()
            total_loss.backward()
            optimizer.step()

    plt.figure(figsize=(15, 10))

    plt.suptitle("Plots Showing 1-day ahead and 7-day Losses for all outcome variables.")

    x_vals = np.arange(i_to_loop)  # Ensure correct x-axis scaling




    plt.subplot(6,1,1)
    plt.title("TS loss")
    plt.plot(x_vals, ts_losses, label="ts_te")
    plt.grid()

    plt.subplot(6,1,2)
    plt.title("Dur loss")
    plt.plot(x_vals, dur_losses, label="ts_te")
    plt.grid()


    plt.subplot(6,1,3)
    plt.title("Distance loss")
    plt.plot(x_vals, distance_losses, label="distance")
    plt.grid()

    plt.subplot(6,1,4)
    plt.title("Purpouse loss")
    plt.plot(x_vals, purpouse_losses, label="purpouse")
    plt.grid()

    plt.subplot(6,1,5)
    plt.title("Entropy")
    plt.plot(x_vals, entropy_evol, label="entropy")
    plt.grid()


    plt.subplot(6,1,6)
    plt.title("Total loss")
    plt.plot(x_vals, total_losses)

    plt.grid()

    plt.tight_layout()

    plt.savefig("/home/trapfishscott/Cambridge24.25/D200_ML_econ/ProblemSets/Project/Plots/Losses.png")

    # Creating Travel DFs

    full_df = pd.concat(complete_travel_diaries_dfs)

    stub_cols = ["TripStart", "Duration", "Distance", "Purpouse", "IsTrip"]
    target_cols = [f"{col}_{i}" for col in stub_cols for i in range(1, max_journey_seq+1)]

    target_cols+= ["i_id"]

    full_df.columns = target_cols

    full_df = full_df.reset_index()

    full_df = full_df.rename(columns={"index": "DoW"})

    long_full_df = pd.wide_to_long(full_df, stubnames=stub_cols, i=["i_id", "DoW"], sep="_", j="TripNum").reset_index()

    long_full_df = long_full_df[long_full_df["Purpouse"] != 0]

    # Return fully trained model wide and long complete travel dfs

    return rnn_model, full_df, long_full_df


if  __name__ == "__main__":

    # Checking shape of tensors

    show_config = False

    if show_config:

        index = 7

        print(f"X: {X.shape}")

        print(f"y_istrip: {y_istrip.shape}")
        print(y_istrip[index,0,:,:])

        print(f"y_ts: {y_ts.shape}")
        print(y_ts[index,0,:,:])

        print(f"y_te: {y_te.shape}")
        print(y_te[index,0,:,:])

        print(f"y_distance: {y_distance.shape}")
        print(y_distance[index,0,:,:])

        print(y_purpouse[index,0,:,:])
        print(f"y_purpouse: {y_purpouse.shape}")

        print(f"y_istrip: {y_istrip.shape}")
        print(y_istrip[index,0,:,:].shape)

        print(f"y_duration: {y_duration.shape}")
        print(y_duration[index,0,:,:])

        print("Final CE Weights:", ce_weighting)

        # Printing model parameters

        model = RNNmodel()

        for a,b in model.named_parameters():
            print(a,b.shape)

        # Configure basic logging
        logging.basicConfig(level=logging.INFO, force=True, format='%(levelname)s: %(message)s')

    # Train NN

    TravNet, wide_diaries, long_diaries = train_TravNet(i_to_loop=X.shape[0])  # Training on all samples

    # TODO absolute Models folder

    Modules_folder = "/home/trapfishscott/Cambridge24.25/D200_ML_econ/ProblemSets/Project/Models"

    with open(Modules_folder + "/TravNet.pkl", "wb") as f:
        pickle.dump(TravNet, f)

    with open(Modules_folder + "/wide_diaries.pkl", "wb") as f:
        pickle.dump(wide_diaries, f)

    with open(Modules_folder + "/long_diaries.pkl", "wb") as f:
        pickle.dump(long_diaries, f)