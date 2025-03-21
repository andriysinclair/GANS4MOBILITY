import torch
import torch.nn as nn
import pickle

#Load tensors
with open("/home/trapfishscott/Cambridge24.25/D200_ML_econ/ProblemSets/Project/tensors/tensors.pkl", "rb") as f:
    (X, y_cont, y_cat) = pickle.load(f)

X = X.to(torch.float32)
y_cont = y_cont.to(torch.float32)
y_cat = y_cat.to(torch.long)

print(f"Input shape: {X.shape}")
print(f"Cont Output shape: {y_cont.shape}")
print(f"Cat Output shape: {y_cat.shape}")


# Defining parameters
INPUT_SIZE = X.shape[3]
HIDDEN_SIZE = 9
NUM_LAYERS = 1
OUTPUT_SIZE_CONT = y_cont.shape[2]
OUTPUT_SIZE_CAT = y_cat.shape[2]

NUM_CLASSES = 24
print(NUM_CLASSES)



class TravelNet(nn.Module):
    def __init__(self):
        super().__init__()

        # Define RNN layer

        self.rnn = nn.RNN(INPUT_SIZE, HIDDEN_SIZE)

        # Output layers

        self.output_cont = nn.Linear(HIDDEN_SIZE, OUTPUT_SIZE_CONT)
        self.output_cat = nn.Linear(HIDDEN_SIZE, OUTPUT_SIZE_CAT)


    def forward(self, X):

        out, hh = self.rnn(X)

        y_cont_hat_vector = self.output_cont(hh)

        y_cat_hat= self.output_cat(hh)

        y_cat_hat = y_cat_hat.permute(0,2,1)
        y_cat_hat = torch.cat([y_cat_hat]*NUM_CLASSES, dim=2)
        y_cat_hat = y_cat_hat.reshape(OUTPUT_SIZE_CAT, NUM_CLASSES)

        # stacking downward NUM_CLASSES times
        #y_cat_hat = y_cat_hat.repeat()
        #print(y_cat_hat)

        y_cont_hat = y_cont_hat_vector[0,0,:]

        y_cont_hat = y_cont_hat.to(torch.float32)
        y_purpouse_pred = y_cat_hat.to(torch.float32)

        # appplying relu so that continous values are non-negative and maxed at 1
        y_tripstart_pred = torch.clamp(y_cont_hat[:10], min=0, max=1)
        y_tripend_pred = torch.clamp(y_cont_hat[10:20], min=0, max=1)
        y_distance_pred = torch.clamp(y_cont_hat[20::], min=0)

        # Applying a m

        return y_tripstart_pred, y_tripend_pred, y_distance_pred, y_purpouse_pred