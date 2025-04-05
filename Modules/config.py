import os
from sklearn.preprocessing import OneHotEncoder, MinMaxScaler

# Absolute Paths

root_folder = os.getcwd()
data_folder = root_folder + "/data"
Models_folder = root_folder + "/Models"
tensors_folder = root_folder + "/tensors"
Results_folder = root_folder + "/Results"

# Variable Selection

numerical_outcome_vars = ["TripStart", "TripEnd", "TripDisExSW"]
categorical_outcome_vars = ["TripPurpose_B01ID", "IsTrip"]

extra_vars = ["IndividualID_x", "JourSeq"]
features_one_hot = ["PSUGOR_B02ID"]

features_numerical = [  "DrivLic_B02ID",
                        "VehAnMileage",
                        "HHoldEmploy_B01ID",
                        "VehComMile_B01ID",
                        "EcoStat_B02ID",
                        "HHoldNumPeople",
                        "WkPlace_B01ID",
                        "Age_B01ID",
                        "HHoldStruct_B02ID",
                        "HRPWorkStat_B02ID",
                        "HHoldNumChildren",
                        "EcoStat_B03ID",
                        "EducN_B01ID",
                        "TravelYear"]


features_cyclical = ["TWSMonth", "TravelWeekDay_B01ID"]

features = features_one_hot + features_numerical + features_cyclical
outcomes = numerical_outcome_vars + categorical_outcome_vars

# Scikit-learn scalers

standard_mms = MinMaxScaler()
ohe = OneHotEncoder(sparse_output=False)
