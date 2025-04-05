import numpy as np
from sklearn.preprocessing import OneHotEncoder
import pandas as pd

# Cyclical encoder

def apply_cyclical_encoding(column, type_, max_val):
    """
    apply_cyclical_encoding 

    Applies cyclical encoding

    Args:
        column (str): col to encode
        type_ (str): cos or sine, need one of each for cyclical encoding
        max_val (int): largest value in the column

    Returns:
        int: returns the cyclical encoding for a numper, usually used with pd.apply
    """    

    if type_ == "cos":
        return np.cos(2 * np.pi * column/ max_val)
    else:
        return np.sin(2 * np.pi * column/ max_val)


def custom_numerical_scaler(x, x_min, x_max, inverse=False):
    """
    custom_numerical_scaler 

    Designed to scale TripStart and TripEnd

    Args:
        x (int): column entry
        x_min (int): minimum value, usually 0  
        x_max (int): maximum, usually 60*24
        inverse (bool, optional): inverse. Defaults to False.

    Returns:
        int: to be used with pd.apply
    """    
    if not inverse:
        x_scaled = (x-x_min)/(x_max - x_min)
        return x_scaled
    else:
        x_unscaled = x*(x_max - x_min) + x_min
        return x_unscaled
    

def log_transformer(x, inverse=False):
    if not inverse:
        return np.log1p(x)
    else:
        return np.expm1(x)
    

def return_correlated_columns(df, ro, outcome_col="TripPurpose_B01ID"):

    df = df.copy()
    df = df.drop(columns=["TripPurpose_B02ID", "TripPurpose_B04ID"], axis=1, errors="ignore")

    if outcome_col == "TripPurpose_B01ID":

        one_hot_columns = ["TripPurpose_B01ID"]
        

        # Apply one-hot to categorical
        ohe = OneHotEncoder(sparse_output=False)

        # Careful not to run twice

        for col in one_hot_columns:
            df[col] = df[col].astype(int)

        ohe.fit_transform(df[one_hot_columns])


        ohe_array = ohe.transform(df[one_hot_columns])
        ohe_df = pd.DataFrame(ohe_array, columns=ohe.get_feature_names_out(one_hot_columns))

        # Reset index to avoid misalignment
        df.reset_index(drop=True, inplace=True)
        ohe_df.reset_index(drop=True, inplace=True)

        full_df = pd.concat([df, ohe_df], axis=1)

        after_one_hot_columns = ohe.get_feature_names_out(one_hot_columns)

        corr_matrix = full_df.corr()

        outcome_cols = corr_matrix[after_one_hot_columns].columns

    else:
        df = df.drop(columns=["TripPurpose_B01ID"], axis=1)
        
        outcome_cols = [outcome_col]

        corr_matrix = df.corr()

    useful_columns = []

    for col in outcome_cols:
        #print(col)
        col_ascending = corr_matrix[col].abs().sort_values(ascending=False)
        
        for key, value in col_ascending.items():
            if value > ro:
                #print(f"{key:<25} {value}")
                if key not in useful_columns and key not in outcome_cols and key != 'TripPurpose_B01ID':
                    useful_columns.append(key)

    print(f"Number of useful columns found: {len(useful_columns)}")

    return useful_columns
    