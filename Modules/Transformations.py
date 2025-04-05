import numpy as np
from sklearn.preprocessing import OneHotEncoder
import pandas as pd

# Cyclical encoder

def apply_cyclical_encoding(column, type_, max_val):
    """
    Apply cyclical encoding to a time-based or periodic variable.

    Converts a scalar value into its cyclical representation using sine or cosine transformation.
    Commonly used for features like hours of the day or days of the week.

    Args:
        column (float): Value to encode.
        type_ (str): Type of transformation to apply. Must be either "cos" or "sine".
        max_val (int): Maximum value of the cycle (e.g., 24 for hours in a day).

    Returns:
        float: Transformed cyclical value.
    """

    if type_ == "cos":
        return np.cos(2 * np.pi * column/ max_val)
    else:
        return np.sin(2 * np.pi * column/ max_val)


def custom_numerical_scaler(x, x_min, x_max, inverse=False):
    """
    Scale or unscale a numerical value to the [0, 1] range.

    Designed for values like TripStart or TripEnd time in minutes. Supports inverse scaling.

    Args:
        x (float): Value to scale or unscale.
        x_min (float): Minimum value in the original scale.
        x_max (float): Maximum value in the original scale.
        inverse (bool, optional): Whether to apply inverse transformation. Defaults to False.

    Returns:
        float: Scaled or unscaled value.
    """
    if not inverse:
        x_scaled = (x-x_min)/(x_max - x_min)
        return x_scaled
    else:
        x_unscaled = x*(x_max - x_min) + x_min
        return x_unscaled
    

def log_transformer(x, inverse=False):
    """
    Apply log or exponential transformation.

    Transforms a non-negative variable using log(1 + x), or reverses it using exp(x) - 1.

    Args:
        x (float): Value to transform.
        inverse (bool, optional): Whether to apply the inverse transformation. Defaults to False.

    Returns:
        float: Transformed or inverse-transformed value.
    """
    if not inverse:
        return np.log1p(x)
    else:
        return np.expm1(x)
    

def return_correlated_columns(df, ro, outcome_col="TripPurpose_B01ID"):
    """
    Identify columns that are correlated with a specified outcome variable.

    Optionally applies one-hot encoding to categorical outcomes before computing correlation.
    Returns a list of numerical features that exceed the given correlation threshold.

    Args:
        df (pd.DataFrame): Input DataFrame.
        ro (float): Correlation threshold (absolute value).
        outcome_col (str, optional): Name of the target column. Defaults to "TripPurpose_B01ID".

    Returns:
        List[str]: List of column names with absolute correlation above threshold.
    """

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
    