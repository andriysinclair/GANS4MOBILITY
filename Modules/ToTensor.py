import pandas as pd
import random
import sys
import torch
import numpy as np
from sklearn.preprocessing import OneHotEncoder, MinMaxScaler
import logging

def impute_missing_travel_week_for_i(i_df, 
                                     i_id, 
                                     full_week_encoding, 
                                     features,
                                     outcomes, 
                                     features_numerical,
                                     features_one_hot,
                                     extra_vars,
                                     cyclical_encoder,
                                     custom_numerical_scaler,
                                     log_transformer,
                                     scikit_minmax_scaler,
                                     scikit_onehot_scaler) -> pd.DataFrame:
    """
    Impute missing weekdays with no travel for a given individual.

    Adds rows with zero-valued outcome variables for weekdays on which the individual has no travel data.
    Also applies cyclical, log, and min-max scaling transformations to the resulting dataframe.

    Args:
        i_df (pd.DataFrame): Subset of the data for an individual.
        i_id (int): ID of the individual.
        full_week_encoding (set): Set of all days in the week (e.g., {1,2,...,7}).
        features (list): List of feature column names.
        outcomes (list): List of outcome column names.
        features_numerical (list): List of numerical feature names.
        features_one_hot (list): List of one-hot encoded feature names.
        extra_vars (list): List of extra variables not used in modeling.
        cyclical_encoder (function): Function to apply cyclical encoding.
        custom_numerical_scaler (function): Function to scale numerical values to [0,1].
        log_transformer (function): Function to apply log1p transform.
        scikit_minmax_scaler (MinMaxScaler): Pre-fitted MinMaxScaler.
        scikit_onehot_scaler (OneHotEncoder): Pre-fitted OneHotEncoder.

    Returns:
        pd.DataFrame: Transformed and imputed dataframe for the individual.
    """
 
    break_flag = False

    # Travel days with travel 
    included_travel_day = i_df["TravelWeekDay_B01ID"].to_list()

    # Travel days with no travel
    travel_day_no_drive = list(set(full_week_encoding) - set(included_travel_day))

    # These values will repeat for empty-travel travel days
    imputed_travel_df = pd.DataFrame({
        "TravelWeekDay_B01ID": travel_day_no_drive,
        "IndividualID_x": [i_id]*len(travel_day_no_drive),
        "JourSeq": [1]*len(travel_day_no_drive)
    })


    
    # Looping through all the columns in the original df
    for col in i_df.columns:

        # For days with no travel all outcomes vars will take 0
        if col in outcomes:
            imputed_travel_df[col] = [0]*len(travel_day_no_drive)

    
        else:
        
            if col not in extra_vars + ["TravelWeekDay_B01ID"]:
                if len(i_df[col].unique()) != 1:
                    print(f"{col} is erroneous for {i_id}")
                    print(f"Unique vals: {i_df[col].unique()}")
                    break_flag = True
                    break
                else:
                    imputed_travel_df[col] = i_df[col].unique()[0]

    if break_flag:
        print("Continuing to next individual")
        return
    

    #display(imputed_travel_df)
    #display(i_df)

    # Merging on IndividualID_x and TravelWeekDay_B01ID
    #full_df = i_df.merge(imputed_travel_df, on=["IndividualID_x", "TravelWeekDay_B01ID"], how="left")

    # Concatenating df to include empty travel days
    full_df = pd.concat([i_df, imputed_travel_df])
    #display(full_df)

    full_df = full_df.sort_values(["TravelYear", "TWSMonth", "TravelWeekDay_B01ID", "JourSeq", "TripStart", "TripEnd"])

    full_df.loc[:, "TWSMonth_cos"] = cyclical_encoder(column=full_df["TWSMonth"], type_="cos", max_val=12)
    full_df.loc[:, "TWSMonth_sin"] = cyclical_encoder(column=full_df["TWSMonth"], type_="sin", max_val=12)

    full_df.loc[:, "TravelWeekDay_B01ID_cos"] = cyclical_encoder(column=full_df["TravelWeekDay_B01ID"], type_="cos", max_val=7)
    full_df.loc[:, "TravelWeekDay_B01ID_sin"] = cyclical_encoder(column=full_df["TravelWeekDay_B01ID"], type_="sin", max_val=7)

    if full_df["TripStart"].max() > 1.5:
        full_df.loc[:,"TripStart"] = full_df["TripStart"].apply(lambda x: custom_numerical_scaler(x, x_max=60*24, x_min=0))

    if full_df["TripEnd"].max() > 1.5:
        full_df.loc[:,"TripEnd"] = full_df["TripEnd"].apply(lambda x: custom_numerical_scaler(x, x_max=60*24, x_min=0))

    full_df.loc[:,"TripDisExSW"] = full_df.loc[:,"TripDisExSW"].apply(lambda x: log_transformer(x))

    full_df.loc[:,features_numerical] = scikit_minmax_scaler.transform(full_df[features_numerical])

    ohe_array = scikit_onehot_scaler.transform(full_df[features_one_hot])
    ohe_df = pd.DataFrame(ohe_array, columns=scikit_onehot_scaler.get_feature_names_out(features_one_hot))

    # Reset index to avoid misalignment
    full_df.reset_index(drop=True, inplace=True)
    ohe_df.reset_index(drop=True, inplace=True)

    full_df = pd.concat([full_df, ohe_df], axis=1)

    #display(full_df)


    return full_df


def transform_to_wide_for_i(i_df, 
                            i_id, 
                            full_week_encoding, 
                            features,
                            outcomes, 
                            categorical_outcome_vars,
                            features_numerical,
                            features_cyclical,
                            features_one_hot,
                            extra_vars,
                            cyclical_encoder,
                            custom_numerical_scaler,
                            log_transformer,
                            scikit_minmax_scaler,
                            scikit_onehot_scaler,
                            max_journey_seq=10,
                            seq_length=7) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Transform a long-format individual dataframe into wide format.

    Pivot the individual's dataframe so each row represents one day of the week, with journey features
    encoded in wide format. Also separates continuous and categorical outcomes for modeling.

    Args:
        i_df (pd.DataFrame): Subset of the data for an individual.
        i_id (int): ID of the individual.
        full_week_encoding (set): Set of all weekdays (e.g., {1,2,...,7}).
        features (list): Input feature columns.
        outcomes (list): Target variable columns.
        categorical_outcome_vars (list): List of categorical target column names.
        features_numerical (list): Numerical input feature columns.
        features_cyclical (list): Names of cyclical feature columns.
        features_one_hot (list): One-hot encoded feature columns.
        extra_vars (list): Extra non-modeled columns to remove.
        cyclical_encoder (function): Function to encode cyclical features.
        custom_numerical_scaler (function): Function to scale numerical features.
        log_transformer (function): Function to log-transform distance features.
        scikit_minmax_scaler (MinMaxScaler): Fitted scaler for numerical features.
        scikit_onehot_scaler (OneHotEncoder): Fitted scaler for one-hot features.
        max_journey_seq (int, optional): Max journeys per day. Defaults to 10.
        seq_length (int, optional): Sequence length (usually 7 for weekdays). Defaults to 7.

    Returns:
        Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]: 
            - Full wide-format dataframe.
            - Continuous target columns.
            - Categorical target columns.
    """

    df = i_df.copy()

    # Making column names based off all outcome categories

    TS_TE = []
    # TripStart TripEnd
    for TS,TE in zip(["TripStart"], ["TripEnd"]):
        for i in range(1, max_journey_seq+1):
            TS_TE.append(f"{TS}_{i}")
            TS_TE.append(f"{TE}_{i}")

    #print(TS_TE)

    expected_all = [f"{col}_{i}" for col in ["TripDisExSW", "TripPurpose_B01ID", "IsTrip"] for i in range(1, max_journey_seq+1)]

    expected_all = TS_TE + expected_all
    #print(expected_all)

    # Making column names for categorical outcomes only
    expected_categorical = [f"{col}_{i}" for col in categorical_outcome_vars for i in range(1, max_journey_seq+1)]

    # Applying max daily journeys cut off
    df = df[df["JourSeq"]<=max_journey_seq]

    # Transforming to wide
    df_wide = df.pivot(index="TravelWeekDay_B01ID",
                  columns = "JourSeq",
                  values = outcomes)
    
    df_wide.columns = [f"{col[0]}_{int(col[1])}" for col in df_wide.columns]

    for col in expected_all:
        if col not in df_wide.columns:
            df_wide[col] = 0
    
    # Ensure column order is consistent
    df_wide = df_wide[expected_all]
    
    df_wide = df_wide.fillna(0)

    df_wide.reset_index(inplace=True)

    # Dropping outcome columns
    df.drop(columns=outcomes + extra_vars, axis=1, inplace = True)
    df.drop_duplicates(subset=["TravelWeekDay_B01ID"], inplace=True)

    df_wide = df_wide.merge(df, on="TravelWeekDay_B01ID", how="left")

    top_row = df_wide.head(1).copy()

    for col in expected_all:
        top_row[col] = 0
        top_row["TravelWeekDay_B01ID"] = 0

    repeated_rows = pd.concat([top_row] * seq_length, ignore_index=True)

    df_wide = pd.concat([repeated_rows, df_wide], ignore_index=True)

    df_wide.drop(columns=features_one_hot + features_cyclical, inplace=True, axis=1)

    #df_wide.drop(columns=features_cyclical + features_one_hot, axis=1, inplace=True)

    targets_only = df_wide.drop(columns=features + extra_vars, axis=1, errors="ignore")

    targets_only = targets_only.iloc[seq_length:,:]

    targets_cont = targets_only[expected_all]
    targets_cont = targets_cont.copy()
    targets_cont.drop(columns=expected_categorical, axis=1, inplace=True)


    targets_cat = targets_only[expected_categorical]

    return df_wide, targets_cont, targets_cat


def prepare_data_for_LSTM(long_df, 
                          features,
                          outcomes,
                          categorical_outcome_vars,
                          features_numerical,
                          features_cyclical,
                          features_one_hot,
                          extra_vars,
                          cyclical_encoder,
                          custom_numerical_scaler,
                          log_transformer,
                          scikit_minmax_scaler,
                          scikit_onehot_scaler,
                          impute_missing_travel_weeks=True, 
                          transform_to_wide=False, 
                          transform_to_tensor=False, 
                          debug=False,
                          max_journey_seq=10, 
                          seq_length = 7):# -> Index[str] | tuple[Tensor, Tensor, Tensor] | DataFrame | Any:# -> Index[str] | tuple[Tensor, Tensor, Tensor] | DataFrame | Any:# -> Index[str] | tuple[Tensor, Tensor, Tensor] | DataFrame | Any:# -> Index[str] | tuple[Tensor, Tensor, Tensor] | DataFrame | Any:# -> Index[str] | tuple[Tensor, Tensor, Tensor] | DataFrame | Any:# -> Index[str] | tuple[Tensor, Tensor, Tensor] | DataFrame | Any:# -> Index[str] | tuple[Tensor, Tensor, Tensor] | DataFrame | Any:# -> Index[str] | tuple[Tensor, Tensor, Tensor] | DataFrame | Any:# -> Index[str] | tuple[Tensor, Tensor, Tensor] | DataFrame | Any:# -> Index[str] | tuple[Tensor, Tensor, Tensor] | DataFrame | Any:
    """
    Prepare input features and targets for LSTM training from long-format travel data.

    Applies imputation, wide-format transformation, feature engineering, and tensor conversion
    on a per-individual basis. Optionally returns debug information or intermediate data.

    Args:
        long_df (pd.DataFrame): Full long-format dataset (filtered for relevant features).
        features (list): Feature column names.
        outcomes (list): Target column names.
        categorical_outcome_vars (list): List of categorical target column names.
        features_numerical (list): Numerical input features.
        features_cyclical (list): Cyclical input features.
        features_one_hot (list): One-hot encoded input features.
        extra_vars (list): Extra variables to be ignored or dropped.
        cyclical_encoder (function): Function for cyclical encoding.
        custom_numerical_scaler (function): Function for scaling time variables.
        log_transformer (function): Function for transforming trip distance.
        scikit_minmax_scaler (MinMaxScaler): Fitted scaler for numerical columns.
        scikit_onehot_scaler (OneHotEncoder): Fitted encoder for categorical columns.
        impute_missing_travel_weeks (bool, optional): Whether to add missing travel days. Defaults to True.
        transform_to_wide (bool, optional): Whether to reshape to wide format. Defaults to False.
        transform_to_tensor (bool, optional): Whether to convert final output to torch tensors. Defaults to False.
        debug (bool, optional): If True, prints example processing steps for one individual. Defaults to False.
        max_journey_seq (int, optional): Max journeys per weekday. Defaults to 10.
        seq_length (int, optional): Number of time steps (weekdays). Defaults to 7.

    Returns:
        torch.Tensor | pd.DataFrame | List[str]: 
            - If `transform_to_tensor`: Tensors (X, y_cont, y_cat).
            - If `transform_to_wide`: List of wide-format DataFrames.
            - If `debug`: List of column names for a single debug example.
    """
    
    df = long_df.copy()

    # Fit encoders

    scikit_minmax_scaler.fit_transform(df[features_numerical])
    scikit_onehot_scaler.fit_transform(df[features_one_hot])
           

    #df = df[~df["DVLALengthBand_B01ID"].isin([-8, -10])]

    # All unique individual id's to loop over
    individual_ids = df["IndividualID_x"].unique()

    # Apply numerical encoding to numerical column
    #

    df_chunks = []

    full_week_encoding = list(range(1,8))

    if debug:
        random_index = random.randint(0, len(individual_ids))

        debug_df = df[df["IndividualID_x"] == individual_ids[random_index]]

        #display(debug_df)

        debug_df = impute_missing_travel_week_for_i(debug_df, 
                                                    i_id=individual_ids[random_index], 
                                                    full_week_encoding=full_week_encoding,
                                                    features=features,
                                                    outcomes=outcomes,
                                                    features_numerical=features_numerical,
                                                    features_one_hot=features_one_hot,
                                                    extra_vars=extra_vars,
                                                    cyclical_encoder=cyclical_encoder,
                                                    custom_numerical_scaler=custom_numerical_scaler,
                                                    log_transformer=log_transformer,
                                                    scikit_minmax_scaler=scikit_minmax_scaler,
                                                    scikit_onehot_scaler=scikit_onehot_scaler)

        #display(debug_df)

        debug_df, debug_targets_cont, debug_targets_cat = transform_to_wide_for_i(debug_df, 
                                                                                  i_id=individual_ids[random_index],
                                                                                  max_journey_seq=max_journey_seq,
                                                                                  full_week_encoding=full_week_encoding,
                                                                                  features=features,
                                                                                  outcomes=outcomes,
                                                                                  categorical_outcome_vars=categorical_outcome_vars,
                                                                                  features_numerical=features_numerical,
                                                                                  features_cyclical=features_cyclical,
                                                                                  features_one_hot=features_one_hot,
                                                                                  extra_vars=extra_vars,
                                                                                  cyclical_encoder=cyclical_encoder,
                                                                                  custom_numerical_scaler=custom_numerical_scaler,
                                                                                  log_transformer=log_transformer,
                                                                                  scikit_onehot_scaler=scikit_onehot_scaler,
                                                                                  scikit_minmax_scaler=scikit_minmax_scaler)

        
        for i, col in enumerate(debug_df.columns):
            print(f"{i}: {col}")
        print("")
        '''
        for i, col in enumerate(debug_targets_cont.columns):
            print(f"{i}: {col}")
        print("")
        for i, col in enumerate(debug_targets_cat.columns):
            print(f"{i}: {col}")'
        '''

        #print(debug_df.iloc[:,[0,1,2,3,20,31]].to_latex())

        #display(debug_df.iloc[0:7,[0,1,2,46]])

        display(debug_df)

        display(debug_targets_cont)

        display(debug_targets_cat)

        return debug_df.columns
    
    if transform_to_tensor:
        individual_tensors = []
        target_cont_tensors = []
        target_cat_tensors = []
    
    if impute_missing_travel_weeks:

        for i, individual_id in enumerate(individual_ids[:]):

            i_df = df[df["IndividualID_x"] == individual_id]

            full_df = impute_missing_travel_week_for_i(i_df=i_df, 
                                                        i_id=i, 
                                                        full_week_encoding=full_week_encoding,
                                                        features=features,
                                                        outcomes=outcomes,
                                                        features_numerical=features_numerical,
                                                        features_one_hot=features_one_hot,
                                                        extra_vars=extra_vars,
                                                        cyclical_encoder=cyclical_encoder,
                                                        custom_numerical_scaler=custom_numerical_scaler,
                                                        log_transformer=log_transformer,
                                                        scikit_minmax_scaler=scikit_minmax_scaler,
                                                        scikit_onehot_scaler=scikit_onehot_scaler)

            #display(full_df)

            if full_df is not None:
                if not transform_to_wide:
                    df_chunks.append(full_df)

                else:

                    full_df, targets_cont, targets_cat = transform_to_wide_for_i(i_df=full_df, 
                                                                                i_id=i,
                                                                                  max_journey_seq=max_journey_seq,
                                                                                  full_week_encoding=full_week_encoding,
                                                                                  features=features,
                                                                                  outcomes=outcomes,
                                                                                  categorical_outcome_vars=categorical_outcome_vars,
                                                                                  features_numerical=features_numerical,
                                                                                  features_cyclical=features_cyclical,
                                                                                  features_one_hot=features_one_hot,
                                                                                  extra_vars=extra_vars,
                                                                                  cyclical_encoder=cyclical_encoder,
                                                                                  custom_numerical_scaler=custom_numerical_scaler,
                                                                                  log_transformer=log_transformer,
                                                                                  scikit_onehot_scaler=scikit_onehot_scaler,
                                                                                  scikit_minmax_scaler=scikit_minmax_scaler)

                    logging.debug(f"{full_df.shape}, {targets_cont.shape}, {targets_cat.shape}")
                    
                    if transform_to_tensor:

                        if full_df.shape[0] == 14 and targets_cont.shape[0] == 7 and targets_cat.shape[0] == 7:
                            full_arr = full_df.to_numpy()
                            full_arr = np.expand_dims(full_arr, axis=1)

                            targets_cont_arr = targets_cont.to_numpy()
                            targets_cat_arr = targets_cat.to_numpy()

                            full_i_tensor = torch.tensor(full_arr)
                            target_cont_i_tensor = torch.tensor(targets_cont_arr)
                            target_cat_i_tensor = torch.tensor(targets_cat_arr)

                            individual_tensors.append(full_i_tensor)
                            target_cont_tensors.append(target_cont_i_tensor)
                            target_cat_tensors.append(target_cat_i_tensor)


                    else:

                        #display(full_df)
                        print("")
                        #display(targets)
                        df_chunks.append(full_df)

            logging.debug(f"Individual {i+1} out of {len(individual_ids)} Complete!")

        if transform_to_tensor:
            individual_tensors = torch.stack(individual_tensors, dim=0)
            target_cont_tensors = torch.stack(target_cont_tensors, dim=0)
            target_cat_tensors = torch.stack(target_cat_tensors, dim=0)
            return individual_tensors, target_cont_tensors, target_cat_tensors
        
        else:

            df_to_return = pd.concat(df_chunks)

            return df_to_return



    else:
        return df
