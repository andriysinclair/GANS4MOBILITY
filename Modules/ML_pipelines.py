import pandas as pd
import logging
import joblib
import pickle
import os
from pathlib import Path
from sklearn.ensemble import RandomForestRegressor

##### OUTCOME COLUMNS OF INTEREST #####

# TripStart
# TripEnd
# TripType
# NumTrips
# TripDisExSW

project_folder = str(Path(__file__).resolve().parent.parent)


# Setting logging
logging.basicConfig(
    filename= project_folder + "/logs/debug_log.log",
    level = logging.DEBUG,
    format="%(asctime)s - %(levelname)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S"
)

def mean_encoder(df, outcome=None, max_categories=25, explore_categories=False):

    if not isinstance(max_categories, int):
        raise ValueError(f"Max categories must be int, max_categories = {max_categories}")
    
    if explore_categories is False and outcome not in df.columns:
        raise ValueError(f"If explore_outcomes is False, outcome must be in df.columns. outcome: {outcome}")

    df = df.copy()

    categorical = [outcome]
     
    for col, val in df.nunique().sort_values().items():
            if explore_categories is True:
                print(f"{col:<25} {val}")
                
            else:
                if val <= max_categories and col != outcome:
                    categorical.append(col)

    if not explore_categories:
        logging.debug(f"Categorical categories")
        logging.debug(f"{categorical}")
        for cat in categorical[1:]:

             mapping_df = df[[outcome, cat]].groupby(cat).mean()
              
             mapping = dict(zip(mapping_df.index, mapping_df[outcome]))
             logging.debug(f"Final mapping for {cat}")
             
             for k, v in mapping.items():
                logging.debug(f"{k:<25} {v}")

             df[cat] = df[cat].map(mapping)

             logging.debug(f"Final mapped column for {cat}:")
             logging.debug("\n" + df[[cat]].head(10).to_string(index=True))

        return df
    

def remove_correlations(corr_matrix, outcome_var, drop_vars=False, drop_cut_off=None):
    sorted_corr = corr_matrix[outcome_var].abs().sort_values(ascending=False)  
    vars_to_drop = []

    for column in sorted_corr.index:
        value = corr_matrix[outcome_var][column]  
        if drop_vars:
            if abs(value)>=drop_cut_off and column != outcome_var:
                vars_to_drop.append(column)

        else:
            if abs(value) > 0.05:
                print(f"{column:<25} {value:.2f}")

    if drop_vars:
        logging.debug(f"vars_to_drop with drop_cut_off: {drop_cut_off}")
        logging.debug(vars_to_drop)
        return vars_to_drop
        
def ML_pipeline(df, outcome, corr_cut_off=None, info=True, max_categories=25, cols_to_exclude = []):
    df_for_corr = df.sample(n=min(50000, len(df)))

    if info:

        print("############################")
        # Identify categories based on orignal df
        print(f"Identifying categories for {outcome}")
        mean_encoder(df, explore_categories=True)
        print("")


    logging.debug(f"Mean encoding...")
    # Already mean encoding everything
    df_for_corr = mean_encoder(df_for_corr, outcome=outcome, explore_categories=False, max_categories=max_categories)

    logging.debug("Exploring correlations")
    corr_matrix = df_for_corr.corr()

    if info:

        # Based on mean encoded df
        print(f"Identifying correlations with {outcome}")
        remove_correlations(corr_matrix=corr_matrix, outcome_var=outcome)
        # Exploring categories
        print("")
        print("############################")



        return None

    else:
        # Dropping vars from correlation matrix
        vars_to_drop = remove_correlations(corr_matrix=corr_matrix, outcome_var=outcome, drop_vars=True, drop_cut_off=corr_cut_off)
        
        y = df_for_corr[outcome]
        X = df_for_corr.drop(columns=vars_to_drop, axis=1, errors="ignore")
        X = X.drop(columns = outcome, axis=1)
        X = X.drop(columns = cols_to_exclude, axis=1, errors="ignore")

        #X = mean_encoder(X, outcome=outcome, explore_categories=False, max_categories=max_categories)

        return X, y
    
def rf_for_fi(df, outcome, corr_cut_off, max_categories, cols_to_exclude = [], refit=False):

    rf_save_file_name = "/Models/rf_model" + "_" + outcome + ".pkl"

    X, y = ML_pipeline(df, outcome=outcome, info=False, corr_cut_off=corr_cut_off, max_categories=max_categories, cols_to_exclude=cols_to_exclude)

    logging.info(len(X.columns))
    # Check if model exists
    if os.path.exists(project_folder + rf_save_file_name):
        logging.info(f"Model: {rf_save_file_name} already exists!")

        if not refit:

            logging.info(f"Refit set to: {refit}. Loading")
            with open(project_folder + rf_save_file_name, "rb") as f:

                rf = pickle.load(f)

                feature_importances = rf.feature_importances_
                logging.info(len(feature_importances))

                columns = rf.feature_names_in_
                logging.info(len(columns))


        if refit:
            logging.info(f"Refit set to: {refit}. Fitting")

            rf = RandomForestRegressor(min_samples_split=20, verbose=0, n_estimators=50)
            rf.fit(X, y)

            with open(project_folder + rf_save_file_name, "wb") as f:
                pickle.dump(rf, f)

            feature_importances = rf.feature_importances_
            logging.info(len(feature_importances))

            columns = rf.feature_names_in_
            logging.info(len(columns))


    else:  # Fit a new one

        logging.info(f"Model: {rf_save_file_name} does not exist. Fiting...")
       
        rf = RandomForestRegressor(min_samples_split=20, verbose=0, n_estimators=50)
        rf.fit(X, y)

        with open(project_folder + rf_save_file_name, "wb") as f:
            pickle.dump(rf, f)

        feature_importances = rf.feature_importances_

        columns = rf.feature_names_in_


    #for col, imp in zip(columns, feature_importances):
        
    fi_df = pd.DataFrame({"features": columns, "FI": feature_importances})

    fi_df = fi_df.sort_values("FI", ascending=False)

    for feature, fi in zip(fi_df["features"], fi_df["FI"]):
        if fi >= 0.01:
            print(f"{feature:<25} {fi:.2f}")
        
    print("")


if __name__ == "__main__":
    
    print(project_folder)
    distance_df = pd.read_pickle("/home/trapfishscott/Cambridge24.25/D200_ML_econ/ProblemSets/Project/data/merged_df_analysis.pkl")

    ## Run to see correlations (for each outcome var) and unique features per column ## 

    cont_outcomes = ["TripStart", "TripEnd", "NumTrips", "TripDisExSW"]


    ML_pipeline(df=distance_df, outcome=cont_outcomes[3], max_categories=55)
