from pathlib import Path
import pandas as pd
import numpy as np
import logging
import pickle
import config as cfg

#TODO Make a more dynamic path system later

data_folder = str(Path(__file__).resolve().parent.parent / "data")

nts_trip = data_folder + "/trip_eul_2002-2023.tab"
nts_i = data_folder + "/individual_eul_2002-2023.tab"
nts_vehicle = data_folder + "/vehicle_eul_2002-2023.tab"
nts_household = data_folder + "/household_eul_2002-2023.tab"
nts_psu = data_folder + "/psu_eul_2002-2023.tab"
nts_day = data_folder + "/day_eul_2002-2023.tab"


def wrangler(merged_df, drop_fraction  = 0.3) -> pd.DataFrame:
    """
    Applied to the merged DataFrame. Drops columns with excessive missing values. and performs basic cleaning.
    Used in conjunction with loader function.

    Args:
        merged_df (pd.DataFrame): The merged df.    
        drop_fraction (float, optional): The minimum number of missing values before columns is dropped. Defaults to 0.3.

    Returns:
        pd.DataFrame: Cleaned DataFrame
    """
    merged_df = merged_df.copy()

    # Taking small sample to drop duplicates

    #logging.debug(merged_df)

    merged_df_duplicate = merged_df.sample(n=min(1000, len(merged_df)))

    #logging.debug(merged_df_duplicate)

    logging.debug(f"Num columns before dropping duplicates: {len(merged_df_duplicate.columns)}")

    #logging.debug(merged_df_duplicate.T)

    merged_df_duplicate = merged_df_duplicate.T.drop_duplicates().T
    logging.debug(f"Num columns after dropping duplicates: {len(merged_df_duplicate.columns)}")

    dupe_cols = set(merged_df.columns) - set(merged_df_duplicate.columns)
    dupe_cols = list(dupe_cols)

    logging.debug(f"Duplicate columns...")
    logging.debug(dupe_cols)

    merged_df_analysis = merged_df.drop(columns=dupe_cols, axis=1)

    cols_missing = []

    # Dealing with missing values
    logging.debug("% missing vals per column")
    for col,missing_count in merged_df_analysis.isna().sum().items():
        if missing_count/len(merged_df_analysis) >= 0.001:
            logging.debug(f"{col}: {missing_count/len(merged_df_analysis):.2f}")

        if missing_count/len(merged_df_analysis) >= drop_fraction:
            cols_missing.append(col)


    # Trip start/ end are vital variables so we shall drop them

    # TripStartHours: 7042
    # TripStartMinutes: 7042
    # TripStart: 7042
    # TripEndHours: 7396
    # TripEndMinutes: 7396
    # TripEnd: 7396

    vars_to_drop_na = ["TripStartHours", "TripStartMinutes", "TripStart", "TripEndHours", "TripEndMinutes", "TripEnd" ]

    # Dropping all rows
    merged_df_analysis = merged_df_analysis.dropna(subset=vars_to_drop_na)

    merged_df_analysis = merged_df_analysis.drop(columns=cols_missing, axis=1, errors="ignore")

    # All other columns are insignificant --> drop

    #merged_df_analysis = merged_df_analysis.dropna()
    merged_df_analysis = merged_df_analysis.copy()
    merged_df_analysis.reset_index(drop=True, inplace=True)

    # Columns were loaded in as string now converting to numeric

# Converting all to float

    logging.debug(f"DF was loaded in as a string --> converting to float")

    faulty_cols = []


    for col in merged_df_analysis.columns:
        try:
            merged_df_analysis[col] = merged_df_analysis[col].astype(float)

        except:
            logging.debug(f"conversion did not work for {col}")
            faulty_cols.append(col)

    logging.debug("'TWSDate' is completely empty so dropping")

    merged_df_analysis.drop(columns="TWSDate", axis=1, inplace=True, errors="ignore")

    if "TWSDate" in faulty_cols:
        faulty_cols.remove("TWSDate")

    faulty_indices = []

    for index, row in merged_df_analysis[faulty_cols].iterrows():
        for col in faulty_cols:
            try:
                float(row[col])
            except:
                logging.debug(f"{row[col]} is faulty")
                faulty_indices.append(index)

    merged_df_analysis = merged_df_analysis.drop(index=faulty_indices)

    # Converting all to float

    for col in merged_df_analysis.columns:
        merged_df_analysis[col] = merged_df_analysis[col].astype(float)

    # Dropping Id cols except individual ID and SurveyYear because we need TravelYear

    id_cols = ["TripID", "DayID", "HouseholdID_x", "PSUID_x", "SurveyYear"]

    merged_df_analysis.drop(columns=id_cols, axis=1, inplace=True, errors="ignore")

    merged_df_analysis = merged_df_analysis.copy()

    # encoding num trips

    merged_df_analysis["NumTrips"] = merged_df_analysis.groupby(["IndividualID_x", "TravelWeekDay_B01ID"])["TravelWeekDay_B01ID"].transform("count")

    num_trips_mapping = dict(zip(merged_df_analysis.groupby("IndividualID_x")["JourSeq"].max().index, merged_df_analysis.groupby("IndividualID_x")["JourSeq"].max().values))
    merged_df_analysis["NumTrips"] = merged_df_analysis["IndividualID_x"].map(num_trips_mapping)
    
    
    merged_df_analysis["IsTrip"] = 1

    # Dropping old cols

    merged_df_analysis.drop(columns=["TripPurpFrom_B01ID", "TripPurpTo_B01ID"], axis=1, inplace=True, errors="ignore")

    return merged_df_analysis




def loader(output_file_name, wrangle_func=wrangler, nts_trip=nts_trip, nts_vehicle=nts_vehicle, nts_i=nts_i, nts_household=nts_household, 
           nts_psu=nts_psu, nts_day=nts_day, chunksize = 100000, 
            survey_years=[2017], drop_fraction=0.3, return_raw=False, features = cfg.features, outcomes = cfg.outcomes,
           extra_vars = cfg.extra_vars, features_one_hot = cfg.features_one_hot) -> pd.DataFrame:
    """
    loader 

    Loads Individual NTS datasets and merges on unique identifiers. Afterwords, can subset on variables in config.py

    Args:
        output_file_name (str): file name - saved in \data
        wrangle_func (func, optional): Function used to wrangle data. Defaults to wrangler.
        nts_trip (str, optional): path to trip data. Defaults to nts_trip.
        nts_vehicle (str, optional): path to vehicle data. Defaults to nts_vehicle.
        nts_i (str, optional): path to individual data. Defaults to nts_i.
        nts_household (str, optional): path to household data. Defaults to nts_household.
        nts_psu (str, optional): path to PSU data. Defaults to nts_psu.
        nts_day (str, optional): path to day data. Defaults to nts_day.
        chunksize (int, optional): Chunks to load at once as dataset is HUGE. Defaults to 100000.
        survey_years (list, optional): years to extract MUST BE IN LIST. Defaults to 2017.
        drop_fraction (float, optional): min % of missing values to drop a column. Defaults to 0.3.
        return_raw (bool, optional): Returns data subsetted based on config.py. Otherwise returns data with all columns. Defaults to False.
        features (_type_, optional): features (if return_raw=False). Defaults to cfg.features.
        outcomes (_type_, optional): outcomes (if return_raw=False). Defaults to cfg.outcomes.
        extra_vars (_type_, optional): extra_vars (if return_raw=False). Defaults to cfg.extra_vars.
        features_one_hot (_type_, optional): features_one_hot (if return_raw=False). Defaults to cfg.features_one_hot.

    Returns:
        pd.DataFrame: Either full merged data frame or subsetted based on config.py (depending on return_raw)
    """    
    # Load in vehicle df

    # Converting everything to string as everything is loaded in a string
    survey_years = [str(year) for year in survey_years]

    vehicle_df = pd.read_csv(nts_vehicle, sep="\t",  dtype=str)

    # Load in Individual df

    i_df = pd.read_csv(nts_i, sep="\t",  dtype=str)

    # Household

    household_df = pd.read_csv(nts_household, sep="\t",  dtype=str)

    # Load in Postcode ID

    psu_df = pd.read_csv(nts_psu, sep="\t",  dtype=str)

    # Load in day

    day_df = pd.read_csv(nts_day, sep="\t",  dtype=str)

    #logging.debug(f"Filter data frames for SurveyYear == {survey_year}")
    #i_df = i_df[i_df["SurveyYear"] == survey_year]
    #vehicle_df = vehicle_df[vehicle_df["SurveyYear"] == survey_year]
    #psu_df = psu_df[psu_df["SurveyYear"] == survey_year]
    #household_df = household_df[household_df["SurveyYear"] == survey_year]
    #day_df = day_df[day_df["SurveyYear"] == survey_year]
    #logging.debug("Complete!")

    # Dropping survey year as it is a nuisance column and not needed

    vehicle_df = vehicle_df.drop(columns="SurveyYear", axis=1, errors="ignore")
    vehicle_df = vehicle_df.drop(columns="SurveyYear", axis=1, errors="ignore")
    day_df = day_df.drop(columns="SurveyYear", axis=1, errors="ignore")
    i_df = i_df.drop("SurveyYear", axis=1, errors="ignore")

# Load Trip DF and merging in chunks

    output_chunks_file = data_folder + f"/{output_file_name}"

    merged_chunks = []

    for i,trip_df in enumerate(pd.read_csv(nts_trip, sep="\t", chunksize=chunksize, dtype=str)):
        # Filter by car only
        # Taking sample
        #trip_df = trip_df.sample(n=sample_size)
        

        #logging.debug(trip_df["SurveyYear"].unique())

        if "3" in trip_df["MainMode_B04ID"].unique():
            trip_df = trip_df[trip_df["MainMode_B04ID"] == "3"]
            
            #logging.debug(trip_df)

            for year in trip_df["SurveyYear"].unique():
                if year in survey_years:

            #if str(survey_year) in trip_df["SurveyYear"].unique():

            #logging.debug(trip_df["SurveyYear"].unique())
            #logging.debug(trip_df["MainMode_B04ID"].unique())

                
                    trip_df = trip_df[trip_df["SurveyYear"].isin(survey_years)]

                    #logging.debug(trip_df)
                    chunk = trip_df.merge(i_df, on="IndividualID", how="left")
                    #logging.debug("1st merge")
                    #logging.debug(chunk)
                    chunk = chunk.merge(vehicle_df, on="VehicleID", how="left")
                    #logging.debug("2nd merge")
                    #logging.debug(chunk)
                    chunk = chunk.merge(psu_df, on="PSUID", how="left")
                    #logging.debug("3rd merge")
                    #logging.debug(chunk)
                    chunk.drop(columns=["PSUID", "HouseholdID"], axis=1, inplace=True, errors="ignore")
                    chunk = chunk.merge(day_df, on="DayID", how="left")
                    #logging.debug("4th merge")
                    #logging.debug(chunk)
                    chunk.drop(columns="PSUID", axis=1, inplace=True, errors="ignore")
                    chunk = chunk.merge(household_df, on="HouseholdID", how="left")
                    #logging.debug("5th merge")
                    #logging.debug(chunk)
                    # Apply wrangler func

                    #logging.debug(chunk)

                    #chunk = wrangle_func(chunk)

                    merged_chunks.append(chunk)

                    print(f"\rchunk: {i+1} complete!", end="", flush=True)

                else:
                    print(f"\rSurveyYear = {survey_years} not found in chunk {i+1}. Continuing", end="", flush=True)
                    continue

        else:
            print(f"\rMainMode_B04ID = 3 not found in chunk {i+1}. Continuing", end="", flush=True)
            continue

    merged_df = pd.concat(merged_chunks, ignore_index=True)

    # Apply missing mapping

    missing_mapping ={
    str(-8): np.nan,
    str(-9): np.nan,
    str(-10): np.nan}

    merged_df = merged_df.replace(missing_mapping).infer_objects(copy=False)

    merged_df = wrangle_func(merged_df)

    merged_df = merged_df.fillna(0)

    # Adding a NumTrips variable

    #merged_df["NumTrips"] = merged_df.groupby(["IndividualID_x", "TravelWeekDay_B01ID"])["TravelWeekDay_B01ID"].transform("count")

    #merged_df = pd.concat(merged_chunks, ignore_index=True)

    if return_raw:

        return merged_df
    
    else:

        ts_df = merged_df[features + outcomes + extra_vars]

        # Apply cyclical encoding to cyclical column, assuming the same categories that appear in 2017 appear elsewhere


        for col in features_one_hot:
            ts_df.loc[:,col] = ts_df.loc[:,col].astype(int)

        ts_df.loc[:, "TravelWeekDay_B01ID"] = ts_df.loc[:, "TravelWeekDay_B01ID"].astype(int)

        with open(output_chunks_file, "wb") as f:
            pickle.dump(ts_df, f)   

        print("\nMerged chunks saved to pickle!")

        return ts_df



if __name__ == "__main__":

    # Configure basic logging
    logging.basicConfig(level=logging.INFO, force=True, format='%(levelname)s: %(message)s')

    # Define data range

    years_to_extract = list(range(2017,2018))

    df = loader(output_file_name="merged_df2017.pkl", chunksize=100000, sample_size=100000, survey_years=years_to_extract)

