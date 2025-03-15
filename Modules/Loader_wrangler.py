from pathlib import Path
import pandas as pd
import re
import logging
import pickle

#TODO Make a more dynamic path system later

data_folder = str(Path(__file__).resolve().parent.parent / "data")

nts_trip = data_folder + "/trip_eul_2002-2023.tab"
nts_i = data_folder + "/individual_eul_2002-2023.tab"
nts_vehicle = data_folder + "/vehicle_eul_2002-2023.tab"
nts_household = data_folder + "/household_eul_2002-2023.tab"
nts_psu = data_folder + "/psu_eul_2002-2023.tab"
nts_day = data_folder + "/day_eul_2002-2023.tab"

trip_type_mapping = {('Home', 'Work'): 0,
                     ('Home', 'Other'): 1,
                     ('Other', 'Home'): 2,
                     ('Work', 'Home'): 3,
                     ('Other', 'Other'): 4,
                     ('Work', 'Other'): 5,
                     ('Other', 'Work'): 6,
}


def wrangler(merged_df):

    """
    Cleans and processes the merged trip dataset.

    This function performs the following steps:
    - Drops duplicate columns based on a sample of 1000 rows.
    - Removes rows with missing values in crucial trip-related columns (e.g., Trip Start/End times).
    - Drops additional columns with high missing values or redundant information.
    - Converts all numerical columns to float, handling any conversion errors.
    - Encodes the number of trips per individual.
    - Maps trip purposes into simplified categories (Work, Home, Other).
    - Creates a "TripType" column based on start and destination trip purposes.
    - Drops unnecessary ID columns and old categorical columns after mapping.

    Parameters:
    -----------
    merged_df : pandas.DataFrame
        A DataFrame containing merged trip, individual, vehicle, household, PSU, and day data.

    Returns:
    --------
    pandas.DataFrame
        A cleaned and processed DataFrame ready for further analysis or modeling.
    """

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

    # Dealing with missing values
    logging.debug("% missing vals per column")
    for col,missing_count in merged_df_analysis.isna().sum().items():
        if missing_count/len(merged_df_analysis) >= 0.001:
            logging.debug(f"{col}: {missing_count/len(merged_df_analysis):.2f}")

    # Trip start/ end are vital variables so we shall drop them

    # TripStartHours: 7042
    # TripStartMinutes: 7042
    # TripStart: 7042
    # TripEndHours: 7396
    # TripEndMinutes: 7396
    # TripEnd: 7396

    vars_to_drop_na = ["TripStartHours", "TripStartMinutes", "TripStart", "TripEndHours", "TripEndMinutes", "TripEnd" ]

    merged_df_analysis = merged_df_analysis.dropna(subset=vars_to_drop_na)

    '''
    QLeaHous: 413346   -> "How many times did you leave the house yesterday - actual number"
    FarWalk: 363596 -> "Time last long walk took - minutes - actual time" # Has a banded version: "Time last long walk took - minutes - banded time"

    DistWalk: 523930 -> "Distance of last long walk - miles - actual distance", has a banded version
    IntPlane: 120431 -> "Number of international plane trips in last 12 months - actual number", has a banded version
    DTJbYear: 395771 -> "Date left last paid job - year element"
    SchAgeAcc: 524130 -> "Age first unaccompanied to school - actual age"
    ReNDNaM_B01ID: 366718 -> "Main reason do not drive"
    ReNDNbM_B01ID: 366718 -> "Main reason do not drive"
    CycMore_B01ID: 366718 -> "Amount of cycling compared to this time last year"
    Cycle4w_B01ID: 366718 -> "Whether ridden a bicycle during the last 4 weeks"
    ResMNCy_B01ID: 366718 -> "Main reason for not cycling more"
    DVLALength: 94114 -> "Vehicle length from the DVLA database - mm - actual length", contains a banded version
    VehComMile: 22566 -> "Annual vehicle commuting mileage - actual mileage", contains banded
    VehBusMile: 22562 -> "Annual vehicle business mileage - actual mileage", contains banded
    VehPriMile: 22566 -> "Annual vehicle private mileage - actual mileage", contains banded

    ---> Drop all
    
    '''

    cols_to_drop_missing = [
        "QLeaHous",
        "FarWalk",
        "DistWalk",
        "IntPlane",
        "DTJbYear",
        "ReNDNaM_B01ID",
        "CycMore_B01ID",
        "Cycle4w_B01ID",
        "ResMNCy_B01ID",
        "DVLALength",
        "VehComMile",
        "VehBusMile",
        "VehPriMile"
    ]

    merged_df_analysis = merged_df_analysis.drop(columns=cols_to_drop_missing, axis=1, errors="ignore")

    # All other columns are insignificant --> drop

    merged_df_analysis = merged_df_analysis.dropna()
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

    '''
    num_trips_mapping = dict(zip(merged_df_analysis.groupby("IndividualID_x")["JourSeq"].max().index, merged_df_analysis.groupby("IndividualID_x")["JourSeq"].max().values))
    merged_df_analysis["NumTrips"] = merged_df_analysis["IndividualID_x"].map(num_trips_mapping)
    '''

    # Removing Na and Dead
    merged_df_analysis = merged_df_analysis[~merged_df_analysis["TripPurpFrom_B01ID"].isin([-8,-10])]
    merged_df_analysis = merged_df_analysis[~merged_df_analysis["TripPurpTo_B01ID"].isin([-8,-10])]

    # Simpler Mappings

    

    trip_purpose_mapping = {
        1: "Work", 2: "Other", 3: "Other", 4: "Other", 5: "Other",
        6: "Other", 7: "Other", 8: "Other", 9: "Other", 10: "Other",
        11: "Other", 12: "Other", 13: "Other", 14: "Other", 15: "Other",
        16: "Other", 17: "Other", 18: "Other", 19: "Other", 20: "Other",
        21: "Other", 22: "Other", 23: "Home",
    }

    merged_df_analysis["TripPurpFrom_B01ID"] = merged_df_analysis["TripPurpFrom_B01ID"].map(trip_purpose_mapping)
    merged_df_analysis["TripPurpTo_B01ID"] = merged_df_analysis["TripPurpTo_B01ID"].map(trip_purpose_mapping)

    merged_df_analysis["TripType"] = list(zip(merged_df_analysis["TripPurpFrom_B01ID"], merged_df_analysis["TripPurpTo_B01ID"]))


    trip_type_mapping = {}

    for i,type in enumerate(merged_df_analysis["TripType"].unique()):
        trip_type_mapping[type] = i
        
    print(f"Trip type mapping")
    for k,v in trip_type_mapping.items():
        print(f"{k}: {v}")

    # Export mapping
    with open(data_folder + "/TripType_mapping.pkl", "wb") as f:
        pickle.dump(trip_type_mapping, f)   


    merged_df_analysis["TripType"] = merged_df_analysis["TripType"].map(trip_type_mapping)

    # Dropping old cols

    merged_df_analysis.drop(columns=["TripPurpFrom_B01ID", "TripPurpTo_B01ID"], axis=1, inplace=True, errors="ignore")

    return merged_df_analysis




def loader(output_file_name, wrangle_func=wrangler, nts_trip=nts_trip, nts_vehicle=nts_vehicle, nts_i=nts_i, nts_household=nts_household, nts_psu=nts_psu, nts_day=nts_day, chunksize = 100000, sample_size = 10000, survey_year=2017):

    """
    Loads, merges, and processes National Travel Survey (NTS) datasets in chunks.

    This function performs the following steps:
    - Loads multiple NTS datasets: trip, individual, vehicle, household, PSU, and day.
    - Merges these datasets on appropriate keys (e.g., IndividualID, VehicleID, HouseholdID).
    - Filters trips to include only car trips (`MainMode_B04ID == "3"`).
    - Drops unnecessary columns such as "SurveyYear".
    - Processes the dataset in chunks to handle large file sizes efficiently.
    - Applies a user-defined wrangling function (`wrangle_func`) to clean and preprocess the merged data.
    - Concatenates processed chunks into a single DataFrame.

    Parameters:
    -----------
    wrangle_func : function, optional (default=wrangler)
        A function to apply preprocessing and feature engineering to the merged dataset.
    
    nts_trip : str, optional
        File path for the trip dataset.

    nts_vehicle : str, optional
        File path for the vehicle dataset.

    nts_i : str, optional
        File path for the individual dataset.

    nts_household : str, optional
        File path for the household dataset.

    nts_psu : str, optional
        File path for the PSU dataset.

    nts_day : str, optional
        File path for the day dataset.

    chunksize : int, optional (default=100000)
        Number of rows to load at a time for efficient processing of large datasets.

    sample_size: int, optional (default=10000)
        Number of rows to draw from each chunk (to save memory)

    return_raw : bool, optional (default=False)
        If True, returns the raw merged DataFrame before applying wrangling.

    info : bool, optional (default=False)
        If True, prints additional debug information during processing.

    Returns:
    --------
    pandas.DataFrame
        A fully merged and processed DataFrame containing travel data.
    """
    
    # Load in vehicle df

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

            if str(survey_year) in trip_df["SurveyYear"].unique():

            #logging.debug(trip_df["SurveyYear"].unique())
            #logging.debug(trip_df["MainMode_B04ID"].unique())

                
                trip_df = trip_df[trip_df["SurveyYear"] == str(survey_year)]

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
                print(f"\rSurveyYear = {survey_year} not found in chunk {i+1}. Continuing", end="", flush=True)
                continue

        else:
            print(f"\rSurveyYear = {survey_year} not found in chunk {i+1}. Continuing", end="", flush=True)
            continue

    merged_df = pd.concat(merged_chunks, ignore_index=True)

    merged_df = wrangle_func(merged_df)

    # Adding a NumTrips variable

    #merged_df["NumTrips"] = merged_df.groupby(["IndividualID_x", "TravelWeekDay_B01ID"])["TravelWeekDay_B01ID"].transform("count")

    with open(output_chunks_file, "wb") as f:
        pickle.dump(merged_df, f)   

    print("\nMerged chunks saved to pickle!")

    #merged_df = pd.concat(merged_chunks, ignore_index=True)

    return merged_df



if __name__ == "__main__":
    print(data_folder)

