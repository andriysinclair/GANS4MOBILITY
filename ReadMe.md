# TravNet

## An RNN-based Car Trip Simulator

This travel simulator is built using the UK National Travel Survey (link in the acknowledgements). It uses descriptive features on the individual, vehicle, day, household and region level to generate a weekly (Monday - Sunday) car travel diary for an individual. We generate the following trip-level variables for:

* `i_id`: The unique identifier.
* `DoW`: The day of the week on which the trip took place (1-7).
* `TripNum`: The trip's order for a given day.
* `TripStart`: The start time of the trip (minutes from midnight).
* `Duration`: Duration of the trip (minutes).
* `Distance`: Distance of the trip (miles).
* `Purpouse`: Purpouse of the trip (1 of 23 categories):
    * 1: Commuting
    * 2: Business
    * 3: Other work
    * 4: Education
    * 5: Food shopping
    * 6: Non food shopping
    * 7: Personal business medical
    * 8: Personal business eat / drink
    * 9: Personal business other
    * 10: Visit friends at private home
    * 11: Eat / drink with friends
    * 12: Other social
    * 13: Entertain / public activity
    * 14: Sport: participate
    * 15: Holiday: base
    * 16: Day trip
    * 17: Just walk
    * 18: Other non-escort
    * 19: Escort commuting
    * 20: Escort business & other work
    * 21: Escort education
    * 22: Escort shopping / personal business
    * 23: Escort home (not own) & other escort
    

## Getting Started

### Installing

**From Source**
1. Clone the repository.
2. Open terminal.
3. If using conda:
    1. `cd path_to_repo`
    2. `conda env create -f environment.yml`
    3. `conda activate TravNet`
4. If using pip:
    1. `cd path_to_repo`
    2. `pip install -e .`

**From PyPi**

#TBC

### Executing program

**From a Python session:**
1. Run ` from Modules.TravNetUser import generate_travel_data`
2. Run `wide_df, long_df = generate_travel_data(N)`, where `N` is the number of individuals for whom you want to generate weekly travel schedules.
3. This returns travel schedules in wide and long format and saves the long format in `/Results` as a `.pkl`.
4. Analysis pipeline #TBC

**From CLI**
#TBC

### Developer

**Directory Structure**

#TBC work out how to add docstrings of different functions #FIRST PRIORITY after wright-up

## Version History

* v0.0.1 - Pre-release
    * Neural network running, but no rationality conditions to resemble actual travel data.
    * See [v0.0.1](https://github.com/andriysinclair/TravNet/releases/tag/v0.0.1) 
* v0.0.2 - Pre-release
    * Added rationality conditions.
    * See [v0.0.2](https://github.com/andriysinclair/TravNet/releases/tag/v0.0.2) 
* 0.0.3 - Pre-release
    * Matrix based losses
    * See [v0.0.3](https://github.com/andriysinclair/TravNet/releases/tag/v0.0.3) 

## Acknowledgments

**Dataset**
Department for Transport. (2023). National Travel Survey. [data series]. 8th Release. UK Data Service. SN: 2000037, DOI: http://doi.org/10.5255/UKDA-Series-2000037


