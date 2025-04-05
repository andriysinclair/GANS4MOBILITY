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

`analysis.ipynb` showcases all the main functions of this package in around 1 minute. Just press play :)

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

`analysis.ipynb` is a brief showcase of all the major functions.

**From a Python session:**
1. Run ` from Modules.TravNetUser import TravNet` to import the `TravNet` class.
2. Create an instance `travnet = TravNet()`
3. Run `travnet.generate_travel_data(N)`, where `N` is the number of individuals for whom you want to generate weekly travel schedules.
4. if you run `travnet.generate_travel_data(N, return_df=True)`This returns travel schedules (pd.DataFrame) in wide and long format and saves the long format in `/Results_N` as a `.pkl`. If `return_df=False` (default) then the long format is just saved as a pkl.
5. Run `travnet.output_aggregate_stats()` to print aggregate stats from real and generated data.
6. Run `travnet.plot_histograms()` to plot histograms, which will save to `/Plots`.

*Note*
* If one does not run `travnet.generate_travel_data(N)` prior to running `travnet.output_aggregate_stats()` and/or `travnet.plot_histograms()`. Then, the original generated data for 10,000 individuals will be used as the results will reflect those of the paper.

### Developer

**Directory Structure**

.
├── Analysis.ipynb               # Jupyter notebook showcasing main features
├── Models/                      # Trained models
├── Modules/                     # Core Python modules and configs
├── Plots/                       # Visualizations of losses and histograms
├── Results/                     # Pickled result outputs
├── data/                        # Raw and processed NTS data
├── tensors/                     # Processed tensors for model training
├── environment.yml              # Conda environment specification
├── pyproject.toml               # Project metadata
└── ReadMe.md                    # This file

**Under \Modules we have:**
* `config.py`: Encoders, features etc. that are used to set up data for model training.
* `loader_wrangler.py`: Functions to merge NTS datasets and perform basic cleaning.
* `ToTensor.py`: Transforms data to wide format for every individual in the dataset then makes tensors for training.
* `Transformations.py`: Contains custom transformers like cyclical, log, MinMax and a function to plot correlations for variable selection.
* `TravNet.py`: Contains the PyTorch model class and training loop.
* `TravNetUser.py`: Contains the `TravNet` class, that contains all user-level functions for generating data and plotting.


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


