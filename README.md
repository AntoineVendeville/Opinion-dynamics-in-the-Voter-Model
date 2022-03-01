# How Opinions Crystallise

This repository contains code for various works related to the analysis of opinion dynamics under the voter model.

- All code used for <i>Depolarising Social Networks: Optimisation of Exposure to Adverse Opinions in the Presence of a Backfire Effect</i> is in `./SNAM/`. The notebook `./SNAM/SNAM.ipynb` contains the code to produce the plots. `util_snam.py`contains various functions used in the notebook. To obtain the data, go to https://voteview.com/data and download as follows
  - Data Type: Congressional Parties
  - Chamber: Both
  - Congress: All
  - File Format: CSV. The resulting file is the one we use in our analysis and should be named `HSall_parties.csv`. If you use it, don't forget to cite the authors (see website)!

- All code used for <i>Towards control of opinion diversity by introducing zealots into a polarised social group</i> is in `opinion_control.ipynb`. Folder `simu_results/` contains results of the simulations performed for the paper and used for plots.

- All code used for <i>Forecasting elections results via the voter model with stubborn agents</i> is in `elections.ipynb`. The data is stored in `elections_files/`.

`util.py` file contains custom functions used throughout the various notebooks.
