# CS Energy Internship Repo
Github base containing all project data & code.

![skeleton](https://i.giphy.com/media/3o6fJ5z2bgCLBshZUA/giphy.webp)


Directory Structure
-------------------
```
├── README.md               <- Top level README file
│
├── data
│   ├── external            <- Data from third party sources (not the client).
│   ├── raw                 <- Original immutable data dump (no code should alter this)
│   ├── interim             <- Intermediate data that has been transformed.
│   └── processed           <- Final processed data used for modelling.
│
├── containers
│   └── base                <- Container with the module & its dependencies
│
├── models                  <- Trained/serialized model files, model predictions and
│                              model summaries
│
├── notebooks               <- All jupyter notebooks. Naming convention is a number
│   │                          (for ordering execution) along with authors name
│   │                          or initials and short '-' delimited description
│   │
│   └── 1.0-mk-initial-notebook-example.ipynb
│
├── src                     <- Source code for use in this project
│   ├── helpers             <- Directory containing helper functions/files for project-wide use
│   ├── data                <- Scripts for downloading/generating data
│   ├── preprocessing       <- Scripts for preprocessing
│   ├── feature_engineering <- Scripts for engineering features
│   └── modelling           <- Scripts for training/evaluating models
│
└── pyproject.toml          <- Project configuration file
```
