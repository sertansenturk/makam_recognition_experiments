[![DOI](https://zenodo.org/badge/21104/sertansenturk/makam_recognition_experiments.svg)](https://zenodo.org/badge/latestdoi/21104/sertansenturk/makam_recognition_experiments) ![GitHub release (latest by date)](https://img.shields.io/github/v/release/sertansenturk/makam_recognition_experiments) [![Build Status](https://travis-ci.com/sertansenturk/ds-template.svg?branch=master)](https://travis-ci.com/sertansenturk/makam_recognition_experiments) [![codecov](https://codecov.io/gh/sertansenturk/makam_recognition_experiments/branch/master/graph/badge.svg)](https://codecov.io/gh/sertansenturk/makam_recognition_experiments) [![License: AGPL v3](https://img.shields.io/badge/License-AGPL%20v3-ff69b4.svg)](http://www.gnu.org/licenses/agpl-3.0) [![License: CC BY-NC-SA 4.0](https://img.shields.io/badge/License-CC%20BY--NC--SA%204.0-ff69b4.svg)](http://creativecommons.org/licenses/by-nc-sa/4.0/)

# [ConferenceXX] Ottoman-Turkish Makam Recognition Experiments

This repository hosts the experiments conducted in the paper:

> TBD

The dataset used in the experiments was curated as part of the paper:

> Karakurt, A., Şentürk S., & Serra X. (2016). [MORTY: A Toolbox for Mode Recognition and Tonic Identification](http://mtg.upf.edu/node/3538). 3rd International Digital Libraries for Musicology Workshop. pages 9-16, New York, USA

Please cite the papers above, if you are using the contents of this repository for your works.

## Repository Structure

Rewrite XX

- The scripts are located in the base folder along with several miscallenaeous files (the license, readme, setup and requirement files).
- The folder [./data](https://github.com/sertansenturk/makam_recognition_experiments/tree/master/data) links to the relevant commit in [our makam recognition dataset](https://github.com/MTG/otmm_makam_recognition_dataset/releases/tag/dlfm2016), the folds and the summary of the evaluation obtained from all experiments.
- By running the [Jupyter notebooks](#scripts) in this repository, you can reproduce the extensive experiments reported in the paper. The outputs will also be saved to the folder [./data](https://github.com/sertansenturk/makam_recognition_experiments/tree/master/data). However the experiments might run for days (in a local machine), unless you use a cluster. For this reason, the computed features, training models, results and evaluation files are also dowloadable from Zenodo ([link](https://zenodo.org/record/57999)).
- The folder [./dlfm_code](https://github.com/sertansenturk/makam_recognition_experiments/tree/master/dlfm_code) has the relevant Python and MATLAB modules for the training, testing and evaluation.

## Setup

### Data

First, you should initialize and update the [dataset](https://github.com/MTG/otmm_makam_recognition_dataset/releases/tag/dlfm2016), which is linked as a submodule:

    ```bash
    cd path/to/makam_recognition_experiments
    git submodule init
    git submodule update
    ```

Zenodo data XX

### Docker

For the sake of reproducibility, you can run the experiments within Docker compose. To run Docker, you need to setup the Docker engine. Please refer to the [documentation](https://docs.docker.com/install/) to how to install the free, community version.

Currently, the docker-compose stack consists of the services below:

1. A customized [Jupyter](https://jupyter.org/) service. The "makam recognition experiment code" (in short `mre`) installed in editable mode and the **base folder of the repo is mounted** to the Jupyter docker service.
2. An [mlflow](https://mlflow.org/) tracking server to store experiments
3. A [postgresql](https://www.postgresql.org/) database, which stores mlflow tracking information

## Running Experiments

**Note:** We suggest you to use a cluster to run the training and testing steps. Otherwise it might take **days** to reproduce the experiments.

Build the service and run:

    ```bash
    make
    ```

Once the service is running, you will see a link on the terminal, e.g., http://127.0.0.1:8888/?token=3c321..., which you can follow to access the experiment notebooks from your browser.

## Test and Development

You can test the services (e.g., if `mlflow` handles logging correctly from the Jupyter service) automatically by running the docker-compose stack in test mode. You can run the test stack locally by:

    ```bash
    make test
    ```

We automate build, test, code style, and linting checks of the `mre` Python package using `tox` in a docker environment. You can run `tox` by:

    ```bash
    make tox
    ```

In addition, the repo has Travis CI integration ([link](https://travis-ci.com/github/sertansenturk/makam-recognition-experiments)), which runs all of the checks mentioned above automatically after each push. Travis CI also generates code coverage reports for the Python package, which can be viewed on codecov ([link](https://codecov.io/gh/sertansenturk/makam-recognition-experiments/)).

## License

The source code hosted in this repository is licenced under [Affero GPL version 3](https://www.gnu.org/licenses/agpl-3.0.en.html). The data (the features, models,  figures, results etc.) are licenced under [Creative Commons Attribution-NonCommercial-ShareAlike 4.0 International License](http://creativecommons.org/licenses/by-nc-sa/4.0/).
