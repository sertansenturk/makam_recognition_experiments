[![DOI](https://zenodo.org/badge/21104/sertansenturk/makam_recognition_experiments.svg)](https://zenodo.org/badge/latestdoi/21104/sertansenturk/makam_recognition_experiments)

# [ConferenceXX] Ottoman-Turkish Makam Recognition Experiments

This repository hosts the experiments conducted in the paper:

> TBD

The dataset used in the experiments was curated as part of the paper:

> Karakurt, A., Şentürk S., & Serra X. (2016).  [MORTY: A Toolbox for Mode Recognition and Tonic Identification](http://mtg.upf.edu/node/3538). 3rd International Digital Libraries for Musicology Workshop. pages 9-16, New York, USA

Please cite the papers above, if you are using the contents of this repository for your works.

# Repository Structure

Rewrite XX

- The scripts are located in the base folder along with several miscallenaeous files (the license, readme, setup and requirement files).
- The folder [./data](https://github.com/sertansenturk/makam_recognition_experiments/tree/master/data) links to the relevant commit in [our makam recognition dataset](https://github.com/MTG/otmm_makam_recognition_dataset/releases/tag/dlfm2016), the folds and the summary of the evaluation obtained from all experiments.
- By running the [Jupyter notebooks](#scripts) in this repository, you can reproduce the extensive experiments reported in the paper. The outputs will also be saved to the folder [./data](https://github.com/sertansenturk/makam_recognition_experiments/tree/master/data). However the experiments might run for days (in a local machine), unless you use a cluster. For this reason, the computed features, training models, results and evaluation files are also dowloadable from Zenodo ([link](https://zenodo.org/record/57999)).
- The folder [./dlfm_code](https://github.com/sertansenturk/makam_recognition_experiments/tree/master/dlfm_code) has the relevant Python and MATLAB modules for the training, testing and evaluation.

# Setup

## Data

First, you should initialize and update the [dataset](https://github.com/MTG/otmm_makam_recognition_dataset/releases/tag/dlfm2016), which is linked as a submodule:

    ```bash
    cd path/to/makam_recognition_experiments
    git submodule init
    git submodule update
    ```

Zenodo data XX

## Docker

For the sake of reproducibility, you can run the experiments within Docker. To run Docker, you need to setup the Docker engine. Please refer to the [documentation](https://docs.docker.com/install/) to how to install the free, community version.

## Running Experiments

**Note:** We suggest you to use a cluster to run the training and testing steps. Otherwise it might take **days** to reproduce the experiments.

To run the container simply run on the terminal:

    ```bash
    docker-compose up
    ```

Then follow the link in the terminal to open [Jupyter](https://jupyter.org).

Further instructions XX.

## Development

We use `tox` on a virtualenv to automate unittests, linting. The `Makefile` automates the steps. On the terminal simply run:

    ```bash
    make dev
    ```

which will create a virtualenv called `venv` and install `tox`. To test your work, then run:

    ```bash
    make tox
    ```

## License

The source code hosted in this repository is licenced under [Affero GPL version 3](https://www.gnu.org/licenses/agpl-3.0.en.html). The data (the features, models,  figures, results etc.) are licenced under [Creative Commons Attribution-NonCommercial-ShareAlike 4.0 International License](http://creativecommons.org/licenses/by-nc-sa/4.0/).
