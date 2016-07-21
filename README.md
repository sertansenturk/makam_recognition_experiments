# DLfM Ottoman-Turkish Makam Recognition and Tonic Identification Experiments

This repository hosts the experiments conducted, which demonstrates [MORTY](https://github.com/altugkarakurt/morty) (our mode and tonic estmation toolbox) in the paper:

> Karakurt, A., Şentürk S., & Serra X. (2016).  [MORTY: A Toolbox for Mode Recognition and Tonic Identification](http://mtg.upf.edu/node/3538). 3rd International Digital Libraries for Musicology Workshop. New York, USA

Please cite the paper above, if you are using the contents of this repository for your works.


### Structure of the Repository
- The scripts are located in the base folder along with several miscallenaeous files (the license, readme, setup and requirement files).
- The folder [./data](https://github.com/sertansenturk/makam_recognition_experiments/tree/master/data) links to the relevant commit in [our makam recognition dataset](https://github.com/MTG/otmm_makam_recognition_dataset/releases/tag/dlfm2016), the folds and the summary of the evaluation obtained from all experiments. Due to file size constraints features, training models, results and evaluation files are not included in this folder and stored in [Zenodo](https://zenodo.org/record/57999) instead.
- The folder [./dlfm_code](https://github.com/sertansenturk/makam_recognition_experiments/tree/master/dlfm_code) has the relevant Python and MATLAB modules for the training, testing and evaluation.

### Installation

If you want to install the Python package, it is recommended to install the package and dependencies into a virtualenv. In the terminal, do the following:

    virtualenv env
    source env/bin/activate
    python setup.py install

The package and some of its dependencies use several modules in Essentia. Follow the [instructions](essentia.upf.edu/documentation/installing.html) to install the library.

Now you can install the rest of the dependencies:

    pip install -r requirements
    
### Scripts



### LICENCE

The source code hosted in this repository is licenced under [Affero GPL version 3](https://www.gnu.org/licenses/agpl-3.0.en.html). The data (the features, models,  figures, results etc.) are licenced under [Creative Commons Attribution-NonCommercial-ShareAlike 4.0 International License](http://creativecommons.org/licenses/by-nc-sa/4.0/).
