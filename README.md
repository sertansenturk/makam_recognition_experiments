[![DOI](https://zenodo.org/badge/21104/sertansenturk/makam_recognition_experiments.svg)](https://zenodo.org/badge/latestdoi/21104/sertansenturk/makam_recognition_experiments)

# DLfM Ottoman-Turkish Makam Recognition and Tonic Identification Experiments

This repository hosts the experiments conducted, which demonstrates [MORTY](https://github.com/altugkarakurt/morty) (our mode and tonic estmation toolbox) in the paper:

> Karakurt, A., Şentürk S., & Serra X. (2016).  [MORTY: A Toolbox for Mode Recognition and Tonic Identification](http://mtg.upf.edu/node/3538). 3rd International Digital Libraries for Musicology Workshop. New York, USA

Please cite the paper above, if you are using the contents of this repository for your works. [The companion page](http://compmusic.upf.edu/node/319) of the paper is accesible via the CompMusic website.

### Structure of the Repository
- The scripts are located in the base folder along with several miscallenaeous files (the license, readme, setup and requirement files).
- The folder [./data](https://github.com/sertansenturk/makam_recognition_experiments/tree/master/data) links to the relevant commit in [our makam recognition dataset](https://github.com/MTG/otmm_makam_recognition_dataset/releases/tag/dlfm2016), the folds and the summary of the evaluation obtained from all experiments. 
- By running the [Jupyter notebooks](#scripts) in this repository, you can reproduce the extensive experiments reported in the paper. The outputs will also be saved to the folder [./data](https://github.com/sertansenturk/makam_recognition_experiments/tree/master/data). However the experiments might run for days (in a local machine), unless you use a cluster. For this reason, the computed features, training models, results and evaluation files are also dowloadable from Zenodo ([link](https://zenodo.org/record/57999)).
- The folder [./dlfm_code](https://github.com/sertansenturk/makam_recognition_experiments/tree/master/dlfm_code) has the relevant Python and MATLAB modules for the training, testing and evaluation.

### Setup

If you want to install the Python package, it is recommended to install the package and dependencies into a virtualenv. In the terminal, do the following (don't forget to change `path_to_env` with the actual path of the virtualenv, you'd like to create):

    virtualenv path_to_env --system-site-packages
    source path_to_env/bin/activate

The virtualenv is created with the `--system-site-packages` option, because of a [bug](http://www.stevenmaude.co.uk/posts/installing-matplotlib-in-virtualenv) related to matplotlib package, which is one of our requirements.

The package and some of its dependencies use several modules in Essentia. Follow the [instructions](http://essentia.upf.edu/documentation/installing.html) to install the library. Then you should link the python bindings of Essentia in the virtual environment:

    ln -s path_to_essentia_bindings path_to_env/lib/python2.7/site-packages
    
Don't forget to change the `path_to_essentia_bindings` and `path_to_env` with the actual path of the installed Essentia Python bindings and the path of your virtualenv, respectively. Depending on the Essentia version, the default installation path of the Essentia bindings is either `/usr/local/lib/python2.7/dist-packages/essentia` or `/usr/local/lib/python2.7/site-packages/essentia`.

Next, enter to the directory of the repository in the terminal:

    cd path_to_makam_recognition_experiments

You should initialize and update the [dataset](https://github.com/MTG/otmm_makam_recognition_dataset/releases/tag/dlfm2016), which is linked as a submodule:

    git submodule init
    git submodule update

Now you can install the rest of the dependencies:

    pip install -r requirements

Note that you might need to install several additional packages for the dependencies depending on your operating system. 
    
### Experimentation Scripts
<a name="scripts"></a>

We use Jupyter notebooks for the general experimentation and MATLAB for statistical significance tests. To open the notebooks, simply run `jupyter notebook` in your terminal __with the virtualenv activated.__

Note that the notebooks use parallelization to speed up the run time. You have to open a __second terminal with the virtualenv activated__ (`source path_to_env/bin/activate`) and run `ipcluster start` in that terminal before running the scripts.

To reproduce the experiments, run the scripts in the order given below. 

1. [setup_feature_training.ipynb](https://github.com/sertansenturk/makam_recognition_experiments/blob/master/setup_feature_training.ipynb): Create the folds in the stratified 10 fold scheme, extract distribution features, train models. 
2. [testing_evaluation.ipynb](https://github.com/sertansenturk/makam_recognition_experiments/blob/master/testing_evaluation.ipynb): Find optimal minimum peak ratio (Figure 3 in the paper), testing and evaluation.
3. [plot_tonicdist_confusionmat.ipynb](https://github.com/sertansenturk/makam_recognition_experiments/blob/master/plot_tonicdist_confusionmat.ipynb): Display the tonic identification errors for all parameter sets with 7.5 cent bin size and the confusion matrix in mode recognition for the best parameter set
4. [summarize_evaluation.m](https://github.com/sertansenturk/makam_recognition_experiments/blob/master/summarize_evaluation.m): Read and store a summarization of the evaluation obtained for all parameter sets
5. [stat_significance.m](https://github.com/sertansenturk/makam_recognition_experiments/blob/master/stat_significance.m): Conduct statistical significance tests. The last block in the MATLAB code is where the tests are carried semi-automatically. The parameters that are checked for significance are commented.

 We suggest you to use a cluster to run the training and testing steps. Otherwise it might take __days__ to reproduce the experiments.

### License

The source code hosted in this repository is licenced under [Affero GPL version 3](https://www.gnu.org/licenses/agpl-3.0.en.html). The data (the features, models,  figures, results etc.) are licenced under [Creative Commons Attribution-NonCommercial-ShareAlike 4.0 International License](http://creativecommons.org/licenses/by-nc-sa/4.0/).
