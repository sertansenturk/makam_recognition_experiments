from fileoperations.fileoperations import get_filenames_in_dir
from morty.pitchdistribution import PitchDistribution
from morty.evaluator import Evaluator
from morty.converter import Converter
from matplotlib import pyplot as plt
from dlfm_code import io
from morty.classifiers.knnclassifier import KNNClassifier
import os
import json
import numpy as np
import copy


def test(step_size, kernel_width, distribution_type,
         model_type, fold_idx, experiment_type, dis_measure, k_neighbor,
         min_peak_ratio, rank, save_folder, overwrite=False):

    # file to save the results
    res_dict = {'saved': [], 'failed': [], 'skipped': []}
    test_folder = os.path.abspath(os.path.join(io.get_folder(
        os.path.join(save_folder, 'testing', experiment_type), model_type,
        distribution_type, step_size, kernel_width, dis_measure,
        k_neighbor, min_peak_ratio), 'fold{0:d}'.format(fold_idx)))
    results_file = os.path.join(test_folder, 'results.json')
    if not os.path.exists(test_folder):
        os.makedirs(test_folder)
    else:
        if os.path.exists(results_file):
            return u"{0:s} already has results.".format(test_folder)

    # load fold
    fold_file = os.path.join(save_folder, 'folds.json')
    folds = json.load(open(fold_file))
    test_fold = []
    for f in folds:
        if f[0] == fold_idx:
            test_fold = f[1]['testing']
            break

    assert len(test_fold) == 100, "There should be 100 samples in the test " \
                                  "fold"

    # load training model
    training_folder = os.path.abspath(io.get_folder(
        os.path.join(save_folder, 'training'), model_type,
        distribution_type, step_size, kernel_width))

    model_file = os.path.join(training_folder,
                              u'fold{0:d}.json'.format(fold_idx))
    model = json.load(open(model_file))
    # instantiate the PitchDistributions
    for i, m in enumerate(model):
        try:  # filepath given
            model[i] = json.load(open(os.path.join(save_folder, m)))
        except TypeError:  # dict already loaded
            assert isinstance(m['feature'], dict), "Unknown model."
        model[i]['feature'] = PitchDistribution.from_dict(
            model[i]['feature'])

    for test_sample in test_fold:
        # get MBID from pitch file
        mbid = test_sample['source']
        save_file = os.path.join(test_folder, u'{0:s}.json'.format(mbid))
        if not overwrite and os.path.exists(save_file):
            res_dict['skipped'].append(save_file)
            continue

        # instantiate the classifier and evaluator object
        classifier = KNNClassifier(
            step_size=step_size, kernel_width=kernel_width,
            feature_type=distribution_type, model=copy.deepcopy(model))

        # if the model_type is multi and the test data is in the model,
        # remove it
        if model_type == 'multi':
            for i, m in enumerate(classifier.model):
                if mbid in m:
                    del classifier.model[i]
                    break

        try:
            # we use the pitch instead of the distribution already computed in
            # the feature extraction. those distributions are normalized wrt
            # tonic to one of the bins centers will exactly correspond to
            # the tonic freq. therefore it would be cheating
            pitch = np.loadtxt(test_sample['pitch'])
            if experiment_type == 'tonic':  # tonic identification
                results = classifier.estimate_tonic(
                    pitch, test_sample['mode'], min_peak_ratio=min_peak_ratio,
                    distance_method=dis_measure, k_neighbor=k_neighbor,
                    rank=rank)
            elif experiment_type == 'mode':  # mode recognition
                results = classifier.estimate_mode(
                    pitch, test_sample['tonic'], distance_method=dis_measure,
                    k_neighbor=k_neighbor, rank=rank)
            elif experiment_type == 'joint':  # joint estimation
                results = classifier.estimate_joint(
                    pitch, min_peak_ratio=min_peak_ratio,
                    distance_method=dis_measure, k_neighbor=k_neighbor,
                    rank=rank)
            else:
                raise ValueError("Unknown experiment_type")

            # save results
            json.dump(results, open(save_file, 'w'))
            res_dict['saved'].append(save_file)
        except:
            res_dict['failed'].append(save_file)

    if not res_dict['failed']:
        computed = get_filenames_in_dir(test_folder, keyword='*.json')[0]
        assert len(computed) == 100, 'There should have been 100 tested files.'

        results = {}
        for c in computed:
            mbid = os.path.splitext(os.path.split(c)[-1])[0]
            results[mbid] = json.load(open(c))

        json.dump(results, open(results_file, 'w'), indent=4)
        for c in computed:
            os.remove(c)
    return res_dict


def search_min_peak_ratio(step_size, kernel_width, distribution_type,
                          min_peak_ratio):
    base_folder = os.path.join('data', 'features')
    feature_folder = os.path.abspath(io.get_folder(
        base_folder, distribution_type, step_size, kernel_width))
    files = get_filenames_in_dir(feature_folder, keyword='*pdf.json')[0]
    evaluator = Evaluator()
    num_peaks = 0
    num_tonic_in_peaks = 0
    for f in files:
        dd = json.load(open(f))
        dd['feature'] = PitchDistribution.from_dict(dd['feature'])

        peak_idx = dd['feature'].detect_peaks(min_peak_ratio=min_peak_ratio)[0]
        peak_cents = dd['feature'].bins[peak_idx]
        peak_freqs = Converter.cent_to_hz(peak_cents, dd['tonic'])

        ev = [evaluator.evaluate_tonic(pp, dd['tonic'])['tonic_eval']
              for pp in peak_freqs]

        num_tonic_in_peaks += any(ev)
        num_peaks += len(ev)

    return num_tonic_in_peaks, num_peaks


def plot_min_peak_ratio(min_peak_ratios, ratio_tonic, num_peak,
                        prob_tonic=None, num_exps=None):
    fig, ax1 = plt.subplots()
    ax1.plot(min_peak_ratios, ratio_tonic, 'bd-',
             label='Ratio of the tests with the tonic')
    if prob_tonic is not None:
        ax1.plot(min_peak_ratios, prob_tonic, 'b*-',
                 label='Prior probability of tonic')
    ax1.set_ylabel('Probability of getting the tonic\namong the '
                   'detected peaks', color='b')
    ax1.set_ylim([0, 1])
    for tl in ax1.get_yticklabels():
        tl.set_color('b')
    plt.setp(ax1, xticks=[])

    ax2 = ax1.twinx()
    ax2.plot(min_peak_ratios, num_peak, 'r.-', label='Total number of peaks')
    ax2.set_ylabel('# peaks', color='r')
    for tl in ax2.get_yticklabels():
        tl.set_color('r')

    plt.setp(ax2, xticks=min_peak_ratios)
    ax1.set_xticklabels(min_peak_ratios, rotation=-60)
    ax1.set_xlabel('Minimum Peak Ratio')

    # h1, l1 = ax1.get_legend_handles_labels()
    # h2, l2 = ax2.get_legend_handles_labels()
    # ax1.legend(h1 + h2, l1 + l2)

    if num_exps is not None:
        plt.title(
            'Results wrt minimum_peak_ratio values computed\nusing '
            '{0:d} recordings in {1:d} experiments'.format(1000 * num_exps,
                                                           num_exps))

    plt.show()
