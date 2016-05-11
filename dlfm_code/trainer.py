import json
import os

import numpy as np
from fileoperations.fileoperations import get_filenames_in_dir
from morty.classifiers.knnclassifier import KNNClassifier
from morty.pitchdistribution import PitchDistribution
from dlfm_code import io


def compute_recording_distributions(step_size, kernel_width, distribution_type,
                                    anno, overwrite=False):
    # get mbid
    mbid = os.path.split(anno['mbid'])[-1]

    base_folder = os.path.join('.', 'data', 'features')
    feature_folder = os.path.abspath(io.get_folder(
        base_folder, distribution_type, step_size, kernel_width))
    if not os.path.exists(feature_folder):
        os.makedirs(feature_folder)

    raw_save_file = os.path.join(feature_folder,
                                 u'{0:s}--hist.json'.format(mbid))
    norm_save_file = os.path.join(feature_folder,
                                  u'{0:s}--pdf.json'.format(mbid))

    if not overwrite and os.path.exists(norm_save_file):
        return norm_save_file + ' skipped.'

    pitch_file = os.path.abspath(os.path.join(
        anno['dataset_path'], 'data', anno['makam'], mbid + '.pitch'))
    pitch = np.loadtxt(pitch_file)

    # compute histogram (for single training sample per mode)
    feature = PitchDistribution.from_hz_pitch(
        pitch, ref_freq=anno['tonic'], kernel_width=kernel_width,
        step_size=step_size, norm_type=None)
    if distribution_type == 'pcd':
        feature.to_pcd()

    dp = {'feature': feature.to_dict(), 'mode': anno['makam'],
          'source': anno['mbid'], 'tonic': anno['tonic']}
    json.dump(dp, open(raw_save_file, 'w'))

    # compute probability density function (for multi training sample per
    # mode)
    feature.normalize()
    dp = {'feature': feature.to_dict(), 'mode': anno['makam'],
          'source': anno['mbid'], 'tonic': anno['tonic']}
    json.dump(dp, open(norm_save_file, 'w'))

    return norm_save_file + ' computed.'


def train_single(step_size, kernel_width, distribution_type, fold_tuple,
                 overwrite=False):
    training_file = io.get_training_file(
        step_size, kernel_width, distribution_type, 'single', fold_tuple)

    if not overwrite and os.path.exists(training_file):
        return training_file + ' skipped.'

    # get features
    base_folder = os.path.join('.', 'data', 'features')
    feature_folder = os.path.abspath(io.get_folder(
        base_folder, distribution_type, step_size, kernel_width))
    # get histogram files computed from the audio recordings
    feature_files = get_filenames_in_dir(feature_folder,
                                         keyword='*hist.json')[0]

    # initialize temporary model. use dict for easy mapping
    training = fold_tuple[1]['training']
    makams = set(training['modes'])
    tmp_model = {}
    for makam in makams:
        tmp_model[makam] = {'mode': makam, 'sources': [], 'feature': None}

    # compute the single distribution from the distribution
    for ff in feature_files:
        for i, mbid in enumerate(training['sources']):
            if mbid in ff:
                data = json.load(open(ff))

                tmp_model[data['mode']]['sources'].append(mbid)
                if tmp_model[data['mode']]['feature'] is None:
                    tmp_model[data['mode']]['feature'] = PitchDistribution.\
                        from_dict(data['feature'])
                else:
                    tmp_model[data['mode']]['feature'] = \
                        tmp_model[data['mode']]['feature'].merge(
                            PitchDistribution.from_dict(data['feature']))

    # verify the computed distribution of the modes and normalize features
    # and collapse the temporary model dict
    model = []
    for val in tmp_model.values():
        assert len(val['sources']) == 45, 'The mode should have been 45 ' \
                                          'recordings to train'
        val['feature'].normalize()
        model.append(val)

    # save the model
    KNNClassifier.model_to_json(model, training_file)

    return training_file + ' created.'


def train_multi(step_size, kernel_width, distribution_type, fold_tuple,
                overwrite=False):
    # check if the model is already trained
    training_file = io.get_training_file(
        step_size, kernel_width, distribution_type, 'multi', fold_tuple)

    if not overwrite and os.path.exists(training_file):
        return training_file + ' skipped.'

    # get feature files
    base_folder = os.path.join('.', 'data', 'features')
    feature_folder = io.get_folder(base_folder, distribution_type, step_size,
                                   kernel_width)
    # get probability density functions computed from audio recordings
    feature_files = get_filenames_in_dir(feature_folder,
                                         keyword='*pdf.json')[0]

    # gather the distribution files of the training model
    training = fold_tuple[1]['training']
    model_files = []
    for ff in feature_files:
        for i, mbid in enumerate(training['sources']):
            if mbid in ff:  # keep the filenames for compactness
                model_files.append(ff)

    assert len(model_files) == 900, 'The model should have been 900 ' \
                                    'recordings to train'

    # save the model
    json.dump(model_files, open(training_file, 'w'))

    return training_file + ' created.'
