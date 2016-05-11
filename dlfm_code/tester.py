from fileoperations.fileoperations import get_filenames_in_dir
from morty.pitchdistribution import PitchDistribution
from morty.evaluator import Evaluator
from morty.converter import Converter
from matplotlib import pyplot as plt
from dlfm_code import io
import os
import json


def search_min_peak_ratio(step_size, kernel_width, distribution_type,
                          min_peak_ratio):
    base_folder = 'data/features'

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


def plot_min_peak_ratio(min_peak_ratios, per_tonic, num_peak):
    fig, ax1 = plt.subplots()
    ax1.plot(min_peak_ratios, per_tonic, 'bd-')
    ax1.set_ylabel('% cases where the tonic is in the peaks', color='b')
    for tl in ax1.get_yticklabels():
        tl.set_color('b')
    plt.setp(ax1, xticks=[])

    ax2 = ax1.twinx()
    ax2.plot(min_peak_ratios, num_peak, 'r.-')
    ax2.set_ylabel('Total number of peaks', color='r')
    for tl in ax2.get_yticklabels():
        tl.set_color('r')

    plt.setp(ax2, xticks=min_peak_ratios)
    ax1.set_xticklabels(min_peak_ratios, rotation=-60)
    plt.show()
