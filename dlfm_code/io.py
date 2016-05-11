import os
import numbers


def get_folder(base_folder, *params):
    # convert numbers to string with the dot replaced with underscore
    params = [str(p).replace('.', '_') if isinstance(p, numbers.Number) else p
              for p in params]

    # join the parameters
    tmp_str = '--'.join(params)

    # specify save path
    folder = os.path.join(base_folder, tmp_str)

    return folder


def get_training_file(step_size, kernel_width, distribution_type,
                      model_type, fold_tuple):
    # check if the model is already trained
    training_folder = get_folder(
        os.path.join('.', 'data', 'training'), model_type,
        distribution_type, step_size, kernel_width)
    fold_idx = fold_tuple[0]
    training_file = os.path.join(training_folder,
                                 u"fold{0:d}.json".format(fold_idx))

    if not os.path.exists(training_folder):
        os.makedirs(training_folder)

    return training_file
