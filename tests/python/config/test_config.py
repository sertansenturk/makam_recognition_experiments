import os

import mre.config.config as config


def test_get_config_filepath():
    os.path.exists(config.get_config_filepath())
