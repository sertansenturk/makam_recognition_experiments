import os

from mre.config import config


def test_get_config_filepath():
    os.path.exists(config._get_config_filepath())
