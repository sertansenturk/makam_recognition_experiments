import configparser
import os


def read():
    config = configparser.ConfigParser()
    config_file = _get_config_filepath()
    config.read(config_file)

    return config


def _get_config_filepath():
    return os.path.join(os.path.dirname(__file__), 'config.ini')
