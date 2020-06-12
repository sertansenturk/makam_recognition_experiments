import configparser
import os


def read_config():
    config = configparser.ConfigParser()
    config_file = get_config_filepath()
    config.read(config_file)

    return config


def get_config_filepath():
    return os.path.join(os.path.dirname(__file__), 'config.ini')
