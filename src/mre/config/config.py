import configparser
import os


def read() -> configparser.ConfigParser:
    """reads mre configuration file and returns the parser

    Returns
    -------
    configparser.ConfigParser
        configparser with mre configuration
    """
    cfg = configparser.ConfigParser()
    config_file = _get_config_filepath()
    cfg.read(config_file)

    return cfg


def _get_config_filepath() -> str:
    """returns the path of the mre configuration file

    Returns
    -------
    str
        path of the mre configuration file
    """
    return os.path.join(os.path.dirname(__file__), 'config.ini')
