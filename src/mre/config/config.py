import configparser
import os


def read() -> configparser.ConfigParser:
    """reads mre configuration file and returns as a config parser

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


def read_secrets() -> configparser.ConfigParser:
    """reads mre secrets file and returns as a config parser

    Returns
    -------
    configparser.ConfigParser
        configparser with mre secrets configuration
    """
    secrets = configparser.ConfigParser()
    secrets_file = _get_secrets_filepath()
    secrets.read(secrets_file)

    return secrets


def _get_secrets_filepath() -> str:
    """returns the path of the mre secrets file

    Returns
    -------
    str
        path of the mre secrets file
    """
    return os.path.join(os.path.dirname(__file__), 'secrets.ini')
