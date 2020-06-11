import configparser


def read_config():
    config = configparser.ConfigParser()
    config_file = config.read('config.ini')

    return config
