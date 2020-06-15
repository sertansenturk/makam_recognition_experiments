import os

import mock
import unittest

from mre.config import config


class ReadTestCase(unittest.TestCase):
    @mock.patch.object(config.configparser.ConfigParser, 'read')
    def test_read(self, mock_read):
        # set up mock
        config.read()

        # test that rm called os.remove with the right parameters
        mock_read.assert_called_with(config._get_config_filepath())


def test_get_config_filepath():
    # GIVEN
    pass

    # WHEN
    path = config._get_config_filepath()

    # THEN
    assert os.path.exists(path)
