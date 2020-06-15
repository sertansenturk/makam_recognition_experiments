import os
from unittest import mock

import pytest

import pandas as pd

from mre.data import Annotation


class TestAnnotation(object):
    @pytest.mark.parametrize("data,expected", [
        (pd.DataFrame([{"dunya_uid": None, "mbid": "some_mbid"}]),
         pd.DataFrame([{"dunya_uid": "some_mbid", "mbid": "some_mbid"}])),
        (pd.DataFrame([{"dunya_uid": "some_uid", "mbid": "some_mbid"}]),
         pd.DataFrame([{"dunya_uid": "some_uid", "mbid": "some_mbid"}])),
        (pd.DataFrame([{"dunya_uid": "some_uid", "mbid": None}]),
         pd.DataFrame([{"dunya_uid": "some_uid", "mbid": None}]))])
    @mock.patch('mre.data.Annotation.__init__')
    def test_patch_dunya_uids(self, mock_init, data, expected):
        # set up mock
        mock_init.return_value = None
        annotations = Annotation()

        # GIVEN
        annotations.data = data

        # WHEN
        annotations._patch_dunya_uids()

        # THEN
        assert annotations.data.equals(expected)



# def test_head():
#     assert True


# def mock_annotation():
#     assert True


# def test_get_num_recs_per_makam():
#     assert True


# def test_validate():
#     assert True


# def test_parse_mbid_urls():
#     assert True


def test_patch_dunya_uids():
    # GIVEN

    # WHEN

    # THEN

    assert True
