import os

import mock
import pytest
import unittest

import pandas as pd

from mre.data import Annotation


class AnnotationTestCase(unittest.TestCase):
    @pytest.mark.parametrize("data,expected", [
        (pd.DataFrame([{"dunya_uid": None, "mbid": "some_mbid"}]),
         pd.DataFrame([{"dunya_uid": "some_mbid", "mbid": "some_mbid"}]))])
    @mock.patch('mre.data.Annotation.__init__')
    def test_patch_dunya_uids(self, mock_init, data, expected):
        # set up mock
        mock_init.return_value = None
        annotations = Annotation()

        # GIVEN
        data = pd.DataFrame([{
            "dunya_uid": None,
            "mbid": "some_mbid"
        }])
        annotations.data = data

        # WHEN
        annotations._patch_dunya_uids()

        import pdb; pdb.set_trace()
        # THEN
        annotations.data.equals(expected)

# [
#         (pd.DataFrame([{"dunya_uid": None, "mbid": "some_mbid"}]),
#          pd.DataFrame([{"dunya_uid": "some_mbid", "mbid": "some_mbid"}])),
#         (pd.DataFrame([{"dunya_uid": "some_uid", "mbid": "some_mbid"}]),
#          pd.DataFrame([{"dunya_uid": "some_uid", "mbid": "some_mbid"}])),
#         (pd.DataFrame([{"dunya_uid": "some_uid", "mbid": None}]),
#          pd.DataFrame([{"dunya_uid": "some_uid", "mbid": None}]))]

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
