import os
from unittest import mock

import pandas as pd
from pandas.testing import assert_frame_equal

import pytest
from mre.data import Annotation


@pytest.fixture
def mock_annotation(scope="session"):
    with mock.patch("mre.data.Annotation.__init__", return_value=None):
        annotations = Annotation()
    return annotations


class TestAnnotation:
    @pytest.mark.parametrize("invalid_url", [
        None,
        "",
        "string",
        "string/some_mbid",
        "http://wrongdomain.com/some_mbid",
        "http://musicbrainz.org/work/some_mbid",  # not recording
    ])
    def test_parse_mbid_urls_invalid_url(self, mock_annotation, invalid_url):
        # GIVEN
        annotations = mock_annotation
        annotations.data = pd.DataFrame([{"mbid": invalid_url}])

        # WHEN; THEN
        with pytest.raises(ValueError):
            annotations._parse_mbid_urls()

    def test_parse_mbid_urls(self, mock_annotation):
        # GIVEN
        annotations = mock_annotation
        mb_url = "http://musicbrainz.org/recording/some_mbid"
        annotations.data = pd.DataFrame([{"mbid": mb_url}])

        # WHEN
        annotations._parse_mbid_urls()

        # THEN
        expected = pd.DataFrame([{"mb_url": mb_url, "mbid": "some_mbid"}])

        assert_frame_equal(annotations.data, expected, check_like=True)

    @pytest.mark.parametrize("data,expected", [
        (pd.DataFrame([{"dunya_uid": None, "mbid": "some_mbid"}]),
         pd.DataFrame([{"dunya_uid": "some_mbid", "mbid": "some_mbid"}])),
        (pd.DataFrame([{"dunya_uid": "some_uid", "mbid": "some_mbid"}]),
         pd.DataFrame([{"dunya_uid": "some_uid", "mbid": "some_mbid"}])),
        (pd.DataFrame([{"dunya_uid": "some_uid", "mbid": None}]),
         pd.DataFrame([{"dunya_uid": "some_uid", "mbid": None}]))])
    def test_patch_dunya_uids(self, mock_annotation, data, expected):
        # GIVEN
        annotations = mock_annotation
        annotations.data = data

        # WHEN
        annotations._patch_dunya_uids()

        # THEN
        assert_frame_equal(annotations.data, expected, check_like=True)



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
