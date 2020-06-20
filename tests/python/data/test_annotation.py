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





# def test_validate():
#     assert True


class TestAnnotation:
    @mock.patch('pandas.read_json')
    def test_read_from_github(self, mock_read_json, mock_annotation):
        # WHEN
        annotations = mock_annotation
        annotations.url = "mock_url"
        _ = annotations._read_from_github()

        # THEN
        mock_read_json.assert_called_once_with(annotations.url)

    def test_get_num_recs_per_makam_inbalanced_dataset(self, mock_annotation):
        # GIVEN
        annotations = mock_annotation
        annotations.data = pd.DataFrame([  # inbalanced
            {"makam": "makam1"}, {"makam": "makam1"}, {"makam": "makam2"}])

        # WHEN; THEN
        with pytest.raises(ValueError):
            annotations._get_num_recs_per_makam()


    @pytest.mark.parametrize("data,expected", [
        (pd.DataFrame([{"makam": "makam1"}]), 1),
        (pd.DataFrame([{"makam": "makam1"}, {"makam": "makam2"}]), 1),
        (pd.DataFrame([{"makam": "makam1"}, {"makam": "makam1"},
                       {"makam": "makam2"}, {"makam": "makam2"}]), 2)])
    def test_get_num_recs_per_makam(self, mock_annotation, data, expected):
        # GIVEN
        annotations = mock_annotation
        annotations.data = data

        # WHEN
        result = annotations._get_num_recs_per_makam()

        # THEN
        assert result == expected


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
