import os
from unittest import mock

import numpy as np
import pandas as pd
from pandas.testing import assert_frame_equal

import pytest
from mre.data import Annotation
from mre.config import config


@pytest.fixture
def mock_annotation(scope="session") -> Annotation:
    with mock.patch("mre.data.Annotation.__init__",
                    return_value=None,
                    autospec=True):
        return Annotation()


class TestAnnotation:
    @mock.patch('pandas.read_json', autospec=True)
    def test_read_from_github(self, mock_read_json, mock_annotation):
        # GIVEN
        mock_annotation.URL = "mock_url"

        # WHEN
        _ = mock_annotation._read_from_github()

        # THEN
        mock_read_json.assert_called_once_with(mock_annotation.URL)

    def test_validate_num_recordings(self, mock_annotation):
        # GIVEN
        mock_annotation.data = pd.DataFrame(
            [{"col1": "val1"}, {"col1": "val2"}])
        mock_annotation.EXPECTED_NUM_RECORDINGS = 2

        # WHEN; THEN
        mock_annotation._validate_num_recordings()

    def test_validate_mbids(self, mock_annotation):
        # GIVEN
        mock_annotation.data = pd.DataFrame(
            [{"mbid": "id1"}, {"mbid": "id2"}])

        # WHEN; THEN
        mock_annotation._validate_mbids()

    def test_validate_nonunique_mbids(self, mock_annotation):
        # GIVEN
        mock_annotation.data = pd.DataFrame(
            [{"mbid": "id1"}, {"mbid": "id1"}])

        # WHEN; THEN
        with pytest.raises(ValueError):
            mock_annotation._validate_mbids()

    @pytest.mark.parametrize("data", [
        pd.DataFrame([{"mbid": ""}]),
        pd.DataFrame([{"mbid": None}]),
        pd.DataFrame([{"mbid": np.nan}])])
    def test_validate_missing_mbid(self, mock_annotation, data):
        # GIVEN
        mock_annotation.data = data

        # WHEN; THEN
        with pytest.raises(ValueError):
            mock_annotation._validate_mbids()

    @pytest.mark.parametrize("data,expected", [
        (pd.DataFrame([{"makam": "makam1"}]), 1),
        (pd.DataFrame([{"makam": "makam1"}, {"makam": "makam2"}]), 1),
        (pd.DataFrame([{"makam": "makam1"}, {"makam": "makam1"},
                       {"makam": "makam2"}, {"makam": "makam2"}]), 2)])
    def test_val_num_recs_per_makam(self, mock_annotation, data, expected):
        # GIVEN
        mock_annotation.data = data
        mock_annotation.EXPECTED_NUM_RECORDINGS_PER_MAKAM = expected

        # WHEN; THEN
        mock_annotation._validate_num_recordings_per_makam()

    def test_val_num_recs_per_makam_unbalanced(self, mock_annotation):
        # GIVEN
        mock_annotation.data = pd.DataFrame([  # unbalanced
            {"makam": "makam1"}, {"makam": "makam1"}, {"makam": "makam2"}])

        # WHEN; THEN
        with pytest.raises(ValueError):
            mock_annotation._validate_num_recordings_per_makam()

    def test_val_num_recs_per_makam_unexpected(self, mock_annotation):
        # GIVEN
        mock_annotation.data = pd.DataFrame([  # 2 recordings per makams
            {"makam": "makam1"}, {"makam": "makam2"}])
        mock_annotation.EXPECTED_NUM_RECORDINGS_PER_MAKAM = 5  # not 2

        # WHEN; THEN
        with pytest.raises(ValueError):
            mock_annotation._validate_num_recordings_per_makam()

    @pytest.mark.parametrize("data,expected", [
        (pd.DataFrame([{"makam": "makam1"}]), 1),
        (pd.DataFrame([{"makam": "makam1"}, {"makam": "makam2"}]), 2)])
    def test_validate_num_makams(self, mock_annotation, data, expected):
        # GIVEN
        mock_annotation.data = data
        mock_annotation.EXPECTED_NUM_MAKAMS = expected

        # WHEN; THEN
        mock_annotation._validate_num_makams()

    def test_validate_num_makams_incorrect(self, mock_annotation):
        # GIVEN
        mock_annotation.data = pd.DataFrame([  # 3 makams
            {"makam": "makam1"}, {"makam": "makam2"}, {"makam": "makam3"}])
        mock_annotation.EXPECTED_NUM_MAKAMS = 5  # not 2

        # WHEN; THEN
        with pytest.raises(ValueError):
            mock_annotation._validate_num_makams()

    def test_parse_mbid_urls(self, mock_annotation):
        # GIVEN
        mb_url = "http://musicbrainz.org/recording/some_mbid"
        mock_annotation.data = pd.DataFrame([{"mbid": mb_url}])

        # WHEN
        mock_annotation._parse_mbid_urls()

        # THEN
        expected = pd.DataFrame([{"mb_url": mb_url, "mbid": "some_mbid"}])

        assert_frame_equal(mock_annotation.data, expected, check_like=True)

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
        mock_annotation.data = pd.DataFrame([{"mbid": invalid_url}])

        # WHEN; THEN
        with pytest.raises(ValueError):
            mock_annotation._parse_mbid_urls()

    @pytest.mark.parametrize("data,expected", [
        (pd.DataFrame([{"dunya_uid": None, "mbid": "some_mbid"}]),
         pd.DataFrame([{"dunya_uid": "some_mbid", "mbid": "some_mbid"}])),
        (pd.DataFrame([{"dunya_uid": "some_uid", "mbid": "some_mbid"}]),
         pd.DataFrame([{"dunya_uid": "some_uid", "mbid": "some_mbid"}])),
        (pd.DataFrame([{"dunya_uid": "some_uid", "mbid": None}]),
         pd.DataFrame([{"dunya_uid": "some_uid", "mbid": None}]))])
    def test_patch_dunya_uids(self, mock_annotation, data, expected):
        # GIVEN
        mock_annotation.data = data

        # WHEN
        mock_annotation._patch_dunya_uids()

        # THEN
        assert_frame_equal(mock_annotation.data, expected, check_like=True)
