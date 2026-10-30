"""``VideoTrackingData.get_object_data``: animal ids match exactly, other names by a unique substring."""

import pandas as pd
import pytest

from video.tracking_import import VideoTrackingData

APT_KEYS = ['rat613', 'rat615', 'rat630', 'rat631', 'rat634', 'rat635']


def _tracking(keys):
    return VideoTrackingData(animal_id='631', session_id='20251216',
                             parsed_data={k: pd.DataFrame({'name': [k]}) for k in keys})


def _name(df):
    return None if df is None else df['name'].iloc[0]


@pytest.mark.parametrize('query', ['631', 'rat631', 'RAT631'])
def test_full_ids_resolve(query):
    assert _name(_tracking(APT_KEYS).get_object_data(query)) == 'rat631'


def test_the_apt_prefix_resolves_to_the_normalised_key():
    assert _name(_tracking(APT_KEYS).get_object_data('rat4635')) == 'rat635'


@pytest.mark.parametrize('query', ['63', '13', 'rat63', '6310'])
def test_partial_or_longer_ids_never_match_by_substring(query):
    assert _tracking(APT_KEYS).get_object_data(query) is None


def test_other_names_match_one_key_by_substring():
    nwb = _tracking(['mouse_center', 'reward_port'])
    assert _name(nwb.get_object_data('mouse')) == 'mouse_center'
    assert nwb.get_object_data('lever') is None


def test_an_ambiguous_name_raises():
    with pytest.raises(ValueError, match='ambiguous'):
        _tracking(['mouse_center', 'mouse_nose']).get_object_data('mouse')
