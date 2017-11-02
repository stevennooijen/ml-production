import pandas as pd
from pandas.util.testing import assert_series_equal

from shelter import data


def test_convert_camel_case():
    assert data.convert_camel_case('CamelCase') == 'camel_case'
    assert data.convert_camel_case('CamelCASE') == 'camel_case'


def test_is_dog():
    s = pd.Series(['Dog', 'Cat'])
    result = data.check_is_dog(s)
    expected = pd.Series([True, False])
    assert_series_equal(result, expected)


def test_get_sex():
    s = pd.Series(['Neutered Male', 'Spayed Female', 'Intact Male',
                   'Intact Female', 'Unknown', 'whale'])
    result = data.get_sex(s)
    expected = pd.Series(['male', 'female', 'male',
                          'female', 'unknown', 'unknown'])
    assert_series_equal(result, expected)


def test_get_neutered():
    s = pd.Series(['Neutered Male', 'Spayed Female', 'Intact Male',
                   'Intact Female', 'Unknown', 'whale'])
    result = data.get_neutered(s)
    expected = pd.Series(['fixed', 'fixed', 'intact',
                          'intact', 'unknown', 'unknown'])
    assert_series_equal(result, expected)


def test_get_hair_type():
    s = pd.Series(['Shetland Sheepdog Mix', 'Pit Bull Mix',
                   'Cairn Terrier/Chihuahua Shorthair',
                   'Domestic Medium Hair Mix', 'Chihuahua Longhair Mix'])
    result = data.get_hair_type(s)
    expected = pd.Series(['unknown', 'unknown', 'shorthair', 'medium hair',
                          'longhair'])
    assert_series_equal(result, expected)


def test_compute_days_upon_outcome():
    s = pd.Series(['1 year', '2 years', '1 month', '2 months',
                   '1 weeks', '2 week', '1 days', '2 day'])
    result = data.compute_days_upon_outcome(s)
    expected = pd.Series([365.0, 2*365.0, 30.0, 2*30.0,
                          7.0, 14.0, 1.0, 2.0])
    assert_series_equal(result, expected)
