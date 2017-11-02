import logging
import re
import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


def load_data(path):
    """Load the data and convert the column names.

    Parameters
    ----------
    path : str
        Path to data

    Returns
    -------
    df : pandas.DataFrame
        DataFrame with data
    """
    logger.info('Reading data from %s', path)
    df = (
        pd.read_csv(path, parse_dates=['DateTime'])
        .rename(columns=lambda x: x.replace('upon', 'Upon'))
        .rename(columns=convert_camel_case)
        .fillna('Unknown')
    )
    logger.info('Read %i rows', len(df))
    return df


def convert_camel_case(name):
    """Convert CamelCase to snake_case.

    Parameters
    ----------
    name : str
        CamelCase string

    Returns
    -------
    result : str
        snake_case string
    """
    s1 = re.sub('(.)([A-Z][a-z]+)', r'\1_\2', name)
    return re.sub('([a-z0-9])([A-Z])', r'\1_\2', s1).lower()


def add_features(df):
    """Add some features to our data.

    Parameters
    ----------
    df : pandas.DataFrame
        DataFrame with data (see load_data)

    Returns
    -------
    with_features : pandas.DataFrame
        DataFrame with some column features added
    """
    newdf = df.copy()
    newdf['is_dog'] = check_is_dog(newdf['animal_type'])
    newdf['has_name'] = check_has_name(newdf['name'])
    newdf['sex'] = get_sex(newdf['sex_upon_outcome'])
    newdf['neutered'] = get_neutered(newdf['sex_upon_outcome'])
    newdf['hair_type'] = get_hair_type(newdf['breed'])
    newdf['days_upon_outcome'] = compute_days_upon_outcome(newdf['age_upon_outcome'])
    return newdf


def check_is_dog(animal_type):
    """Check if the animal is a dog, otherwise return False.

    Parameters
    ----------
    animal_type : pandas.Series
        Type of animal

    Returns
    -------
    result : pandas.Series
        Dog or not
    """
    is_cat_dog = animal_type.str.lower().isin(['dog', 'cat'])
    if not is_cat_dog.all():
        logging.error('Found something else but dogs and cats:\n%s',
                      animal_type[~is_cat_dog])
    return animal_type.str.lower() == 'dog'


def check_has_name(name):
    """Check if the animal is not called 'unknown'.

    Parameters
    ----------
    name : pandas.Series
        Animal name

    Returns
    -------
    result : pandas.Series
        Unknown or not.
    """
    # Check if it has a name.
    return name.str.lower() != 'unknown'


def get_sex(sex_upon_outcome):
    """Determine if the sex was 'Male', 'Female' or unknown.

    Parameters
    ----------
    sex_upon_outcome : pandas.Series
        Sex and fixed state when coming in

    Returns
    -------
    sex : pandas.Series
        Sex when coming in
    """
    sex = pd.Series('unknown', index=sex_upon_outcome.index)
    sex.loc[sex_upon_outcome.str.endswith('Female')] = 'female'
    sex.loc[sex_upon_outcome.str.endswith('Male')] = 'male'
    return sex


def get_neutered(sex_upon_outcome):
    """Determine if an animal was intact or not.

    Parameters
    ----------
    sex_upon_outcome : pandas.Series
        Sex and fixed state when coming in

    Returns
    -------
    sex : pandas.Series
        Intact, fixed or unknown
    """
    sex_upon_outcome.loc[sex_upon_outcome.str.lower().str.contains('neutered')] = 'fixed'
    sex_upon_outcome.loc[sex_upon_outcome.str.lower().str.contains('spayed')] = 'fixed'
    sex_upon_outcome.loc[sex_upon_outcome.str.lower().str.contains('intact')] = 'intact'
    sex_upon_outcome.loc[~sex_upon_outcome.isin(['fixed', 'intact'])] = 'unknown'
    return sex_upon_outcome


def get_hair_type(breed):
    """Get hair type of a breed.

    Parameters
    ----------
    breed : pandas.Series
        Breed of animal

    Returns
    -------
    hair_type : pandas.Series
        Hair type
    """
    hair_type = breed.str.lower()
    valid_hair_types = ['shorthair', 'medium hair', 'longhair']
    for hair in valid_hair_types:
        is_hair_type = hair_type.str.contains(hair)
        hair_type[is_hair_type] = hair
    hair_type[~hair_type.isin(valid_hair_types)] = 'unknown'
    return hair_type


def compute_days_upon_outcome(age_upon_outcome):
    """Compute age in days upon outcome.

    Parameters
    ----------
    age_upon_outcome : pandas.Series
        Age as string

    Returns
    -------
    days_upon_outcome : pandas.Series
        Age in days
    """
    split__age = age_upon_outcome.str.split()
    time = split__age.apply(lambda x: x[0] if x[0] != 'Unknown' else np.nan)
    period = split__age.apply(lambda x: x[1] if x[0] != 'Unknown' else None)
    period__mapping = {'year': 365, 'years': 365, 'weeks': 7, 'week': 7,
                       'month': 30, 'months': 30, 'days': 1, 'day': 1}
    days_upon_outcome = time.astype(float) * period.map(period__mapping)
    assert isinstance(days_upon_outcome, object)
    return days_upon_outcome
