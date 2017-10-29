# Machine Learning in Production

[![Build Status](https://travis-ci.org/hgrif/ml-production.svg?branch=master)](https://travis-ci.org/hgrif/ml-production)


## Getting started


Clone this repository.

Make sure [conda](https://conda.io/miniconda.html) is installed, `cd` into the project root and create a new virtual environment:

```bash
$ conda env create -f environment.yml
```

And activate it (you might have to use `activate` instead of `source activate` if you're on Windows):

```bash
(ml-production) $ source activate ml-production
```

Install the `shelter` package located in the project root with development mode:

```bash
(ml-production) $ python setup.py develop
```

Start the Juypyter Notebook server:

```bash
(ml-production) $ jupyter notebook
```


## Morning

We'll start with building a Machine Learning model.
Open the Notebook `01-machine-learning-model.ipynb` in the folder `notebooks/` and follow the instructions.


## Afternoon

Having made our first steps with building, we'll now focus on same development best practices.
The material can be found in the Notebook `02-machine-learning-in-production.ipynb`.
