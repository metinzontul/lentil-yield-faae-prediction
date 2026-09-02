# -*- coding: utf-8 -*-


import random

import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.model_selection import train_test_split

import config


def set_seed(seed=config.SEED):
    """Fix numpy, random, and tensorflow seeds."""
    np.random.seed(seed)
    random.seed(seed)
    tf.random.set_seed(seed)


def load_dataset(path=config.DATA_PATH, drop_features=None):

    drop_features = drop_features or []
    df = pd.read_csv(path, sep=config.CSV_SEP, encoding=config.CSV_ENCODING).dropna()

    target_name = df.columns[-1]
    feature_names = [c for c in df.columns[:-1] if c not in drop_features]

    missing = [c for c in drop_features if c not in df.columns]
    if missing:
        raise ValueError(f"Feature(s) to drop not found in dataset: {missing}")

    X = df[feature_names].values
    y = df[target_name].values.reshape(-1, 1)

    return X, y, feature_names, target_name, df


def split_data(X, y, test_size=config.TEST_SIZE, seed=config.SEED):
    """Train/test split."""
    return train_test_split(X, y, test_size=test_size, random_state=seed)
