from collections.abc import Iterable

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, KFold
import os

TEST_SIZE = 0.1
VAL_SIZE = 0.1
RANDOM_STATE = 42
N_SPLITS = 5

# Creates a train,validation and test split
def train_val_test_split(X, y, test_size=TEST_SIZE, val_size=VAL_SIZE, random_state=RANDOM_STATE):
    # First split: Train + Validation vs Test
    X_train_val, X_test, y_train_val, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state
    )

    # Second split: Train vs Validation
    val_size_adjusted = val_size / (1 - test_size)  # Adjust validation size relative to remaining data
    X_train, X_val, y_train, y_val = train_test_split(
        X_train_val, y_train_val, test_size=val_size_adjusted, random_state=random_state)


    return X_train, y_train, X_val, y_val, X_test, y_test


# Creates a train, cross-validation and test split
def train_cv_test_split(X, y, test_size=TEST_SIZE, n_splits=N_SPLITS, random_state=RANDOM_STATE):
    #Split into Train + Test
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state
    )

    # Create K-Fold cross-validator for the training set
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=random_state)

    return X_train, y_train, X_test, y_test, kf

def write_data_frames(file_path, dfs, data_types=('train', 'val', 'test'), zs: Iterable ="Xy"):
    if isinstance(dfs[0], (pd.DataFrame, pd.Series)):
        if len(zs) == 1:
            dfs = [[df] for df in dfs]
        elif len(data_types) == 1:
            dfs = [dfs]
        else:
            dfs = [(dfs[i], dfs[i+1]) for i in range(0, len(dfs), 2)]
    os.makedirs(file_path, exist_ok=True)
    for type_, XY_types in zip(data_types, dfs):
        for Z, df in zip(zs, XY_types):
            df: pd.DataFrame = df
            df.to_csv(f"{file_path}/{Z}_{type_}.csv", index=False)

def read_data_frames(file_path, data_types=('train', 'val', 'test'), zs: Iterable ="Xy"):
    return [
        [
            pd.read_csv(f"{file_path}/{Z}_{type_}.csv").to_numpy()
            for Z in zs
        ]
        for type_ in data_types
    ]


def write_train():
    df = pd.read_csv('Datasets/train_data.csv')
    df = df.dropna(subset=['SurvivalTime']) # should this be done here? ans: yes, according to Task 3's initial text
    df = df.drop(columns =['Unnamed: 0', 'id'], errors='ignore')

    # Unaltered Dataset
    y = df['SurvivalTime']
    X = df.drop(columns=['SurvivalTime'])
    ret = (X_train, y_train, X_val, y_val, X_test, y_test) = train_val_test_split(X, y)
    Xs = [X_train, X_val, X_test]
    write_data_frames('datasets/split', ret)
    write_data_frames('datasets/split', [X_["Censored"] for X_ in Xs], zs="c")
    write_data_frames('datasets/split', [X_.drop(columns=["Censored"]) for X_ in Xs], zs=["X_uncensored"])

    # Pruned Dataset
    df = df[df['Censored'] != 1] # removing all censored values
    df = df.dropna(axis=1) # removes all columns that have any NaN value

    y = df['SurvivalTime']
    X = df.drop(columns=['SurvivalTime'])
    ret = (X_train, y_train, X_val, y_val, X_test, y_test) = train_val_test_split(X, y)
    Xs = [X_train, X_val, X_test]
    write_data_frames('datasets/pruned', ret)
    write_data_frames('datasets/pruned', [X_["Censored"] for X_ in Xs],  zs="c")
    write_data_frames('datasets/pruned', [X_.drop(columns=["Censored"]) for X_ in Xs], zs=["X_uncensored"])

def read_pruned_dataset():
    """
    returns 3 lists (train, val and test) each with 3 elements: their X, y and c, where X doesn't contain c.
    c will probably always be 0
    :return: list of lists of dataframes
    """
    return read_data_frames(
        'datasets/pruned',
        data_types = ['train', 'val', 'test'],
        zs = ['X_uncensored', 'y', 'c']
    )

def read_pruned_dataset_train_test_full():
    """
    returns 3 lists (train, val and test) each with 3 elements: their X, y and c, where X doesn't contain c.
    c will probably always be 0
    :return: list of lists of dataframes
    """
    ((X_train, y_train, c_train), (X_val, y_val, c_val), (X_test, y_test, c_test)) = read_data_frames(
        'datasets/pruned',
        data_types = ['train', 'val', 'test'],
        zs = ['X_uncensored', 'y', 'c']
    )
    return (
        np.concat([X_train, X_val], axis=0),
        np.concat([y_train, y_val], axis=0),
        np.concat([c_train, c_val], axis=0)
    ), (X_test, y_test, c_test)


def read_pruned_dataset_c_uc():
    """
        returns the pruned datasets as per request of:
        X_train, y_train, X_val, y_val, X_test, y_test, Censored_X_train, Censored_X_val, Censored_X_test
        :return: list of lists of dataframes
        """
    ((X_uc_train, y_train, X_c_train), (X_uc_val, y_val, X_c_val), (X_uc_test, y_test, X_c_test)) = read_data_frames(
        'datasets/pruned',
        data_types = ['train', 'val', 'test'],
        zs = ['X_uncensored', 'y', 'X']
    )
    return X_uc_train, y_train, X_uc_val, y_val, X_uc_test, y_test, X_c_train, X_c_val, X_c_test

def read_x_test():
    X_test = pd.read_csv('Datasets/test_data.csv')
    X_test = X_test.drop(columns=['Unnamed: 0', 'id'], errors='ignore')
    return X_test

def read_x_test_averaged_nans():
    X_test = read_x_test()
    if X_test.isna().any().any():
        X_test_default = X_test.fillna(X_test.mean())
        X_test_default = X_test_default.drop(columns=['ComorbidityIndex', 'GeneticRisk', 'TreatmentResponse'])
        X_test = X_test_default
    return X_test

def create_kdata_fold():
    return KFold(n_splits=N_SPLITS, shuffle=True, random_state=RANDOM_STATE)

def main():
    write_train()


if __name__ == '__main__':
    main()



