import numpy as np
import pandas as pd
import pytest

from wittgenstein import preprocess
from wittgenstein.irep import IREP
from wittgenstein.ripper import RIPPER

DF = pd.read_csv("house-votes-84.csv").head(10)
CLASS_FEAT = "Party"
DEFAULT_CLASS_FEAT = "Class"
POS_CLASS = "democrat"
SPLIT_SIZE = 0.6

X_DF = DF.drop(CLASS_FEAT, axis=1)
Y_DF = DF[CLASS_FEAT]

XY_NP = DF.values
X_NP = X_DF.values
Y_NP = Y_DF.values
NP_CLASS_FEAT = 0


def test_preprocess_training_data_Xy_df():
    preprocess_params = {
        "trainset": DF,
        "y": None,
        "class_feat": CLASS_FEAT,
        "pos_class": POS_CLASS,
        "feature_names": None,
        "n_discretize_bins": None,
        "verbosity": 0,
    }
    (
        df,
        ret_class_feat,
        ret_pos_class,
        ret_bin_transformer_,
    ) = preprocess.preprocess_training_data(preprocess_params)
    assert ret_class_feat == CLASS_FEAT
    assert ret_pos_class == POS_CLASS
    assert ret_bin_transformer_.isempty()


def test_preprocess_training_data_X_df():
    preprocess_params = {
        "trainset": X_DF,
        "y": Y_DF,
        "class_feat": CLASS_FEAT,
        "pos_class": POS_CLASS,
        "feature_names": None,
        "n_discretize_bins": None,
        "verbosity": 0,
    }
    (
        df,
        ret_class_feat,
        ret_pos_class,
        ret_bin_transformer_,
    ) = preprocess.preprocess_training_data(preprocess_params)
    assert ret_class_feat == CLASS_FEAT
    assert ret_pos_class == POS_CLASS
    assert ret_bin_transformer_.isempty()


def test_preprocess_training_data_X_np():
    preprocess_params = {
        "trainset": X_NP,
        "y": Y_NP,
        "class_feat": None,
        "pos_class": POS_CLASS,
        "feature_names": None,
        "n_discretize_bins": None,
        "verbosity": 0,
    }
    (
        ret_df,
        ret_class_feat,
        ret_pos_class,
        ret_bin_transformer_,
    ) = preprocess.preprocess_training_data(preprocess_params)
    x_ret_df = ret_df[ret_df.columns[:-1]]
    y_ret_df = ret_df[ret_df.columns[-1]]
    assert np.array_equal(x_ret_df.values, X_NP)
    assert np.array_equal(y_ret_df.values, Y_NP)
    assert ret_pos_class == POS_CLASS
    assert ret_bin_transformer_.isempty()


def test_preprocess_training_data_XY_np():
    preprocess_params = {
        "trainset": XY_NP,
        "y": None,
        "class_feat": NP_CLASS_FEAT,
        "pos_class": POS_CLASS,
        "feature_names": None,
        "n_discretize_bins": None,
        "verbosity": 0,
    }
    (
        ret_df,
        ret_class_feat,
        ret_pos_class,
        ret_bin_transformer_,
    ) = preprocess.preprocess_training_data(preprocess_params)
    assert np.array_equal(ret_df.values, XY_NP)
    assert ret_class_feat == NP_CLASS_FEAT
    assert ret_pos_class == POS_CLASS
    assert ret_bin_transformer_.isempty()


def test_preprocess_training_data_XY_rename_columns():

    # With xy
    preprocess_params = {
        "trainset": XY_NP,
        "y": None,
        "class_feat": CLASS_FEAT,
        "pos_class": POS_CLASS,
        "feature_names": DF.columns,
        "n_discretize_bins": None,
        "verbosity": 0,
    }
    (
        ret_df,
        ret_class_feat,
        ret_pos_class,
        ret_bin_transformer_,
    ) = preprocess.preprocess_training_data(preprocess_params)
    assert ret_df.equals(DF)
    assert ret_class_feat == CLASS_FEAT
    assert ret_pos_class == POS_CLASS
    assert ret_bin_transformer_.isempty()

    # With x_y
    preprocess_params = {
        "trainset": X_NP,
        "y": Y_NP,
        "class_feat": CLASS_FEAT,
        "pos_class": POS_CLASS,
        "feature_names": DF.drop(CLASS_FEAT, axis=1).columns,
        "n_discretize_bins": None,
        "verbosity": 0,
    }
    (
        ret_df,
        ret_class_feat,
        ret_pos_class,
        ret_bin_transformer_,
    ) = preprocess.preprocess_training_data(preprocess_params)
    x_ret_df = ret_df[ret_df.columns[:-1]]
    y_ret_df = ret_df[ret_df.columns[-1]]
    assert np.array_equal(x_ret_df.values, X_NP)
    assert np.array_equal(y_ret_df.values, Y_NP)
    assert x_ret_df.columns.tolist() == X_DF.columns.tolist()
    assert y_ret_df.name == CLASS_FEAT
    assert ret_class_feat == CLASS_FEAT
    assert ret_pos_class == POS_CLASS
    assert ret_bin_transformer_.isempty()

    # With x_y and don't rename class feature
    preprocess_params = {
        "trainset": X_NP,
        "y": Y_NP,
        "class_feat": None,
        "pos_class": POS_CLASS,
        "feature_names": DF.drop(CLASS_FEAT, axis=1).columns,
        "n_discretize_bins": None,
        "verbosity": 0,
    }
    (
        ret_df,
        ret_class_feat,
        ret_pos_class,
        ret_bin_transformer_,
    ) = preprocess.preprocess_training_data(preprocess_params)
    x_ret_df = ret_df[ret_df.columns[:-1]]
    y_ret_df = ret_df[ret_df.columns[-1]]
    assert np.array_equal(x_ret_df.values, X_NP)
    assert np.array_equal(y_ret_df.values, Y_NP)
    assert x_ret_df.columns.tolist() == X_DF.columns.tolist()
    assert y_ret_df.name == DEFAULT_CLASS_FEAT
    assert ret_class_feat == DEFAULT_CLASS_FEAT
    assert ret_pos_class == POS_CLASS
    assert ret_bin_transformer_.isempty()


"""
def test_deprecated_bin_transformer():
    deprecated_bin_transformer = {
        "A11": [(0, 1), (1, 2), (2, 4), (4, 8), (8, 17), (17, 67)],
        "A15": [
            (0, 1),
            (1, 9),
            (10, 105),
            (108, 351),
            (351, 1004),
            (1058, 4607),
            (4700, 100000),
        ],
        "A3": [
            (0.0, 0.415),
            (0.415, 0.79),
            (0.79, 1.375),
            (1.375, 2.04),
            (2.04, 3.04),
            (3.04, 4.71),
            (4.75, 7.04),
            (7.08, 10.665),
            (10.75, 14.585),
            (14.79, 28.0),
        ],
        "A8": [
            (0.0, 0.04),
            (0.04, 0.165),
            (0.165, 0.335),
            (0.335, 0.71),
            (0.75, 1.25),
            (1.25, 1.835),
            (1.835, 2.79),
            (3.0, 5.04),
            (5.085, 13.0),
            (13.5, 28.5),
        ],
    }
    df = pd.read_csv("credit.csv")
    irep = IREP()
    irep.fit(df, class_feat="Class", pos_class="+")
    irep.bin_transformer_ = deprecated_bin_transformer
    preds = irep.predict(df)

    rip = RIPPER()
    rip.fit(df, class_feat="Class", pos_class="+")
    rip.bin_transformer_ = deprecated_bin_transformer
    preds = rip.predict(df)
""";
