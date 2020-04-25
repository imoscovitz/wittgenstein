import numpy as np
import pandas as pd
import pytest

from wittgenstein import preprocess

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
    assert ret_bin_transformer_ == None


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
    assert ret_bin_transformer_ == None


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
    assert ret_bin_transformer_ == None


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
    assert ret_bin_transformer_ == None


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
    assert ret_bin_transformer_ == None

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
    assert ret_bin_transformer_ == None

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
    assert ret_bin_transformer_ == None
