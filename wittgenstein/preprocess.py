# Author: Ilan Moscovitz <ilan.moscovitz@gmail.com>
# License: MIT

import numpy as np

import pandas as pd

from wittgenstein.check import (
    _check_any_datasets_not_empty,
    _check_model_features_present,
    _warn_only_single_class,
)
from wittgenstein.discretize import BinTransformer
from wittgenstein import utils


def preprocess_training_data(preprocess_params):

    # Get params
    trainset = preprocess_params["trainset"]
    y = preprocess_params["y"]
    class_feat = preprocess_params["class_feat"]
    pos_class = preprocess_params["pos_class"]
    user_requested_feature_names = preprocess_params["feature_names"]
    n_discretize_bins = preprocess_params["n_discretize_bins"]
    verbosity = preprocess_params["verbosity"]

    # Error check
    _check_valid_input_data(
        trainset,
        y,
        class_feat,
        user_requested_feature_names=user_requested_feature_names,
    )

    # Determine class_feat
    class_feat = _get_class_feat_name(class_feat, y)

    # Build new DataFrame containing both X and y.
    df = _convert_to_training_df(
        trainset,
        y,
        class_feat,
        user_requested_feature_names=user_requested_feature_names,
    )

    # Define pos_class
    pos_class = _get_pos_class(df, class_feat, pos_class)

    # Infer correct datatypes
    df = df.infer_objects()

    # Bin, if necessary
    bin_transformer_ = BinTransformer(
        n_discretize_bins=n_discretize_bins, verbosity=verbosity
    )
    df = bin_transformer_.fit_transform(df, ignore_feats=[class_feat])

    # Done
    return df, class_feat, pos_class, bin_transformer_


def preprocess_prediction_data(preprocess_params):

    X = preprocess_params["X"]
    class_feat = preprocess_params["class_feat"]
    pos_class = preprocess_params["pos_class"]
    user_requested_feature_names = preprocess_params["user_requested_feature_names"]
    selected_features_ = preprocess_params["selected_features_"]
    trainset_features_ = preprocess_params["trainset_features_"]
    bin_transformer_ = preprocess_params["bin_transformer_"]
    verbosity = preprocess_params["verbosity"]

    # Error check
    _check_valid_input_data(
        X,
        y=None,
        class_feat=class_feat,
        requires_label=False,
        user_requested_feature_names=user_requested_feature_names,
    )

    # Build new DataFrame containing both X and y.
    df = _convert_to_prediction_df(
        X,
        class_feat=class_feat,
        user_requested_feature_names=user_requested_feature_names,
    )

    # Make sure selected features are present
    _check_model_features_present(df, selected_features_)

    # Infer correct datatypes
    df = df.infer_objects()

    # Bin, if necessary
    if bin_transformer_:
        df = bin_transformer_.transform(df)

    # Done
    return df


def _preprocess_recalibrate_proba_data(preprocess_params):

    # Get params
    X_or_Xy = preprocess_params["X_or_Xy"]
    y = preprocess_params["y"]
    class_feat = preprocess_params["class_feat"]
    pos_class = preprocess_params["pos_class"]
    user_requested_feature_names = preprocess_params["user_requested_feature_names"]
    bin_transformer_ = preprocess_params["bin_transformer_"]
    verbosity = preprocess_params["verbosity"]

    # Error check
    _check_valid_input_data(
        X_or_Xy,
        y,
        class_feat,
        user_requested_feature_names=user_requested_feature_names,
    )

    # Build new DataFrame containing both X and y.
    df = _convert_to_training_df(
        X_or_Xy,
        y,
        class_feat,
        user_requested_feature_names=user_requested_feature_names,
    )

    # Infer correct datatypes
    df = df.infer_objects()

    # Bin, if necessary
    if bin_transformer_:
        df = bin_transformer_.transform(df)

    # Done
    return df


def _preprocess_y_score_data(y):
    """Return python iterable of y values."""

    def raise_wrong_ndim():
        raise IndexError(f"y input data has wrong number dimensions. It should have 1.")

    # If it's pandas or np...
    if hasattr(y, "ndim"):
        if y.ndim == 1:
            return y.tolist()
        else:
            raise_wrong_ndim()

    # Otherwise try for python iterable
    if hasattr(y, "__iter__"):  # it's an iterable
        # No super clean way to check for 1D, but this should be pretty decent
        if any([hasattr(item, "__iter__") and type(item) is not str for item in y]):
            raise_wrong_ndim()
        else:
            return y

    # Otherwise, no idea what's going on
    else:
        raise TypeError(
            f"Could not identify valid type for y input data: {type(y)}. Recommended types are 1D python iterable, pandas Series, or 1D numpy array."
        )


def _check_valid_input_data(
    X_or_Xy,
    y=None,
    class_feat=None,
    user_requested_feature_names=None,
    requires_label=True,
):

    # Make sure there is data
    if not _check_any_datasets_not_empty([X_or_Xy]):
        raise ValueError("No data provided!")

    # If labels aren't needed, we're done
    if not requires_label:
        return

    # Ensure class feature is provided
    if (y is None) and (class_feat is None):
        raise ValueError("y or class_feat param is required")

    # Ensure target data exists if class feat is provided
    if y is None:
        if user_requested_feature_names is not None:
            feature_names = user_requested_feature_names
        elif hasattr(X_or_Xy, "columns"):
            feature_names = X_or_Xy.columns
        else:
            feature_names = list(range(len(X_or_Xy[0])))

        if class_feat not in feature_names:
            raise IndexError(
                f"Dataset does not include class feature name {class_feat}. Training set features: {feature_names}"
            )

    # If both y and class_feat provided, ensure no name mismatch between them.
    if (
        y is not None
        and class_feat is not None
        and hasattr(y, "name")
        and y.name != class_feat
    ):
        raise NameError(
            f"Feature name mismatch between params y {y.name} and class_feat {class_feat}. Besides, you only need to provide one of them."
        )


def _convert_to_training_df(X_or_Xy, y, class_feat, user_requested_feature_names=None):
    """Make a labeled Xy DataFrame from data."""

    # Create df from X_or_Xy
    if isinstance(X_or_Xy, pd.DataFrame):
        df = X_or_Xy.copy()
    else:
        df = pd.DataFrame(X_or_Xy)

    # Set feature names
    if user_requested_feature_names is not None:
        df.columns = list(
            user_requested_feature_names
        )  # list in case the type is df.columns or something

    # If necessary, merge y into df
    if y is not None:
        # If y is pd or np type, add it
        try:
            df = df.set_index(y.index)
            df[class_feat] = y.copy()
        # If that doesn't work, it's likely a python iterable
        except:
            df[class_feat] = y
    return df


def _convert_to_prediction_df(X_or_Xy, class_feat, user_requested_feature_names=None):
    """Make a labeled X DataFrame from data."""

    # Create df from X_or_Xy
    if isinstance(X_or_Xy, pd.DataFrame):
        df = X_or_Xy.copy()
    else:
        df = pd.DataFrame(X_or_Xy)

    # Drop class feature if present
    if class_feat in df.columns:
        df.drop(class_feat, axis=1, inplace=True)

    # Set feature names
    if user_requested_feature_names:
        df.columns = [f for f in user_requested_feature_names if not f == class_feat]
    return df


def _get_pos_class(df, class_feat, pos_class):
    """Get or infer the positive class name."""
    # Pos class already known

    def raise_fail_infer_pos_class():
        raise NameError(
            f"Couldn't infer name of positive target class from class feature: {class_feat}. Try using parameter pos_class to specify which class label should be treated as positive, or renaming your classes as booleans or 0, 1."
        )

    # pos class is already known
    if pos_class is not None:
        return pos_class

    # Check if pos class can be inferred as True or 1
    class_values = df[class_feat].unique()

    # More than two classes
    if len(class_values) > 2:
        raise_fail_infer_pos_class()

    # Only one class
    elif len(class_values) == 1:
        only_value = utils.try_np_tonum(class_values[0])
        if only_value is 0:
            pos_class = 1
        elif only_value is False:
            pos_class = True
        else:
            pos_class = only_value
        _warn_only_single_class(
            only_value=only_value,
            pos_class=pos_class,
            filename="preprocess.py",
            funcname="_get_pos_class",
        )
        return pos_class

    # Exactly two class. Check if they are 01 or TrueFalse
    else:
        class_values.sort()
        class_values = [utils.try_np_tonum(val) for val in class_values]
        if class_values[0] is 0 and class_values[1] is 1:
            return 1
        elif class_values[0] is False and class_values[1] is True:
            return True

    # Can't infer classes
    raise_fail_infer_pos_class()


def _get_class_feat_name(class_feat, y):

    if class_feat is not None:
        return class_feat

    if y is not None and hasattr(y, "name"):
        # If y is a pandas Series, try to get its name
        class_feat = y.name
    else:
        # Create a name for it
        class_feat = "Class"

    return class_feat


def _upgrade_bin_transformer_ifdepr(obj):
    old_bin_transformer_ = getattr(obj, "bin_transformer_")
    if type(old_bin_transformer_) == dict:
        new_bin_transformer_ = BinTransformer()
        new_bin_transformer_.bins_ = old_bin_transformer_
        setattr(obj, "bin_transformer_", new_bin_transformer_)
