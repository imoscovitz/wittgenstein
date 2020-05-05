import numpy as np

import pandas as pd

from wittgenstein.base_functions import truncstr, rnd
from wittgenstein.check import (
    _check_any_datasets_not_empty,
    _check_model_features_present,
)


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
    df, bin_transformer_ = _try_bin_fit_or_fittransform_(
        df,
        n_discretize_bins=n_discretize_bins,
        ignore_feats=[class_feat],
        verbosity=verbosity,
    )

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
    df, _ = _try_bin_fit_or_fittransform_(
        df,
        bin_transformer_=bin_transformer_,
        ignore_feats=[class_feat],
        verbosity=verbosity,
    )

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
    df, bin_transformer_ = _try_bin_fit_or_fittransform_(
        df,
        bin_transformer_=bin_transformer_,
        ignore_feats=[class_feat],
        verbosity=verbosity,
    )

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
    if hasattr(y, "__getitem__"):
        # No super clean way to check for 1D, but this should be pretty good
        if any([hasattr(item, "__getitem__") and type(item) is not str for item in y]):
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
    if pos_class is not None:
        return pos_class

    # Check if pos class can be inferred as True or 1
    class_values = sorted(df[class_feat].unique())
    if len(class_values) == 2:
        # Convert numpy int64 to native python int
        try:
            class_values = [class_values[0].item(), class_values[1].item()]
        except:
            pass
        if class_values[0] is 0 and class_values[1] is 1:
            return 1
        elif class_values[0] is False and class_values[1] is True:
            return True

    # Can't infer classes
    raise NameError(
        f"Couldn't infer name of positive target class from class feature: {class_feat}. Try using parameter pos_class to specify which class label should be treated as positive, or renaming your classes as booleans or 0, 1."
    )


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


def _try_bin_fit_or_fittransform_(
    df, ignore_feats=[], n_discretize_bins=None, bin_transformer_=None, verbosity=0
):

    # Binning has already been fit
    if bin_transformer_:
        bin_transform(df, bin_transformer_)
        return df, bin_transformer_

    # Binning disabled
    elif not n_discretize_bins:
        return df, bin_transformer_

    # Binning enabled, and binner needs to be fit
    else:
        df, bin_transformer_ = bin_df(
            df,
            n_discretize_bins=n_discretize_bins,
            ignore_feats=ignore_feats,
            verbosity=verbosity,
        )
        return df, bin_transformer_


def bin_df(df, n_discretize_bins=10, ignore_feats=[], verbosity=0):
    """Return df with seemingly continuous features binned, and the bin_transformer or None depending on whether binning occurs."""

    if n_discretize_bins is None:
        return df, None

    isbinned = False
    continuous_feats = find_continuous_feats(
        df, n_discretize_bins=n_discretize_bins, ignore_feats=ignore_feats
    )
    if n_discretize_bins:
        if continuous_feats:
            if verbosity == 1:
                print(f"binning data...\n")
            elif verbosity >= 2:
                print(f"binning features {continuous_feats}...")
            binned_df = df.copy()
            bin_transformer = fit_bins(
                binned_df,
                n_bins=n_discretize_bins,
                output=False,
                ignore_feats=ignore_feats,
                verbosity=verbosity,
            )
            binned_df = bin_transform(binned_df, bin_transformer)
            isbinned = True
    else:
        n_unique_values = sum(
            [len(u) for u in [df[f].unique() for f in continuous_feats]]
        )
        warning_str = f"There are {len(continuous_feats)} features to be treated as continuous: {continuous_feats}. \n Treating {n_unique_values} numeric values as nominal or discrete. To auto-discretize features, assign a value to parameter 'n_discretize_bins.'"
        _warn(warning_str, RuntimeWarning, filename="base", funcname="bin_df")
    if isbinned:
        return binned_df, bin_transformer
    else:
        return df, None


def find_continuous_feats(df, n_discretize_bins, ignore_feats=[]):
    """Return names of df features that seem to be continuous."""

    if not n_discretize_bins:
        return []

    # Find numeric features
    cont_feats = df.select_dtypes(np.number).columns

    # Remove discrete features
    cont_feats = [f for f in cont_feats if len(df[f].unique()) > n_discretize_bins]

    # Remove ignore features
    cont_feats = [f for f in cont_feats if f not in ignore_feats]

    return cont_feats


def fit_bins(df, n_bins=5, output=False, ignore_feats=[], verbosity=0):
    """
    Returns a dict definings fits for numerical features
    A fit is an ordered list of tuples defining each bin's range (min is exclusive; max is inclusive)

    Returned dict allows for fitting to training data and applying the same fit to test data
    to avoid information leak.
    """

    def bin_fit_feat(df, feat, n_bins=10):
        """
        Returns list of tuples defining bin ranges for a numerical feature
        """
        n_bins = min(
            n_bins, len(df[feat].unique())
        )  # In case there are fewer unique values than n_bins
        bin_size = len(df) // n_bins
        sorted_df = df.sort_values(by=[feat])
        sorted_values = sorted_df[feat].tolist()

        if verbosity >= 4:
            print(
                f"{feat}: fitting {len(df[feat].unique())} unique vals into {n_bins} bins"
            )
        bin_ranges = []
        finish_i = -1
        sizes = []

        bin = 0
        start_i = 0
        while bin < n_bins and start_i < len(sorted_values):
            finish_i = min(start_i + bin_size, len(sorted_df) - 1)
            while (
                finish_i < len(sorted_df) - 1
                and finish_i != 0
                and sorted_df.iloc[finish_i][feat] == sorted_df.iloc[finish_i - 1][feat]
            ):  # ensure next bin begins on a new value
                finish_i += 1
            sizes.append(finish_i - start_i)
            if verbosity >= 4:
                print(
                    f"bin #{bin}, start_idx {start_i} value: {sorted_df.iloc[start_i][feat]}, finish_idxx {finish_i} value: {sorted_df.iloc[finish_i][feat]}"
                )
            start_val = sorted_values[start_i]
            finish_val = sorted_values[finish_i]
            bin_range = (start_val, finish_val)
            bin_ranges.append(bin_range)

            start_i = finish_i + 1
            bin += 1

        if verbosity >= 4:
            print(
                f"-bin sizes {sizes}; dataVMR={rnd(np.var(df[feat])/np.mean(df[feat]))}, binVMR={rnd(np.var(sizes)/np.mean(sizes))}"
            )  # , axis=None, dtype=None, out=None, ddof=0)})
        return bin_ranges

    # Create dict to store fit definitions for each feature
    fit_dict = {}
    feats_to_fit = find_continuous_feats(df, n_bins, ignore_feats=ignore_feats)
    if verbosity == 2:
        print(f"fitting bins for features {feats_to_fit}")
    if verbosity >= 2:
        print()

    # Collect fits in dict
    count = 1
    for feat in feats_to_fit:
        fit = bin_fit_feat(df, feat, n_bins=n_bins)
        fit_dict[feat] = fit
    return fit_dict


def bin_transform(df, fit_dict, names_precision=2):
    """
    Uses a pre-collected dictionary of fits to transform df features into bins.
    Returns the fit df rather than modifying inplace.
    """

    def bin_transform_feat(df, feat, bin_fits, names_precision=names_precision):
        """
        Returns new dataframe with n_bin bins replacing each numerical feature
        """

        def renamed(bin_fit_list, value, names_precision=names_precision):
            """
            Returns bin string name for a given numerical value
            Assumes bin_fit_list is ordered
            """
            min_val, min_bin = bin_fit_list[0][0], bin_fit_list[0]
            max_val, max_bin = bin_fit_list[-1][1], bin_fit_list[-1]
            for bin_fit in bin_fits:
                if value <= bin_fit[1]:
                    start_name = str(round(bin_fit[0], names_precision))
                    finish_name = str(round(bin_fit[1], names_precision))
                    bin_name = "-".join([start_name, finish_name])
                    return bin_name
            if value <= min_val:
                return min_bin
            elif value >= max_val:
                return max_bin
            else:
                raise ValueError("No bin found for value", value)

        renamed_values = []
        for value in df[feat]:
            bin_name = renamed(bin_fits, value, names_precision)
            renamed_values.append(bin_name)
        return renamed_values

    # Replace each feature with bin transformations
    for feat, bin_fits in fit_dict.items():
        if feat in df.columns:
            feat_transformation = bin_transform_feat(
                df, feat, bin_fits, names_precision=names_precision
            )
            df[feat] = feat_transformation
    return df


def _try_rename_features(df, class_feat, feature_names):
    """ Rename df columns according to user request. """
    # Rename if same number of features
    df_columns = [col for col in df.columns.tolist() if col != class_feat]
    if len(df_columns) == len(feature_names):
        col_replacements_dict = {
            old: new for old, new in zip(df_columns, feature_names)
        }
        df = df.rename(columns=col_replacements_dict)
        return df
    # Wrong number of feature names
    else:
        return None
