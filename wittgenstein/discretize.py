# Author: Ilan Moscovitz <ilan.moscovitz@gmail.com>
# License: MIT

from copy import deepcopy
from collections import defaultdict
import numpy as np
import pandas as pd

from wittgenstein.base_functions import truncstr
from wittgenstein.utils import rnd


class BinTransformer:
    def __init__(self, n_discretize_bins=10, names_precision=2, verbosity=0):
        self.n_discretize_bins = n_discretize_bins
        self.names_precision = names_precision
        self.verbosity = verbosity
        self.bins_ = None

    def __str__(self):
        return str(self.bins_)

    __repr__ = __str__

    def __bool__(self):
        return not not self.bins_

    def isempty(self):
        return not self.bins_ is None and not self.bins_

    def fit_or_fittransform_(self, df, ignore_feats=[]):
        """Transform df using pre-fit bins, or, if unfit, fit self and transform df"""

        # Binning has already been fit
        if self.bins_:
            return self.transform(df)

        # Binning disabled
        elif not self.n_discretize_bins:
            return df

        # Binning enabled, and binner needs to be fit
        else:
            self.fit(df, ignore_feats=ignore_feats)
            df, bins = self.transform(df, ignore_feats=ignore_feats)
            self.bins = bins
            return df

    def fit_transform(self, df, ignore_feats=[]):
        self.fit(df, ignore_feats=ignore_feats)
        return self.transform(df)

    def fit(self, df, output=False, ignore_feats=[]):
        """
        Returns a dict defining fits for numerical features
        A fit is an ordered list of tuples defining each bin's range (min is exclusive; max is inclusive)

        Returned dict allows for fitting to training data and applying the same fit to test data
        to avoid information leak.
        """

        def _fit_feat(df, feat):
            """Return list of tuples defining bin ranges for a numerical feature using simple linear search"""

            if len(df) == 0:
                return []

            n_discretize_bins = min(self.n_discretize_bins, len(df[feat].unique()))

            # Collect intervals
            bins = pd.qcut(
                df[feat],
                q=self.n_discretize_bins,
                precision=self.names_precision,
                duplicates="drop",
            )
            if (
                len(bins.unique()) < 2
            ):  # qcut can behave weirdly in heavily-skewed distributions
                bins = pd.cut(
                    df[feat],
                    bins=self.n_discretize_bins,
                    precision=self.names_precision,
                    duplicates="drop",
                )

            # Drop empty bins and duplicate intervals to create bins
            bin_counts = bins.value_counts()
            bins = bin_counts[bin_counts > 0].index
            bins = sorted(bins.unique())

            # Extend min/max to -inf, +inf to capture any ranges not present in training set
            bins[0] = pd.Interval(float("-inf"), bins[0].right)
            bins[-1] = pd.Interval(bins[-1].left, float("inf"))
            bins = self._intervals_to_strs(bins)

            if self.verbosity >= 3:
                print(
                    f"{feat}: fit {len(df[feat].unique())} unique vals into {len(bins)} bins"
                )
            return bins

        # Begin fitting
        feats_to_fit = self.find_continuous_feats(df, ignore_feats=ignore_feats)

        if feats_to_fit:
            if self.verbosity == 1:
                print(f"discretizing {len(feats_to_fit)} features")
            elif self.verbosity == 2:
                print(f"discretizing {len(feats_to_fit)} features: {feats_to_fit}\n")

        self.bins_ = {}
        for feat in feats_to_fit:
            self.bins_[feat] = _fit_feat(df, feat)
        return self.bins_

    def transform(self, df):
        """Transform DataFrame using fit bins."""

        def _transform_feat(df, feat):

            if self.bins_ is None:
                return df

            res = deepcopy(df[feat])
            bins = self._strs_to_intervals(self.bins_[feat])
            res = pd.cut(df[feat], bins=pd.IntervalIndex(bins))
            res = res.map(
                lambda x: {i: s for i, s in zip(bins, self.bins_[feat])}.get(x)
            )
            return res

        # Exclude any feats already transformed into valid intervals
        already_transformed_feats = self._find_transformed(df, raise_invalid=True)

        res = df.copy()
        for feat in self.bins_.keys():
            if feat in res.columns and feat not in already_transformed_feats:
                res[feat] = _transform_feat(res, feat)
        return res

    def find_continuous_feats(self, df, ignore_feats=[]):
        """Return names of df features that seem to be continuous."""

        if not self.n_discretize_bins:
            return []

        # Find numeric features
        cont_feats = df.select_dtypes(np.number).columns

        # Remove discrete features
        cont_feats = [
            f for f in cont_feats if len(df[f].unique()) > self.n_discretize_bins
        ]

        # Remove ignore features
        cont_feats = [f for f in cont_feats if f not in ignore_feats]

        return cont_feats

    def _strs_to_intervals(self, strs):
        return [self._str_to_interval(s) for s in strs]

    def _str_to_interval(self, s):
        floor, ceil = self._str_to_floor_ceil(s)
        return pd.Interval(floor, ceil)

    def _intervals_to_strs(self, intervals):
        """Replace a list of intervals with their string representation."""
        return [self._interval_to_str(interval) for interval in intervals]

    def _interval_to_str(self, interval):
        if interval.left == float("-inf"):
            return f"<{interval.right}"
        elif interval.right == float("inf"):
            return f">{interval.left}"
        else:
            return f"{interval.left}-{interval.right}"

    def _str_to_floor_ceil(self, value):
        """Find min, max separated by a dash"""  # . Return None if invalid pattern."""
        if "<" in value:
            floor, ceil = "-inf", value.replace("<", "")
        elif ">" in value:
            floor, ceil = value.replace(">", ""), "inf"
        else:
            split_idx = 0
            for i, char in enumerate(value):
                # Found a possible split and it's not the first number's minus sign
                if char == "-" and i != 0:
                    if split_idx is not None and not split_idx:
                        split_idx = i
                    # Found a - after the split, and it's not the minus of a negative number
                    elif i > split_idx + 1:
                        return None
            floor = value[:split_idx]
            ceil = value[split_idx + 1 :]
        return float(floor), float(ceil)

    def construct_from_ruleset(self, ruleset):
        MIN_N_DISCRETIZED_BINS = 10

        bt = BinTransformer()
        bt.bins_ = self._bin_prediscretized_features(ruleset)
        bt.n_discretize_bins = (
            max(
                (MIN_N_DISCRETIZED_BINS, max([len(bins) for bins in bt.bins_.values()]))
            )
            if bt.bins_
            else MIN_N_DISCRETIZED_BINS
        )
        bt.names_precision = self._max_dec_precision(bt.bins_)
        return bt

    def _bin_prediscretized_features(self, ruleset):
        def is_valid_decimal(s):
            try:
                float(s)
            except:
                return False
            return True

        def find_floor_ceil(value):
            """id min, max separated by a dash. Return None if invalid pattern."""
            split_idx = 0
            for i, char in enumerate(value):
                # Found a possible split and it's not the first number's minus sign
                if char == "-" and i != 0:
                    if split_idx is not None and not split_idx:
                        split_idx = i
                    # Found a - after the split, and it's not the minus of a negative number
                    elif i > split_idx + 1:
                        return None

            floor = value[:split_idx]
            ceil = value[split_idx + 1 :]
            if is_valid_decimal(floor) and is_valid_decimal(ceil):
                return (floor, ceil)
            else:
                return None

        # _bin_prediscretized_features
        discrete = defaultdict(list)
        for cond in ruleset.get_conds():
            floor_ceil = self.find_floor_ceil(cond.val)
            if floor_ceil:
                discrete[cond.feature].append(floor_ceil)
        for feat, ranges in discrete.items():
            ranges.sort(key=lambda x: float(x[0]))
        return dict(discrete)

    def _max_dec_precision(self, bins_dict):
        def dec_precision(value):
            try:
                return len(value) - value.index(".") - 1
            except:
                return 0

        max_prec = 0
        for bins in bins_dict.values():
            for bin_ in bins:
                for value in bin_:
                    cur_prec = dec_precision(value)
                    if cur_prec > max_prec:
                        max_prec = cur_prec
        return max_prec

    def _find_transformed(self, df, raise_invalid=True):
        """Find columns that appear to have already been transformed. Raise error if there is a range that doesn't match a fit bin."""

        check_feats = df.select_dtypes(include=["category", "object"]).columns.tolist()
        invalid_feats = {}
        transformed_feats = []
        for feat, bins in self.bins_.items():
            if feat in check_feats:
                transformed_feats.append(feat)
                invalid_values = set(df[feat].tolist()) - set(bins)
                if invalid_values:
                    invalid_feats[feat] = invalid_values
        if invalid_feats and raise_invalid:
            raise ValueError(
                f"The following input values seem to be transformed but ranges don't match fit bins: {invalid_feats}"
            )
        return transformed_feats
