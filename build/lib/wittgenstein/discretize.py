import numpy as np

from wittgenstein.base_functions import truncstr, rnd


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

    def transform(self, df, ignore_feats=[]):
        """Return df with seemingly continuous features binned, and the bin_transformer or None depending on whether binning occurs."""

        if n_discretize_bins is None:
            return df

        if self.bins_ == {}:
            return df

        isbinned = False
        continuous_feats = find_continuous_feats(df, ignore_feats=ignore_feats)
        if self.n_discretize_bins:
            if continuous_feats:
                if self.verbosity == 1:
                    print(f"binning data...\n")
                elif self.verbosity >= 2:
                    print(f"binning features {continuous_feats}...")
                binned_df = df.copy()
                bin_transformer = fit_bins(
                    binned_df, output=False, ignore_feats=ignore_feats,
                )
                binned_df = bin_transform(binned_df, bin_transformer)
                isbinned = True
        else:
            n_unique_values = sum(
                [len(u) for u in [df[f].unique() for f in continuous_feats]]
            )
            warning_str = f"There are {len(continuous_feats)} features to be treated as continuous: {continuous_feats}. \n Treating {n_unique_values} numeric values as nominal or discrete. To auto-discretize features, assign a value to parameter 'n_discretize_bins.'"
            _warn(warning_str, RuntimeWarning, filename="base", funcname="transform")
        if isbinned:
            self.bins_ = bin_transformer
            return binned_df, bin_transformer
        else:
            return df

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

    def fit(self, df, output=False, ignore_feats=[]):
        """
        Returns a dict definings fits for numerical features
        A fit is an ordered list of tuples defining each bin's range (min is exclusive; max is inclusive)

        Returned dict allows for fitting to training data and applying the same fit to test data
        to avoid information leak.
        """

        def _fit_feat(df, feat):
            """Return list of tuples defining bin ranges for a numerical feature using simple linear search"""

            if len(df) == 0:
                return []

            n_discretize_bins = min(
                self.n_discretize_bins, len(df[feat].unique())
            )  # In case there are fewer unique values than n_discretize_bins
            bin_size = len(df) // n_discretize_bins
            sorted_df = df.sort_values(by=[feat])
            sorted_values = sorted_df[feat].tolist()

            sizes = []  # for verbosity output
            if self.verbosity >= 4:
                print(
                    f"{feat}: fitting {len(df[feat].unique())} unique vals into {n_discretize_bins} bins"
                )

            bin_ranges = []  # result
            bin_num = 0  # current bin number

            ceil_i = -1  # current bin ceiling index
            ceil_val = None  # current bin upper bound

            floor_i = 0  # current bin start index
            floor_val = sorted_df.iloc[0][feat]  # current bin floor value

            prev_finish_val = None  # prev bin upper bound
            while bin_num < n_discretize_bins and floor_i < len(sorted_values):
                # jump to tentative ceiling index
                ceil_i = min(floor_i + bin_size, len(sorted_df) - 1)
                ceil_val = sorted_df.iloc[ceil_i][feat]

                # increment ceiling index until encounter a new value to ensure next bin size is correct
                while (
                    ceil_i < len(sorted_df) - 1  # not last bin
                    and sorted_df.iloc[ceil_i][feat]
                    == ceil_val  # keep looking for a new value
                ):
                    ceil_i += 1

                # found ceiling index. update values
                if self.verbosity >= 4:
                    sizes.append(ceil_i - floor_i)
                    print(
                        f"bin #{bin_num}, floor idx {floor_i} value: {sorted_df.iloc[floor_i][feat]}, ceiling idx {ceil_i} value: {sorted_df.iloc[ceil_i][feat]}"
                    )
                bin_range = (floor_val, ceil_val)
                bin_ranges.append(bin_range)

                # update for next bin
                floor_i = ceil_i + 1
                floor_val = ceil_val
                bin_num += 1

            # Guarantee min and max values
            bin_ranges[0] = (sorted_df.iloc[0][feat], bin_ranges[0][1])
            bin_ranges[-1] = (bin_ranges[-1][0], sorted_df.iloc[-1][feat])

            if self.verbosity >= 4:
                print(
                    f"-bin sizes {sizes}; dataVMR={rnd(np.var(df[feat])/np.mean(df[feat]))}, binVMR={rnd(np.var(sizes)/np.mean(sizes))}"
                )  # , axis=None, dtype=None, out=None, ddof=0)})
            return bin_ranges

        # Create dict to store fit definitions for each feature
        fit_dict = {}
        feats_to_fit = self.find_continuous_feats(df, ignore_feats=ignore_feats)
        if self.verbosity == 2:
            print(f"fitting bins for features {feats_to_fit}")
        if self.verbosity >= 2:
            print()

        # Collect fits in dict
        count = 1
        for feat in feats_to_fit:
            fit = _fit_feat(df, feat)
            fit_dict[feat] = fit
        self.bins_ = fit_dict

    def transform(self, df):
        """
        Uses a pre-collected dictionary of fits to transform df features into bins.
        Returns the fit df rather than modifying inplace.
        """

        if self.bins_ is None:
            return df

        # Replace each feature with bin transformations
        for feat, bin_fit_list in self.bins_.items():
            if feat in df.columns:
                df[feat] = df[feat].map(
                    lambda x: self._transform_value(x, bin_fit_list)
                )
        return df

    def _transform_value(self, value, bin_fit_list):
        """Return bin string name for a given numerical value. Assumes bin_fit_list is ordered."""
        min_val, min_bin = bin_fit_list[0][0], bin_fit_list[0]
        max_val, max_bin = bin_fit_list[-1][1], bin_fit_list[-1]
        for bin_fit in bin_fit_list:
            if value <= bin_fit[1]:
                start_name = str(round(bin_fit[0], self.names_precision)) if self.names_precision else str(int(bin_fit[0]))
                finish_name = str(round(bin_fit[1], self.names_precision)) if self.names_precision else str(int(bin_fit[1]))
                bin_name = "-".join([start_name, finish_name])
                return bin_name
        if value <= min_val:
            return min_bin
        elif value >= max_val:
            return max_bin
        else:
            raise ValueError("No bin found for value", value)

    def _try_rename_features(self, df, class_feat, feature_names):
        """Rename df columns according to user request."""
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
