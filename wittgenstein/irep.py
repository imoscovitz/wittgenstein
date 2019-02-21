"""
This module implements incremental reduced error pruning (IREP*) algorithm
for growing classification rulesets.
See https://www.let.rug.nl/nerbonne/teach/learning/cohen95fast.pdf
"""

# Author: Ilan Moscovitz <ilan.moscovitz@gmail.com>
# License: MIT

import pandas as pd
import math
import random
import copy

from ruleset import base
from .base import Cond, Rule, Ruleset
from .base import rnd, fit_bins, bin_transform, score_accuracy

class IREP:
    """ Class for generating ruleset classification models. """

    def __init__(self, prune_size=.33, verbosity=0):
        """ Creates a new IREP object.

            args:
                prune_size (optional):   proportion of training set to be used for pruning (defailt=.33).
                                           Set to None to skip pruning (not recommended).
                verbosity (optional):    output information about the training process (default=0)
                                           1: Show results of each major phase
                                           2: Show Ruleset grow/optimization steps
                                           3: Show Ruleset grow/optimization calculations
                                           4: Show Rule grow/prune steps
                                           5: Show Rule grow/prune calculations
        """
        self.prune_size = prune_size
        self.verbosity = verbosity

    def __str__(self):
        """ Returns string representation of an IREP object. """
        fitstr = f'with fit ruleset' if hasattr(self,'ruleset_') else '(unfit)'
        return f'<IREP object {fitstr}>'
    __repr__ = __str__

    def fit(self, df, y=None, class_feat=None, pos_class=None, n_discretize_bins=None, random_state=None):
        """ Fit a Ruleset model using a training DataFrame.

            args:
                df <DataFrame>: categorical training dataset
                y: <iterable>: class labels corresponding to df rows. Parameter y or class_feat (see next) must be provided.
                class_feat: column name of class feature (Use if class feature is still in df.)

                pos_class (optional): name of positive class. If not provided, defaults to class of first training example.
                n_discretize_bins (optional): try to fit apparent numeric attributes into n_discretize_bins discrete bins.
                                              Pass None to disable auto-discretization. (default=None)
                random_state: (optional) random state to allow for repeatable results
        """

        # Stage 0: Setup

        # Set up trainset, set class feature name, and set pos class name
        df, self.class_feat, self.pos_class = base.trainset_classfeat_posclass(df, y=y, class_feat=class_feat, pos_class=pos_class)

        # Anything to discretize?
        numeric_feats = base.find_numeric_feats(df, min_unique=n_discretize_bins, ignore_feats=[self.class_feat])
        if numeric_feats:
            if n_discretize_bins is not None:
                if self.verbosity==1:
                    print(f'binning data...\n')
                elif self.verbosity>=2:
                    print(f'binning features {numeric_feats}...')
                self.bin_transformer_ = fit_bins(df, n_bins=n_discretize_bins, output=False, ignore_feats=[self.class_feat], verbosity=self.verbosity)
                binned_df = bin_transform(df, self.bin_transformer_)
            else:
                n_unique_values = sum([len(u) for u in [df[f].unique() for f in numeric_feats]])
                warnings.warn(f'Optional param n_discretize_bins=None, but there are apparent numeric features: {numeric_feats}. \n Treating {n_unique_values} numeric values as nominal', RuntimeWarning)
                binned_df=None
        else:
            binned_df=None

        # Split df into pos, neg classes
        pos_df, neg_df = base.pos_neg_split(df, self.class_feat, self.pos_class) if binned_df is None else base.pos_neg_split(binned_df, self.class_feat, self.pos_class)
        pos_df = pos_df.drop(self.class_feat,axis=1)
        neg_df = neg_df.drop(self.class_feat,axis=1)

        # Stage 1 (of 1): Grow Ruleset
        self.ruleset_ = Ruleset()
        self.ruleset_ = self._grow_ruleset(pos_df, neg_df,
            prune_size=self.prune_size, random_state=random_state)
        if self.verbosity >= 1:
            print()
            print('GREW RULESET:')
            self.ruleset_.out_pretty()
            print()

    def predict(self, X_df, give_reasons=False):
        """ Predict classes of data using a IREP-fit model.

            args:
                X_df <DataFrame>: examples to make predictions on.

                give_reasons (optional) <bool>: whether to provide reasons for each prediction made.

            returns:
                list of <bool> values corresponding to examples. True indicates positive predicted class; False non-positive class.

                If give_reasons is True, returns a tuple that contains the above list of predictions
                    and a list of the corresponding reasons for each prediction;
                    for each positive prediction, gives a list of all the covering Rules, for negative predictions, an empty list.
        """

        if not hasattr(self, 'ruleset_'):
            raise AttributeError('You should fit an IREP object before making predictions with it.')
        else:
            return self.ruleset_.predict(X_df, give_reasons=give_reasons)

    def score(self, X, y, score_function=score_accuracy):
        """ Test performance of an IREP-fit model.

            X: <DataFrame> of independent attributes
            y: <DataFrame> or <iterable> of matching dependent target values

            score_function (optional): function that takes two parameters: actuals <iterable<bool>>, predictions <iterable<bool>>,
                                       containing class values. (default=accuracy)
                                       this parameter is intended to be compatible with sklearn's scoring functions:
                                       https://scikit-learn.org/stable/modules/model_evaluation.html#classification-metrics
        """

        predictions = self.predict(X)
        if type(y)==pd.core.frame.DataFrame:
            actuals = [yi==self.pos_class for yi in y.tolist()]
        else:
            actuals = [yi==self.pos_class for yi in y]
        return score_function(actuals, predictions)

    def _grow_ruleset(self, pos_df, neg_df, prune_size, random_state=None, verbosity=0):
        """ Grow a Ruleset with (optional) pruning. """

        ruleset = Ruleset()
        ruleset._set_possible_conds(pos_df, neg_df)

        if not prune_size: prune_size = 0 # If not pruning, use all the data for growing
        pos_remaining = pos_df.copy()
        neg_remaining = neg_df.copy()
        self.rules = []
        while len(pos_remaining) > 0: # Stop adding disjunctions if there are no more positive examples to cover
            # Grow-prune split remaining uncovered examples (if applicable)
            pos_growset, pos_pruneset = base.df_shuffled_split(pos_remaining, (1-prune_size), random_state=random_state)
            neg_growset, neg_pruneset = base.df_shuffled_split(neg_remaining, (1-prune_size), random_state=random_state)
            if self.verbosity>=2:
                print(f'pos_growset {len(pos_growset)} pos_pruneset {len(pos_pruneset)}')
                print(f'neg_growset {len(neg_growset)} neg_pruneset {len(neg_pruneset)}')
                if not prune_size: print(f'(pruning is turned off)')

            # Grow Rule
            grown_rule = base.grow_rule(pos_growset, neg_growset, ruleset.possible_conds, verbosity=self.verbosity)

            # If not pruning, add Rule to Ruleset and drop only the covered positive examples
            if not prune_size:
                ruleset.add(grown_rule)
                if self.verbosity>=2:
                    print(f"updated ruleset: {ruleset.truncstr(direction='right')}")
                    print()
                rule_covers_pos = grown_rule.covers(pos_remaining)
                pos_remaining = pos_remaining.drop(rule_covers_pos.index, axis=0)
                if self.verbosity>=3:
                    print(f'examples remaining: {len(pos_remaining)} pos, {len(neg_remaining)} neg')
                    print()

            # If pruning, prune Rule, assess if it's time to stop, and drop all covered examples
            else:
                pruned_rule = base.prune_rule(grown_rule, _IREP_prune_metric, pos_pruneset, neg_pruneset, verbosity=self.verbosity)

                # Stop if the Rule is bad
                prune_precision = base.precision(pruned_rule, pos_pruneset, neg_pruneset)
                if not prune_precision or prune_precision < .50:
                    break
                # Otherwise, add new Rule, remove covered examples, and continue
                else:
                    ruleset.add(pruned_rule)
                    if self.verbosity>=2:
                        print(f"updated ruleset: {ruleset.truncstr(direction='right')}")
                        print()
                    pos_remaining, neg_remaining = base.rm_covered(pruned_rule, pos_remaining, neg_remaining)
                    if self.verbosity>=3:
                        print(f'examples remaining: {len(pos_remaining)} pos, {len(neg_remaining)} neg')
                        print()
        return ruleset

##### Metric #####

def _IREP_prune_metric(self, pos_pruneset, neg_pruneset):
    """ Returns the prune value of a candidate Rule """

    P = len(pos_pruneset)
    N = len(neg_pruneset)
    p = self.num_covered(pos_pruneset)
    n = self.num_covered(neg_pruneset)
    return (p+(N - n)) / (P + N)
