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
import warnings
import numpy as np

from wittgenstein import base
from .base import Cond, Rule, Ruleset
from .base import rnd, score_accuracy, bin_df

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

    def out_model(self):
        """ Prints trained Ruleset model line-by-line: V represents 'or'; ^ represents 'and'. """
        if hasattr(self,'ruleset_'):
            self.ruleset_.out_pretty()
        else:
            print('no model fitted')

    def fit(self, df, y=None, class_feat=None, pos_class=None, n_discretize_bins=10, max_rules=None, max_rule_conds=None, max_total_conds=None, random_state=None):
        """ Fit a Ruleset model using a training DataFrame.

            args:
                df <DataFrame>: categorical training dataset
                y: <iterable>: class labels corresponding to df rows. Parameter y or class_feat (see next) must be provided.
                class_feat: column name of class feature (Use if class feature is still in df.)

                pos_class (optional): name of positive class. If not provided, defaults to class of first training example.
                n_discretize_bins (optional): Fit apparent numeric attributes into a maximum of n_discretize_bins discrete bins, inclusive on upper part of range.
                                              Pass None to disable auto-discretization. (default=10)

                random_state: (optional) random state to allow for repeatable results

                options to stop early. Intended for improving model interpretability or limiting training time on noisy datasets. Not specifically intended for use as a hyperparameter, since pruning already occurs during training, though it is certainly possible that tuning could improve model performance.
                max_rules (optional): maximum number of rules. default=None
                max_rule_conds (optional): maximum number of conds per rule. default=None
                max_total_conds (optional): maximum number of total conds in entire ruleset. default=None

        """

        # Stage 0: Setup

        # Set up trainset, set class feature name, and set pos class name
        df, self.class_feat, self.pos_class = base.trainset_classfeat_posclass(df, y=y, class_feat=class_feat, pos_class=pos_class)

        # Anything to discretize?
        df, self.bin_transformer_ = bin_df(df, n_discretize_bins=n_discretize_bins, ignore_feats=[self.class_feat], verbosity=self.verbosity)

        # Split df into pos, neg classes
        pos_df, neg_df = base.pos_neg_split(df, self.class_feat, self.pos_class)
        pos_df = pos_df.drop(self.class_feat,axis=1)
        neg_df = neg_df.drop(self.class_feat,axis=1)

        # Stage 1 (of 1): Grow Ruleset
        self.ruleset_ = Ruleset()
        self.ruleset_ = self._grow_ruleset(pos_df, neg_df,
            prune_size=self.prune_size, max_rules=max_rules, max_rule_conds=max_rule_conds, max_total_conds=max_total_conds, random_state=random_state)
        if self.verbosity >= 1:
            print()
            print('GREW RULESET:')
            self.ruleset_.out_pretty()
            print()

        # Fit probas
        self.recalibrate_proba(df, min_samples=None, require_min_samples=False, discretize=False)

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

        if not self.bin_transformer_:
            return self.ruleset_.predict(X_df, give_reasons=give_reasons)
        else:
            binned_X = X_df.copy()
            base.bin_transform(binned_X, self.bin_transformer_)
            return self.ruleset_.predict(binned_X, give_reasons=give_reasons)

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

    def predict_proba(self, X_df, give_reasons=False, ret_n=False, min_samples=1, discretize=True):
        # Drop class feature if user forgot to:
        df = X_df if self.class_feat not in X_df.columns else X_df.drop(self.class_feat, axis=1)
        return self.ruleset_.predict_proba(df, give_reasons=give_reasons, ret_n=ret_n, min_samples=min_samples, discretize=True, bin_transformer=self.bin_transformer_)

    def recalibrate_proba(self, Xy_df, min_samples=20, require_min_samples=True, discretize=True):
        """ Recalibrate a classifier's probability estimations using unseen labeled data. May improve .predict_proba generalizability.
            Does not affect the underlying model or which predictions it makes -- only probability estimates. Use params min_samples and require_min_samples to select desired behavior.

            Note1: RunTimeWarning will occur as a reminder when min_samples and require_min_samples params might result in unintended effects.
            Note2: It is possible recalibrating could result in some positive .predict predictions with <0.5 .predict_proba positive probability.

            Xy_df <DataFrame>: labeled data

            min_samples <int> (optional): required minimum number of samples per Rule
                                          default=10. set None to ignore min sampling requirement so long as at least one sample exists.
            require_min_samples <bool> (optional): True: halt (with warning) in case min_samples not achieved for all Rules
                                                   False: warn, but still replace Rules that have enough samples
            discretize <bool> (optional): If the classifier has already fit a discretization, automatically discretize recalibrate_proba's training data
                                          default=True
        """

        # Recalibrate
        self.ruleset_.recalibrate_proba(Xy_df, class_feat=self.class_feat, pos_class=self.pos_class, min_samples=min_samples, require_min_samples=require_min_samples, discretize=discretize, bin_transformer=self.bin_transformer_)

    def _grow_ruleset(self, pos_df, neg_df, prune_size, max_rules=None, max_rule_conds=None, max_total_conds=None, random_state=None, verbosity=0):
        """ Grow a Ruleset with (optional) pruning. """

        ruleset = Ruleset()
        ruleset._set_possible_conds(pos_df, neg_df)

        if not prune_size: prune_size = 0 # If not pruning, use all the data for growing
        pos_remaining = pos_df.copy()
        neg_remaining = neg_df.copy()
        self.rules = []

        # Stop adding disjunctions if there are no more positive examples to cover
        while (len(pos_remaining) > 0):

            # If applicable, check for user-specified early stopping
            if (max_rules is not None and len(ruleset.rules) >= max_rules) or (max_total_conds is not None and ruleset.count_conds() >= max_total_conds):
                break

            # Grow-prune split remaining uncovered examples (if applicable)
            pos_growset, pos_pruneset = base.df_shuffled_split(pos_remaining, (1-prune_size), random_state=random_state)
            neg_growset, neg_pruneset = base.df_shuffled_split(neg_remaining, (1-prune_size), random_state=random_state)
            if self.verbosity>=2:
                print(f'pos_growset {len(pos_growset)} pos_pruneset {len(pos_pruneset)}')
                print(f'neg_growset {len(neg_growset)} neg_pruneset {len(neg_pruneset)}')
                if not prune_size: print(f'(pruning is turned off)')

            # Grow Rule
            grown_rule = base.grow_rule(pos_growset, neg_growset, ruleset.possible_conds, max_rule_conds=max_rule_conds, verbosity=self.verbosity)

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

        # If applicable, trim total conds
        ruleset.trim_conds(max_total_conds=max_total_conds)

        # Return new ruleset
        return ruleset

##### Metric #####

def _IREP_prune_metric(self, pos_pruneset, neg_pruneset):
    """ Returns the prune value of a candidate Rule """

    P = len(pos_pruneset)
    N = len(neg_pruneset)
    p = self.num_covered(pos_pruneset)
    n = self.num_covered(neg_pruneset)
    return (p+(N - n)) / (P + N)
