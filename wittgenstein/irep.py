"""
Implementation of incremental reduced error pruning (IREP*) algorithm for growing classification rulesets.
See https://www.let.rug.nl/nerbonne/teach/learning/cohen95fast.pdf
"""

# Author: Ilan Moscovitz <ilan.moscovitz@gmail.com>
# License: MIT

import pandas as pd
import random
import copy
import numpy as np

from wittgenstein import base, base_functions
from .base import Cond, Rule, Ruleset, bin_df
from .base_functions import score_accuracy, stop_early

from .catnap import CatNap

class IREP:
    """ Class for generating ruleset classification models. """

    def __init__(self, prune_size=.33, max_rules=None, max_rule_conds=None, max_total_conds=None,
                 n_discretize_bins=10, verbosity=0, random_state=None):
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
        self.POSSIBLE_HYPERPARAM_NAMES = {'prune_size','max_rules','max_rule_conds','max_total_conds', 'n_discretize_bins'}

        self.prune_size = prune_size
        self.max_rules = max_rules
        self.max_rule_conds = max_rule_conds
        self.max_total_conds = max_total_conds

        self.n_discretize_bins = n_discretize_bins

        self.random_state = random_state
        self.verbosity = verbosity

    def __str__(self):
        """ Returns string representation of an IREP object. """
        params=str(self.get_params())+'>'
        params=params.replace(': ','=').replace("'",'').replace('{',"(").replace('}',')')
        return f'<IREP{params}'
    __repr__ = __str__

    def out_model(self):
        """ Prints trained Ruleset model line-by-line: V represents 'or'; ^ represents 'and'. """
        if hasattr(self,'ruleset_'):
            self.ruleset_.out_pretty()
        else:
            print('no model fitted')

    def fit(self, X_or_Xy, y=None, class_feat=None, pos_class=None, feature_names=None, cn_optimize=True):
        """ Fit a Ruleset model.

            args:
                X_or_Xy: categorical training dataset <pandas DataFrame, numpy array, or other Python iterable>
                y: class labels corresponding to trainset rows. Parameter y or class_feat (see next) must be provided.
                class_feat: column name of class feature (Use if class feature is still in df.)

                feature_names (optional): if trainset inputted as non-DataFrame iterable (such as numpy array), sets feature names. If None, and training data not a DatFrame, feature for arrays and python iterables will be set to column indices. (default=None)
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

        # Preprocess training data
        preprocess_params = {'X_or_Xy':X_or_Xy,
                             'y':y,
                             'class_feat':class_feat,
                             'pos_class':pos_class,
                             #'bin_transformer':
                             'n_discretize_bins':self.n_discretize_bins,
                             'user_requested_feature_names':feature_names,
                             'verbosity':self.verbosity
                            }
        df, self.class_feat, self.pos_class, self.bin_transformer_ = base_functions.preprocess_training_data(preprocess_params)

        # Create CatNap
        # possible minor speedup if pass cond_subset of only pos_class conds?
        if cn_optimize:
            self.cn = CatNap(df, feat_subset=None, cond_subset=None, class_feat=self.class_feat, pos_class=None)

        # Split df into pos, neg classes
        pos_df, neg_df = base_functions.pos_neg_split(df, self.class_feat, self.pos_class)
        pos_df = pos_df.drop(self.class_feat,axis=1)
        neg_df = neg_df.drop(self.class_feat,axis=1)

        # Stage 1 (of 1): Grow Ruleset
        if self.verbosity>=1: print('\nbuilding Ruleset...')
        self.ruleset_ = Ruleset()
        if not cn_optimize:
            self.ruleset_ = self._grow_ruleset(pos_df, neg_df)
        else:
            self.ruleset_ = self._grow_ruleset_cn(pos_df, neg_df)

        if self.verbosity >= 1:
            print('\nGREW RULESET:')
            self.ruleset_.out_pretty()

        # Issue warning if Ruleset is universal or empty
        self.ruleset_._check_allpos_allneg(warn=True, warnstack=[('irep','fit')])

        # Set selected and trainset features
        self.selected_features_ = self.ruleset_.get_selected_features()
        self.trainset_features_ = df.drop(self.class_feat,axis=1).columns.tolist()

        # Fit probas
        #if self.verbosity>=1: print('\ncalibrating probas for predict_proba...\n')
        self.recalibrate_proba(df, min_samples=None, require_min_samples=False, discretize=False)

        # Cleanup
        if cn_optimize: del(self.cn)

    def predict(self, X, give_reasons=False, feature_names=None):
        """ Predict classes of data using a IREP-fit model.

            args:
                X: examples to make predictions on. Should include at least the features selected by model.

                give_reasons (optional) <bool>: whether to provide reasons for each prediction made.
                feature_names (optional) <list>: specify different feature names for X to ensure they match up with the names of selected features.

            returns:
                list of <bool> values corresponding to examples. True indicates positive predicted class; False non-positive class.

                If give_reasons is True, returns a tuple that contains the above list of predictions
                    and a list of the corresponding reasons for each prediction;
                    for each positive prediction, gives a list of all the covering Rules, for negative predictions, an empty list.
        """

        if not hasattr(self, 'ruleset_'):
            raise AttributeError('You should fit an IREP object with .fit method before making predictions with it.')

        # Preprocess prediction data
        preprocess_params = {'X':X,
                             'class_feat':self.class_feat,
                             'pos_class':self.pos_class,
                             'bin_transformer_':self.bin_transformer_,

                             'user_requested_feature_names':feature_names,
                             'selected_features_':self.selected_features_,
                             'trainset_features_':self.trainset_features_,
                             'verbosity':self.verbosity
                            }

        X_df = base_functions.preprocess_prediction_data(preprocess_params)

        return self.ruleset_.predict(X_df, give_reasons=give_reasons)

    def score(self, X, y, score_function=score_accuracy):
        """ Test performance of an IREP-fit model.

            X: 2-D independent attributes values
            y: 1-D corresponding target values

            score_function (optional): function that takes two parameters: actuals <iterable<bool>>, predictions <iterable<bool>>,
                                       containing class values. (default=accuracy)
                                       this parameter is intended to be compatible with sklearn's scoring functions:
                                       https://scikit-learn.org/stable/modules/model_evaluation.html#classification-metrics
        """

        predictions = self.predict(X)
        actuals = [yi==self.pos_class for yi in base_functions.aslist(y)]
        return score_function(actuals, predictions)

    def predict_proba(self, X, feature_names=None, give_reasons=False):

        if not hasattr(self, 'ruleset_'):
            raise AttributeError('You should fit an IREP object with .fit method before making predictions with it.')

        # Preprocess prediction data
        preprocess_params = {'X':X,
                             'class_feat':self.class_feat,
                             'pos_class':self.pos_class,
                             'bin_transformer_':self.bin_transformer_,

                             'user_requested_feature_names':feature_names,
                             'selected_features_':self.selected_features_,
                             'trainset_features_':self.trainset_features_,
                             'verbosity':self.verbosity
                            }

        X_df = base_functions.preprocess_prediction_data(preprocess_params)

        return self.ruleset_.predict_proba(X_df, give_reasons=give_reasons)

    def recalibrate_proba(self, X_or_Xy, y=None, feature_names=None, min_samples=20, require_min_samples=True, discretize=True):
        """ Recalibrate a classifier's probability estimations using unseen labeled data. May improve .predict_proba generalizability.
            Does not affect the underlying model or which predictions it makes -- only probability estimates. Use params min_samples and require_min_samples to select desired behavior.

            Note1: RunTimeWarning will occur as a reminder when min_samples and require_min_samples params might result in unintended effects.
            Note2: It is possible recalibrating could result in some positive .predict predictions with <0.5 .predict_proba positive probability.

            Xy: labeled data

            min_samples <int> (optional): required minimum number of samples per Rule
                                          default=10. set None to ignore min sampling requirement so long as at least one sample exists.
            require_min_samples <bool> (optional): True: halt (with warning) in case min_samples not achieved for all Rules
                                                   False: warn, but still replace Rules that have enough samples
            discretize <bool> (optional): If the classifier has already fit a discretization, automatically discretize recalibrate_proba's training data
                                          default=True
        """

        # Preprocess training data
        preprocess_params = {'X_or_Xy':X_or_Xy,
                             'y':y,
                             'class_feat':self.class_feat,
                             'pos_class':self.pos_class,
                             'bin_transformer_':self.bin_transformer_ if discretize else None,

                             'user_requested_feature_names':feature_names,
                             'min_samples':min_samples,
                             'require_min_samples':require_min_samples,
                             'verbosity':self.verbosity
                            }

        df = base_functions._preprocess_recalibrate_proba_data(preprocess_params)

        # Recalibrate
        base_functions.recalibrate_proba(self.ruleset_, Xy_df=df, class_feat=self.class_feat, pos_class=self.pos_class, min_samples=min_samples, require_min_samples=require_min_samples)

    def get_params(self, deep=True):
        return {param:self.__dict__.get(param) for param in self.POSSIBLE_HYPERPARAM_NAMES}
    def set_params(self, **parameters):
        for parameter, value in parameters.items():
            #if parameter in self.POSSIBLE_HYPERPARAM_NAMES:
            setattr(self, parameter, value)
        return self

    def _grow_ruleset(self, pos_df, neg_df):
        """ Grow a Ruleset with (optional) pruning. """

        ruleset = Ruleset()
        ruleset._set_possible_conds(pos_df, neg_df)

        prune_size = self.prune_size if self.prune_size is not None else 0 # If not pruning, use all the data for growing
        pos_remaining = pos_df.copy()
        neg_remaining = neg_df.copy()
        self.rules = []

        # Stop adding disjunctions if there are no more positive examples to cover
        while (len(pos_remaining) > 0):

            # If applicable, check for user-specified early stopping
            if stop_early(ruleset, self.max_rules, self.max_total_conds):
                break

            # Grow-prune split remaining uncovered examples (if applicable)
            pos_growset, pos_pruneset = base_functions.df_shuffled_split(pos_remaining, (1-prune_size), random_state=self.random_state)
            neg_growset, neg_pruneset = base_functions.df_shuffled_split(neg_remaining, (1-prune_size), random_state=self.random_state)
            if self.verbosity>=2:
                print(f'pos_growset {len(pos_growset)} pos_pruneset {len(pos_pruneset)}')
                print(f'neg_growset {len(neg_growset)} neg_pruneset {len(neg_pruneset)}')
                if not prune_size: print(f'(pruning is turned off)')

            # Grow Rule
            grown_rule = base_functions.grow_rule(pos_growset, neg_growset, ruleset.possible_conds, max_rule_conds=self.max_rule_conds, verbosity=self.verbosity)

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
                pruned_rule = base_functions.prune_rule(grown_rule, _IREP_prune_metric, pos_pruneset, neg_pruneset, verbosity=self.verbosity)

                # Stop if the Rule is bad
                prune_precision = base_functions.precision(pruned_rule, pos_pruneset, neg_pruneset)
                if not prune_precision or prune_precision < .50:
                    break
                # Otherwise, add new Rule, remove covered examples, and continue
                else:
                    ruleset.add(pruned_rule)
                    if self.verbosity>=2:
                        print(f"updated ruleset: {ruleset.truncstr(direction='right')}")
                        print()
                    pos_remaining, neg_remaining = base_functions.rm_covered(pruned_rule, pos_remaining, neg_remaining)
                    if self.verbosity>=3:
                        print(f'examples remaining: {len(pos_remaining)} pos, {len(neg_remaining)} neg')
                        print()

        # If applicable, trim total conds
        ruleset.trim_conds(max_total_conds=self.max_total_conds)

        # Return new ruleset
        return ruleset

    def _grow_ruleset_cn(self, pos_df, neg_df):
        """ Grow a Ruleset with (optional) pruning. """

        ruleset = Ruleset()
        ruleset.possible_conds = self.cn.conds
        #ruleset._set_possible_conds(pos_df, neg_df)

        prune_size = self.prune_size if self.prune_size is not None else 0 # If not pruning, use all the data for growing
        pos_remaining_idx = set(pos_df.index.tolist())
        neg_remaining_idx = set(neg_df.index.tolist())
        #self.rules = []

        # Stop adding disjunctions if there are no more positive examples to cover
        while (len(pos_remaining_idx) > 0):

            # If applicable, check for user-specified early stopping
            if stop_early(ruleset, self.max_rules, self.max_total_conds):
                break

            # Grow-prune split remaining uncovered examples (if applicable)
            pos_growset_idx, pos_pruneset_idx = base_functions.set_shuffled_split(pos_remaining_idx, (1-prune_size), random_state=self.random_state)
            neg_growset_idx, neg_pruneset_idx = base_functions.set_shuffled_split(neg_remaining_idx, (1-prune_size), random_state=self.random_state)

            if self.verbosity>=2:
                print(f'pos_growset {len(pos_growset_idx)} pos_pruneset {len(pos_pruneset_idx)}')
                print(f'neg_growset {len(neg_growset_idx)} neg_pruneset {len(neg_pruneset_idx)}')
                if not prune_size: print(f'(pruning is turned off)')

            # Grow Rule
            grown_rule = base_functions.grow_rule_cn(self.cn, pos_growset_idx, neg_growset_idx, initial_rule=Rule(), max_rule_conds=self.max_rule_conds, verbosity=self.verbosity)

            # If not pruning, add Rule to Ruleset and drop only the covered positive examples
            if not prune_size:
                ruleset.add(grown_rule)
                if self.verbosity>=2:
                    print(f"updated ruleset: {ruleset.truncstr(direction='right')}")
                    print()
                pos_remaining_idx = pos_remaining_idx - self.cn.rule_covers(grown_rule, pos_remaining_idx)
                if self.verbosity>=3:
                    print(f'examples remaining: {len(pos_remaining_idx)} pos, {len(neg_remaining_idx)} neg')
                    print()

            # If pruning, prune Rule, assess if it's time to stop, and drop all covered examples
            else:
                pruned_rule = base_functions.prune_rule_cn(self.cn, grown_rule, _IREP_prune_metric_cn, pos_pruneset_idx, neg_pruneset_idx, verbosity=self.verbosity)

                # Stop if the Rule is bad
                prune_precision = base_functions.rule_precision_cn(self.cn, pruned_rule, pos_pruneset_idx, neg_pruneset_idx)
                if not prune_precision or prune_precision < .50:
                    break
                # Otherwise, add new Rule, remove covered examples, and continue
                else:
                    ruleset.add(pruned_rule)
                    if self.verbosity>=2:
                        print(f"updated ruleset: {ruleset.truncstr(direction='right')}")
                        print()
                    # Remove ruleset-covered rules from training
                    pos_remaining_idx, neg_remaining_idx = base_functions.rm_rule_covers_cn(self.cn, pruned_rule, pos_remaining_idx, neg_remaining_idx)
                    if self.verbosity>=3:
                        print(f'examples remaining: {len(pos_remaining_idx)} pos, {len(neg_remaining_idx)} neg')
                        print()

        # If applicable, trim total conds
        ruleset.trim_conds(max_total_conds=self.max_total_conds)

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

def _IREP_prune_metric_cn(cn, rule, pos_idxs, neg_idxs):
    """ Returns the prune value of a candidate Rule """

    P = len(pos_idxs)
    N = len(neg_idxs)
    p = len(cn.rule_covers(rule, pos_idxs))
    n = len(cn.rule_covers(rule, neg_idxs))
    return (p+(N - n)) / (P + N)
