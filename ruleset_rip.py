"""
This module implements incremental reduced error pruning (IREP) algorithm
for growing classification rulesets.
See https://www.let.rug.nl/nerbonne/teach/learning/cohen95fast.pdf
"""

# Author: Ilan Moscovitz <ilan.moscovitz@gmail.com>
# License: MIT

import pandas as pd


import base
from base import Ruleset, Rule, Cond

class IREP:
    """ Class for generating ruleset classification models. """

    def __init__(self, class_feat, pos_class=None, prune_size=.33):
        self.class_feat = class_feat
        self.pos_class = pos_class
        self.prune_size = prune_size

    def __str__(self):
        fitstr = f'fit ruleset={self.ruleset_}' if hasattr(self,'ruleset_') else 'not fit'
        return f'<IREP object {fitstr}>'
    __repr__ = __str__

    def fit(self, df, prune=True, seed=None, display=False):
        """ Fit an IREP to data by growing a classification Ruleset in disjunctive normal form.
            class_feat: name of DataFrame class feature
            prune=True/False: whether to prune Ruleset's Rules during its growth
            seed: (optional) random state for grow/prune split (if pruning)
        """

        # If not given by __init__, define positive class here
        if not self.pos_class:
            self.pos_class = df.iloc[0][class_feat]

        # Split df into pos, neg classes
        pos_df, neg_df = base.pos_neg_split(df, self.class_feat, self.pos_class)
        pos_df = pos_df.drop(self.class_feat,axis=1)
        neg_df = neg_df.drop(self.class_feat,axis=1)

        # Grow Ruleset
        self.ruleset_ = Ruleset()
        if prune:
            self.ruleset_ = self.grow_pruned_ruleset(pos_df, neg_df, prune_size=self.prune_size, seed=seed, display=display)
        else:
            self.ruleset_ = self.grow_unpruned_ruleset(pos_df, neg_df, display=display)

    def predict(self, X):
        """ Predict classes of X """

        if hasattr(self, 'ruleset_'):
            covered_indices = self.ruleset_.covers(X).index
            return [covered_i in covered_indices for covered_i in X.index]
        else:
            raise AttributeError('You should fit an IREP object before making predictions with it.')

    def score(self, X, y, score_function):
        """ Test performance of fit Ruleset.
            score_function: function that takes parameters (actuals, predictions)
            Examples: https://scikit-learn.org/stable/modules/model_evaluation.html#classification-metrics
        """

        predictions = self.predict(X)
        actuals = [yi==self.pos_class for yi in y]
        return score_function(actuals, predictions)

    def grow_unpruned_ruleset(self, pos_df, neg_df, display=False):
        """ Grow a Ruleset without pruning. Not recommended. """

        remaining_pos = pos_df.copy()
        remaining_neg = neg_df.copy()
        ruleset = Ruleset()

        while len(remaining_pos) > 0: # Stop adding disjunctions if there are no more positive examples to cover
            rule = base.grow_rule(remaining_pos, remaining_neg, display=display)
            if rule is None:
                break
            pos_covered = rule.covers(remaining_pos)
            neg_covered = rule.covers(remaining_neg)
            remaining_pos.drop(pos_covered.index, axis=0, inplace=True)
            ruleset.add(rule)
        return ruleset

    def grow_pruned_ruleset(self, pos_df, neg_df, prune_size=.33, seed=None, display=False):
        """ Grow a Ruleset with pruning. """

        pos_remaining = pos_df.copy()
        neg_remaining = neg_df.copy()
        ruleset = Ruleset()

        while len(pos_remaining) > 0: # Stop adding disjunctions if there are no more positive examples to cover
            pos_growset, pos_pruneset = base.df_shuffled_split(pos_remaining, prune_size, seed=seed)
            neg_growset, neg_pruneset = base.df_shuffled_split(neg_remaining, prune_size, seed=seed)
            grown_rule = base.grow_rule(pos_growset, neg_growset, display=display)

            if grown_rule is None:
                break
            pruned_rule = base.prune_rule(grown_rule, prune_metric, pos_pruneset, neg_pruneset)

            prune_precision = base.precision(pruned_rule, pos_pruneset, neg_pruneset)
            if not prune_precision or prune_precision < .50:
                break
            else:
                ruleset.add(pruned_rule)
                pos_remaining.drop(pruned_rule.covers(pos_remaining).index, axis=0, inplace=True)
                neg_remaining.drop(pruned_rule.covers(neg_remaining).index, axis=0, inplace=True)
        return ruleset

def prune_metric(rule, pos_pruneset, neg_pruneset):
    """ Returns the prune value of a candidate Rule """

    P = len(pos_pruneset)
    N = len(neg_pruneset)
    p = rule.num_covered(pos_pruneset)
    n = rule.num_covered(neg_pruneset)
    return (p+(N - n)) / (P + N)
