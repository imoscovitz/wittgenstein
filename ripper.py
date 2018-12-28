"""
This module implements ... (RIPPER) algorithm
for growing classification rulesets.
See https://www.let.rug.nl/nerbonne/teach/learning/cohen95fast.pdf
"""

# Author: Ilan Moscovitz <ilan.moscovitz@gmail.com>
# License: MIT

import pandas as pd
import copy
import math

import base
from base import Ruleset, Rule, Cond

# Number of ruleset bits allowed beyond minimum for MDL
#PRUNE_SIZE = .33

class RIPPER:
    """ Class for generating ruleset classification models. """

    def __init__(self, class_feat, pos_class=None, prune_size=.33, dl_allowance=64):
        self.class_feat = class_feat
        self.pos_class = pos_class
        self.prune_size = prune_size
        self.dl_allowance = dl_allowance

    def __str__(self):
        fitstr = f'fit ruleset={self.ruleset_}' if hasattr(self,'ruleset_') else 'not fit'
        return f'<IREP object {fitstr}>'
    __repr__ = __str__

    def fit(self, df, prune=True, seed=None, verbose=False):
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
        self.ruleset_ = self.grow_ruleset(pos_df, neg_df,
            prune_size=self.prune_size, dl_allowance=self.dl_allowance,
            seed=seed, verbose=verbose)

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


    def grow_ruleset(self, pos_df, neg_df, prune_size, dl_allowance, seed=None, verbose=False):
        """ Grow a Ruleset with pruning. """

        pos_remaining = pos_df.copy()
        neg_remaining = neg_df.copy()
        ruleset = Ruleset()
        ruleset_dl = None
        mdl = None      # Minimum description length (in bits)

        while (len(pos_remaining) > 0) and (ruleset_dl is None or (ruleset_dl - mdl) <= self.dl_allowance):
            pos_growset, pos_pruneset = base.df_shuffled_split(pos_remaining, prune_size, seed=seed)
            neg_growset, neg_pruneset = base.df_shuffled_split(neg_remaining, prune_size, seed=seed)

            # Grow Rule
            grown_rule = base.grow_rule(pos_growset, neg_growset)
            if grown_rule is None:
                break
            if verbose:
                print(f'grew rule: {grown_rule}')

            # Prune Rule
            pruned_rule = base.prune_rule(grown_rule, prune_metric, pos_pruneset, neg_pruneset)
            if verbose:
                print(f'pruned rule: {pruned_rule}')

            # Add rule; calculate new description length
            ruleset.add(pruned_rule) # "*After* each rule is added, the total description length of the rule set and examples is computed."
            if ruleset_dl is None:   # First Rule to be added
                ruleset_dl = total_bits(pruned_rule, pos_remaining, neg_remaining)
                mdl = ruleset_dl
            else:
                ruleset_dl += total_bits(pruned_rule, pos_remaining, neg_remaining)
                mdl = ruleset_dl if ruleset_dl<mdl else mdl

            # Update dataset
            pos_remaining.drop(pruned_rule.covers(pos_remaining).index, axis=0, inplace=True)
            neg_remaining.drop(pruned_rule.covers(neg_remaining).index, axis=0, inplace=True)

            if verbose:
                print(f'examples remaining: {len(pos_remaining)} pos, {len(neg_remaining)} neg')
                print(f'mdl {round(mdl,2)}')
                print()
        return ruleset

    def optimize_ruleset(self, pos_df, neg_df, prune_size):
        pos_remaining = pos_df.copy()
        neg_remaining = neg_df.copy()
        optimized_ruleset = copy.deepcopy(self.ruleset_)

        for i, rule in enumerate(current_ruleset):
            current_ruleset = copy.deepcopy(self.ruleset_)
            replacement = grow_replacement_rule(self, i, current_ruleset)
            print(f'rule {rule} replacement {replacement}')

    def grow_replacement_rule(self, i, ruleset, pos_df, neg_df):
        """ Fit a new rule to add to a ruleset """
        rule0 = Rule()
        rule1 = Rule()
        while len(rule0.covers(neg_df)) > 0 and rule1 is not None: # Stop refining rule if no negative examples remain
            rule1 = rule0.best_successor(pos_df, neg_df)
            #print(f'growing rule... {rule1}')
            if rule1 is not None:
                rule0 = rule1

        if rule0.isempty():
            return None
        else:
            return rule0


def prune_metric(rule, pos_pruneset, neg_pruneset):
    """ RIPPER/IREP* prune metric.
        Returns the prune value of a candidate Rule.
    """

    p = rule.num_covered(pos_pruneset)
    n = rule.num_covered(neg_pruneset)
    print(f'on pruneset iter, p_covered {p} n_covered {n}')
    return (p - n + 1) / (p + n + 1)         # Unclear from the paper how they handle divzero when p+n=0, so for now I Laplaced it
                                             # Weka modified the formula to (p+1)/(p+n+2), but it gives different values

    #if (p+n) == 0:
    #    return None
    #else:
    #    return (p - n) / (p + n)

def rule_bits(rule, pos_df, neg_df):
    k = len(rule.conds)                                 # Number of rule conditions
    n = len(Rule().successors(pos_df, neg_df))          # Number of possible conditions
    pr = k/n                                            #

    S = k*math.log2(1/pr) + (n-k)*math.log2((1/(1-pr)))  # S(n, k, pr)
    K = math.log2(k)                                     # Number bits need to send integer k
    rule_dl = 0.5*(K + S)                                # Divide by 2 a la Quinlan and Cohen "to adjust for possible redundency in attributes"
    print(f'rule_bits| rule {rule} k {k} n {n} pr {pr}: rule_bits {round(rule_dl,2)}')

    return rule_dl

def exceptions_bits(rule, pos_df, neg_df):
    P = len(pos_df)                         # Total number of positive examples
    p = rule.num_covered(pos_df)            # Number positive examples covered by rule
    fp = rule.num_covered(neg_df)           # Number false positives = negatives covered by the rule
    fn = P - p                              # Number false negatives = positives not covered by the rule
    exceptions_dl = math.log2(base.nCr(p,fp)) + math.log2(base.nCr((P-p),fn))
    print(f'exceptions_bits| P {P} p {p} fp {fp} fn {fn}: exceptions_bits {round(exceptions_dl,2)}')

    return exceptions_dl

def total_bits(rule, pos_df, neg_df):
    total = rule_bits(rule, pos_df, neg_df) + exceptions_bits(rule, pos_df, neg_df)
    print(f'Rule {rule} total_bits {round(total,2)}')
    return total


#def exceptions_bits(rule, pos_df, neg_df):



#def bits(rule, pos_remaining, neg_remaining):#
#    data_bits =







"""
def grow_unpruned_ruleset(self, pos_df, neg_df, display=False):


    remaining_pos = pos_df.copy()
    remaining_neg = neg_df.copy()
    ruleset = Ruleset()

    while len(remaining_pos) > 0: # Stop adding disjunctions if there are no more positive examples to cover
        rule = self.grow_rule(remaining_pos, remaining_neg, display=display)
        if rule is None:
            break
        pos_covered = rule.covers(remaining_pos)
        neg_covered = rule.covers(remaining_neg)
        remaining_pos.drop(pos_covered.index, axis=0, inplace=True)
        ruleset.add(rule)
    return ruleset
"""



"""
def bits(rule, pos_remaining, neg_remaining):
    P = len(pos_remaining)                  # Total number of positive examples
    p = rule.num_covered(pos_remaining)     # Number positive examples covered by rule
    fp = rule.num_covered(neg_remaining)    # Number false positives = negatives covered by the rule
    fn = P - p                              # Number false negatives = positives not covered by the rule

    return math.log2(p/fp) + math.log2((P-p)/fn)
"""
