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

class IREP:
    """ Class for generating ruleset classification models. """

    def __init__(self, class_feat, pos_class=None, prune_size=.33):
        self.class_feat = class_feat
        self.pos_class = pos_class
        self.prune_size = prune_size

    def __str__(self):
        fitstr = f'fit ruleset={self.ruleset}' if hasattr(self,'ruleset_') else 'not fit'
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
        pos_df, neg_df = pos_neg_split(df, self.class_feat, self.pos_class)
        pos_df = pos_df.drop(self.class_feat,axis=1)
        neg_df = neg_df.drop(self.class_feat,axis=1)

        # Grow Ruleset
        self.ruleset_ = Ruleset()
        if prune:
            self.ruleset_.grow_pruned(pos_df, neg_df, prune_size=self.prune_size, seed=seed, display=display)
        else:
            self.ruleset_.grow_unpruned(pos_df, neg_df, display=display)

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

class Ruleset:
    """ Base IREP model.
        Implements collection of Rules in disjunctive normal form.
    """

    def __init__(self, rules=[]):
        self.rules = rules
        self.cond_count = 0

    def __str__(self):
        ruleset_str = str([str(rule) for rule in self.rules]).replace(',','v').replace("'","").replace(' ','')
        return ruleset_str

    def __repr__(self):
        ruleset_str = str([str(rule) for rule in self.rules]).replace(',','v').replace("'","").replace(' ','')
        return f'<Ruleset object: {ruleset_str}>'

    def covers(self, df):
        """ Returns instances covered by the Ruleset (i.e. those which are not in contradiction with it). """

        if not self.rules:
            return df
        else:
            covered = self.rules[0].covers(df).copy()
            for rule in self.rules[1:]:
                covered = covered.append(rule.covers(df))
            return covered

    def grow_unpruned(self, pos_df, neg_df, display=False):
        """ Grow a Ruleset without pruning. Not recommended. """

        remaining_pos = pos_df.copy()
        remaining_neg = neg_df.copy()
        self.rules = []
        while len(remaining_pos) > 0: # Stop adding disjunctions if there are no more positive examples to cover
            #if display: print('Remaining pos:',num_pos(remaining, class_feat, pos_class))

            rule = Rule()
            rule.grow(remaining_pos, remaining_neg, display=display)
            #if display:print(rule)
            #if display: print('Grown:',rule,'\n')

            if rule is None:
                #if display: print(f'finished growing {self}')
                break

            pos_covered = rule.covers(remaining_pos)
            neg_covered = rule.covers(remaining_neg)

            #if display:print(f'rule covered {rule_covers}')
            #if display: print('new_rule_covers:',rule_covers,'\n')


            #if display:print(f'pos covered {rule_covers_pos}')
            #if display: print('rule_covers_pos',rule_covers)

            remaining_pos.drop(pos_covered.index, axis=0, inplace=True)

            #if display:print(f'remaining {len(remaining)}')
            self.rules.append(rule)
            #if display:print(f'Ruleset is {self}')
            #if display:print()

        # Count cond complexity
        self.cond_count = sum([len(rule.conds) for rule in self.rules])

    def grow_pruned(self, pos_df, neg_df, prune_size=.33, seed=None, display=False):
        """ Grow a Ruleset with pruning. """

        pos_remaining = pos_df.copy()
        neg_remaining = neg_df.copy()
        self.rules = []
        while len(pos_remaining) > 0: # Stop adding disjunctions if there are no more positive examples to cover
            pos_growset, pos_pruneset = df_shuffled_split(pos_remaining, prune_size, seed=seed)
            neg_growset, neg_pruneset = df_shuffled_split(neg_remaining, prune_size, seed=seed)
            grown_rule = Rule()
            grown_rule.grow(pos_growset, neg_growset, display=display)
            #if display: print("Grown:",grown_rule)

            pruned_rule = grown_rule.pruned(pos_pruneset, neg_pruneset) # Pruned is a make -- not a modifier -- method
            #if display: print("Pruned to:",pruned_rule)

            prune_precision = precision(pruned_rule, pos_pruneset, neg_pruneset)
            if not prune_precision or prune_precision < .50:
                break
            else:
                self.rules.append(pruned_rule)
                pos_remaining.drop(pruned_rule.covers(pos_remaining).index, axis=0, inplace=True)
                neg_remaining.drop(pruned_rule.covers(neg_remaining).index, axis=0, inplace=True)
                #if display: print("Updated ruleset:",self,'\n')

        # Count cond complexity
        self.cond_count = sum([len(rule.conds) for rule in self.rules])

class Rule:
    """ Class implementing conjunctions of Conds. """

    def __init__(self, conds=[]):
        self.conds = conds

    def __str__(self):
        rule_str = str([str(cond) for cond in self.conds]).replace(',','^').replace("'","").replace(' ','')
        return rule_str

    def __repr__(self):
        rule_str = str([str(cond) for cond in self.conds]).replace(', ','^').replace("'","").replace(' ','')
        return f'<Rule object: {rule_str}>'

    def __add__(self, cond):
        if type(cond)==Cond:
            return Rule(self.conds+[cond])
        else:
            raise TypeError(f'{self} + {cond}: Rule objects can only conjoin Cond objects.')

    def covers(self, df):
        """ Returns instances covered by the Rule (i.e. those which are not in contradiction with it). """

        covered = df.copy()
        for cond in self.conds:
            covered = cond.covers(covered)
        return covered

    def num_covered(self, df):
        return len(self.covers(df))

    def covered_feats(self):
        """ Returns list of features covered by the Rule """
        return [cond.feature for cond in self.conds]

    def grow(self, pos_df, neg_df, display=False, sleep=False):
        """ Fit a new rule to add to a ruleset """
        while len(self.covers(neg_df)) > 0: # Stop refining rule if no negative examples remain
            #if display:print(f'neg {num_neg(self.covers(df), class_feat, pos_class)}')
            #if display: print("Update:", self)
            #if sleep: time.sleep(1)

            best_successor_rule = self.best_successor(pos_df, neg_df, display=display)
            #if display:print(f'best_successor_rule {best_successor_rule}')
            if best_successor_rule is not None:
                self.conds = best_successor_rule.conds
            else:
                break

    def pruned(self, pos_pruneset, neg_pruneset, display=False):
        """ Returns a pruned version of the Rule by removing Conds """

        # Currently-best pruned rule and its prune value
        best_rule = copy.deepcopy(self)
        best_v = 0
        # Iterative test rule
        current_rule = copy.deepcopy(self)

        while current_rule.conds:
            v = current_rule.prune_value(pos_pruneset, neg_pruneset)
            if v > best_v:
                best_v = v
                best_rule = copy.deepcopy(current_rule)
            current_rule.conds.pop(-1)
        return best_rule

    #############################################
    ##### Rule::grow/prune helper functions #####
    #############################################

    def best_successor(self, pos_df, neg_df, display=False):
        """ Returns for a Rule its best successor Rule based on information gain metric. """

        best_gain = 0
        best_successor_rule = None
        for successor in self.successors(pos_df, neg_df):
            g = self.gain(successor, pos_df, neg_df)
            if g > best_gain:
                best_gain = g
                best_successor_rule = successor
            #print(f'try successor {g, successor}')

        #if display: print("Best Gain:", str(best_gain))
        return best_successor_rule

    def successors(self, pos_df, neg_df):
        """ Returns a list of all valid successor rules. """

        successor_rules = []
        for feat in pos_df.columns.values:
            for val in set(pos_df[feat].unique()).intersection(set(neg_df[feat].unique())): # Could optimize by calculating this once during fit and passing it along
                if feat not in self.covered_feats(): # Conds already in Rule and Conds that contradict Rule aren't valid successors
                    successor_rules.append(self+Cond(feat, val))
        return successor_rules

    def gain(self, other, pos_df, neg_df):
        """ Returns the information gain from self (rule0) to other (rule1) """

        p0count = self.num_covered(pos_df)
        p1count = other.num_covered(pos_df)
        n0count = self.num_covered(neg_df)
        n1count = other.num_covered(neg_df)
        return p1count * (math.log2((p1count + 1) / (p1count + n1count + 1)) - math.log2((p0count + 1) / (p0count + n0count + 1)))

    def prune_value(self, pos_pruneset, neg_pruneset):
        """ Returns the prune value of a candidate Rule """

        P = len(pos_pruneset)
        N = len(neg_pruneset)
        p = self.num_covered(pos_pruneset)
        n = self.num_covered(neg_pruneset)
        return (p+(N - n)) / (P + N)

class Cond:
    """ Class implementing conditional. """

    def __init__(self, feature, val):
        self.feature = feature
        self.val = val

    def __str__(self):
        return f'{self.feature}={self.val}'

    def __repr__(self):
        return f'<Cond object: {self.feature}={self.val}>'

    def __eq__(self, other):
        return self.feature == other.feature and self.val == other.val

    def covers(self, df):
        """ Returns instances covered by the Cond (i.e. those which are not in contradiction with it). """
        return df[df[self.feature]==self.val]
        #return [(Xi, yi) for Xi, yi in zip(X, y) if Xi[self.feat_index  ]==self.val]
    def num_covered(self, df):
        return len(self.covers(df))

    ###################
    ##### METRICS #####
    ###################

def precision(object, pos_df, neg_df):
    """ Returns precision value of object's classification.
        object: Cond, Rule, or Ruleset
    """

    pos_covered = object.covers(pos_df)
    neg_covered = object.covers(neg_df)
    total_n_covered = len(pos_covered)+len(neg_covered)
    if total_n_covered == 0:
        return None
    else:
        return len(pos_covered) / total_n_covered


def give_reasons(irep_, df):
    """ Experimental """
    def pos_reasons(example):
        print(example)
        assert len(example)==1
        return [rule for rule in irep_.ruleset.rules if len(rule.covers(example))==1]

    return [pos_reasons(df[df.index==i]) for i in df.index]


    ###################
    ##### HELPERS #####
    ###################

def pos_neg_split(df, class_feat, pos_class):
    """ Split df into pos and neg classes. """
    pos_df = pos(df, class_feat, pos_class)
    neg_df = neg(df, class_feat, pos_class)
    return pos_df, neg_df

def df_shuffled_split(df, split_size, seed=None):
    """ Returns tuple of shuffled and split DataFrame.
        split_size: proportion of rows to include in tuple[0]
    """
    df = df.sample(frac=1, random_state=seed).reset_index(drop=True)
    split_at = int(len(df)*split_size)
    return (df[:split_at], df[split_at:])

def pos(df, class_feat, pos_class):
    """ Returns subset of instances that are labeled positive. """
    #""" Returns X,y subset that are labeled positive """
    return df[df[class_feat] == pos_class]
    #return [(Xi, yi) for Xi, yi in zip(X, y) if y==pos_class]

def neg(df, class_feat, pos_class):
    """ Returns subset of instances that are NOT labeled positive. """
    #""" Returns X,y subset that are NOT labeled positive """
    return df[df[class_feat] != pos_class]
    #return [(Xi, yi) for Xi, yi in zip(X, y) if y!=pos_class]

def num_pos(df, class_feat, pos_class):
    """ Returns number of instances that are labeled positive. """
    #""" Returns X,y subset that are labeled positive """
    return len(df[df[class_feat] == pos_class])
    #return len(_pos(X, y, pos_class))

def num_neg(df, class_feat, pos_class):
    """ Returns number of instances that are NOT labeled positive. """
    #""" Returns X,y subset that are NOT labeled positive """
    return len(df[df[class_feat] != pos_class])
    #return len(_neg(X, y, pos_class))