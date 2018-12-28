""" Base classes and functions for ruleset classifiers """

import math
import operator as op
from functools import reduce
import copy

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

    def num_covered(self, df):
        return len(self.covers(df))

    def add(self, rule):
        self.rules.append(rule)

    def count_conds(self):
        return sum([len(r.conds) for r in self.rules])

class Rule:
    """ Class implementing conjunctions of Conds """

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

    def isempty(self):
        return len(self.conds)==0

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

    #############################################
    ##### Rule::grow/prune helper functions #####
    #############################################

    def best_successor(self, pos_df, neg_df, eval_with=None):
        """ Returns for a Rule its best successor Rule based on FOIL information gain metric.
            eval_with: option to evaluate gain with extra disjoined rules (for use with RIPPER's post-optimization)
        """
        best_gain = 0
        best_successor_rule = None
        for successor in self.successors(pos_df, neg_df):
            g = gain(self, successor, pos_df, neg_df)
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

##########################
##### BASE FUNCTIONS #####
##########################

def grow_rule(pos_df, neg_df):
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

def prune_rule(rule, prune_metric, pos_pruneset, neg_pruneset):
    """ Returns a pruned version of the Rule by removing Conds """

    # Currently-best pruned rule and its prune value
    best_rule = copy.deepcopy(rule)
    best_v = 0

    # Iterative test rule
    current_rule = copy.deepcopy(rule)

    while current_rule.conds:
        v = prune_metric(current_rule, pos_pruneset, neg_pruneset)
        if v is None:
            return None
        if v > best_v:
            best_v = v
            best_rule = copy.deepcopy(current_rule)
        current_rule.conds.pop(-1)
    return best_rule


    ###################
    ##### METRICS #####
    ###################

def gain(self, other, pos_df, neg_df):
    """ Returns the information gain from self to other """

    p0count = self.num_covered(pos_df)
    p1count = other.num_covered(pos_df)
    n0count = self.num_covered(neg_df)
    n1count = other.num_covered(neg_df)
    return p1count * (math.log2((p1count + 1) / (p1count + n1count + 1)) - math.log2((p0count + 1) / (p0count + n0count + 1)))

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

def nCr(n, r):
    """ Returns number of combinations C(n, r) """
    def product(numbers):
        return reduce(op.mul, numbers, 1)

    num = product(range(n, n-r, -1))
    den = product(range(1, r+1))
    return num/den
