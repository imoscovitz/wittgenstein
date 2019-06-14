""" Base functions for Ruleset classifiers """

# Author: Ilan Moscovitz <ilan.moscovitz@gmail.com>
# License: MIT

from wittgenstein.base import Cond, Rule, Ruleset, bin_df, bin_transform, rnd, _warn
from wittgenstein.catnap import CatNap

import math
from functools import reduce
import operator as op
import copy
import pandas as pd
import numpy as np
from random import shuffle, seed

##########################
##### BASE FUNCTIONS #####
##########################

# Possible optimization: remove data after each added cond?
def grow_rule(pos_df, neg_df, possible_conds, initial_rule=Rule(), max_rule_conds=None, verbosity=0):
    """ Fit a new rule to add to a ruleset """

    rule0 = copy.deepcopy(initial_rule)
    if verbosity>=4:
        print(f'growing rule from initial rule: {rule0}')

    rule1 = copy.deepcopy(rule0)
    while (len(rule0.covers(neg_df)) > 0 and rule1 is not None) and (max_rule_conds is None or len(rule1.conds) < max_rule_conds): # Stop refining rule if no negative examples remain
        rule1 = best_successor(rule0, possible_conds, pos_df, neg_df, verbosity=verbosity)
        if rule1 is not None:
            rule0 = rule1
            if verbosity>=4:
                print(f'negs remaining {len(rule0.covers(neg_df))}')

    if not rule0.isempty():
        if verbosity>=2: print(f'grew rule: {rule0}')
        return rule0
    else:
        #warning_str = f"grew an empty rule: {rule0} over {len(pos_idx)} pos and {len(neg_idx)} neg"
        #_warn(warning_str, RuntimeWarning, filename='base_functions', funcname='grow_rule')
        return rule0

# Possible optimization: remove data after each added cond?
def grow_rule_cn(cn, pos_idx, neg_idx, initial_rule=Rule(), max_rule_conds=None, verbosity=0):
    """ Fit a new rule to add to a ruleset """

    rule0 = copy.deepcopy(initial_rule)
    rule1 = copy.deepcopy(rule0)
    if verbosity>=4:
        print(f'growing rule from initial rule: {rule0}')

    num_neg_covered = len(cn.rule_covers(rule0, subset=neg_idx))
    user_halt = (max_rule_conds is not None and len(rule1.conds) >= max_rule_conds)
    while num_neg_covered > 0: # Stop refining rule if no negative examples remain
        rule1 = best_rule_successor_cn(cn, rule0, pos_idx, neg_idx, verbosity=verbosity)
        if rule1 is None: break
        rule0 = rule1
        num_neg_covered = len(cn.rule_covers(rule0, neg_idx))
        if verbosity>=4:
            print(f'negs remaining: {num_neg_covered}')

    if not rule0.isempty():
        if verbosity>=2: print(f'grew rule: {rule0}')
        return rule0
    else:
        #warning_str = f"grew an empty rule: {rule0} over {len(pos_idx)} pos and {len(neg_idx)} neg"
        #_warn(warning_str, RuntimeWarning, filename='base_functions', funcname='grow_rule_cn')
        return rule0

def prune_rule(rule, prune_metric, pos_pruneset, neg_pruneset, eval_index_on_ruleset=None, verbosity=0):
    """ Returns a pruned version of the Rule by removing Conds

        rule: Rule to prune
        prune_metric: function that returns value to maximize
        pos_pruneset: df of positive class examples
        neg_pruneset: df of non-positive class examples

        eval_index_on_ruleset (optional): tuple(rule_index, ruleset)
            pass the rest of the Rule's Ruleset (excluding the Rule in question),
            in order to prune the rule based on the performance of its entire Ruleset,
            rather than on the rule alone. For use during optimization stage.
    """

    if rule.isempty():
        #warning_str = f"can't prune empty rule: {rule}"
        #_warn(warning_str, RuntimeWarning, filename='base_functions', funcname='prune_rule')
        return rule

    if not eval_index_on_ruleset:

        # Currently-best pruned rule and its prune value
        best_rule = copy.deepcopy(rule)
        best_v = 0

        # Iterative test rule
        current_rule = copy.deepcopy(rule)

        while current_rule.conds:
            v = prune_metric(current_rule, pos_pruneset, neg_pruneset)
            if verbosity>=5: print(f'prune value of {current_rule}: {rnd(v)}')
            if v is None:
                return None
            if v >= best_v:
                best_v = v
                best_rule = copy.deepcopy(current_rule)
            current_rule.conds.pop(-1)

        if verbosity>=2:
            if len(best_rule.conds)!=len(rule.conds):
                print(f'pruned rule: {best_rule}')
            else:
                print(f'pruned rule unchanged')
        return best_rule

    else:
        # Check if index matches rule to prune
        rule_index, ruleset = eval_index_on_ruleset
        if ruleset.rules[rule_index] != rule:
            raise ValueError(f'rule mismatch: {rule} - {ruleset.rules[rule_index]} in {ruleset}')

        current_ruleset = copy.deepcopy(ruleset)
        current_rule = current_ruleset.rules[rule_index]
        best_ruleset = copy.deepcopy(current_ruleset)
        best_v = 0

        # Iteratively prune and test rule over ruleset.
        # This is unfortunately expensive.
        while current_rule.conds:
            v = prune_metric(current_ruleset, pos_pruneset, neg_pruneset)
            if verbosity>=5: print(f'prune value of {current_rule}: {rnd(v)}')
            if v is None:
                return None
            if v >= best_v:
                best_v = v
                best_rule = copy.deepcopy(current_rule)
                best_ruleset = copy.deepcopy(current_ruleset)
            current_rule.conds.pop(-1)
            current_ruleset.rules[rule_index] = current_rule
        return best_rule

def prune_rule_cn(cn, rule, prune_metric_cn, pos_idx, neg_idx, eval_index_on_ruleset=None, verbosity=0):
    """ Returns a pruned version of the Rule by removing Conds

        rule: Rule to prune
        prune_metric: function that returns value to maximize
        pos_pruneset: df of positive class examples
        neg_pruneset: df of non-positive class examples

        eval_index_on_ruleset (optional): tuple(rule_index, ruleset)
            pass the rest of the Rule's Ruleset (excluding the Rule in question),
            in order to prune the rule based on the performance of its entire Ruleset,
            rather than on the rule alone. For use during optimization stage.
    """

    if rule.isempty():
        #warning_str = f"can't prune empty rule: {rule}"
        #_warn(warning_str, RuntimeWarning, filename='base_functions', funcname='prune_rule_cn')
        return rule

    if not eval_index_on_ruleset:

        # Currently-best pruned rule and its prune value
        best_rule = copy.deepcopy(rule)
        best_v = 0

        # Iterative test rule
        current_rule = copy.deepcopy(rule)

        while current_rule.conds:
            v = prune_metric_cn(cn, current_rule, pos_idx, neg_idx)
            if verbosity>=5: print(f'prune value of {current_rule}: {rnd(v)}')
            if v is None:
                return None
            if v >= best_v:
                best_v = v
                best_rule = copy.deepcopy(current_rule)
            current_rule.conds.pop(-1)

        if verbosity>=2:
            if len(best_rule.conds)!=len(rule.conds):
                print(f'pruned rule: {best_rule}')
            else:
                print(f'pruned rule unchanged')
        return best_rule

    # cn is Untouched below here
    else:
        # Check if index matches rule to prune
        rule_index, ruleset = eval_index_on_ruleset
        if ruleset.rules[rule_index] != rule:
            raise ValueError(f'rule mismatch: {rule} - {ruleset.rules[rule_index]} in {ruleset}')

        current_ruleset = copy.deepcopy(ruleset)
        current_rule = current_ruleset.rules[rule_index]
        best_ruleset = copy.deepcopy(current_ruleset)
        best_v = 0

        # Iteratively prune and test rule over ruleset.
        while current_rule.conds:
            v = prune_metric_cn(cn, current_rule, pos_idx, neg_idx)
            if verbosity>=5: print(f'prune value of {current_rule}: {rnd(v)}')
            if v is None:
                return None
            if v >= best_v:
                best_v = v
                best_rule = copy.deepcopy(current_rule)
                best_ruleset = copy.deepcopy(current_ruleset)
            current_rule.conds.pop(-1)
            current_ruleset.rules[rule_index] = current_rule
        return best_rule

def recalibrate_proba(ruleset, Xy_df, class_feat, pos_class, min_samples=10, require_min_samples=True):
    """ Recalibrate a Ruleset's probability estimations using unseen labeled data. May improve .predict_proba generalizability.
        Does not affect the underlying model or which predictions it makes -- only probability estimates. Use params min_samples and require_min_samples to select desired behavior.

        Note1: RunTimeWarning will occur as a reminder when min_samples and require_min_samples params might result in unintended effects.
        Note2: It is possible recalibrating could result in some positive .predict predictions with <0.5 .predict_proba positive probability.

        Xy_df <DataFrame>: labeled data

        min_samples <int> (optional): required minimum number of samples per Rule. (default=10) Regardless of min_samples, at least one sample of the correct class will always be required.
        require_min_samples <bool> (optional): True: halt (with warning) in case min_samples not achieved for all Rules
                                               False: warn, but still replace Rules that have enough samples
    """

    # At least this many samples per rule (or neg) must be of correct class
    required_correct_samples = 1

    # If not using min_samples, set it to 1
    if not min_samples or min_samples < 1:
        min_samples = 1

    # Collect each Rule's pos and neg frequencies in list "rule_class_freqs"
    # Store rules that lack enough samples in list "insufficient_rules"
    df = Xy_df

    rule_class_freqs = [None]*len(ruleset.rules)
    insufficient_rules = []
    for i, rule in enumerate(ruleset.rules):
        npos_pred = num_pos(rule.covers(df), class_feat=class_feat, pos_class=pos_class)
        nneg_pred = num_neg(rule.covers(df), class_feat=class_feat, pos_class=pos_class)
        neg_pos_pred = (nneg_pred, npos_pred)
        rule_class_freqs[i] = neg_pos_pred
        # Rule has insufficient samples if fewer than minsamples or lacks at least one correct sample
        if sum(neg_pos_pred) < min_samples or sum(neg_pos_pred) < 1 or neg_pos_pred[0] < required_correct_samples:
            insufficient_rules.append(rule)

    # Collect class frequencies for negative predictions
    uncovered = df.drop(ruleset.covers(df).index)
    neg_freq = num_neg(uncovered, class_feat=class_feat, pos_class=pos_class)
    tn_fn = (neg_freq, len(uncovered) - neg_freq)

    # Issue warnings if trouble with sample size
    if require_min_samples:
        if insufficient_rules: # WARN if/which rules lack enough samples
            pretty_insufficient_rules = '\n'.join([str(r) for r in insufficient_rules])
            warning_str = f'param min_samples={min_samples}; insufficient number of samples or fewer than {required_correct_samples} correct samples for rules {pretty_insufficient_rules}'
            _warn(warning_str, RuntimeWarning, filename='base_functions', funcname='recalibrate_proba')
        if neg_freq < min_samples or tn_fn[1]<1: # WARN if neg lacks enough samples
            warning_str = f'param min_samples={min_samples}; insufficient number of negatively labled samples'
            _warn(warning_str, RuntimeWarning, filename='base_functions', funcname='recalibrate_proba')
        if insufficient_rules or sum(tn_fn) < min_samples:
            if require_min_samples: # WARN if require_min_samples -> halting recalibration
                warning_str = f'Recalibrating halted. to recalibrate, try using more samples, lowering min_samples, or set require_min_samples to False'
                _warn(warning_str, RuntimeWarning, filename='base_functions', funcname='recalibrate_proba')
                return
            else: # GO AHEAD EVEN THOUGH NOT ENOUGH SAMPLES
                pass
                #warning_str = f'Because require_min_samples=False, recalibrating probabilities for any rules with enough samples min_samples>={min_samples} that have at least {required_correct_samples} correct samples even though not all rules have enough samples. Probabilities for any rules that lack enough samples will be retained.'
                #_warn(warning_str, RuntimeWarning, filename='base_functions', funcname='recalibrate_proba')

    # Assign collected frequencies to Rules
    for rule, freqs in zip(ruleset.rules, rule_class_freqs):
        if sum(freqs) >= min_samples and freqs[0]>=required_correct_samples:
            rule.class_freqs = freqs
        else:
            pass # ignore Rules that don't have enough samples

    # Assign Ruleset's uncovered frequencies
    if not hasattr(ruleset,'uncovered_class_freqs') or (neg_freq >= min_samples and tn_fn[1]>=required_correct_samples):
        ruleset.uncovered_class_freqs = tn_fn

    # Warn if no pos or no neg samples
    nopos = (sum([freqs[0] for freqs in rule_class_freqs])+ruleset.uncovered_class_freqs[0])==0
    noneg = (sum([freqs[1] for freqs in rule_class_freqs])+ruleset.uncovered_class_freqs[1])==0
    if nopos:
        warning_str=f"No positive samples"
        _warn(warning_str, RuntimeWarning, filename='base_functions', funcname='recalibrate_proba')
    if noneg:
        warning_str=f"No negative samples."
        _warn(warning_str, RuntimeWarning, filename='base_functions', funcname='recalibrate_proba')

###################
##### METRICS #####
###################

def possible_conds(df):
    """ Create and return a list of all possible conds from a categorical dataframe. """

    possible_conds = []
    for feat in pos_df.columns.values:
        for val in set(pos_df[feat].unique()):
            self.possible_conds.append(Cond(feat, val))
    return possible_conds

    ###################
    ##### METRICS #####
    ###################

def gain(before, after, pos_df, neg_df):
    """ Returns the information gain from self to other """
    p0count = before.num_covered(pos_df)
    p1count = after.num_covered(pos_df)
    n0count = before.num_covered(neg_df)
    n1count = after.num_covered(neg_df)
    return p1count * (math.log2((p1count + 1) / (p1count + n1count + 1)) - math.log2((p0count + 1) / (p0count + n0count + 1)))

def gain_cn(cn, cond_step, rule_covers_pos_idx, rule_covers_neg_idx):
    """ Returns the information gain from self to other """
    #print('gain_cn', len(rule_covers_pos_idx), len(rule_covers_neg_idx))

    p0count = len(rule_covers_pos_idx)
    p1count = len(cn.cond_covers(cond_step, subset=rule_covers_pos_idx))
    n0count = len(rule_covers_neg_idx)
    n1count = len(cn.cond_covers(cond_step, subset=rule_covers_neg_idx))
    g = p1count * (math.log2((p1count + 1) / (p1count + n1count + 1)) - math.log2((p0count + 1) / (p0count + n0count + 1)))
    #if p1count > p0count or n1count > n0count:
    #    print(cond_step, p0count, p1count, n0count, n1count, rnd(g))
    return g

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

def rule_precision_cn(cn, rule, pos_idx, neg_idx):
    pos_covered = cn.rule_covers(rule, pos_idx)
    neg_covered = cn.rule_covers(rule, neg_idx)
    total_n_covered = len(pos_covered)+len(neg_covered)
    if total_n_covered == 0:
        return None
    else:
        return len(pos_covered) / total_n_covered

def score_accuracy(predictions, actuals):
    """ For evaluating trained model on test set.

        predictions: <iterable<bool>> True for predicted positive class, False otherwise
        actuals:     <iterable<bool>> True for actual positive class, False otherwise
    """
    t = [pr for pr,act in zip(predictions, actuals) if pr==act]
    n = predictions
    return len(t)/len(n)

def _accuracy(object, pos_pruneset, neg_pruneset):
    """ Returns accuracy value of object's classification.
        object: Cond, Rule, or Ruleset
    """
    P = len(pos_pruneset)
    N = len(neg_pruneset)
    if P + N == 0:
        return None

    tp = len(object.covers(pos_pruneset))
    tn = N - len(object.covers(neg_pruneset))
    return (tp + tn) / (P + N)

def _rule_accuracy_cn(cn, rule, pos_pruneset_idx, neg_pruneset_idx):
    """ Returns accuracy value of object's classification.
        object: Cond, Rule, or Ruleset
    """
    P = len(pos_pruneset_idx)
    N = len(neg_pruneset_idx)
    if P + N == 0:
        return None

    tp = len(cn.rule_covers(rule, pos_pruneset_idx))
    tn = N - len(cn.rule_covers(rule, neg_pruneset_idx))
    return (tp + tn) / (P + N)

def best_successor(rule, possible_conds, pos_df, neg_df, verbosity=0):
    """ Returns for a Rule its best successor Rule according to FOIL information gain metric.

        eval_on_ruleset: option to evaluate gain with extra disjoined rules (for use with RIPPER's post-optimization)
    """

    best_gain = 0
    best_successor_rule = None

    for successor in rule.successors(possible_conds, pos_df, neg_df):
        g = gain(rule, successor, pos_df, neg_df)
        if g > best_gain:
            best_gain = g
            best_successor_rule = successor
    if verbosity>=5: print(f'gain {rnd(best_gain)} {best_successor_rule}')
    return best_successor_rule

# Can possibly further optimize by not rechecking rule each time but just cond
def best_rule_successor_cn(cn, rule, pos_idx, neg_idx, verbosity=0):
    best_cond = None
    best_gain = float('-inf')

    rule_covers_pos_idx = cn.rule_covers(rule, pos_idx)
    rule_covers_neg_idx = cn.rule_covers(rule, neg_idx)

    for cond_action_step in cn.conds:
        g = gain_cn(cn, cond_action_step, rule_covers_pos_idx, rule_covers_neg_idx)
        if g > best_gain:
            best_gain = g
            best_cond = cond_action_step
    if verbosity>=5: print(f'gain {rnd(best_gain)} {best_cond}')
    return Rule(rule.conds+[best_cond]) if best_gain > 0 else None

#def rule_successors_cn(cn, rule):
#    successor_rules = []
#    for c in cn.conds:
#        successor_rules.append(Rule(rule.conds+[c]))
#    return successor_rules

    ###################
    ##### HELPERS #####
    ###################

def pos_neg_split(df, class_feat, pos_class):
    """ Split df into pos and neg classes. """
    pos_df = pos(df, class_feat, pos_class)
    neg_df = neg(df, class_feat, pos_class)
    return pos_df, neg_df

def df_shuffled_split(df, split_size, random_state=None):
    """ Returns tuple of shuffled and split DataFrame.
        split_size: proportion of rows to include in tuple[0]
    """
    df = df.sample(frac=1, random_state=random_state).reset_index(drop=True)
    split_at = int(len(df)*split_size)
    return (df[:split_at], df[split_at:])

# Make sure seed is behaving correctly
def set_shuffled_split(set_to_split, split_size, random_state=None):
    list_to_split = list(set_to_split)
    seed(random_state)
    shuffle(list_to_split)
    split_at = int(len(list_to_split)*split_size)
    return (set(list_to_split[:split_at]), set(list_to_split[split_at:]))

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
    return num//den

def argmin(list_):
    """ Returns index of minimum value. """
    lowest_val = list_[0]
    lowest_i = 0
    for i, val in enumerate(list_):
        if val < lowest_val:
            lowest_val = val
            lowest_i = i
    return lowest_i

def i_replaced(list_, i, value):
    """ Returns a new list with element i replaced by value.
        Pass None to value to return list with element i removed.
    """
    if value is not None:
        return list_[:i]+[value]+list_[i+1:]
    else:
        return list_[:i]+list_[i+1:]

def rm_covered(object, pos_df, neg_df):
    """ Return pos and neg dfs of examples that are not covered by object """
    return (pos_df.drop(object.covers(pos_df).index, axis=0, inplace=False),\
            neg_df.drop(object.covers(neg_df).index, axis=0, inplace=False))

def rm_rule_covers_cn(cn, rule, pos_idx, neg_idx):
    return (pos_idx - cn.rule_covers(rule, pos_idx), \
            neg_idx - cn.rule_covers(rule, neg_idx))

def preprocess_training_data(preprocess_params):

    # Get params
    X_or_Xy = preprocess_params['X_or_Xy']
    y = preprocess_params['y']
    class_feat = preprocess_params['class_feat']
    pos_class = preprocess_params['pos_class']
    user_requested_feature_names = preprocess_params['user_requested_feature_names']
    n_discretize_bins = preprocess_params['n_discretize_bins']
    verbosity = preprocess_params['verbosity']

    # STEP 0: ERROR CHECKING
    _check_valid_input_data(X_or_Xy, y, class_feat)

    # STEP 1: DETERMINE class_feat
    class_feat = _get_class_feat_name(class_feat, y)

    # STEP 2: BUILD DataFrame from X_or_Xy (and if necessary y). DO NOT MODIFY ORIGINAL DATA
    df = _convert_to_df(X_or_Xy, y, class_feat)

    # STEP 3: DEFINE pos_class
    pos_class = get_pos_class(df, class_feat, pos_class)

    # STEP 4: GET_OR_SET FEATURE NAMES
    df = _get_or_set_feature_names(df, class_feat, user_requested_feature_names=user_requested_feature_names)

    # STEP 5: INFER FEATURE DTYPES
    df = df.infer_objects()

    # STEP 6: BIN FIT_TRANSFORM, TRANSFORM, or NOTHING
    df, bin_transformer_ = _try_bin_fit_or_fittransform_(df, n_discretize_bins=n_discretize_bins, ignore_feats=[class_feat], verbosity=verbosity)

    # ALL DONE
    return df, class_feat, pos_class, bin_transformer_

def preprocess_prediction_data(preprocess_params):

    X = preprocess_params['X']
    class_feat = preprocess_params['class_feat']
    pos_class = preprocess_params['pos_class']
    user_requested_feature_names = preprocess_params['user_requested_feature_names']
    selected_features_ = preprocess_params['selected_features_']
    trainset_features_ = preprocess_params['trainset_features_']
    bin_transformer_ = preprocess_params['bin_transformer_']
    verbosity = preprocess_params['verbosity']

    # STEP 0: ERROR CHECKING
    _check_valid_input_data(X, y=None, class_feat=class_feat, requires_label=False)

    # STEP 1: BUILD DataFrame from X_or_Xy (and if necessary y). DO NOT MODIFY ORIGINAL DATA
    df = _convert_to_df(X, y=None, class_feat=class_feat, requires_label=False)

    # STEP 2: CHECK THAT ALL MODEL FEAETURES ARE PRESENT IN X
    df = _get_or_set_feature_names(df, class_feat, user_requested_feature_names=user_requested_feature_names, selected_features_=selected_features_, trainset_features_=trainset_features_)

    # STEP 3: ASK PANDAS TO CORRECT DTYPES
    df = df.infer_objects()

    # STEP 4: BIN, IF NECESSARY
    df, _ = _try_bin_fit_or_fittransform_(df, bin_transformer_=bin_transformer_, ignore_feats=[class_feat], verbosity=verbosity)

    return df

def _preprocess_recalibrate_proba_data(preprocess_params):

    # Get params
    X_or_Xy = preprocess_params['X_or_Xy']
    y = preprocess_params['y']
    class_feat = preprocess_params['class_feat']
    pos_class = preprocess_params['pos_class']
    user_requested_feature_names = preprocess_params['user_requested_feature_names']
    bin_transformer_ = preprocess_params['bin_transformer_']
    verbosity = preprocess_params['verbosity']

    # STEP 0: ERROR CHECKING
    _check_valid_input_data(X_or_Xy, y, class_feat)

    # STEP 2: BUILD DataFrame from X_or_Xy (and if necessary y). DO NOT MODIFY ORIGINAL DATA
    df = _convert_to_df(X_or_Xy, y, class_feat)

    # STEP 4: GET_OR_SET FEATURE NAMES
    df = _get_or_set_feature_names(df, class_feat, user_requested_feature_names=user_requested_feature_names)

    # STEP 5: INFER FEATURE DTYPES
    df = df.infer_objects()

    # STEP 6: BIN FIT_TRANSFORM, TRANSFORM, or NOTHING
    df, bin_transformer_ = _try_bin_fit_or_fittransform_(df, bin_transformer_=bin_transformer_, ignore_feats=[class_feat], verbosity=verbosity)

    # ALL DONE
    return df

def _preprocess_y_score_data(y):
    """ Return python iterable of y values """

    def raise_wrong_ndim():
        raise IndexError(f'y input data has wrong number dimensions. It should have 1.')

    # If it's pandas or np...
    if hasattr(y, 'ndim'):
        if y.ndim==1:
            return y.tolist()
        else:
            raise_wrong_ndim()

    # Otherwise try for python iterable
    if hasattr(y, '__getitem__'):
        # No super clean way to check for 1D, but this should be pretty good
        if any([hasattr(item, '__getitem__') and type(item) is not str for item in y]):
            raise_wrong_ndim()
        else:
            return y

    # Otherwise, no idea what's going on
    else:
        raise TypeError(f'Could not identify valid type for y input data: {type(y)}. Recommended types are 1D python iterable, pandas Series, or 1D numpy array.')

def _check_valid_input_data(X_or_Xy, y=None, class_feat=None, requires_label=True):

    # Make sure there is data
    if not _check_some_datasets_not_empty([X_or_Xy]):
        raise ValueError('No data provided!')

    # If labels aren't needed, we're done
    if not requires_label:
        return

    # Ensure class feature is provided or can be inferred
    if (y is None) and (class_feat is None):
        raise ValueError('y or class_feat param is required')

    # Ensure target data exists if class feat is provided
    if (y is None):
        # X_or_Xy is a df: does target column exist?
        if hasattr(X_or_Xy, 'columns') and (class_feat not in X_or_Xy.columns):
            raise IndexError(f'Data does not contain class feature {class_feat}')
        # X_or_Xy is an array or iterable: does target index exist?
        elif not hasattr(X_or_Xy, 'columns'):
            try:
                if len(X_or_Xy[0]) > class_feat:
                    raise IndexError(f'Data does not contain class index {class_feat}')
            except:
                raise IndexError(f'Data does not contain class index {class_feat}')

    # Ensure no class feature name mismatch
    if y is not None and class_feat is not None \
            and hasattr(y, 'name') \
            and y.name != class_feat:
        raise NameError(f'Feature name mismatch between params y {y.name} and class_feat {class_feat}. Besides, you only need to provide one of them.')

def _get_class_feat_name(class_feat, y):

    if class_feat is not None:
        return class_feat

    if y is not None and hasattr(y, 'name'):
        # If y is a pandas Series, try to get its name
        class_feat = y.name
    else:
        # Create a name for it
        class_feat = 'Class'

    return class_feat

def _convert_to_df(X_or_Xy, y, class_feat, requires_label=True):
    """ Create a labeled trainset from input data. If original data was of pandas type, make deepcopy. """

    # Create df from X_or_Xy
    if isinstance(X_or_Xy, pd.DataFrame):
        df = X_or_Xy.copy()
    else:
        df = pd.DataFrame(X_or_Xy)

    # If necessary, merge y into df
    if y is not None:
        # If y is pd or np type, add it
        try:
            df = df.set_index(y.index)
            df[class_feat]=y.copy()
        # If that doesn't work, it's likely a python iterable
        except:
            df[class_feat]=y
    return df

def get_pos_class(df, class_feat, pos_class):

    # Pos class already known
    if pos_class:
        return pos_class

    # Check if pos class can be inferred as True or 1
    y_unique = df[class_feat].unique()
    if len(y_unique) == 2:
        class_labels = sorted(y_unique.tolist())
        if class_labels[0] is 0 and class_labels[1] is 1: # 'is' operator is essential to distinguish 1,True, etc.
            return 1
        elif class_labels[0] is False and class_labels[1] is True:
            return True

    # Can't infer classes
    raise NameError(f"Couldn't infer name of positive target class from class feature: {class_feat}. Try using parameter pos_class to specify which class label should be treated as positive, or renaming your classes as booleans or 0,1.")

def _try_bin_fit_or_fittransform_(df, ignore_feats=[], n_discretize_bins=None, bin_transformer_=None, verbosity=0):

    # Binning has already been fit
    if bin_transformer_:
        bin_transform(df, bin_transformer_)
        return df, bin_transformer_

    # Binning disabled
    elif not n_discretize_bins:
        return df, bin_transformer_

    # Binning enabled, and binner needs to be fit
    else:
        df, bin_transformer_ = bin_df(df, n_discretize_bins=n_discretize_bins, ignore_feats=ignore_feats, verbosity=verbosity)
        return df, bin_transformer_

def _get_or_set_feature_names(df, class_feat, user_requested_feature_names=None, selected_features_=None, trainset_features_=None):
    """ Assign feature names in order of these preferences:
        1) user_requested_feature_names
        2) fine if no missing features or no model has been trained
        3) trainset_features_ (if they exist)
    """

    # 1) User wants to rename features. Raise error if fail
    if user_requested_feature_names:
        original_feat_names = [col for col in df.columns if col!=class_feat]
        df = _try_rename_features(df, class_feat, user_requested_feature_names) # Returns None if can't rename
        if df is not None:
            return df
        else:
            raise IndexError(f'The number of requested features names ({len(user_requested_feature_names)}) does not match the number of non-class features in dataset: ({len(original_feat_names)}).\nParam feature_names: {truncstr(user_requested_feature_names,10)}\nTraining set features names: {truncstr(original_feat_names,10)}')

    # 2) If no model trained, use defaults
    if selected_features_ is None:
        return df

    # 3) Are all model features present? If so, we're good.
    missing_feats = _get_missing_selected_features(df, selected_features_)
    if not missing_feats:
        return df

    # 4) Some feats are missing: try to apply training data. Warn on success; raise error on fail
    if missing_feats:
        missing_feats_str = f'Some features selected by model are missing: {missing_feats}.\nEither ensure prediction dataset includes all Ruleset-selected features with same names as training set, or use param feature_names to specify the names of dataset features.\nAttempted feature names:{truncstr(df.columns.tolist())}\n'
        df = _try_rename_features(df, class_feat, trainset_features_) # returns None if feature length mismatch
        if df is not None:
            _warn(missing_feats_str, RuntimeWarning, filename='base_functions', funcname='_get_or_set_feature_names')
            trainset_feats_str = f"Mismatch between Ruleset training feature names and prediction feature names:{truncstr(df.columns.tolist())}\nPredicting under on the assumption that the features provided for prediction are the same as those the model was trained on, and that they are being provided in the same order.\nYou probably want to ensure your prediction dataset includes the same feature names as your Ruleset selected features, or alternately, to use param feature_names to specify wha prediction feature names should be used."
            _warn(trainset_feats_str, RuntimeWarning, filename='base_functions', funcname='_get_or_set_feature_names')
            return df
        else:
            raise NameError(missing_feats_str)

def _try_rename_features(df, class_feat, feature_names):
    """ Rename df columns according to user request. """
    # Rename if same number of features
    df_columns = [col for col in df.columns.tolist() if col!=class_feat]
    if len(df_columns)==len(feature_names):
        col_replacements_dict = {old:new for old,new in zip(df_columns, feature_names)}
        df = df.rename(columns=col_replacements_dict)
        return df
    # Wrong number of feature names
    else:
        return None

def _get_missing_selected_features(df, model_selected_features):
    df_feats = df.columns.tolist()
    return [f for f in model_selected_features if f not in df_feats]

def trainset_classfeat_posclass(df, y=None, class_feat=None, pos_class=None):
    """ Process params into trainset, class feature name, and pos class, for use in .fit methods. """

    if not _check_some_datasets_not_empty([df]) and not y:
        warnings_str="Can't train on an empty dataset!"
        _warn(warnings_str, RuntimeWarning, filename='base_functions', funcname='grow_rule_cn')

    # Ensure class feature is provided or inferable
    if (y is None) and (class_feat is None):
        raise ValueError('y or class_feat argument is required')

    if (y is None) and (class_feat not in df.columns):
        raise ValueError(f'Training data does not contain class feature: {class_feat}')

    # Ensure no class feature name mismatch
    if y is not None and class_feat is not None \
            and hasattr(y, 'name') \
            and y.name != class_feat:
        raise ValueError(f'Value mismatch between params y {y.name} and class_feat {class_feat}. Besides, you only need to provide one of them.')

    # Set class feature name
    if class_feat is not None:
        # (IOW, pass)
        class_feat = class_feat
    elif y is not None and hasattr(y, 'name'):
        # If y is a pandas Series, try to get its name
        class_feat = y.name
    else:
        # Create a name for it
        class_feat = 'Class'

    # If necessary, merge y into df
    if y is not None:
        df[class_feat] = y

    # If provided, define positive class name. Otherwise, assign one.
    if pos_class is not None:
        pos_class = pos_class
    else:
        pos_class = df.iloc[0][class_feat]

    return (df, class_feat, pos_class)

def aslist(data):
    if type(data)==pd.core.frame.DataFrame or type(data)==np.ndarray:
        return data.tolist()
    else:
        return data

def truncstr(iterable, limit=5, direction='left'):
    """ Return Ruleset string representation limited to a specified number of rules.

        limit: how many rules to return
        direction: which part to return. (valid options: 'left', 'right')
    """
    if len(iterable)>=limit:
        if direction=='left':
            return iterable[:limit].__str__()+'...'
        elif direction=='right':
            return '...'+iterable[-limit:].__str__()
        else:
            raise ValueError('direction param must be "left" or "right"')
    else:
        return self.__str__()

def infer_columns(df, expected_columns):
    if df.columns.tolist()==expected_columns:
        return expected_columns
    problem_str = f'Did not receive expected feature names. Received {truncstr(df.columns.tolist())}. Expected {expected_columns}'
    if len(df.columns)==len(expected_columns):
        _warn(problem_str, RuntimeWarning, filename='base_functions', funcname='infer_columns')
        return pd.DataFrame(df, columns=expected_columns)
    else:
        raise IndexError(problem_str)
        return df.columns.tolist()

    #####################
    ##### CHECKERS ######
    #####################

def _check_some_datasets_not_empty(datasets):
    return any([len(dataset)>0 for dataset in datasets])

def stop_early(ruleset, max_rules, max_total_conds):
    return (max_rules is not None and len(ruleset.rules) >= max_rules) or (max_total_conds is not None and ruleset.count_conds() >= max_total_conds)

##### RANDO ######

def _check_all_model_features_in_X(df, trainset_features, model_selected_features):

    df_feats = df.columns.tolist()
    missing_feats = [f for f in model_selected_features if f not in df_feats]

    # User supplied same number of features in prediction set as training set, but wrong names.
    # Assume they are same but give stern warning
    if missing_feats:
        if len(trainset_features)==len(df_feats):
            warnings_str=f"Mismatch between Ruleset training feature names and prediction feature names.\nPredicting under on the assumption that the features provided for prediction are the same as those the model was trained on, and that they are being provided in the same order.\nYou probably want to ensure 1) you include all selected features, correctly named; 2) your prediction dataset feature names are identical with your training data; or 3) use .predict param 'feature_names' to specify what prediction feature names should be.\nCurrent prediction feature names:{truncstr(df.columns.tolist())}"
            _warn(warnings_str, RuntimeWarning, filename='base_functions', funcname='_check_all_model_features_in_X')
            print([type(f) for f in missing_feats])
            print([type(f) for f in df_feats])
            df.columns = trainset_features
            return df

    # Didn't provide all model selected features.
        else:
            raise IndexError(f'The features selected by Ruleset model need to be present in prediction dataset.\nMissing features: {missing_feats}.\nEither ensure prediction dataset includes all Ruleset-selected features with same names as training set, or use .predict param feature_names to specify the names of prediction dataset features.\n')
            return
