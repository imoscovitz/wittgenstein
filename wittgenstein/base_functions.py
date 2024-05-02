# Author: Ilan Moscovitz <ilan.moscovitz@gmail.com>
# License: MIT

import copy
from functools import reduce
import math
import numpy as np
import operator as op

import pandas as pd
from random import shuffle, seed

from wittgenstein.base import Cond, Rule, Ruleset
from wittgenstein.check import (
    _warn,
    _check_model_features_present,
)
from wittgenstein.utils import rnd

##########################
##### BASE FUNCTIONS #####
##########################


def grow_rule(
    pos_df,
    neg_df,
    possible_conds,
    initial_rule=Rule(),
    min_rule_samples=None,
    max_rule_conds=None,
    verbosity=0,
):
    """Fit a new rule to add to a ruleset"""

    rule0 = copy.deepcopy(initial_rule)
    if verbosity >= 4:
        print(f"growing rule from initial rule: {rule0}")

    rule1 = copy.deepcopy(rule0)
    num_neg_covered = len(rule0.covers(neg_df))
    while num_neg_covered > 0 and rule1 is not None: # Stop refining rule if no negative examples remain
        user_halt = (max_rule_conds is not None and len(rule1.conds) >= max_rule_conds) or \
            (min_rule_samples is not None and num_neg_covered + rule0.num_covered(pos_df) < min_rule_samples)
        if user_halt:
            if min_rule_samples is not None and rule0.conds:
                rule0.conds.pop(-1)
            break

        rule1 = best_successor(
            rule0, possible_conds, pos_df, neg_df, verbosity=verbosity
        )
        if rule1 is not None:
            rule0 = rule1
            if verbosity >= 4:
                print(f"negs remaining {len(rule0.covers(neg_df))}")
        num_neg_covered = len(rule0.covers(neg_df))

    if not rule0.isempty():
        if verbosity >= 2:
            print(f"grew rule: {rule0}")
        if verbosity >= 4:
            print(f"covered {len(rule0.covers(neg_df)) + len(rule0.covers(pos_df))} samples")
        return rule0
    else:
        # warning_str = f"grew an empty rule: {rule0} over {len(pos_idx)} pos and {len(neg_idx)} neg"
        # _warn(warning_str, RuntimeWarning, filename='base_functions', funcname='grow_rule')
        return rule0

def grow_rule_cn(
    cn,
    pos_idx,
    neg_idx,
    initial_rule=Rule(),
    max_rule_conds=None,
    min_rule_samples=None,
    verbosity=0
):
    """Fit a new rule to add to a ruleset. (Optimized version.)"""

    rule0 = copy.deepcopy(initial_rule)
    rule1 = copy.deepcopy(rule0)
    if verbosity >= 4:
        print(f"growing rule from initial rule: {rule0}")

    num_neg_covered = len(cn.rule_covers(rule0, subset=neg_idx))
    while num_neg_covered > 0:  # Stop refining rule if no negative examples remain
        user_halt = (max_rule_conds is not None and len(rule1.conds) >= max_rule_conds) or \
            (min_rule_samples is not None and (num_neg_covered + len(cn.rule_covers(rule0, subset=pos_idx))) < min_rule_samples)
        if user_halt:
            if min_rule_samples is not None and rule0.conds:
                rule0.conds.pop(-1)
            break

        rule1 = best_rule_successor_cn(cn, rule0, pos_idx, neg_idx, verbosity=verbosity)
        if rule1 is None:
            break
        rule0 = rule1
        num_neg_covered = len(cn.rule_covers(rule0, neg_idx))
        if verbosity >= 4:
            print(f"negs remaining: {num_neg_covered}")

    if not rule0.isempty():
        if verbosity >= 2:
            print(f"grew rule: {rule0}")
        if verbosity >= 4:
            print(f"covered {len(cn.rule_covers(rule0, subset=neg_idx)) + len(cn.rule_covers(rule0, subset=pos_idx))} samples")
        return rule0
    else:
        # warning_str = f"grew an empty rule: {rule0} over {len(pos_idx)} pos and {len(neg_idx)} neg"
        # _warn(warning_str, RuntimeWarning, filename='base_functions', funcname='grow_rule_cn')
        return rule0


def prune_rule(
    rule,
    prune_metric,
    pos_pruneset,
    neg_pruneset,
    eval_index_on_ruleset=None,
    verbosity=0,
):
    """Return a pruned version of the Rule by removing Conds.

    rule : Rule
        Rule to prune.
    prune_metric : function
        Function that returns a value to maximize.
    pos_pruneset : DataFrame
        Positive class examples.
    neg_pruneset : DataFrame
        Negative class examples.

    eval_index_on_ruleset : tuple(rule_index, Ruleset), default=None
        Pass the rest of the Rule's Ruleset (excluding the Rule in question),
        in order to prune the rule based on the performance of its entire Ruleset,
        rather than on the rule alone. Used during optimization stage of RIPPER.
    verbosity : int (0-5), default=0
        Output verbosity.
    """

    if rule.isempty():
        # warning_str = f"can't prune empty rule: {rule}"
        # _warn(warning_str, RuntimeWarning, filename='base_functions', funcname='prune_rule')
        return rule

    if not eval_index_on_ruleset:

        # Currently-best pruned rule and its prune value
        best_rule = copy.deepcopy(rule)
        best_v = 0

        # Iterative test rule
        current_rule = copy.deepcopy(rule)

        while current_rule.conds:
            v = prune_metric(current_rule, pos_pruneset, neg_pruneset)
            if verbosity >= 5:
                print(f"prune value of {current_rule}: {rnd(v)}")
            if v is None:
                return None
            if v >= best_v:
                best_v = v
                best_rule = copy.deepcopy(current_rule)
            current_rule.conds.pop(-1)

        if verbosity >= 2:
            if len(best_rule.conds) != len(rule.conds):
                print(f"pruned rule: {best_rule}")
            else:
                print(f"pruned rule unchanged")
        return best_rule

    else:
        # Check if index matches rule to prune
        rule_index, ruleset = eval_index_on_ruleset
        if ruleset.rules[rule_index] != rule:
            raise ValueError(
                f"rule mismatch: {rule} - {ruleset.rules[rule_index]} in {ruleset}"
            )

        current_ruleset = copy.deepcopy(ruleset)
        current_rule = current_ruleset.rules[rule_index]
        best_ruleset = copy.deepcopy(current_ruleset)
        best_v = 0

        # Iteratively prune and test rule over ruleset.
        # This is unfortunately expensive.
        while current_rule.conds:
            v = prune_metric(current_ruleset, pos_pruneset, neg_pruneset)
            if verbosity >= 5:
                print(f"prune value of {current_rule}: {rnd(v)}")
            if v is None:
                return None
            if v >= best_v:
                best_v = v
                best_rule = copy.deepcopy(current_rule)
                best_ruleset = copy.deepcopy(current_ruleset)
            current_rule.conds.pop(-1)
            current_ruleset.rules[rule_index] = current_rule
        return best_rule


def prune_rule_cn(
    cn, rule, prune_metric_cn, pos_idx, neg_idx, eval_index_on_ruleset=None, verbosity=0
):
    """Return a pruned version of the Rule by removing Conds. (Optimized version.)

    rule : Rule
        Rule to prune.
    prune_metric : function
        Function that returns a value to maximize.
    pos_pruneset : DataFrame
        Positive class examples.
    neg_pruneset : DataFrame
        Negative class examples.

    eval_index_on_ruleset : tuple(rule_index, Ruleset), default=None
        Pass the rest of the Rule's Ruleset (excluding the Rule in question),
        in order to prune the rule based on the performance of its entire Ruleset,
        rather than on the rule alone. Used during optimization stage of RIPPER.
    verbosity : int (0-5), default=0
        Output verbosity.
    """

    if rule.isempty():
        # warning_str = f"can't prune empty rule: {rule}"
        # _warn(warning_str, RuntimeWarning, filename='base_functions', funcname='prune_rule_cn')
        return rule

    if not eval_index_on_ruleset:

        # Currently-best pruned rule and its prune value
        best_rule = copy.deepcopy(rule)
        best_v = 0

        # Iterative test rule
        current_rule = copy.deepcopy(rule)

        while current_rule.conds:
            v = prune_metric_cn(cn, current_rule, pos_idx, neg_idx)
            if verbosity >= 5:
                print(f"prune value of {current_rule}: {rnd(v)}")
            if v is None:
                return None
            if v >= best_v:
                best_v = v
                best_rule = copy.deepcopy(current_rule)
            current_rule.conds.pop(-1)

        if verbosity >= 2:
            if len(best_rule.conds) != len(rule.conds):
                print(f"pruned rule: {best_rule}")
            else:
                print(f"pruned rule unchanged")
        return best_rule

    # cn is Untouched below here
    else:
        # Check if index matches rule to prune
        rule_index, ruleset = eval_index_on_ruleset
        if ruleset.rules[rule_index] != rule:
            raise ValueError(
                f"rule mismatch: {rule} - {ruleset.rules[rule_index]} in {ruleset}"
            )

        current_ruleset = copy.deepcopy(ruleset)
        current_rule = current_ruleset.rules[rule_index]
        best_ruleset = copy.deepcopy(current_ruleset)
        best_v = 0

        # Iteratively prune and test rule over ruleset.
        while current_rule.conds:
            v = prune_metric_cn(cn, current_rule, pos_idx, neg_idx)
            if verbosity >= 5:
                print(f"prune value of {current_rule}: {rnd(v)}")
            if v is None:
                return None
            if v >= best_v:
                best_v = v
                best_rule = copy.deepcopy(current_rule)
                best_ruleset = copy.deepcopy(current_ruleset)
            current_rule.conds.pop(-1)
            current_ruleset.rules[rule_index] = current_rule
        return best_rule


def recalibrate_proba(
    ruleset, Xy_df, class_feat, pos_class, min_samples=1, require_min_samples=False, alpha=1.0, verbosity=0,
):
    """Recalibrate a Ruleset's probability estimations using unseen labeled data without changing the underlying model. May improve .predict_proba generalizability.
    Does not affect the underlying model or which predictions it makes -- only probability estimates. Use params min_samples and require_min_samples to select desired behavior.

    Note1: RunTimeWarning will occur as a reminder when min_samples and require_min_samples params might result in unintended effects.
    Note2: It is possible recalibrating could result in some positive .predict predictions with <0.5 .predict_proba positive probability.

    ruleset : Ruleset
        Ruleset to recalibrate.
    Xy_df : DataFrame
        Labeled dataset.
    class_feat : str
        Name of class feature column in Xy_df.
    pos_class : value, typically str or int
        Positive class value.

    min_samples : int, default=10
        Required minimum number of samples per Rule. Regardless of min_samples, at least one sample of the correct class is always required.
    require_min_samples : bool, default=True
        Halt (with warning) if any Rule lacks the minimum number of samples.
        Setting to False will warn, but still replace Rules probabilities even if the minimum number of samples is not met.
    alpha : float, default=1.0
        additive smoothing parameter
    """

    _check_model_features_present(Xy_df, ruleset.get_selected_features())

    # At least this many samples per rule (or neg) must be of correct class
    required_correct_samples = 1

    # If not using min_samples, set it to 1
    if not min_samples or min_samples < 1:
        min_samples = 1

    # Collect each Rule's pos and neg frequencies in rule.class_freqs
    # Store rules that lack enough samples in insufficient_rules
    df = Xy_df

    # Positive Predictions
    insufficient_rules = []
    n_samples = len(df)
    for i, rule in enumerate(ruleset.rules):
        p_preds = rule.covers(df)
        num_pos_pred = len(p_preds)
        tp = num_pos(p_preds, class_feat=class_feat, pos_class=pos_class)
        fp = num_pos_pred - tp
        tpr = tp / num_pos_pred if num_pos_pred else 0
        fpr = fp / num_pos_pred if num_pos_pred else 0
        rule.class_ns_ = (fp, tp)
        rule.class_freqs_ = (fpr, tpr)
        rule.smoothed_class_freqs_ = additive_smoothed_rates((fp, tp), alpha=alpha)
        if verbosity >= 2:
            print('rule', i, rule)
            print('n_samples:', n_samples, \
                'tp:', tp, 'fp:', fp, 'tpr:', tpr, 'fpr:', fpr, \
                'class_ns:', rule.class_ns_, 'class_freqs:', rule.class_freqs_,
                'smoothed_class_freqs:', rule.smoothed_class_freqs_)
        # Rule has insufficient samples if fewer than minsamples or lacks at least one correct sample
        if n_samples < min_samples or tp < required_correct_samples:
            insufficient_rules.append(rule)

    # Collect class frequencies for negative predictions
    uncovered = df.drop(ruleset.covers(df).index)
    num_neg_pred = len(uncovered)
    tn = num_neg(uncovered, class_feat=class_feat, pos_class=pos_class)
    fn = num_neg_pred - tn
    tnr = tn / num_neg_pred if num_neg_pred else 0
    fnr = fn / num_neg_pred if num_neg_pred else 0
    ruleset.uncovered_class_ns_ = (tn, fn)
    ruleset.uncovered_class_freqs_ = (tnr, fnr)
    ruleset.smoothed_uncovered_class_freqs_ = additive_smoothed_rates((tn, fn), alpha=alpha)
    if verbosity >= 2:
        print('tn:', tn, 'fn:', fn, 'tnr:', tnr, 'fnr:', fnr, \
              'ruleset_uncovered_class_ns:', ruleset.uncovered_class_ns_,
              'ruleset_uncovered_class_freqs', ruleset.uncovered_class_freqs_,
              'smoothed_uncovered_class_freqs_', ruleset.smoothed_uncovered_class_freqs_)

    """
    # Issue warnings if trouble with sample size
    sample_failure = False
    if insufficient_rules:  # WARN if any rules lack enough samples
        pretty_insufficient_rules = "\n".join([f'{r} class counts {sum(r.class_ns_)}' for r in insufficient_rules])
        warning_str = f"param min_samples={min_samples}; insufficient number of samples or fewer than {required_correct_samples} true positives for rules {pretty_insufficient_rules}"
        _warn(
            warning_str,
            RuntimeWarning,
            filename="base_functions",
            funcname="recalibrate_proba",
        )
        sample_failure = True
    if num_neg(df, class_feat=class_feat, pos_class=pos_class) < min_samples:  # WARN if negative class lacks enough samples
        warning_str = f"param min_samples={min_samples}; insufficient number of negatively labled samples"
        _warn(
            warning_str,
            RuntimeWarning,
            filename="base_functions",
            funcname="recalibrate_proba",
        )
        sample_failure = True
    if n_samples < min_samples:  # WARN if negative class lacks enough samples
        warning_str = f"param min_samples={min_samples}; insufficient number of negatively labled samples"
        _warn(
            warning_str,
            RuntimeWarning,
            filename="base_functions",
            funcname="recalibrate_proba",
        )
    if require_min_samples and sample_failure:
            warning_str = f"Recalibrate proba halted. to recalibrate, try using more samples, lowering min_samples, or set require_min_samples to False"
            _warn(
                warning_str,
                RuntimeWarning,
                filename="base_functions",
                funcname="recalibrate_proba",
            )
            return
    """

def additive_smoothed_rates(counts, alpha=1.0):
    """ counts : iterable of values
            e.g. (2, 0)
        alpha : float, default=1.0
            additive smoothing parameter
        return : iterable
            smoothed rates e.g. (0.75, 0.25)
    """
    n_counts = len(counts)
    denom = sum(counts) + (alpha * n_counts)
    res = [((val+alpha) / denom) for val in counts]
    #print(res)
    #type_ = type(counts)
    #return type_(res)
    return res

    ###################
    ##### METRICS #####
    ###################


def gain(before, after, pos_df, neg_df):
    """Calculates the information gain from before to after."""
    p0count = before.num_covered(pos_df)  # tp
    p1count = after.num_covered(pos_df)  # tp after action step
    n0count = before.num_covered(neg_df)  # fn
    n1count = after.num_covered(neg_df)  # fn after action step
    return p1count * (
        math.log2((p1count + 1) / (p1count + n1count + 1))
        - math.log2((p0count + 1) / (p0count + n0count + 1))
    )


def gain_cn(cn, cond_step, rule_covers_pos_idx, rule_covers_neg_idx):
    """Calculates the information gain from adding a Cond."""
    p0count = len(rule_covers_pos_idx)  # tp
    p1count = len(
        cn.cond_covers(cond_step, subset=rule_covers_pos_idx)
    )  # tp after action step
    n0count = len(rule_covers_neg_idx)  # fn
    n1count = len(
        cn.cond_covers(cond_step, subset=rule_covers_neg_idx)
    )  # fn after action step
    return p1count * (
        math.log2((p1count + 1) / (p1count + n1count + 1))
        - math.log2((p0count + 1) / (p0count + n0count + 1))
    )


def precision(object, pos_df, neg_df):
    """Calculate precision value of object's classification.

    object : Cond, Rule, or Ruleset
    """

    pos_covered = object.covers(pos_df)
    neg_covered = object.covers(neg_df)
    total_n_covered = len(pos_covered) + len(neg_covered)
    if total_n_covered == 0:
        return None
    else:
        return len(pos_covered) / total_n_covered


def rule_precision_cn(cn, rule, pos_idx, neg_idx):
    """Calculate precision value of object's classification.

    object : Cond, Rule, or Ruleset
    """

    pos_covered = cn.rule_covers(rule, pos_idx)
    neg_covered = cn.rule_covers(rule, neg_idx)
    total_n_covered = len(pos_covered) + len(neg_covered)
    if total_n_covered == 0:
        return None
    else:
        return len(pos_covered) / total_n_covered


def score_accuracy(predictions, actuals):
    """Calculate accuracy score of a trained model on a test set.

    predictions : iterable<bool>
        True for predicted positive class, False otherwise.
    actuals : iterable<bool>
        True for actual positive class, False otherwise.
    """
    t = [pr for pr, act in zip(predictions, actuals) if pr == act]
    n = predictions
    return len(t) / len(n)


def _accuracy(object, pos_pruneset, neg_pruneset):
    """Calculate accuracy value of object's classification.

    object : Cond, Rule, or Ruleset
    """
    P = len(pos_pruneset)
    N = len(neg_pruneset)
    if P + N == 0:
        return None

    tp = len(object.covers(pos_pruneset))
    tn = N - len(object.covers(neg_pruneset))
    return (tp + tn) / (P + N)


def _rule_accuracy_cn(cn, rule, pos_pruneset_idx, neg_pruneset_idx):
    """Calculate accuracy value of object's classification.

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
    """Return for a Rule its best successor Rule according to FOIL information gain metric."""

    best_gain = 0
    best_successor_rule = None

    for successor in rule.successors(possible_conds, pos_df, neg_df):
        g = gain(rule, successor, pos_df, neg_df)
        if g > best_gain:
            best_gain = g
            best_successor_rule = successor
    if verbosity >= 5:
        print(f"gain {rnd(best_gain)} {best_successor_rule}")
    return best_successor_rule


def best_rule_successor_cn(cn, rule, pos_idx, neg_idx, verbosity=0):
    """Return for a Rule its best successor Rule according to FOIL information gain metric."""

    best_cond = None
    best_gain = float("-inf")

    rule_covers_pos_idx = cn.rule_covers(rule, pos_idx)
    rule_covers_neg_idx = cn.rule_covers(rule, neg_idx)

    for cond_action_step in cn.conds:
        g = gain_cn(cn, cond_action_step, rule_covers_pos_idx, rule_covers_neg_idx)
        if g > best_gain:
            best_gain = g
            best_cond = cond_action_step
    if verbosity >= 5:
        print(f"gain {rnd(best_gain)} {best_cond}")
    return Rule(rule.conds + [best_cond]) if best_gain > 0 else None


###################
##### HELPERS #####
###################


def pos_neg_split(df, class_feat, pos_class):
    """Split df into pos and neg classes."""
    pos_df = pos(df, class_feat, pos_class)
    neg_df = neg(df, class_feat, pos_class)
    return pos_df, neg_df


def df_shuffled_split(df, split_size=0.66, random_state=None):
    """Return tuple of shuffled and split DataFrame.

    split_size : float
        Proportion of rows to include in return[0].
    random_state : float, default=None
        Random seed.

    Returns
        Tuple of shuffled and split DataFrame.
    """
    idx1, idx2 = random_split(
        df.index, split_size, res_type=set, random_state=random_state
    )
    return df.loc[idx1, :], df.loc[idx2, :]


def set_shuffled_split(set_to_split, split_size, random_state=None):
    """Return tuple of shuffled and split set.

    split_size : float
        Proportion of set to include in return[0].
    random_state : float, default=None
        Random seed.

    Returns
        Tuple of shuffled and split DataFrame.
    """
    list_to_split = list(set_to_split)
    seed(random_state)
    shuffle(list_to_split)
    split_at = int(len(list_to_split) * split_size)
    return (set(list_to_split[:split_at]), set(list_to_split[split_at:]))


def random_split(to_split, split_size, res_type=set, random_state=None):
    """Return tuple of shuffled and split iterable.

    to_split : iterable
        What to shuffle and split.
    split_size : float
        Proportion to include in return[0].
    res_type : type
        Type of items to return.
    random_state : float, default=None
        Random seed.
    Returns
        Tuple of shuffled and split DataFrame.
    """
    to_split = list(to_split)
    seed(random_state)
    shuffle(to_split)
    split_at = int(len(to_split) * split_size)
    return (res_type(to_split[:split_at]), res_type(to_split[split_at:]))


def pos(df, class_feat, pos_class):
    """Return subset of instances that are labeled positive."""
    return df[df[class_feat] == pos_class]


def neg(df, class_feat, pos_class):
    """Return subset of instances that are labeled negative."""
    return df[df[class_feat] != pos_class]


def num_pos(df, class_feat, pos_class):
    """Return number of instances that are labeled positive."""
    return len(df[df[class_feat] == pos_class])


def num_neg(df, class_feat, pos_class):
    """ Return number of instances that are labeled negative."""
    return len(df[df[class_feat] != pos_class])


def nCr(n, r):
    """Return number of combinations C(n, r)."""

    def product(numbers):
        return reduce(op.mul, numbers, 1)

    num = product(range(n, n - r, -1))
    den = product(range(1, r + 1))
    return num // den


def argmin(iterable):
    """Return index of minimum value."""
    lowest_val = iterable[0]
    lowest_i = 0
    for i, val in enumerate(iterable):
        if val < lowest_val:
            lowest_val = val
            lowest_i = i
    return lowest_i


def i_replaced(list_, i, value):
    """Return a new list with element i replaced by value.

    i : value
        Index to replace with value.
    value : value
        Value to replace at index i. None will return original list with element i removed.
    """
    if value is not None:
        return list_[:i] + [value] + list_[i + 1 :]
    else:
        return list_[:i] + list_[i + 1 :]


def rm_covered(object, pos_df, neg_df):
    """Return pos and neg dfs of examples that are not covered by object.

    Parameters
    ----------
    object : Cond, Rule, or Ruleset
        Object whose coverage predictions to invoke.
    pos_df : DataFrame
        Positive examples.
    neg_df : DataFrame
        Negative examples.

    Return
    ------
    tuple<DataFrame>
        Positive and negative examples not covered by object.
    """
    return (
        pos_df.drop(object.covers(pos_df).index, axis=0, inplace=False),
        neg_df.drop(object.covers(neg_df).index, axis=0, inplace=False),
    )


def rm_rule_covers_cn(cn, rule, pos_idx, neg_idx):
    """Return positive and negative indices not covered by object."""
    return (
        pos_idx - cn.rule_covers(rule, pos_idx),
        neg_idx - cn.rule_covers(rule, neg_idx),
    )


def truncstr(iterable, limit=5, direction="left"):
    """Return Ruleset string representation limited to a specified number of rules.

    limit: how many rules to return
    direction: which part to return. (valid options: 'left', 'right')
    """
    if len(iterable) > limit:
        if direction == "left":
            return iterable[:limit].__str__() + "..."
        elif direction == "right":
            return "..." + iterable[-limit:].__str__()
        else:
            raise ValueError('direction param must be "left" or "right"')
    else:
        return str(iterable)


def stop_early(ruleset, pos_remaining, neg_remaining, max_rules, min_ruleset_samples, max_total_conds):
    """Function to decide whether to halt training."""
    return (max_rules is not None and len(ruleset.rules) >= max_rules) \
        or (max_total_conds is not None and ruleset.count_conds() >= max_total_conds) \
        or (min_ruleset_samples is not None and (len(pos_remaining) + len(neg_remaining) < min_ruleset_samples))
