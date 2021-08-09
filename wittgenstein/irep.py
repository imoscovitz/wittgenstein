"""
Implementation of incremental reduced error pruning (IREP*) algorithm for growing classification rulesets.
See https://www.let.rug.nl/nerbonne/teach/learning/cohen95fast.pdf
"""

# Author: Ilan Moscovitz <ilan.moscovitz@gmail.com>
# License: MIT

import copy
import numpy as np
import random

import pandas as pd

from wittgenstein import utils, base, base_functions, preprocess
from .catnap import CatNap
from .check import _check_param_deprecation
from .abstract_ruleset_classifier import AbstractRulesetClassifier
from .base import Cond, Rule, Ruleset, asruleset
from .base_functions import score_accuracy, stop_early


class IREP(AbstractRulesetClassifier):
    """ Class for generating ruleset classification models. """

    def __init__(
        self,
        prune_size=0.33,
        n_discretize_bins=10,
        max_rules=None,
        max_rule_conds=None,
        max_total_conds=None,
        random_state=None,
        verbosity=0,
    ):
        """Create an IREP classifier.

        Parameters
        ----------
        prune_size : int, default=.33
            Proportion of training set to be used for pruning. Set to None to skip pruning (not recommended).
        n_discretize_bins : int, default=10
            Fit apparent numeric attributes into a maximum of n_discretize_bins discrete bins, inclusive on upper part of range. Pass None to disable auto-discretization.

        Limits for early-stopping. Intended for enhancing model interpretability and limiting training time on noisy datasets. Not specifically intended for use as a hyperparameter, since pruning already occurs during training, though it is certainly possible that tuning could improve model performance.
        max_rules : int, default=None
            Maximum number of rules.
        max_rule_conds : int, default=None
            Maximum number of conds per rule.
        max_total_conds : int, default=None
            Maximum number of total conds in entire ruleset.

        random_state : int, default=None
            Random seed for repeatable results.
        verbosity : int, default=0
            Output progress, model development, and/or computation. Each level includes the information belonging to lower-value levels.
               1: Show results of each major phase
               2: Show Ruleset grow/optimization steps
               3: Show Ruleset grow/optimization calculations
               4: Show Rule grow/prune steps
               5: Show Rule grow/prune calculations

        """
        AbstractRulesetClassifier.__init__(
            self,
            algorithm_name="IREP",
            prune_size=prune_size,
            n_discretize_bins=n_discretize_bins,
            max_rules=max_rules,
            max_rule_conds=max_rule_conds,
            max_total_conds=max_total_conds,
            random_state=random_state,
            verbosity=verbosity,
        )

    def __str__(self):
        """Returns string representation."""
        return super().__str__()

    def out_model(self):
        """Prints trained Ruleset model line-by-line: V represents 'or'; ^ represents 'and'."""
        super().out_model()

    def fit(
        self,
        trainset,
        y=None,
        class_feat=None,
        pos_class=None,
        feature_names=None,
        initial_model=None,
        cn_optimize=True,
        **kwargs,
    ):
        """Fit a Ruleset model.

        Parameters
        ----------
        trainset : DataFrame, numpy array, or other iterable
            Training dataset. Optional whether to include or exclude class labels column.
        y : iterable of str, int, bool
            Class labels corresponding to trainset rows. Use if class labels aren't included in trainset.
        class_feat: str, int
            Column name or index of class feature. Use if class feature is still in trainset.
        pos_class : str, optional for boolean target, default=1 or True
            Name of positive class.
        feature_names : list<str>, optional, default=None
            Specify feature names. If None, feature names default to column names for a DataFrame, or indices in the case of indexed iterables such as an array or list.
        initial_model : Ruleset, str, IREP or RIPPER, default=None
            Preexisting model from which to begin training. See also 'init_ruleset'.
        cn_optimize : bool, default=True
            Use algorithmic speed optimization.

        **kwargs
        --------
        The following parameters are moving to the RIPPER constructor (__init__) function. For the time-being, both the constructor and fit functions will accept them, but passing them here using .fit will be deprecated:

        prune_size : float, default=.33
            Proportion of training set to be used for pruning.
        n_discretize_bins : int, default=10
            Fit apparent numeric attributes into a maximum of n_discretize_bins discrete bins, inclusive on upper part of range. Pass None to disable auto-discretization.
        random_state : int, default=None
            Random seed for repeatable results.

        Limits for early-stopping. Intended for enhancing model interpretability and limiting training time on noisy datasets. Not specifically intended for use as a hyperparameter, since pruning already occurs during training, though it is certainly possible that tuning could improve model performance.
        max_rules : int, default=None
            Maximum number of rules.
        max_rule_conds : int, default=None
            Maximum number of conds per rule.
        max_total_conds : int, default=None
            Maximum number of total conds in entire ruleset.

        verbosity : int, default=0
            Output progress, model development, and/or computation. Each level includes the information belonging to lower-value levels.
               1: Show results of each major phase
               2: Show Ruleset grow/optimization steps
               3: Show Ruleset grow/optimization calculations
               4: Show Rule grow/prune steps
               5: Show Rule grow/prune calculations
        """

        # SETUP
        self.ruleset_ = Ruleset() if not initial_model else asruleset(initial_model)

        if self.verbosity >= 1:
            print("initialize model")
            print(self.ruleset_)

        # Handle any hyperparam deprecation
        self._set_deprecated_fit_params(kwargs)

        # Preprocess training data
        preprocess_params = {
            "trainset": trainset,
            "y": y,
            "class_feat": class_feat,
            "pos_class": pos_class,
            "feature_names": feature_names,
            "n_discretize_bins": self.n_discretize_bins,
            "verbosity": self.verbosity,
        }
        (
            df,
            self.class_feat,
            self.pos_class,
            self.bin_transformer_,
        ) = preprocess.preprocess_training_data(preprocess_params)

        # Create CatNap
        # possible minor speedup if pass cond_subset of only pos_class conds?
        if cn_optimize:
            self.cn = CatNap(
                df,
                feat_subset=None,
                cond_subset=None,
                class_feat=self.class_feat,
                pos_class=None,
            )

        # Split df into pos, neg classes
        pos_df, neg_df = base_functions.pos_neg_split(
            df, self.class_feat, self.pos_class
        )
        pos_df = pos_df.drop(self.class_feat, axis=1)
        neg_df = neg_df.drop(self.class_feat, axis=1)

        # TRAINING
        if self.verbosity >= 1:
            print("\ntraining Ruleset...")
        if not cn_optimize:
            self.ruleset_ = self._grow_ruleset(
                pos_df, neg_df, initial_model=initial_model
            )
        else:
            self.ruleset_ = self._grow_ruleset_cn(
                pos_df, neg_df, initial_model=initial_model
            )

        if self.verbosity >= 1:
            print("\nGREW RULESET:")
            self.ruleset_.out_pretty()

        # Issue warning if Ruleset is universal or empty
        self.ruleset_._check_allpos_allneg(warn=True, warnstack=[("irep", "fit")])

        # Set selected and trainset features
        self.selected_features_ = self.ruleset_.get_selected_features()
        self.trainset_features_ = df.drop(self.class_feat, axis=1).columns.tolist()

        # FIT PROBAS
        self.recalibrate_proba(
            df, min_samples=None, require_min_samples=False, discretize=False,
        )

        # CLEANUP
        self.classes_ = np.array([0, 1])

        # Remove any duplicates and trim
        self.ruleset_.rules = utils.remove_duplicates(self.ruleset_.rules)
        self.ruleset_.trim_conds(max_total_conds=self.max_total_conds)
        if cn_optimize:
            del self.cn

    def _grow_ruleset(self, pos_df, neg_df, initial_model=None):
        """Grow a Ruleset with (optional) pruning."""

        ruleset = self._ruleset_frommodel(initial_model)
        ruleset._update_possible_conds(pos_df, neg_df)

        if self.verbosity >= 2:
            print("growing ruleset...")
            print(f"initial model: {ruleset}")
            print()

        prune_size = (
            self.prune_size if self.prune_size is not None else 0
        )  # If not pruning, use all the data for growing
        pos_remaining = pos_df.copy()
        neg_remaining = neg_df.copy()
        self.rules = []

        # Stop adding disjunctions if there are no more positive examples to cover
        while len(pos_remaining) > 0:

            # If applicable, check for user-specified early stopping
            if stop_early(ruleset, self.max_rules, self.max_total_conds):
                break

            # Grow-prune split remaining uncovered examples (if applicable)
            pos_growset, pos_pruneset = base_functions.df_shuffled_split(
                pos_remaining, (1 - prune_size), random_state=self.random_state
            )
            neg_growset, neg_pruneset = base_functions.df_shuffled_split(
                neg_remaining, (1 - prune_size), random_state=self.random_state
            )
            if self.verbosity >= 2:
                print(
                    f"pos_growset {len(pos_growset)} pos_pruneset {len(pos_pruneset)}"
                )
                print(
                    f"neg_growset {len(neg_growset)} neg_pruneset {len(neg_pruneset)}"
                )
                if not prune_size:
                    print(f"(pruning is turned off)")

            # Grow Rule
            grown_rule = base_functions.grow_rule(
                pos_growset,
                neg_growset,
                ruleset.possible_conds,
                max_rule_conds=self.max_rule_conds,
                verbosity=self.verbosity,
            )

            # If not pruning, add Rule to Ruleset and drop only the covered positive examples
            if not prune_size:
                ruleset.add(grown_rule)
                if self.verbosity >= 2:
                    print(f"updated ruleset: {ruleset.truncstr(direction='right')}")
                    print()
                rule_covers_pos = grown_rule.covers(pos_remaining)
                pos_remaining = pos_remaining.drop(rule_covers_pos.index, axis=0)
                if self.verbosity >= 3:
                    print(
                        f"examples remaining: {len(pos_remaining)} pos, {len(neg_remaining)} neg"
                    )
                    print()

            # If pruning, prune Rule, assess if it's time to stop, and drop all covered examples
            else:
                pruned_rule = base_functions.prune_rule(
                    grown_rule,
                    _IREP_prune_metric,
                    pos_pruneset,
                    neg_pruneset,
                    verbosity=self.verbosity,
                )

                # Stop if the Rule is bad
                prune_precision = base_functions.precision(
                    pruned_rule, pos_pruneset, neg_pruneset
                )
                if not prune_precision or prune_precision < 0.50:
                    break
                # Otherwise, add new Rule, remove covered examples, and continue
                else:
                    ruleset.add(pruned_rule)
                    if self.verbosity >= 2:
                        print(f"updated ruleset: {ruleset.truncstr(direction='right')}")
                        print()
                    pos_remaining, neg_remaining = base_functions.rm_covered(
                        pruned_rule, pos_remaining, neg_remaining
                    )
                    if self.verbosity >= 3:
                        print(
                            f"examples remaining: {len(pos_remaining)} pos, {len(neg_remaining)} neg"
                        )
                        print()

        # Return new ruleset
        return ruleset

    def _grow_ruleset_cn(self, pos_df, neg_df, initial_model=None):
        """Grow a Ruleset with (optional) pruning."""

        ruleset = self._ruleset_frommodel(initial_model)
        ruleset.possible_conds = self.cn.conds

        if self.verbosity >= 2:
            print("growing ruleset...")
            print(initial_model)
            print(f"initial model: {ruleset}")
            print()

        prune_size = (
            self.prune_size if self.prune_size is not None else 0
        )  # If not pruning, use all the data for growing
        pos_remaining_idx = set(pos_df.index.tolist())
        neg_remaining_idx = set(neg_df.index.tolist())

        # Stop adding disjunctions if there are no more positive examples to cover
        while len(pos_remaining_idx) > 0:

            # If applicable, check for user-specified early stopping
            if stop_early(ruleset, self.max_rules, self.max_total_conds):
                break

            # Grow-prune split remaining uncovered examples (if applicable)
            pos_growset_idx, pos_pruneset_idx = base_functions.random_split(
                pos_remaining_idx,
                (1 - prune_size),
                res_type=set,
                random_state=self.random_state,
            )
            neg_growset_idx, neg_pruneset_idx = base_functions.random_split(
                neg_remaining_idx,
                (1 - prune_size),
                res_type=set,
                random_state=self.random_state,
            )

            if self.verbosity >= 2:
                print(
                    f"pos_growset {len(pos_growset_idx)} pos_pruneset {len(pos_pruneset_idx)}"
                )
                print(
                    f"neg_growset {len(neg_growset_idx)} neg_pruneset {len(neg_pruneset_idx)}"
                )
                if not prune_size:
                    print(f"(pruning is turned off)")

            # Grow Rule
            grown_rule = base_functions.grow_rule_cn(
                self.cn,
                pos_growset_idx,
                neg_growset_idx,
                initial_rule=Rule(),
                max_rule_conds=self.max_rule_conds,
                verbosity=self.verbosity,
            )

            # If not pruning, add Rule to Ruleset and drop only the covered positive examples
            if not prune_size:
                ruleset.add(grown_rule)
                if self.verbosity >= 2:
                    print(f"updated ruleset: {ruleset.truncstr(direction='right')}")
                    print()
                pos_remaining_idx = pos_remaining_idx - self.cn.rule_covers(
                    grown_rule, pos_remaining_idx
                )
                if self.verbosity >= 3:
                    print(
                        f"examples remaining: {len(pos_remaining_idx)} pos, {len(neg_remaining_idx)} neg"
                    )
                    print()

            # If pruning, prune Rule, assess if it's time to stop, and drop all covered examples
            else:
                pruned_rule = base_functions.prune_rule_cn(
                    self.cn,
                    grown_rule,
                    _IREP_prune_metric_cn,
                    pos_pruneset_idx,
                    neg_pruneset_idx,
                    verbosity=self.verbosity,
                )

                # Stop if the Rule is bad
                prune_precision = base_functions.rule_precision_cn(
                    self.cn, pruned_rule, pos_pruneset_idx, neg_pruneset_idx
                )
                if not prune_precision or prune_precision < 0.50:
                    break
                # Otherwise, add new Rule, remove covered examples, and continue
                else:
                    ruleset.add(pruned_rule)
                    if self.verbosity >= 2:
                        print(f"updated ruleset: {ruleset.truncstr(direction='right')}")
                        print()
                    # Remove ruleset-covered rules from training
                    (
                        pos_remaining_idx,
                        neg_remaining_idx,
                    ) = base_functions.rm_rule_covers_cn(
                        self.cn, pruned_rule, pos_remaining_idx, neg_remaining_idx
                    )
                    if self.verbosity >= 3:
                        print(
                            f"examples remaining: {len(pos_remaining_idx)} pos, {len(neg_remaining_idx)} neg"
                        )
                        print()

        # Return new ruleset
        return ruleset


##### Metrics #####


def _IREP_prune_metric(self, pos_pruneset, neg_pruneset):
    """Returns the prune value of a candidate Rule."""

    P = len(pos_pruneset)
    N = len(neg_pruneset)
    p = self.num_covered(pos_pruneset)
    n = self.num_covered(neg_pruneset)
    return (p + (N - n)) / (P + N)


def _IREP_prune_metric_cn(cn, rule, pos_idxs, neg_idxs):
    """Returns the prune value of a candidate Rule."""

    P = len(pos_idxs)
    N = len(neg_idxs)
    p = len(cn.rule_covers(rule, pos_idxs))
    n = len(cn.rule_covers(rule, neg_idxs))
    return (p + (N - n)) / (P + N)
