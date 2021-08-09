"""
Implementation of the RIPPERk algorithm for growing classification rulesets.
See https://www.let.rug.nl/nerbonne/teach/learning/cohen95fast.pdf
"""

# Author: Ilan Moscovitz <ilan.moscovitz@gmail.com>
# License: MIT

import copy
import math
import numpy as np

import pandas as pd

from wittgenstein import base, base_functions, preprocess
from .abstract_ruleset_classifier import AbstractRulesetClassifier
from .base import Cond, Rule, Ruleset, asruleset
from .base_functions import score_accuracy
from .catnap import CatNap
from .check import _check_is_model_fit
from wittgenstein import utils
from wittgenstein.utils import rnd


class RIPPER(AbstractRulesetClassifier):
    """ Class for generating ruleset classification models.
        See Cohen (1995): https://www.let.rug.nl/nerbonne/teach/learning/cohen95fast.pdf
    """

    def __init__(
        self,
        k=2,
        dl_allowance=64,
        prune_size=0.33,
        n_discretize_bins=10,
        max_rules=None,
        max_rule_conds=None,
        max_total_conds=None,
        random_state=None,
        verbosity=0,
    ):
        """Create a RIPPER classifier.

        Parameters
        ----------
        k : int, default=2
            Number of RIPPERk optimization iterations.
        prune_size : float, default=.33
            Proportion of training set to be used for pruning.
        dl_allowance : int, default=64
            Terminate Ruleset grow phase early if a Ruleset description length is encountered
            that is more than this amount above the lowest description length so far encountered.
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

        AbstractRulesetClassifier.__init__(
            self,
            algorithm_name="RIPPER",
            prune_size=prune_size,
            n_discretize_bins=n_discretize_bins,
            max_rules=max_rules,
            max_rule_conds=max_rule_conds,
            max_total_conds=max_total_conds,
            random_state=random_state,
            verbosity=verbosity,
        )
        self.VALID_HYPERPARAMETERS.update({"k", "dl_allowance"})
        self.k = k
        self.dl_allowance = dl_allowance

    def __str__(self):
        """Return string representation of a RIPPER classifier."""
        params = str(self.get_params()) + ">"
        params = (
            params.replace(": ", "=")
            .replace("'", "")
            .replace("{", "(")
            .replace("}", ")")
        )
        return f"<RIPPER{params}"

    def out_model(self):
        """Print trained Ruleset model line-by-line: V represents 'or'; ^ represents 'and'."""
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

        k : int, default=2
            Number of RIPPERk optimization iterations.
        prune_size : float, default=.33
            Proportion of training set to be used for pruning.
        dl_allowance : int, default=64
            Terminate Ruleset grow phase early if a Ruleset description length is encountered
            that is more than this amount above the lowest description length so far encountered.
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

        ################
        # Stage 0: Setup
        ################

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

        ###############################
        # Stage 1: Grow initial Ruleset
        ###############################

        if cn_optimize:
            pos_idx = set(pos_df.index.tolist())
            neg_idx = set(neg_df.index.tolist())
            self.ruleset_ = self._grow_ruleset_cn(
                pos_idx,
                neg_idx,
                prune_size=self.prune_size,
                dl_allowance=self.dl_allowance,
                max_rules=self.max_rules,
                max_rule_conds=self.max_rule_conds,
                max_total_conds=self.max_total_conds,
                initial_model=initial_model,
                random_state=self.random_state,
            )
        else:
            self.ruleset_ = self._grow_ruleset(
                pos_df,
                neg_df,
                prune_size=self.prune_size,
                dl_allowance=self.dl_allowance,
                max_rules=self.max_rules,
                max_rule_conds=self.max_rule_conds,
                max_total_conds=self.max_total_conds,
                initial_model=initial_model,
                random_state=self.random_state,
            )
        if self.verbosity >= 1:
            print()
            print("GREW INITIAL RULESET:")
            self.ruleset_.out_pretty()
            print()

        ###########################
        # Stage 2: Optimize Ruleset
        ###########################

        for iter in range(1, self.k + 1):
            # Create new but reproducible random_state (if applicable)
            iter_random_state = (
                self.random_state + 100 if self.random_state is not None else None
            )
            # Run optimization iteration
            if self.verbosity >= 1:
                print(f"optimization run {iter} of {self.k}")
            if cn_optimize:
                newset = self._optimize_ruleset_cn(
                    self.ruleset_,
                    pos_idx,
                    neg_idx,
                    prune_size=self.prune_size,
                    random_state=iter_random_state,
                )
            else:
                newset = self._optimize_ruleset(
                    self.ruleset_,
                    pos_df,
                    neg_df,
                    prune_size=self.prune_size,
                    random_state=iter_random_state,
                )

            if self.verbosity >= 1:
                print()
                print("OPTIMIZED RULESET:")
                if self.verbosity >= 2:
                    print(
                        f"iteration {iter} of {self.k}\n modified rules {[i for i in range(len(self.ruleset_.rules)) if self.ruleset_.rules[i]!= newset.rules[i]]}"
                    )
                newset.out_pretty()
                print()

            if iter != self.k and self.ruleset_ == newset:
                if self.verbosity >= 1:
                    print("No changes were made. Halting optimization.")
                break
            else:
                self.ruleset_ = newset

        #############################################
        # Stage 3: Cover any last remaining positives
        #############################################

        if cn_optimize:
            self._cover_remaining_positives_cn(
                df,
                max_rules=self.max_rules,
                max_rule_conds=self.max_rule_conds,
                max_total_conds=self.max_total_conds,
                random_state=self.random_state,
            )
        else:
            self._cover_remaining_positives(
                df,
                max_rules=self.max_rules,
                max_rule_conds=self.max_rule_conds,
                max_total_conds=self.max_total_conds,
                random_state=self.random_state,
            )

        #################################################
        # Stage 4: Remove any rules that don't improve dl
        #################################################

        if self.verbosity >= 2:
            print("Optimizing dl...")
        if cn_optimize:
            mdl_subset, _ = _rs_total_bits_cn(
                self.cn,
                self.ruleset_,
                self.ruleset_.possible_conds,
                pos_idx,
                neg_idx,
                bestsubset_dl=True,
                ret_bestsubset=True,
                verbosity=self.verbosity,
            )
        else:
            mdl_subset, _ = _rs_total_bits(
                self.ruleset_,
                self.ruleset_.possible_conds,
                pos_df,
                neg_df,
                bestsubset_dl=True,
                ret_bestsubset=True,
                verbosity=self.verbosity,
            )
        self.ruleset_ = mdl_subset
        if self.verbosity >= 1:
            print("FINAL RULESET:")
            self.ruleset_.out_pretty()
            print()

        # Issue warning if Ruleset is universal or empty
        self.ruleset_._check_allpos_allneg(warn=True, warnstack=[("ripper", "fit")])

        # Set Ruleset features
        self.selected_features_ = self.ruleset_.get_selected_features()
        self.trainset_features_ = df.drop(self.class_feat, axis=1).columns.tolist()

        # Remove any duplicates and trim
        self.ruleset_.rules = utils.remove_duplicates(self.ruleset_.rules)
        self.ruleset_.trim_conds(max_total_conds=self.max_total_conds)

        # Fit probas
        self.recalibrate_proba(
            df, min_samples=None, require_min_samples=False, discretize=False
        )
        self.classes_ = np.array([0, 1])

        # Cleanup
        if cn_optimize:
            del self.cn

    def score(self, X, y, score_function=score_accuracy):
        """Score the performance of a fit model.

        X : DataFrame, numpy array, or other iterable
            Examples to score.
        y : Series, numpy array, or other iterable
            Class label actuals.

        score_function : function, default=score_accuracy
            Any scoring function that takes two parameters: actuals <iterable<bool>>, predictions <iterable<bool>>, where the elements represent class labels.
            this optional parameter is intended to be compatible with sklearn's scoring functions: https://scikit-learn.org/stable/modules/model_evaluation.html#classification-metrics
        """

        _check_is_model_fit(self)

        predictions = self.predict(X)
        actuals = [yi == self.pos_class for yi in utils.aslist(y)]
        return score_function(actuals, predictions)

    def _set_theory_dl_lookup(self, df, size=15, verbosity=0):
        """Precalculate rule theory dls for various-sized rules."""

        self.dl_dict = {}

        temp = Ruleset()
        temp._update_possible_conds(df, df)

        for n in range(1, size + 1):
            rule = Rule([Cond("_", "_")] * n)
            dl = _r_theory_bits(
                rule, temp.possible_conds, bits_dict=None, verbosity=verbosity
            )
            self.dl_dict[n] = dl
            if verbosity >= 2:
                print(f"updated dl for rule size {n}: {dl}")

    def _grow_ruleset(
        self,
        pos_df,
        neg_df,
        prune_size,
        dl_allowance,
        max_rules=None,
        max_rule_conds=None,
        max_total_conds=None,
        initial_model=None,
        random_state=None,
    ):
        """Grow a Ruleset with pruning."""
        ruleset = self._ruleset_frommodel(initial_model)
        ruleset._update_possible_conds(pos_df, neg_df)

        ruleset_dl = None
        mdl = None  # Minimum encountered description length (in bits)
        dl_diff = 0
        if self.verbosity >= 2:
            print("growing ruleset...")
            print(f"initial model: {ruleset}")
            print()

        pos_remaining = pos_df.copy()
        neg_remaining = neg_df.copy()
        while len(pos_remaining) > 0 and dl_diff <= self.dl_allowance:

            # If applicable, check for user-specified early stopping
            if stop_early(ruleset, max_rules, max_total_conds):
                break

            # Grow-prune split remaining uncovered examples
            pos_growset, pos_pruneset = base_functions.df_shuffled_split(
                pos_remaining, (1 - prune_size), random_state=random_state
            )
            neg_growset, neg_pruneset = base_functions.df_shuffled_split(
                neg_remaining, (1 - prune_size), random_state=random_state
            )
            if self.verbosity >= 2:
                print(
                    f"pos_growset {len(pos_growset)} pos_pruneset {len(pos_pruneset)}"
                )
                print(
                    f"neg_growset {len(neg_growset)} neg_pruneset {len(neg_pruneset)}"
                )
            if len(pos_growset) == 0:
                break  # Probably safe, but a little dicey to only check pos_growset.

            # Grow Rule
            grown_rule = base_functions.grow_rule(
                pos_growset,
                neg_growset,
                ruleset.possible_conds,
                max_rule_conds=max_rule_conds,
                verbosity=self.verbosity,
            )
            if grown_rule.isempty():
                break  # Generated an empty rule b/c no good conds exist

            # Prune Rule
            pruned_rule = base_functions.prune_rule(
                grown_rule,
                _RIPPER_growphase_prune_metric,
                pos_pruneset,
                neg_pruneset,
                verbosity=self.verbosity,
            )

            # Add rule; calculate new description length
            ruleset.add(
                pruned_rule
            )  # Unlike IREP, IREP*/RIPPER stopping condition is inclusive: "After each rule is added, the total description length of the rule set and examples is computed."
            if self.verbosity >= 2:
                print(f"updated ruleset: {ruleset.truncstr(direction='right')}")
                print()

            if ruleset_dl is None:  # First Rule to be added
                rule_dl = _r_theory_bits(
                    pruned_rule, ruleset.possible_conds, verbosity=self.verbosity
                )
                theory_dl = rule_dl
                data_dl = _exceptions_bits(
                    ruleset, pos_df, neg_df, verbosity=self.verbosity
                )
                ruleset_dl = theory_dl + data_dl
                mdl = ruleset_dl
            else:
                rule_dl = _r_theory_bits(
                    pruned_rule, ruleset.possible_conds, verbosity=self.verbosity
                )
                theory_dl += rule_dl
                data_dl = _exceptions_bits(
                    ruleset, pos_df, neg_df, verbosity=self.verbosity
                )
                ruleset_dl = theory_dl + data_dl
                dl_diff = ruleset_dl - mdl

            if self.verbosity >= 3:
                print(f"rule dl: {rnd(rule_dl)}")
                print(f"updated theory dl: {rnd(theory_dl)}")
                print(f"exceptions: {rnd(data_dl)}")
                print(f"total dl: {rnd(ruleset_dl)}")
                if dl_diff <= self.dl_allowance:
                    print(
                        f"mdl {rnd(mdl)} (diff {rnd(dl_diff)} <= {rnd(self.dl_allowance)})"
                    )
                else:
                    print(
                        f"mdl {rnd(mdl)} dl-halt: diff {rnd(dl_diff)} exceeds allowance ({rnd(self.dl_allowance)})"
                    )

            mdl = ruleset_dl if ruleset_dl < mdl else mdl

            # Remove covered examples
            pos_remaining, neg_remaining = base_functions.rm_covered(
                pruned_rule, pos_remaining, neg_remaining
            )

            if self.verbosity >= 3:
                print(
                    f"examples remaining: {len(pos_remaining)} pos, {len(neg_remaining)} neg"
                )
                print()

        return ruleset

    def _grow_ruleset_cn(
        self,
        pos_idx,
        neg_idx,
        prune_size,
        dl_allowance,
        max_rules=None,
        max_rule_conds=None,
        max_total_conds=None,
        initial_model=None,
        random_state=None,
    ):
        """Grow a Ruleset with pruning."""
        ruleset = self._ruleset_frommodel(initial_model)
        ruleset.possible_conds = self.cn.conds

        pos_remaining_idx = pos_idx
        neg_remaining_idx = neg_idx
        ruleset_dl = None
        mdl = None  # Minimum encountered description length (in bits)
        dl_diff = 0
        if self.verbosity >= 2:
            print("growing ruleset...")
            print(f"initial model: {ruleset}")
            print()

        while len(pos_remaining_idx) > 0 and dl_diff <= self.dl_allowance:

            # If applicable, check for user-specified early stopping
            if (max_rules is not None and len(ruleset.rules) >= max_rules) or (
                max_total_conds is not None and ruleset.count_conds() >= max_total_conds
            ):
                break

            # Grow-prune split remaining uncovered examples
            pos_growset_idx, pos_pruneset_idx = base_functions.random_split(
                pos_remaining_idx,
                (1 - prune_size),
                res_type=set,
                random_state=random_state,
            )
            neg_growset_idx, neg_pruneset_idx = base_functions.random_split(
                neg_remaining_idx,
                (1 - prune_size),
                res_type=set,
                random_state=random_state,
            )
            if self.verbosity >= 2:
                print(
                    f"pos_growset {len(pos_growset_idx)} pos_pruneset {len(pos_pruneset_idx)}"
                )
                print(
                    f"neg_growset {len(neg_growset_idx)} neg_pruneset {len(neg_pruneset_idx)}"
                )
            if len(pos_growset_idx) == 0:
                break  # Probably safe, but a little dicey to only check pos_growset.

            # Grow Rule
            grown_rule = base_functions.grow_rule_cn(
                self.cn,
                pos_growset_idx,
                neg_growset_idx,
                initial_rule=Rule(),
                max_rule_conds=max_rule_conds,
                verbosity=self.verbosity,
            )
            if grown_rule.isempty():
                break  # Generated an empty rule b/c no good conds exist

            # Prune Rule
            pruned_rule = base_functions.prune_rule_cn(
                self.cn,
                grown_rule,
                _RIPPER_growphase_prune_metric_cn,
                pos_pruneset_idx,
                neg_pruneset_idx,
                verbosity=self.verbosity,
            )

            # Add rule; calculate new description length
            ruleset.add(
                pruned_rule
            )  # Unlike IREP, IREP*/RIPPER stopping condition is inclusive: "After each rule is added, the total description length of the rule set and examples is computed."
            if self.verbosity >= 2:
                print(f"updated ruleset: {ruleset.truncstr(direction='right')}")
                print()

            if ruleset_dl is None:  # First Rule to be added
                rule_dl = _r_theory_bits(
                    pruned_rule, ruleset.possible_conds, verbosity=self.verbosity
                )
                theory_dl = rule_dl
                data_dl = _exceptions_bits_cn(
                    self.cn, ruleset, pos_idx, neg_idx, verbosity=self.verbosity
                )
                ruleset_dl = theory_dl + data_dl
                mdl = ruleset_dl
            else:
                rule_dl = _r_theory_bits(
                    pruned_rule, ruleset.possible_conds, verbosity=self.verbosity
                )
                theory_dl += rule_dl
                data_dl = _exceptions_bits_cn(
                    self.cn, ruleset, pos_idx, neg_idx, verbosity=self.verbosity
                )
                ruleset_dl = theory_dl + data_dl
                dl_diff = ruleset_dl - mdl

            if self.verbosity >= 3:
                print(f"rule dl: {rnd(rule_dl)}")
                print(f"updated theory dl: {rnd(theory_dl)}")
                print(f"exceptions: {rnd(data_dl)}")
                print(f"total dl: {rnd(ruleset_dl)}")
                if dl_diff <= self.dl_allowance:
                    print(
                        f"mdl {rnd(mdl)} (diff {rnd(dl_diff)} <= {rnd(self.dl_allowance)})"
                    )
                else:
                    print(
                        f"mdl {rnd(mdl)} dl-halt: diff {rnd(dl_diff)} exceeds allowance ({rnd(self.dl_allowance)})"
                    )

            mdl = ruleset_dl if ruleset_dl < mdl else mdl

            # Remove covered examples
            pos_remaining_idx, neg_remaining_idx = base_functions.rm_rule_covers_cn(
                self.cn, pruned_rule, pos_remaining_idx, neg_remaining_idx
            )

            if self.verbosity >= 3:
                print(
                    f"examples remaining: {len(pos_remaining_idx)} pos, {len(neg_remaining_idx)} neg"
                )
                print()

        return ruleset

    def _optimize_ruleset(
        self,
        ruleset,
        pos_df,
        neg_df,
        prune_size,
        max_rule_conds=None,
        random_state=None,
    ):
        """Optimization phase."""

        if self.verbosity >= 2:
            print("optimizing ruleset...")
            print()

        pos_remaining = pos_df.copy()
        neg_remaining = neg_df.copy()
        original_ruleset = copy.deepcopy(ruleset)
        if self.verbosity >= 4:
            print("calculate original ruleset potential dl...")
        original_dl = _rs_total_bits(
            original_ruleset,
            original_ruleset.possible_conds,
            pos_df,
            neg_df,
            bestsubset_dl=True,
            verbosity=self.verbosity,
        )
        if self.verbosity >= 3:
            print(f"original ruleset potential dl: {rnd(original_dl)}")
            print()
        new_ruleset = copy.deepcopy(ruleset)

        for i, rule in enumerate(original_ruleset.rules):
            pos_growset, pos_pruneset = base_functions.df_shuffled_split(
                pos_remaining, (1 - prune_size), random_state=random_state
            )
            neg_growset, neg_pruneset = base_functions.df_shuffled_split(
                neg_remaining, (1 - prune_size), random_state=random_state
            )
            if len(pos_growset) == 0:
                break  # Possible where optimization run > 1

            # Create alternative rules
            if self.verbosity >= 4:
                print(
                    f"creating replacement for {i} of {len(original_ruleset.rules)}: {ruleset.rules[i]}"
                )
            g_replacement = base_functions.grow_rule(
                pos_growset,
                neg_growset,
                original_ruleset.possible_conds,
                initial_rule=Rule(),
                max_rule_conds=max_rule_conds,
                verbosity=self.verbosity,
            )
            replacement_ruleset = Ruleset(
                base_functions.i_replaced(original_ruleset.rules, i, g_replacement)
            )
            pr_replacement = base_functions.prune_rule(
                g_replacement,
                _RIPPER_optimization_prune_metric,
                pos_pruneset,
                neg_pruneset,
                eval_index_on_ruleset=(i, replacement_ruleset),
                verbosity=self.verbosity,
            )
            replacement_ruleset = Ruleset(
                base_functions.i_replaced(original_ruleset.rules, i, pr_replacement)
            )
            if self.verbosity >= 3:
                print(f"grew replacement {g_replacement}")
                print(f"pruned replacement is {pr_replacement}")

            if self.verbosity >= 3:
                print(
                    f"creating revision for {i} of {len(original_ruleset.rules)}: {ruleset.rules[i]}"
                )
            g_revision = base_functions.grow_rule(
                pos_growset,
                neg_growset,
                original_ruleset.possible_conds,
                initial_rule=ruleset.rules[i],
                max_rule_conds=max_rule_conds,
                verbosity=self.verbosity,
            )
            revision_ruleset = Ruleset(
                base_functions.i_replaced(original_ruleset.rules, i, g_revision)
            )
            pr_revision = base_functions.prune_rule(
                g_revision,
                _RIPPER_optimization_prune_metric,
                pos_pruneset,
                neg_pruneset,
                eval_index_on_ruleset=(i, revision_ruleset),
                verbosity=self.verbosity,
            )
            revision_ruleset = Ruleset(
                base_functions.i_replaced(original_ruleset.rules, i, pr_revision)
            )
            if self.verbosity >= 3:
                print(f"grew revision {g_replacement}")
                print(f"pruned revision is {pr_replacement}")
                print()

            # Calculate alternative Rulesets' respective lowest potential dls to identify the best version
            if self.verbosity >= 3:
                print(
                    f"calculate potential dl for ds with replacement {pr_replacement}"
                )
            replacement_dl = (
                _rs_total_bits(
                    replacement_ruleset,
                    original_ruleset.possible_conds,
                    pos_df,
                    neg_df,
                    bestsubset_dl=True,
                    verbosity=self.verbosity,
                )
                if pr_replacement != rule
                else original_dl
            )
            if self.verbosity >= 3:
                print(f"calculate potential dl for ds with revision {pr_revision}")
            revision_dl = (
                _rs_total_bits(
                    revision_ruleset,
                    original_ruleset.possible_conds,
                    pos_df,
                    neg_df,
                    bestsubset_dl=True,
                    verbosity=self.verbosity,
                )
                if pr_revision != rule
                else original_dl
            )
            best_rule = [rule, pr_replacement, pr_revision][
                base_functions.argmin([original_dl, replacement_dl, revision_dl])
            ]

            if self.verbosity >= 2:
                print(f"\nrule {i+1} of {len(original_ruleset.rules)}")
                rep_str = (
                    pr_replacement.__str__() if pr_replacement != rule else "unchanged"
                )
                rev_str = pr_revision.__str__() if pr_revision != rule else "unchanged"
                best_str = best_rule.__str__() if best_rule != rule else "unchanged"
                if self.verbosity == 2:
                    print(f"original: {rule}")
                    print(f"replacement: {rep_str}")
                    print(f"revision: {rev_str}")
                    print(f"*best: {best_str}")
                    if best_rule in new_ruleset:
                        print(
                            f"best already included in optimization -- retaining original"
                        )
                    print()
                else:
                    print(f"original: {rule}) | {rnd(original_dl)} bits")
                    print(f"replacement: {rep_str} | {rnd(replacement_dl)} bits")
                    print(f"revision: {rev_str} | {rnd(revision_dl)} bits")
                    print(
                        f"*best: {best_str} | {rnd(min([replacement_dl, revision_dl, original_dl]))} bits"
                    )
                    if best_rule in new_ruleset:
                        print(
                            f"best already included in optimization -- retaining original"
                        )
                    print()
            if best_rule not in new_ruleset:
                new_ruleset.rules[i] = best_rule
            else:
                new_ruleset.rules[i] = rule

            # Remove covered examples
            pos_remaining, neg_remaining = base_functions.rm_covered(
                rule, pos_remaining, neg_remaining
            )
            if self.verbosity >= 3:
                print(
                    f"examples remaining: {len(pos_remaining)} pos, {len(neg_remaining)} neg"
                )
                print()

            # If there are no pos data remaining to train optimization (could happen if optimization run >1), keep remaining rules the same
            if len(pos_remaining) == 0:
                break

        return new_ruleset

    def _optimize_ruleset_cn(
        self,
        ruleset,
        pos_idx,
        neg_idx,
        prune_size,
        max_rule_conds=None,
        random_state=None,
    ):
        """Optimization phase."""

        if self.verbosity >= 2:
            print("optimizing ruleset...")
            print()

        pos_remaining_idx = pos_idx
        neg_remaining_idx = neg_idx
        original_ruleset = copy.deepcopy(ruleset)
        if self.verbosity >= 4:
            print("calculate original ruleset potential dl...")
        original_dl = _rs_total_bits_cn(
            self.cn,
            original_ruleset,
            original_ruleset.possible_conds,
            pos_idx,
            neg_idx,
            bestsubset_dl=True,
            verbosity=self.verbosity,
        )
        if self.verbosity >= 3:
            print(f"original ruleset potential dl: {rnd(original_dl)}")
            print()
        new_ruleset = copy.deepcopy(ruleset)

        for i, rule in enumerate(original_ruleset.rules):
            pos_growset_idx, pos_pruneset_idx = base_functions.set_shuffled_split(
                pos_remaining_idx, (1 - prune_size), random_state=random_state
            )
            neg_growset_idx, neg_pruneset_idx = base_functions.set_shuffled_split(
                neg_remaining_idx, (1 - prune_size), random_state=random_state
            )
            if len(pos_growset_idx) == 0:
                break  # Possible where optimization run > 1

            # Create alternative rules
            if self.verbosity >= 4:
                print(
                    f"creating replacement for {i} of {len(original_ruleset.rules)}: {ruleset.rules[i]}"
                )
            # g_replacement = base_functions.grow_rule(pos_growset, neg_growset, original_ruleset.possible_conds, initial_rule=Rule(), max_rule_conds=max_rule_conds, verbosity=self.verbosity)
            g_replacement = base_functions.grow_rule_cn(
                self.cn,
                pos_growset_idx,
                neg_growset_idx,
                initial_rule=Rule(),
                max_rule_conds=max_rule_conds,
                verbosity=self.verbosity,
            )
            replacement_ruleset = Ruleset(
                base_functions.i_replaced(original_ruleset.rules, i, g_replacement)
            )
            # pr_replacement = base_functions.prune_rule(g_replacement, _RIPPER_optimization_prune_metric, pos_pruneset, neg_pruneset, eval_index_on_ruleset=(i,replacement_ruleset), verbosity=self.verbosity)
            pr_replacement = base_functions.prune_rule_cn(
                self.cn,
                g_replacement,
                _RIPPER_optimization_prune_metric_cn,
                pos_pruneset_idx,
                neg_pruneset_idx,
                eval_index_on_ruleset=(i, replacement_ruleset),
                verbosity=self.verbosity,
            )
            replacement_ruleset = Ruleset(
                base_functions.i_replaced(original_ruleset.rules, i, pr_replacement)
            )
            if self.verbosity >= 3:
                print(f"grew replacement {g_replacement}")
                print(f"pruned replacement is {pr_replacement}")

            if self.verbosity >= 3:
                print(
                    f"creating revision for {i} of {len(original_ruleset.rules)}: {ruleset.rules[i]}"
                )
            # g_revision = base_functions.grow_rule(pos_growset, neg_growset, original_ruleset.possible_conds, initial_rule=ruleset.rules[i], max_rule_conds=max_rule_conds, verbosity=self.verbosity)
            g_revision = base_functions.grow_rule_cn(
                self.cn,
                pos_growset_idx,
                neg_growset_idx,
                initial_rule=ruleset.rules[i],
                max_rule_conds=max_rule_conds,
                verbosity=self.verbosity,
            )
            revision_ruleset = Ruleset(
                base_functions.i_replaced(original_ruleset.rules, i, g_revision)
            )
            # pr_revision = base_functions.prune_rule(g_revision, _RIPPER_optimization_prune_metric, pos_pruneset, neg_pruneset, eval_index_on_ruleset=(i,revision_ruleset), verbosity=self.verbosity)
            pr_revision = base_functions.prune_rule_cn(
                self.cn,
                g_revision,
                _RIPPER_optimization_prune_metric_cn,
                pos_pruneset_idx,
                neg_pruneset_idx,
                eval_index_on_ruleset=(i, revision_ruleset),
                verbosity=self.verbosity,
            )
            revision_ruleset = Ruleset(
                base_functions.i_replaced(original_ruleset.rules, i, pr_revision)
            )
            if self.verbosity >= 3:
                print(f"grew revision {g_replacement}")
                print(f"pruned revision is {pr_replacement}")
                print()

            # Calculate alternative Rulesets' respective lowest potential dls to identify the best version
            if self.verbosity >= 3:
                print(
                    f"calculate potential dl for ds with replacement {pr_replacement}"
                )
            replacement_dl = (
                _rs_total_bits_cn(
                    self.cn,
                    replacement_ruleset,
                    original_ruleset.possible_conds,
                    pos_idx,
                    neg_idx,
                    bestsubset_dl=False,
                    ret_bestsubset=False,
                    verbosity=0,
                )
                if pr_replacement != rule
                else original_dl
            )
            if self.verbosity >= 3:
                print(f"calculate potential dl for ds with revision {pr_revision}")
            revision_dl = (
                _rs_total_bits_cn(
                    self.cn,
                    revision_ruleset,
                    original_ruleset.possible_conds,
                    pos_idx,
                    neg_idx,
                    bestsubset_dl=False,
                    ret_bestsubset=False,
                    verbosity=0,
                )
                if pr_revision != rule
                else original_dl
            )
            best_rule = [rule, pr_replacement, pr_revision][
                base_functions.argmin([original_dl, replacement_dl, revision_dl])
            ]

            if self.verbosity >= 2:
                print(f"\nrule {i+1} of {len(original_ruleset.rules)}")
                rep_str = (
                    pr_replacement.__str__() if pr_replacement != rule else "unchanged"
                )
                rev_str = pr_revision.__str__() if pr_revision != rule else "unchanged"
                best_str = best_rule.__str__() if best_rule != rule else "unchanged"
                if self.verbosity == 2:
                    print(f"original: {rule}")
                    print(f"replacement: {rep_str}")
                    print(f"revision: {rev_str}")
                    print(f"*best: {best_str}")
                    if best_rule in new_ruleset:
                        print(
                            f"best already included in optimization -- retaining original"
                        )
                    print()
                else:
                    print(f"original: {rule}) | {rnd(original_dl)} bits")
                    print(f"replacement: {rep_str} | {rnd(replacement_dl)} bits")
                    print(f"revision: {rev_str} | {rnd(revision_dl)} bits")
                    print(
                        f"*best: {best_str} | {rnd(min([replacement_dl, revision_dl, original_dl]))} bits"
                    )
                    if best_rule in new_ruleset:
                        print(
                            f"best already included in optimization -- retaining original"
                        )
                    print()
            if best_rule not in new_ruleset:
                new_ruleset.rules[i] = best_rule
            else:
                new_ruleset.rules[i] = rule

            # Remove covered examples
            pos_remaining_idx, neg_remaining_idx = base_functions.rm_rule_covers_cn(
                self.cn, rule, pos_remaining_idx, neg_remaining_idx
            )
            if self.verbosity >= 3:
                print(
                    f"examples remaining: {len(pos_remaining_idx)} pos, {len(neg_remaining_idx)} neg"
                )
                print()

            # If there are no pos data remaining to train optimization (could happen if optimization run >1), keep remaining rules the same
            if len(pos_remaining_idx) == 0:
                break

        return new_ruleset

    def _cover_remaining_positives(
        self,
        df,
        max_rules=None,
        max_rule_conds=None,
        max_total_conds=None,
        random_state=None,
    ):
        """Stage 3: Post-optimization, cover any remaining uncovered positives."""
        pos_remaining, neg_remaining = base_functions.pos_neg_split(
            df, self.class_feat, self.pos_class
        )
        pos_remaining = pos_remaining.drop(self.class_feat, axis=1)
        neg_remaining = neg_remaining.drop(self.class_feat, axis=1)
        pos_remaining, neg_remaining = base_functions.rm_covered(
            self.ruleset_, pos_remaining, neg_remaining
        )
        if len(pos_remaining) >= 1:
            if self.verbosity >= 2:
                print(f"{len(pos_remaining)} pos left. Growing final rules...")
            newset = self._grow_ruleset(
                pos_remaining,
                neg_remaining,
                initial_model=self.ruleset_,
                prune_size=self.prune_size,
                dl_allowance=self.dl_allowance,
                max_rules=max_rules,
                max_rule_conds=max_rule_conds,
                max_total_conds=max_total_conds,
                random_state=random_state,
            )
            if self.verbosity >= 1:
                print("GREW FINAL RULES")
                newset.out_pretty()
                print()
            self.ruleset_ = newset
        else:
            if self.verbosity >= 1:
                print("All pos covered\n")

    def _cover_remaining_positives_cn(
        self,
        df,
        max_rules=None,
        max_rule_conds=None,
        max_total_conds=None,
        random_state=None,
    ):
        """Stage 3: Post-optimization, cover any remaining uncovered positives."""
        pos_remaining_idx, neg_remaining_idx = self.cn.pos_idx_neg_idx(
            df, self.class_feat, self.pos_class
        )

        if len(pos_remaining_idx) >= 1:
            if self.verbosity >= 2:
                print(f"{len(pos_remaining_idx)} pos left. Growing final rules...")
            newset = self._grow_ruleset_cn(
                pos_remaining_idx,
                neg_remaining_idx,
                initial_model=self.ruleset_,
                prune_size=self.prune_size,
                dl_allowance=self.dl_allowance,
                max_rules=max_rules,
                max_rule_conds=max_rule_conds,
                max_total_conds=max_total_conds,
                random_state=random_state,
            )
            if self.verbosity >= 1:
                print("GREW FINAL RULES")
                newset.out_pretty()
                print()
            self.ruleset_ = newset
        else:
            if self.verbosity >= 1:
                print("All positives covered\n")

    def get_params(self, deep=True):
        return {param: self.__dict__.get(param) for param in self.VALID_HYPERPARAMETERS}

    def set_params(self, **parameters):
        for parameter, value in parameters.items():
            # if parameter in self.VALID_HYPERPARAMETERS:
            setattr(self, parameter, value)
        return self


###################################
##### RIPPER-specific Metrics #####
###################################


def _RIPPER_growphase_prune_metric(rule, pos_pruneset, neg_pruneset):
    """RIPPER/IREP* prune metric.
    Returns the prune value of a candidate Rule.

    Cohen's formula is (p-n) / (p+n).
    Unclear from the paper how they handle divzero (where p+n=0), so I Laplaced it.
    Weka's solution was to modify the formula to (p+1)/(p+n+2), but running with this because the (non-NaN) values I got appeared closer to those of the original formula.
    """
    # I imagine Weka's is 1/2 because that's closer to a 50-50 class distribution?
    p = rule.num_covered(pos_pruneset)
    n = rule.num_covered(neg_pruneset)
    return (p - n + 1) / (p + n + 1)


def _RIPPER_growphase_prune_metric_cn(cn, rule, pos_pruneset_idx, neg_pruneset_idx):
    """RIPPER/IREP* prune metric.
    Returns the prune value of a candidate Rule.

    Cohen's formula is (p-n) / (p+n).
    Unclear from the paper how they handle divzero (where p+n=0), so I Laplaced it.
    Weka's solution was to modify the formula to (p+1)/(p+n+2), but the (non-NaN) values I got appeared closer to those of the original formula.
    """
    # I imagine Weka's is 1/2 because that's closer to a 50-50 class distribution?
    p = len(cn.rule_covers(rule, pos_pruneset_idx))
    n = len(cn.rule_covers(rule, neg_pruneset_idx))
    return (p - n + 1) / (p + n + 1)


def _RIPPER_optimization_prune_metric(rule, pos_pruneset, neg_pruneset):
    return base_functions._accuracy(rule, pos_pruneset, neg_pruneset)


def _RIPPER_optimization_prune_metric_cn(cn, rule, pos_pruneset_idx, neg_pruneset_idx):
    return base_functions._rule_accuracy_cn(
        cn, rule, pos_pruneset_idx, neg_pruneset_idx
    )


def _r_theory_bits(rule, possible_conds, bits_dict=None, verbosity=0):
    """Returns description length (in bits) for a single Rule."""

    if hasattr(rule, "dl"):
        return rule.dl
    else:
        # if type(rule) != Rule:
        #    raise TypeError(f'param rule in _r_theory_bits is type {type(rule)}; it should be type Rule')
        k = len(rule.conds)  # Number of rule conditions
        n = len(possible_conds)  # Number of possible conditions
        pr = k / n

        S = k * math.log2(1 / pr) + (n - k) * math.log2((1 / (1 - pr)))  # S(n, k, pr)
        K = math.log2(k)  # Number bits need to send integer k
        rule_dl = 0.5 * (
            K + S
        )  # Divide by 2 a la Quinlan. Cohen: "to adjust for possible redundency in attributes"
        if verbosity >= 5:
            print(
                f"rule theory bits| {rule} k {k} n {n} pr {rnd(pr)}: {rnd(rule_dl)} bits"
            )

        # rule.dl = rule_dl
        return rule_dl


def _rs_theory_bits(ruleset, possible_conds, verbosity=0):
    """Returns theory description length (in bits) for a Ruleset."""

    # if type(ruleset) != Ruleset:
    #    raise TypeError(f'param ruleset in _rs_theory_bits should be type Ruleset')
    total = 0
    for rule in ruleset.rules:
        total += _r_theory_bits(rule, possible_conds, verbosity=verbosity)
        # total += rule_bits(rule, possible_conds, rem_pos, rem_neg, verbosity=verbosity)
        # rem_pos, rem_neg = base.rm_covered(rule, rem_pos, rem_neg)
    if verbosity >= 5:
        print(f"ruleset theory bits| {rnd(total)}")

    # ruleset.dl = total
    return total


def _exceptions_bits(ruleset, pos_df, neg_df, verbosity=0):
    """Returns description length (in bits) for exceptions to a Ruleset's coverage."""

    if type(ruleset) != Ruleset:
        raise TypeError(
            f"to avoid double-counting, _exceptions_bits should calculate exceptions over entire set of rules with type Ruleset"
        )
    N = len(pos_df) + len(neg_df)  # Total number of examples
    p = ruleset.num_covered(pos_df) + ruleset.num_covered(
        neg_df
    )  # Total number of examples classified as positive = total covered
    fp = ruleset.num_covered(
        neg_df
    )  # Number false positives = negatives covered by the ruleset
    fn = len(pos_df) - ruleset.num_covered(
        pos_df
    )  # Number false negatives = positives not covered by the ruleset
    exceptions_dl = math.log2(base_functions.nCr(p, fp)) + math.log2(
        base_functions.nCr((N - p), fn)
    )
    if verbosity >= 5:
        print(
            f"exceptions_bits| {ruleset.truncstr()}: \n N {N} p {p} fp {fp} fn {fn}: exceptions_bits {rnd(exceptions_dl)}"
        )

    return exceptions_dl


def _exceptions_bits_cn(cn, ruleset, pos_idx, neg_idx, verbosity=0):
    """Returns description length (in bits) for exceptions to a Ruleset's coverage."""

    # if type(ruleset) != Ruleset:
    #    raise TypeError(f'to avoid double-counting, _exceptions_bits should calculate exceptions over entire set of rules with type Ruleset')
    N = len(pos_idx) + len(neg_idx)  # Total number of examples
    pos_cov = cn.ruleset_covers(ruleset, subset=pos_idx)
    neg_cov = cn.ruleset_covers(ruleset, subset=neg_idx)
    p = len(pos_cov) + len(
        neg_cov
    )  # Total number of examples classified as positive = total covered
    fp = len(neg_cov)  # Number false positives = negatives covered by the ruleset
    fn = len(pos_idx) - len(
        pos_cov
    )  # Number false negatives = positives not covered by the ruleset
    exceptions_dl = math.log2(base_functions.nCr(p, fp)) + math.log2(
        base_functions.nCr((N - p), fn)
    )
    if verbosity >= 5:
        print(
            f"exceptions_bits| {ruleset.truncstr()}: \n N {N} p {p} fp {fp} fn {fn}: exceptions_bits {rnd(exceptions_dl)}"
        )

    return exceptions_dl


def _rs_total_bits(
    ruleset,
    possible_conds,
    pos_df,
    neg_df,
    bestsubset_dl=False,
    ret_bestsubset=False,
    verbosity=0,
):
    """Returns total description length (in bits) of ruleset -- the sum of its theory dl and exceptions dl.

    bestsubset_dl : bool, default=False
        Whether to return estimated minimum possible dl were all rules that increase dl to be removed.
    ret_bestsubset : bool, default=False
        Whether to return the best subset that was found. Return format will be (<Ruleset>,dl).
    """

    # The RIPPER paper is brief and unclear w/r how to evaluate a ruleset for best potential dl.

    # 1) Do you reevaluate already-visited rules or evaluate each rule independently of one another?
    # Weka's source code comments that you are not supposed to, and that this is "bizarre."
    # Perhaps not recursing so -- and getting a possibly sub-optimal mdl -- could be viewed as a greedy time-saver?
    # After all, this is supposed to be an iterative algorithm, it could optimize more times with future k's,
    # and it's not like we're performing an exhaustive search of every possible combination anyways.

    # 2) In what order are you supposed to evaluate? FIFO or LIFO?
    # Footnote 7 suggests optimization is done FIFO; the previous page suggests IREP* final dl reduction is done LIFO;
    # and context suggests dl reduction should be performed the same way both times.

    # I chose to greedy for #1, and FIFO for #2 but may choose differently in a future version if it seems more appropriate.
    # In any case, RIPPER's strong performance on the test sets vs. RandomForest suggests it may not matter all that much.

    # if type(ruleset) != Ruleset:
    #    raise TypeError(f'param ruleset in _rs_total_bits should be type Ruleset')
    if ret_bestsubset and not bestsubset_dl:
        raise ValueError(
            f"ret_bestsubset must be True in order to return bestsubset_dl"
        )

    if not bestsubset_dl:
        theory_bits = _rs_theory_bits(ruleset, possible_conds, verbosity=verbosity)
        data_bits = _exceptions_bits(ruleset, pos_df, neg_df, verbosity=verbosity)
        if verbosity >= 3:
            print(f"total ruleset bits | {rnd(theory_bits + data_bits)}")
        return theory_bits + data_bits
    else:
        # Collect the dl of each subset
        subset_dls = []
        theory_dl = 0
        if verbosity >= 5:
            print(f"find best potential dl for {ruleset}:")
        for i, rule in enumerate(
            ruleset.rules
        ):  # Separating theory and exceptions dls in this way means you don't have to recalculate theory each time
            subset = Ruleset(ruleset.rules[: i + 1])
            rule_theory_dl = _r_theory_bits(rule, possible_conds, verbosity=verbosity)
            theory_dl += rule_theory_dl
            exceptions_dl = _exceptions_bits(
                subset, pos_df, neg_df, verbosity=verbosity
            )
            subset_dls.append(theory_dl + exceptions_dl)
            if verbosity >= 5:
                print(f"subset 0-{i} | dl: {rnd(subset_dls[i])}")

        # Build up the best Ruleset and calculate the mdl
        mdl_ruleset = Ruleset()
        for i, rule, in enumerate(ruleset.rules):
            if (
                i == 0 or subset_dls[i] <= subset_dls[i - 1]
            ):  # Rule i does not worsen the dl
                mdl_ruleset.add(rule)
        if verbosity >= 5:
            print(f"subset dls: {[(i,rnd(dl)) for i,dl in enumerate(subset_dls)]}")
            print(f"best potential ruleset: {mdl_ruleset}")
        mdl = _rs_total_bits(
            mdl_ruleset,
            possible_conds,
            pos_df,
            neg_df,
            bestsubset_dl=False,
            verbosity=0,
        )  # About to print value below
        if verbosity >= 5:
            print(f"best potential dl was {rnd(mdl)}")
            print()
        if not ret_bestsubset:
            return mdl
        else:
            return (mdl_ruleset, mdl)


def _rs_total_bits_cn(
    cn,
    ruleset,
    possible_conds,
    pos_idx,
    neg_idx,
    bestsubset_dl=False,
    ret_bestsubset=False,
    verbosity=0,
):
    """Returns total description length (in bits) of ruleset -- the sum of its theory dl and exceptions dl.

    bestsubset_dl : bool, default=False
        Whether to return estimated minimum possible dl were all rules that increase dl to be removed.
    ret_bestsubset : bool, default=False
        Whether to return the best subset that was found. Return format will be (<Ruleset>,dl).
    """

    # The RIPPER paper is brief and unclear w/r how to evaluate a ruleset for best potential dl.

    # 1) Do you reevaluate already-visited rules or evaluate each rule independently of one another?
    # Weka's source code comments that you are not supposed to, and that this is "bizarre."
    # Perhaps not recursing so -- and getting a possibly sub-optimal mdl -- could be viewed as a greedy time-saver?
    # After all, this is supposed to be an iterative algorithm, it could optimize more times with future k's,
    # and it's not like we're performing an exhaustive search of every possible combination anyways.

    # 2) In what order are you supposed to evaluate? FIFO or LIFO?
    # Footnote 7 suggests optimization is done FIFO; the previous page suggests IREP* final dl reduction is done LIFO;
    # and context suggests dl reduction should be performed the same way both times.

    # I chose to greedy for #1, and FIFO for #2 but may choose differently in a future version if it seems more appropriate.
    # In any case, RIPPER's strong performance on the test sets vs. RandomForest suggests it may not matter all that much.

    # if type(ruleset) != Ruleset:
    #    raise TypeError(f'param ruleset in _rs_total_bits should be type Ruleset')
    if ret_bestsubset and not bestsubset_dl:
        raise ValueError(
            f"ret_bestsubset must be True in order to return bestsubset_dl"
        )

    if not bestsubset_dl:
        theory_bits = _rs_theory_bits(ruleset, possible_conds, verbosity=verbosity)
        data_bits = _exceptions_bits_cn(
            cn, ruleset, pos_idx, neg_idx, verbosity=verbosity
        )
        if verbosity >= 3:
            print(f"total ruleset bits | {rnd(theory_bits + data_bits)}")
        return theory_bits + data_bits
    else:
        # Collect the dl of each subset
        subset_dls = []
        theory_dl = 0
        if verbosity >= 5:
            print(f"find best potential dl for {ruleset}:")
        for i, rule in enumerate(
            ruleset.rules
        ):  # Separating theory and exceptions dls in this way means you don't have to recalculate theory each time
            subset = Ruleset(ruleset.rules[: i + 1])
            rule_theory_dl = _r_theory_bits(rule, possible_conds, verbosity=verbosity)
            theory_dl += rule_theory_dl
            exceptions_dl = _exceptions_bits_cn(
                cn, subset, pos_idx, neg_idx, verbosity=verbosity
            )
            subset_dls.append(theory_dl + exceptions_dl)
            if verbosity >= 5:
                print(f"subset 0-{i} | dl: {rnd(subset_dls[i])}")

        # Build up the best Ruleset and calculate the mdl
        mdl_ruleset = Ruleset()
        for i, rule, in enumerate(ruleset.rules):
            if (
                i == 0 or subset_dls[i] <= subset_dls[i - 1]
            ):  # Rule i does not worsen the dl
                mdl_ruleset.add(rule)
        if verbosity >= 5:
            print(f"subset dls: {[(i,rnd(dl)) for i,dl in enumerate(subset_dls)]}")
            print(f"best potential ruleset: {mdl_ruleset}")
        mdl = _rs_total_bits_cn(
            cn,
            mdl_ruleset,
            possible_conds,
            pos_idx,
            neg_idx,
            bestsubset_dl=False,
            verbosity=0,
        )  # About to print value below
        if verbosity >= 5:
            print(f"best potential dl was {rnd(mdl)}")
            print()
        if not ret_bestsubset:
            return mdl
        else:
            return (mdl_ruleset, mdl)
