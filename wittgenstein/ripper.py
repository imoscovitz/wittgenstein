"""
Implementation of the RIPPERk algorithm for growing classification rulesets.
See https://www.let.rug.nl/nerbonne/teach/learning/cohen95fast.pdf
"""

# Author: Ilan Moscovitz <ilan.moscovitz@gmail.com>
# License: MIT

import pandas as pd
import copy
import math
import numpy as np

from wittgenstein import base, base_functions
from .base import Cond, Rule, Ruleset, bin_df, rnd
from .base_functions import score_accuracy

from .catnap import CatNap


class RIPPER:
    """ Class for generating ruleset classification models.
        See Cohen (1995): https://www.let.rug.nl/nerbonne/teach/learning/cohen95fast.pdf
    """

    def __init__(
        self,
        k=2,
        prune_size=0.33,
        dl_allowance=64,
        max_rules=None,
        max_rule_conds=None,
        max_total_conds=None,
        verbosity=0,
    ):
        """ Creates a new RIPPER object.

            args:
                k (optional):            number of RIPPERk optimization iterations (default=2)
                prune_size (optional):   proportion of training set to be used for pruning (default=.33)
                dl_allowance (optional): terminate Ruleset grow phase early if a Ruleset description length is encountered
                                            that is more than this amount above the lowest description length so far encountered.
                                            (default=64 bits)
                verbosity (optional):    output information about the training process (default=0)
                                           1: Show results of each major phase
                                           2: Show Ruleset grow/optimization steps
                                           3: Show Ruleset grow/optimization calculations
                                           4: Show Rule grow/prune steps
                                           5: Show Rule grow/prune calculations
        """
        self.POSSIBLE_HYPERPARAM_NAMES = {
            "prune_size",
            "dl_allowance",
            "k",
            "max_rules",
            "max_rule_conds",
            "max_total_conds",
        }

        self.prune_size = prune_size
        self.dl_allowance = dl_allowance
        self.k = k

        self.max_rules = max_rules
        self.max_rule_conds = max_rule_conds
        self.max_total_conds = max_total_conds

        self.verbosity = verbosity

    def __str__(self):
        """ Returns string representation of a RIPPER object. """
        params = str(self.get_params()) + ">"
        params = (
            params.replace(": ", "=")
            .replace("'", "")
            .replace("{", "(")
            .replace("}", ")")
        )
        return f"<RIPPER{params}"

    __repr__ = __str__

    def out_model(self):
        """ Prints trained Ruleset model line-by-line: V represents 'or'; ^ represents 'and'. """
        if hasattr(self, "ruleset_"):
            self.ruleset_.out_pretty()
        else:
            print("no model fitted")

    def fit(
        self,
        trainset,
        y=None,
        columns=None,
        class_feat=None,
        pos_class=None,
        feature_names=None,
        n_discretize_bins=10,
        max_rules=None,
        max_rule_conds=None,
        max_total_conds=None,
        cn_optimize=True,
        random_state=None,
    ):
        """ Fit a Ruleset model.

        args:
            trainset: categorical training dataset <pandas DataFrame, numpy array, or other Python iterable>
            y: class labels corresponding to trainset rows. Parameter y or class_feat (see next) must be provided.
            class_feat: column name of class feature (Use if class feature is still in df.)

            columns (optional): if trainset inputted as non-DataFrame iterable (such as numpy array), sets feature names. If None, feature for iterables will be set to column indices. (default=None)
            pos_class (optional): name of positive class. If not provided, defaults to class of first training example.
            n_discretize_bins (optional): Fit apparent numeric attributes into a maximum of n_discretize_bins discrete bins, inclusive on upper part of range.
                                          Setting to smaller values can improve training speed.
                                          Pass None to disable auto-discretization and treat values as categorical. (default=10)
            random_state: (optional) random state to allow for repeatable results

            options to stop early. Intended for improving model interpretability or limiting training time on noisy datasets. Not specifically intended for use as a hyperparameter, since pruning already occurs during training, though it is certainly possible that tuning could improve model performance.
            Note: small max_rule_conds may result in the k-optimizations stage making fewer changes.
            max_rules (optional): maximum number of rules. default=None
            max_rule_conds (optional): maximum number of conds per rule. default=None
            max_total_conds (optional): maximum number of total conds in entire ruleset. default=None
        """

        ################
        # Stage 0: Setup
        ################

        # Record any hyperparameters
        self.max_rules = max_rules
        self.max_rule_conds = max_rule_conds
        self.max_total_conds = max_total_conds

        # Preprocess training data
        (
            df,
            self.class_feat,
            self.pos_class,
            self.trainset_features_,
            self.bin_transformer_,
        ) = base_functions.preprocess_training_data(
            trainset,
            y=y,
            class_feat=class_feat,
            pos_class=pos_class,
            user_requested_feature_names=feature_names,
            bin_transformer_=None,
            n_discretize_bins=n_discretize_bins,
            verbosity=self.verbosity,
        )

        # CatNap optimization:
        if cn_optimize:
            self.cn = CatNap(
                df,
                feat_subset=None,
                cond_subset=None,
                class_feat=self.class_feat,
                pos_class=None,
            )

        # Split df into pos, neg classes
        if cn_optimize:
            pos_idx, neg_idx = self.cn.pos_idx_neg_idx(
                df, self.class_feat, self.pos_class
            )
        else:
            pos_df, neg_df = base_functions.pos_neg_split(
                df, self.class_feat, self.pos_class
            )
            pos_df = pos_df.drop(self.class_feat, axis=1)
            neg_df = neg_df.drop(self.class_feat, axis=1)

        # Collect possible conds
        self._set_possible_conds(df)

        ###############################
        # Stage 1: Grow initial Ruleset
        ###############################

        self.ruleset_ = Ruleset()
        if cn_optimize:
            self.ruleset_ = self._grow_ruleset_cn(
                pos_idx,
                neg_idx,
                prune_size=self.prune_size,
                dl_allowance=self.dl_allowance,
                max_rules=max_rules,
                max_rule_conds=max_rule_conds,
                max_total_conds=max_total_conds,
                random_state=random_state,
            )
        else:
            self.ruleset_ = self._grow_ruleset(
                pos_df,
                neg_df,
                prune_size=self.prune_size,
                dl_allowance=self.dl_allowance,
                max_rules=max_rules,
                max_rule_conds=max_rule_conds,
                max_total_conds=max_total_conds,
                random_state=random_state,
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
            iter_random_state = random_state + 100 if random_state is not None else None
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
                max_rules=max_rules,
                max_rule_conds=max_rule_conds,
                max_total_conds=max_total_conds,
                random_state=random_state,
            )
        else:
            self._cover_remaining_positives(
                df,
                max_rules=max_rules,
                max_rule_conds=max_rule_conds,
                max_total_conds=max_total_conds,
                random_state=random_state,
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

        # Fit probas
        self.recalibrate_proba(
            df, min_samples=None, require_min_samples=False, discretize=False
        )

        # Cleanup
        if cn_optimize:
            del self.cn

    def predict(self, X, give_reasons=False, feature_names=None):
        """ Predict classes of data using a RIPPER-fit model.

            args:
                X: examples to make predictions on

                give_reasons (optional) <bool>: whether to provide reasons for each prediction made.

            returns:
                list of <bool> values corresponding to examples. True indicates positive predicted class; False non-positive class.

                If give_reasons is True, returns a tuple that contains the above list of predictions
                    and a list of the corresponding reasons for each prediction;
                    for each positive prediction, gives a list of all the covering Rules, for negative predictions, an empty list.
        """

        if not hasattr(self, "ruleset_"):
            raise AttributeError(
                "You should fit a RIPPER object with .fit method before making predictions with it."
            )

        X_df = base_functions.preprocess_prediction_data(
            X,
            trainset_features=self.trainset_features_,
            model_selected_features=self.selected_features_,
            bin_transformer_=self.bin_transformer_,
            user_requested_feature_names=feature_names,
            verbosity=0,
        )

        return self.ruleset_.predict(X_df, give_reasons=give_reasons)

    def score(self, X, y, score_function=score_accuracy):
        """ Test performance of a RIPPER-fit model.

            X: 2-D independent attributes values
            y: 1-D corresponding target values

            score_function (optional): function that takes two parameters: actuals <iterable<bool>>, predictions <iterable<bool>>,
                                       containing class values. (default=accuracy)
                                       this parameter is intended to be compatible with sklearn's scoring functions:
                                       https://scikit-learn.org/stable/modules/model_evaluation.html#classification-metrics
        """

        predictions = self.predict(X)
        actuals = [yi == self.pos_class for yi in base_functions.aslist(y)]
        return score_function(actuals, predictions)

    def predict_proba(self, X_df, give_reasons=False, ret_n=False, min_samples=1):
        # Drop class feature if user forgot to:
        df = (
            X_df
            if self.class_feat not in X_df.columns
            else X_df.drop(self.class_feat, axis=1)
        )
        return self.ruleset_.predict_proba(
            df,
            give_reasons=give_reasons,
            ret_n=ret_n,
            min_samples=min_samples,
            discretize=True,
            bin_transformer=self.bin_transformer_,
        )

    def recalibrate_proba(
        self, Xy, min_samples=20, require_min_samples=True, discretize=True
    ):
        """ Recalibrate a classifier's probability estimations using unseen labeled data. May improve .predict_proba generalizability.
            Does not affect the underlying model or which predictions it makes -- only probability estimates. Use params min_samples and require_min_samples to select desired behavior.

            Note1: RunTimeWarning will occur as a reminder when min_samples and require_min_samples params might result in unintended effects.
            Note2: It is possible recalibrating could result in some positive .predict predictions with <0.5 .predict_proba positive probability.

            Xy: labeled data

            min_samples <int> (optional): required minimum number of samples per Rule
                                          default=10. set None to ignore min sampling requirement so long as at least one sample exists.
            require_min_samples <bool> (optional): True: halt (with warning) in case min_samples not achieved for all Rules
                                                   False: warn, but still replace Rules that have enough samples
            discretize <bool> (optional): if the classifier has already fit a discretization, automatically discretize recalibrate_proba's training data
                                          default=True
        """

        # Recalibrate
        base_functions.recalibrate_proba(
            self.ruleset_,
            Xy_df=Xy,
            class_feat=self.class_feat,
            pos_class=self.pos_class,
            min_samples=min_samples,
            require_min_samples=require_min_samples,
            discretize=discretize,
            bin_transformer=self.bin_transformer_,
        )

    def _set_theory_dl_lookup(self, df, size=15, verbosity=0):
        """ Precalculate rule theory dls for various-sized rules. """

        self.dl_dict = {}

        temp = Ruleset()
        temp._set_possible_conds(df, df)
        possible_conds = temp.possible_conds

        for n in range(1, size + 1):
            rule = Rule([Cond("_", "_")] * n)
            dl = _r_theory_bits(
                rule, possible_conds, bits_dict=None, verbosity=verbosity
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
        initial_ruleset=None,
        max_rules=None,
        max_rule_conds=None,
        max_total_conds=None,
        random_state=None,
    ):
        """ Grow a Ruleset with pruning. """
        pos_remaining = pos_df.copy()
        neg_remaining = neg_df.copy()

        if initial_ruleset is None:
            ruleset = Ruleset()
            ruleset._set_possible_conds(pos_df, neg_df)
        else:
            ruleset = copy.deepcopy(initial_ruleset)

        ruleset_dl = None
        mdl = None  # Minimum encountered description length (in bits)
        dl_diff = 0
        if self.verbosity >= 2:
            print("growing ruleset...")
            print()
        while len(pos_remaining) > 0 and dl_diff <= self.dl_allowance:

            # If applicable, check for user-specified early stopping
            if (max_rules is not None and len(ruleset.rules) >= max_rules) or (
                max_total_conds is not None and ruleset.count_conds() >= max_total_conds
            ):
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

        # If applicable, trim total conds
        ruleset.trim_conds(max_total_conds=max_total_conds)

        return ruleset

    def _grow_ruleset_cn(
        self,
        pos_idx,
        neg_idx,
        prune_size,
        dl_allowance,
        initial_ruleset=None,
        max_rules=None,
        max_rule_conds=None,
        max_total_conds=None,
        random_state=None,
    ):
        """ Grow a Ruleset with pruning. """

        # Initialize new Ruleset
        if initial_ruleset is None:
            ruleset = Ruleset()
            ruleset.possible_conds = self.possible_conds  # cn.conds
        else:
            ruleset = copy.deepcopy(initial_ruleset)

        pos_remaining_idx = pos_idx
        neg_remaining_idx = neg_idx
        ruleset_dl = None
        mdl = None  # Minimum encountered description length (in bits)
        dl_diff = 0
        if self.verbosity >= 2:
            print("growing ruleset...")
            print()
        while len(pos_remaining_idx) > 0 and dl_diff <= self.dl_allowance:

            # If applicable, check for user-specified early stopping
            if (max_rules is not None and len(ruleset.rules) >= max_rules) or (
                max_total_conds is not None and ruleset.count_conds() >= max_total_conds
            ):
                break

            # Grow-prune split remaining uncovered examples
            pos_growset_idx, pos_pruneset_idx = base_functions.set_shuffled_split(
                pos_remaining_idx, (1 - prune_size), random_state=random_state
            )
            neg_growset_idx, neg_pruneset_idx = base_functions.set_shuffled_split(
                neg_remaining_idx, (1 - prune_size), random_state=random_state
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

        # If applicable, trim total conds
        ruleset.trim_conds(max_total_conds=max_total_conds)

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
        """ Optimization phase. """

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
        """ Optimization phase. """

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

    def _set_possible_conds(self, df):
        self.possible_conds = []
        for feat in df.columns.values:
            if feat != self.class_feat:
                for val in df[feat].unique():
                    self.possible_conds.append(Cond(feat, val))

    def _cover_remaining_positives(
        self,
        df,
        max_rules=None,
        max_rule_conds=None,
        max_total_conds=None,
        random_state=None,
    ):
        """ Stage 3: Post-optimization, cover any remaining uncovered positives. """
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
                initial_ruleset=self.ruleset_,
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
        """ Stage 3: Post-optimization, cover any remaining uncovered positives. """
        pos_remaining_idx, neg_remaining_idx = self.cn.pos_idx_neg_idx(
            df, self.class_feat, self.pos_class
        )

        if len(pos_remaining_idx) >= 1:
            if self.verbosity >= 2:
                print(f"{len(pos_remaining_idx)} pos left. Growing final rules...")
            newset = self._grow_ruleset_cn(
                pos_remaining_idx,
                neg_remaining_idx,
                initial_ruleset=self.ruleset_,
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
        return {
            param: self.__dict__.get(param) for param in self.POSSIBLE_HYPERPARAM_NAMES
        }

    def set_params(self, **parameters):
        for parameter, value in parameters.items():
            # if parameter in self.POSSIBLE_HYPERPARAM_NAMES:
            setattr(self, parameter, value)
        return self


#### HELPER Class #####


class RSStats:
    def __init__(self, ruleset=None):
        self.subset_dls = []
        self.ruleset = Ruleset() if ruleset is None else copy.deepcopy(ruleset)


class RulesetStats:
    # This class is not used in the current implementation but could come in handy for future optimization
    # by storing and retreiving calculations that may be repeated.
    # Haven't incorporated it because there are bigger fish to fry, optimization-wise.
    def __init__(self, ruleset=None):
        self.subset_dls = []
        self.ruleset = Ruleset() if ruleset is None else ruleset
        self.dl = 0

    def update(self, ruleset, possible_conds, pos_df, neg_df, verbosity=0):

        # Find first mismatching rule index
        # If there is no mismatch, return
        # If there is a mismatch, update self.ruleset and wipe subsequent dls
        index = 0
        while (
            index < len(self.ruleset)
            and index < len(ruleset)
            and self.ruleset[index] == ruleset[index]
        ):
            index += 1
        if index == len(ruleset) and index == len(self.ruleset):
            if verbosity >= 4:
                print(f"not updating stats -- no ruleset change found")
            return
        if verbosity >= 4:
            print(f"updating stats from index {index}")
        self.ruleset.rules[index:] = ruleset[index:]
        self.subset_dls = self.subset_dls[:index]

        # Beginning with index, update subset dls
        for i in range(index, len(ruleset)):
            rule = ruleset[i]
            subset = Ruleset(ruleset.rules[: i + 1])
            subset_dl = _rs_total_bits(
                subset, possible_conds, pos_df, neg_df, verbosity=verbosity
            )
            self.subset_dls.append(subset_dl)

        self.dl = self.subset_dls[-1]

    def dl_change(self, index):
        return self.subset_dls[index] - self.subset_dls[index - 1]

    def potential_dl_stats(
        self,
        possible_conds,
        pos_df,
        neg_df,
        ret_ruleset=True,
        ret_dl=False,
        verbosity=0,
    ):
        if not any((ret_ruleset, ret_dl)):
            raise ValueError(
                "method dl_pruned_ruleset called without any return values specified"
            )

        tempStats = copy.deepcopy(self)
        i = len(tempStats.ruleset) - 1
        while i > 0:
            if tempStats.dl_change(i) > 0:
                if verbosity >= 4:
                    print(f"rule {i} raised dl -- removing")
                tempStats.update(
                    Ruleset(tempStats.ruleset[:i] + tempStats.ruleset[i + 1 :]),
                    possible_conds,
                    pos_df,
                    neg_df,
                )
                if verbosity >= 4:
                    print(f"new ruleset is {tempStats.ruleset}")
            i -= 1

        return tempStats


###################################
##### RIPPER-specific Metrics #####
###################################


def _RIPPER_growphase_prune_metric(rule, pos_pruneset, neg_pruneset):
    """ RIPPER/IREP* prune metric.
        Returns the prune value of a candidate Rule.

        Cohen's formula is (p-n) / (p+n).
        Unclear from the paper how they handle divzero (where p+n=0), so I Laplaced it.
        Weka's solution was to modify the formula to (p+1)/(p+n+2), but the (non-NaN) values I got appeared closer to those of the original formula.
    """
    # I imagine Weka's is 1/2 because that's closer to a 50-50 class distribution?
    p = rule.num_covered(pos_pruneset)
    n = rule.num_covered(neg_pruneset)
    return (p - n + 1) / (p + n + 1)


def _RIPPER_growphase_prune_metric_cn(cn, rule, pos_pruneset_idx, neg_pruneset_idx):
    """ RIPPER/IREP* prune metric.
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
    """ Returns description length (in bits) for a single Rule. """

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
    """ Returns theory description length (in bits) for a Ruleset. """

    # if type(ruleset) != Ruleset:
    #    raise TypeError(f'param ruleset in _rs_theory_bits should be type Ruleset')
    """ Returns sum of theory bits for each Rule in ruleset """
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
    """ Returns description length (in bits) for exceptions to a Ruleset's coverage. """

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
    """ Returns description length (in bits) for exceptions to a Ruleset's coverage. """

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
    """ Returns total description length (in bits) of ruleset -- the sum of its theory dl and exceptions dl.

        bestsubset_dl (optional, <bool>): whether to return estimated minimum possible dl were all rules that increase dl to be removed
        ret_bestsubset (optional): whether to return the best subset that was found. Return format will be (<Ruleset>,dl).
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
    """ Returns total description length (in bits) of ruleset -- the sum of its theory dl and exceptions dl.

        bestsubset_dl (optional, <bool>): whether to return estimated minimum possible dl were all rules that increase dl to be removed
        ret_bestsubset (optional): whether to return the best subset that was found. Return format will be (<Ruleset>,dl).
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
