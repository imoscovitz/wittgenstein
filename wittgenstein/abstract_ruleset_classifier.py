# Author: Ilan Moscovitz <ilan.moscovitz@gmail.com>
# License: MIT

from abc import ABC, abstractmethod
from copy import deepcopy
import numpy as np

import pandas as pd

from wittgenstein.check import _check_is_model_fit, _warn
from wittgenstein.base import (
    Rule,
    Ruleset,
    asrule,
    asruleset,
    cond_fromstr,
    ruleset_fromstr,
)
import wittgenstein.base_functions as base_functions
import wittgenstein.preprocess as preprocess
from wittgenstein.preprocess import BinTransformer, _upgrade_bin_transformer_ifdepr


class AbstractRulesetClassifier(ABC):
    def __init__(
        self,
        algorithm_name,
        prune_size=0.33,
        n_discretize_bins=10,
        max_rules=None,
        max_rule_conds=None,
        max_total_conds=None,
        random_state=None,
        verbosity=0,
    ):
        super().__init__()

        self.VALID_HYPERPARAMETERS = {
            "prune_size",
            "n_discretize_bins",
            "max_rules",
            "max_rule_conds",
            "max_total_conds",
            "random_state",
            "verbosity",
        }
        self.algorithm_name = algorithm_name
        self.prune_size = prune_size
        self.n_discretize_bins = n_discretize_bins
        self.max_rules = max_rules
        self.max_rule_conds = max_rule_conds
        self.max_total_conds = max_total_conds
        self.random_state = random_state
        self.verbosity = verbosity

        # This is to help keep sklearn ensemble happy should someone want use it
        self._estimator_type = "classifier"

    def __str__(self):
        """Return string representation."""
        isfit_str = (
            " with fit ruleset"
            if (hasattr(self, "ruleset_") and self.ruleset_ is not None)
            else ""
        )
        params = str(self.get_params())
        params = (
            params.replace(": ", "=")
            .replace("'", "")
            .replace("{", "(")
            .replace("}", ")")
        )
        return f"<{self.algorithm_name}{params}{isfit_str}>"

    __repr__ = __str__

    def out_model(self):
        """Print trained Ruleset model line-by-line: V represents 'or'; ^ represents 'and'."""
        if hasattr(self, "ruleset_"):
            self.ruleset_.out_pretty()
        else:
            print("no model fitted")

    def predict(self, X, give_reasons=False, feature_names=None):
        """Predict classes using a fit model.

        Parameters
        ----------
        X: DataFrame, numpy array, or other iterable
            Examples to make predictions on. All selected features of the model should be present.

        give_reasons : bool, default=False
            Whether to provide reasons for each prediction made.
        feature_names : list<str>, default=None
            Specify feature names for X to orient X's features with selected features.

        Returns
        -------
        list<bool>
            Predicted class labels for each row of X. True indicates positive predicted class, False negative class.

        Or, if give_reasons=True, returns

        tuple<list<bool>, <list<list<Rule>>>
            Tuple containing list of predictions and a list of the corresponding reasons for each prediction --
            for each positive prediction, a list of all the covering Rules, for negative predictions, an empty list.
        """

        _check_is_model_fit(self)

        self._ensure_has_bin_transformer()

        _upgrade_bin_transformer_ifdepr(self)

        # Preprocess prediction data
        preprocess_params = {
            "X": X,
            "class_feat": self.class_feat,
            "pos_class": self.pos_class,
            "bin_transformer_": self.bin_transformer_,
            "user_requested_feature_names": feature_names,
            "selected_features_": self.selected_features_,
            "trainset_features_": self.trainset_features_,
            "verbosity": self.verbosity,
        }

        X_df = preprocess.preprocess_prediction_data(preprocess_params)

        return self.ruleset_.predict(X_df, give_reasons=give_reasons)

    def score(self, X, y, score_function=base_functions.score_accuracy):

        """Score the performance of a fit model.

        X : DataFrame, numpy array, or other iterable
            Examples to score.
        y : Series, numpy array, or other iterable
            Class label actuals.

        score_function : function, default=score_accuracy
            Any scoring function that takes two parameters: actuals <iterable<bool>>,
            predictions <iterable<bool>>, where the elements represent class labels.
            This optional parameter is intended to be compatible with sklearn's scoring functions:
            https://scikit-learn.org/stable/modules/model_evaluation.html#classification-metrics
        """

        _check_is_model_fit(self)

        predictions = self.predict(X)
        actuals = [
            yi == self.pos_class for yi in preprocess._preprocess_y_score_data(y)
        ]
        return score_function(actuals, predictions)

    def predict_proba(self, X, give_reasons=False, feature_names=None):
        """Predict class probabilities of data using a fit model.

        Parameters
        ----------
            X: DataFrame, numpy array, or other iterable
                Examples to make predictions on. All selected features of the model should be present.

            give_reasons : bool, default=False
                Whether to provide reasons for each prediction made.
            feature_names : list<str>, default=None
                Specify feature names for X to orient X's features with selected features.

        Returns
        -------
        array
            Predicted class label probabilities for each row of X, ordered neg, pos.
            True indicates positive predicted class, False negative classes.

        or, if give_reasons=True:

        tuple< array, <list<list<Rule>> >
            Tuple containing array of predicted probabilities and a list of the corresponding reasons for each prediction--
            for each positive prediction, a list of all the covering Rules, for negative predictions, an empty list.
        """

        _check_is_model_fit(self)

        _upgrade_bin_transformer_ifdepr(self)

        # Preprocess prediction data
        preprocess_params = {
            "X": X,
            "class_feat": self.class_feat,
            "pos_class": self.pos_class,
            "bin_transformer_": self.bin_transformer_,
            "user_requested_feature_names": feature_names,
            "selected_features_": self.selected_features_,
            "trainset_features_": self.trainset_features_,
            "verbosity": self.verbosity,
        }

        X_df = preprocess.preprocess_prediction_data(preprocess_params)

        # This is to help keep sklearn ensemble happy should someone want use it
        # self.classes_ = np.array([0, 1])
        self.classes_ = np.array([f"not {self.pos_class}", self.pos_class])

        return self.ruleset_.predict_proba(X_df, give_reasons=give_reasons)

    def recalibrate_proba(
        self,
        X_or_Xy,
        y=None,
        feature_names=None,
        min_samples=20,
        require_min_samples=True,
        discretize=True,
    ):
        """Recalibrate a classifier's probability estimations using unseen labeled data. May improve .predict_proba generalizability.
        Does not affect the underlying model or which predictions it makes -- only probability estimates.
        Use params min_samples and require_min_samples to select desired behavior.

        Note1: RunTimeWarning will occur as a reminder when min_samples and require_min_samples params might result in unintended effects.
        Note2: It is possible recalibrating could result in some positive .predict predictions with <0.5 .predict_proba positive probability.

        Xy: labeled data

        min_samples : int, default=20
            Required minimum number of samples per Rule.
            Use None to ignore minimum sampling requirement so long as at least one sample exists.
        require_min_samples : bool, default=True
            True: halt (with warning) in case min_samples not achieved for all Rules
            False: warn, but still replace Rules that have enough samples
        discretize : bool, default=True
            If the classifier has already fit bins, automatically discretize recalibrate_proba's training data
        """

        # Preprocess training data
        preprocess_params = {
            "X_or_Xy": X_or_Xy,
            "y": y,
            "class_feat": self.class_feat,
            "pos_class": self.pos_class,
            "bin_transformer_": self.bin_transformer_ if discretize else None,
            "user_requested_feature_names": feature_names,
            "min_samples": min_samples,
            "require_min_samples": require_min_samples,
            "verbosity": self.verbosity,
        }

        df = preprocess._preprocess_recalibrate_proba_data(preprocess_params)

        # Recalibrate
        base_functions.recalibrate_proba(
            self.ruleset_,
            Xy_df=df,
            class_feat=self.class_feat,
            pos_class=self.pos_class,
            min_samples=min_samples,
            require_min_samples=require_min_samples,
        )

    def copy(self):
        """Return deep copy of classifier."""
        return deepcopy(self)

    def init_ruleset(self, ruleset, class_feat, pos_class):
        self.ruleset_ = self._ruleset_frommodel(ruleset)
        self.class_feat = class_feat
        self.pos_class = pos_class
        self.selected_features_ = self.ruleset_.get_selected_features()
        self.trainset_features_ = self.selected_features_

    def add_rule(self, new_rule):
        self.ruleset_.add(new_rule)

    def replace_rule_at(self, index, new_rule):
        self.ruleset_.replace(index, new_rule)

    def replace_rule(self, old_rule, new_rule):
        self.ruleset_.replace_rule(old_rule, new_rule)

    def remove_rule_at(self, index):
        self.ruleset_.remove(index)

    def remove_rule(self, old_rule):
        self.ruleset_.remove_rule(old_rule)

    def insert_rule_at(self, index, new_rule):
        self.ruleset_.insert(index, new_rule)

    def insert_rule(self, insert_before_rule, new_rule):
        self.ruleset_.insert_rule(insert_before_rule, new_rule)

    def get_params(self, deep=True):
        # parameter deep is a required artifact of sklearn compatability
        return {param: self.__dict__.get(param) for param in self.VALID_HYPERPARAMETERS}

    def set_params(self, **parameters):
        for parameter, value in parameters.items():
            setattr(self, parameter, value)
        return self

    def _ruleset_frommodel(self, model):
        """Return the ruleset from model, which may be a Ruleset, wittgenstein classifier, or str."""
        if not model:
            return Ruleset()
        elif type(model) == Ruleset:
            return deepcopy(model)
        elif type(model) == str:
            return deepcopy(asruleset(model))
        elif isinstance(model, AbstractRulesetClassifier):
            return deepcopy(model.ruleset_)
        else:
            raise AttributeError(
                f"Couldnt recognize type: {type(model)} of model: {model}. Model should be of type Ruleset, str defining a ruleset, or wittgenstein classifier."
            )

    def _ensure_has_bin_transformer(self):
        if hasattr(self, "bin_transformer_") and self.bin_transformer_ is not None:
            return
        else:
            self.bin_transformer_ = BinTransformer()
            self.bin_transformer_.construct_from_ruleset(self.ruleset_)

    def _set_deprecated_fit_params(self, params):
        """Handle setting parameters passed to .fit that should have been passed to __init__"""
        found_deprecated_params = []
        for param, value in params.items():
            if param in self.VALID_HYPERPARAMETERS:
                found_deprecated_params.append(param)
                setattr(self, param, value)
        if found_deprecated_params:
            _warn(
                f"In the future, you should assign these parameters when initializating classifier instead of during model fitting: {found_deprecated_params}",
                DeprecationWarning,
                "irep/ripper",
                "fit",
            )
