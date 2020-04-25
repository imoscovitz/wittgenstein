import numpy as np
import pandas as pd
import pytest

from wittgenstein.base import Cond, Rule, Ruleset
from wittgenstein.irep import IREP
from wittgenstein.ripper import RIPPER
from wittgenstein import preprocess

DF = pd.read_csv("house-votes-84.csv")
CLASS_FEAT = "Party"
DEFAULT_CLASS_FEAT = "Class"
POS_CLASS = "democrat"
SPLIT_SIZE = 0.6

X_DF = DF.drop(CLASS_FEAT, axis=1)
Y_DF = DF[CLASS_FEAT]

XY_NP = DF.values
X_NP = X_DF.values
Y_NP = Y_DF.values
NP_CLASS_FEAT = 0

irep = IREP(random_state=42)
rip = RIPPER(random_state=42)

IREP_RULESET_42 = Ruleset(
    [
        Rule([Cond("physician-fee-freeze", "n")]),
        Rule(
            [Cond("synfuels-corporation-cutback", "y"), Cond("education-spending", "n")]
        ),
    ]
)

RIP_RULESET_42 = Ruleset(
    [
        Rule([Cond("physician-fee-freeze", "n")]),
        Rule(
            [Cond("synfuels-corporation-cutback", "y"), Cond("education-spending", "n")]
        ),
        Rule(
            [
                Cond("synfuels-corporation-cutback", "y"),
                Cond("adoption-of-the-budget-resolution", "y"),
            ]
        ),
    ]
)

FEAT2IDX = {col: i for i, col in enumerate(DF.columns)}


def feat_to_num_rs(ruleset):
    new_ruleset = Ruleset()
    for rule in ruleset.rules:
        new_rule = Rule()
        for cond in rule.conds:
            feat = cond.feature
            val = cond.val
            new_cond = Cond(FEAT2IDX[feat], val)
            new_rule.conds.append(new_cond)
        new_ruleset.rules.append(new_rule)
    return new_ruleset


def test_fit_Xy_df():
    irep.fit(DF, class_feat=CLASS_FEAT, pos_class=POS_CLASS)
    assert irep.ruleset_ == IREP_RULESET_42

    rip.fit(DF, class_feat=CLASS_FEAT, pos_class=POS_CLASS)
    assert rip.ruleset_ == RIP_RULESET_42


def test_fit_X_y_df():
    irep.fit(X_DF, y=Y_DF, class_feat=CLASS_FEAT, pos_class=POS_CLASS)
    assert irep.ruleset_ == IREP_RULESET_42

    rip.fit(X_DF, y=Y_DF, class_feat=CLASS_FEAT, pos_class=POS_CLASS)
    assert rip.ruleset_ == RIP_RULESET_42


def test_fit_X_y_np():
    irep.fit(X_DF, y=Y_DF, pos_class=POS_CLASS)
    assert irep.ruleset_ == IREP_RULESET_42

    rip.fit(X_DF, y=Y_DF, pos_class=POS_CLASS)
    assert rip.ruleset_ == RIP_RULESET_42


def test_fit_Xy_np():
    irep.fit(XY_NP, y=None, class_feat=NP_CLASS_FEAT, pos_class=POS_CLASS)
    assert irep.ruleset_ == feat_to_num_rs(IREP_RULESET_42)

    rip.fit(XY_NP, y=None, class_feat=NP_CLASS_FEAT, pos_class=POS_CLASS)
    assert rip.ruleset_ == feat_to_num_rs(RIP_RULESET_42)


def test_fit_XY_rename_columns():

    # With xy
    irep.fit(
        XY_NP,
        y=None,
        class_feat=CLASS_FEAT,
        pos_class=POS_CLASS,
        feature_names=DF.columns,
    )
    assert irep.ruleset_ == IREP_RULESET_42

    rip.fit(
        XY_NP,
        y=None,
        class_feat=CLASS_FEAT,
        pos_class=POS_CLASS,
        feature_names=DF.columns,
    )
    assert rip.ruleset_ == RIP_RULESET_42

    # With x_y
    irep.fit(
        X_NP,
        y=Y_NP,
        class_feat=CLASS_FEAT,
        pos_class=POS_CLASS,
        feature_names=DF.drop(CLASS_FEAT, axis=1).columns,
    )
    assert irep.ruleset_ == IREP_RULESET_42

    rip.fit(
        X_NP,
        y=Y_NP,
        class_feat=CLASS_FEAT,
        pos_class=POS_CLASS,
        feature_names=DF.drop(CLASS_FEAT, axis=1).columns,
    )
    assert rip.ruleset_ == RIP_RULESET_42


CREDIT_DF = pd.read_csv("credit.csv")
CREDIT_CLASS_FEAT = "Class"
CREDIT_DEFAULT_CLASS_FEAT = "Class"
CREDIT_POS_CLASS = "+"
CREDIT_SPLIT_SIZE = 0.6
CREDIT_IREP_RULESET_42 = Ruleset([Rule([Cond("A9", "t")])])


def test_fit_numeric_dataset():
    irep.fit(
        CREDIT_DF, class_feat=CREDIT_CLASS_FEAT, pos_class=CREDIT_POS_CLASS,
    )
    assert irep.ruleset_ == CREDIT_IREP_RULESET_42
