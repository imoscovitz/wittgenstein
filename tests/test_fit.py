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


CREDIT_DF = pd.read_csv("credit.csv")
CREDIT_CLASS_FEAT = "Class"
CREDIT_POS_CLASS = "+"


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
    irep = IREP(random_state=42)
    rip = RIPPER(random_state=42)

    irep.fit(DF, class_feat=CLASS_FEAT, pos_class=POS_CLASS)
    assert irep.ruleset_ == IREP_RULESET_42

    rip.fit(DF, class_feat=CLASS_FEAT, pos_class=POS_CLASS)
    assert rip.ruleset_ == RIP_RULESET_42


def test_fit_X_y_df():
    irep = IREP(random_state=42)
    rip = RIPPER(random_state=42)

    irep.fit(X_DF, y=Y_DF, class_feat=CLASS_FEAT, pos_class=POS_CLASS)
    assert irep.ruleset_ == IREP_RULESET_42

    rip.fit(X_DF, y=Y_DF, class_feat=CLASS_FEAT, pos_class=POS_CLASS)
    assert rip.ruleset_ == RIP_RULESET_42


def test_fit_X_y_np():
    irep = IREP(random_state=42)
    rip = RIPPER(random_state=42)

    irep.fit(X_DF, y=Y_DF, pos_class=POS_CLASS)
    assert irep.ruleset_ == IREP_RULESET_42

    rip.fit(X_DF, y=Y_DF, pos_class=POS_CLASS)
    assert rip.ruleset_ == RIP_RULESET_42


def test_fit_Xy_np():
    irep = IREP(random_state=42)
    rip = RIPPER(random_state=42)

    irep.fit(XY_NP, y=None, class_feat=NP_CLASS_FEAT, pos_class=POS_CLASS)
    assert irep.ruleset_ == feat_to_num_rs(IREP_RULESET_42)

    rip.fit(XY_NP, y=None, class_feat=NP_CLASS_FEAT, pos_class=POS_CLASS)
    assert rip.ruleset_ == feat_to_num_rs(RIP_RULESET_42)


def test_fit_XY_rename_columns():
    irep = IREP(random_state=42)
    rip = RIPPER(random_state=42)

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

    irep = IREP(random_state=42)
    rip = RIPPER(random_state=42)

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


def test_infer_pos_class():
    irep = IREP(random_state=42)
    rip = RIPPER(random_state=42)

    infer_df = DF.copy()
    infer_df[CLASS_FEAT] = infer_df[CLASS_FEAT].map(
        lambda x: 1 if x == "democrat" else 0
    )

    irep.fit(
        infer_df, class_feat=CLASS_FEAT,
    )
    assert irep.ruleset_ == IREP_RULESET_42
    rip.fit(
        infer_df, class_feat=CLASS_FEAT,
    )
    assert rip.ruleset_ == RIP_RULESET_42


def test_same_inputs_give_same_results():
    for random_state in range(3):
        irep_res = []
        rip_res = []

        irep = IREP(random_state=random_state)
        rip = RIPPER(random_state=random_state)
        irep.fit(DF, class_feat=CLASS_FEAT, pos_class=POS_CLASS)
        irep_res.append(irep.ruleset_)
        rip.fit(DF, class_feat=CLASS_FEAT, pos_class=POS_CLASS)
        rip_res.append(rip.ruleset_)

        irep = IREP(random_state=random_state)
        rip = RIPPER(random_state=random_state)
        irep.fit(X_DF, y=Y_DF, class_feat=CLASS_FEAT, pos_class=POS_CLASS)
        irep_res.append(irep.ruleset_)
        rip.fit(X_DF, y=Y_DF, class_feat=CLASS_FEAT, pos_class=POS_CLASS)
        rip_res.append(rip.ruleset_)

        irep = IREP(random_state=random_state)
        rip = RIPPER(random_state=random_state)
        irep.fit(X_DF, y=Y_DF, pos_class=POS_CLASS)
        irep_res.append(irep.ruleset_)
        rip.fit(X_DF, y=Y_DF, pos_class=POS_CLASS)
        rip_res.append(rip.ruleset_)

        irep = IREP(random_state=random_state)
        rip = RIPPER(random_state=random_state)
        irep.fit(
            XY_NP,
            y=None,
            class_feat=CLASS_FEAT,
            pos_class=POS_CLASS,
            feature_names=DF.columns,
        )
        irep_res.append(irep.ruleset_)
        rip.fit(
            XY_NP,
            y=None,
            class_feat=CLASS_FEAT,
            pos_class=POS_CLASS,
            feature_names=DF.columns,
        )
        rip_res.append(rip.ruleset_)
        irep = IREP(random_state=random_state)
        rip = RIPPER(random_state=random_state)
        irep.fit(
            X_NP,
            y=Y_NP,
            class_feat=CLASS_FEAT,
            pos_class=POS_CLASS,
            feature_names=DF.drop(CLASS_FEAT, axis=1).columns,
        )
        irep_res.append(irep.ruleset_)
        rip.fit(
            X_NP,
            y=Y_NP,
            class_feat=CLASS_FEAT,
            pos_class=POS_CLASS,
            feature_names=DF.drop(CLASS_FEAT, axis=1).columns,
        )
        rip_res.append(rip.ruleset_)

        assert all([res == irep_res[0] for res in irep_res])
        assert all([res == rip_res[0] for res in rip_res])


CREDIT_DF = pd.read_csv("credit.csv")
CREDIT_CLASS_FEAT = "Class"
CREDIT_DEFAULT_CLASS_FEAT = "Class"
CREDIT_POS_CLASS = "+"
CREDIT_SPLIT_SIZE = 0.6
CREDIT_IREP_RULESET_42 = Ruleset([Rule([Cond("A9", "t")])])


def test_fit_numeric_dataset():
    irep = IREP(random_state=42)
    rip = RIPPER(random_state=42)

    irep.fit(
        CREDIT_DF, class_feat=CREDIT_CLASS_FEAT, pos_class=CREDIT_POS_CLASS,
    )
    assert irep.ruleset_ == CREDIT_IREP_RULESET_42


def test_fit_boolean_dataset():
    irep = IREP(random_state=42)
    rip = RIPPER(random_state=42)

    def tobool(x):
        if x == "y":
            return 0
        elif x == "n":
            return 1
        else:
            return 2

    bool_df = DF.copy()
    for col in bool_df.drop("Party", axis=1).columns:
        bool_df[col] = bool_df[col].map(tobool)
    irep.fit(bool_df, class_feat="Party", pos_class="democrat")
    assert not (irep.ruleset_.isuniversal()) and not (irep.ruleset_.isnull())


def test_fit_discrete_dataset():

    irep = IREP(random_state=0, n_discretize_bins=6)
    rip = RIPPER(random_state=0, n_discretize_bins=6)

    discrete_df = CREDIT_DF.select_dtypes(float).applymap(lambda x: int(x % 10))
    discrete_df[CREDIT_CLASS_FEAT] = CREDIT_DF[CREDIT_CLASS_FEAT]

    irep.fit(discrete_df, class_feat=CREDIT_CLASS_FEAT, pos_class=CREDIT_POS_CLASS)
    print(f'this is the ruleset!', irep.ruleset_)
    assert (not irep.ruleset_.isuniversal()) and (not irep.ruleset_.isnull())
    rip.fit(discrete_df, class_feat=CREDIT_CLASS_FEAT, pos_class=CREDIT_POS_CLASS)
    print(rip.ruleset_)
    assert (not rip.ruleset_.isuniversal()) and (not rip.ruleset_.isnull())


def test_verbosity():
    irep_v5 = IREP(random_state=42, verbosity=5)
    rip_v5 = RIPPER(random_state=42, verbosity=5)

    irep_v5.fit(DF, class_feat=CLASS_FEAT, pos_class=POS_CLASS)
    print('irep_v5', irep_v5)#, irep_v5.ruleset_)
    assert irep_v5.ruleset_ == IREP_RULESET_42
    rip_v5.fit(DF, class_feat=CLASS_FEAT, pos_class=POS_CLASS)
    assert rip_v5.ruleset_ == RIP_RULESET_42


def test_random_state():

    # Party dataset
    irep_rulesets = []
    rip_rulesets = []
    for _ in range(3):
        irep = IREP(random_state=72)
        rip = RIPPER(random_state=72)
        irep.fit(DF, class_feat=CLASS_FEAT, pos_class=POS_CLASS)
        rip.fit(DF, class_feat=CLASS_FEAT, pos_class=POS_CLASS)
        irep_rulesets.append(irep.ruleset_)
        rip_rulesets.append(rip.ruleset_)
    assert all(rs == irep_rulesets[0] for rs in irep_rulesets)
    assert all(rs == rip_rulesets[0] for rs in rip_rulesets)

    # Credit dataset
    irep_rulesets = []
    rip_rulesets = []
    for _ in range(3):
        irep = IREP(random_state=72)
        rip = RIPPER(random_state=72)
        irep.fit(CREDIT_DF, class_feat=CREDIT_CLASS_FEAT, pos_class=CREDIT_POS_CLASS)
        rip.fit(CREDIT_DF, class_feat=CREDIT_CLASS_FEAT, pos_class=CREDIT_POS_CLASS)
        irep_rulesets.append(irep.ruleset_)
        rip_rulesets.append(rip.ruleset_)
    assert all(rs == irep_rulesets[0] for rs in irep_rulesets)
    assert all(rs == rip_rulesets[0] for rs in rip_rulesets)


def test_df_isnt_modified():
    # df shouldn't be affected by side-effects during model fitting
    old_df = pd.read_csv("credit.csv")
    df = old_df.copy()
    irep = IREP(random_state=42)
    irep.fit(CREDIT_DF, class_feat=CREDIT_CLASS_FEAT, pos_class=CREDIT_POS_CLASS)
    assert df.equals(old_df)

    old_df = pd.read_csv("credit.csv")
    df = old_df.copy()
    rip = RIPPER(random_state=42)
    rip.fit(CREDIT_DF, class_feat=CREDIT_CLASS_FEAT, pos_class=CREDIT_POS_CLASS)
    assert df.equals(old_df)
