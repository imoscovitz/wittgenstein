import numpy as np
import pandas as pd
import pytest

from wittgenstein import IREP, RIPPER
from wittgenstein.base_functions import df_shuffled_split
from wittgenstein.base import Cond, Rule

DF = pd.read_csv("credit.csv")
CLASS_FEAT = "Class"
DEFAULT_CLASS_FEAT = "Class"
CREDIT_POS_CLASS = "+"
SPLIT_SIZE = 0.6

X_DF = DF.drop(CLASS_FEAT, axis=1)
Y_DF = DF[CLASS_FEAT]

XY_NP = DF.values
X_NP = X_DF.values
Y_NP = Y_DF.values
NP_CLASS_FEAT = -1

######

irep = IREP(random_state=42)
rip = RIPPER(random_state=42)

#####

train, test = df_shuffled_split(DF, random_state=42)
test_x, test_y = test.drop(CLASS_FEAT, axis=1), test[CLASS_FEAT]

irep.fit(train, class_feat=CLASS_FEAT, pos_class=CREDIT_POS_CLASS)
rip.fit(train, class_feat=CLASS_FEAT, pos_class=CREDIT_POS_CLASS)

#####


def test_predict():
    irep_preds = irep.predict(test_x)

    assert all(p in (True, False) for p in irep_preds)
    assert not all(p == True for p in irep_preds)
    assert not all(p == False for p in irep_preds)
    assert sum(irep_preds) == 128

    rip_preds = rip.predict(test_x)
    assert all(p in (True, False) for p in rip_preds)
    assert not all(p == True for p in rip_preds)
    assert not all(p == False for p in rip_preds)
    assert sum(rip_preds) == 99


def test_predict_give_reasons():
    def reason_iff_pos(pred, reason):
        if list(reason) != []:
            return pred
        else:
            return not pred

    irep_preds = irep.predict(test_x, give_reasons=True)
    assert all(reason_iff_pos(p, r) for p, r in zip(*irep_preds))
    rip_preds = rip.predict(test_x, give_reasons=True)
    assert all(reason_iff_pos(p, r) for p, r in zip(*rip_preds))


def test_predict_proba():
    # The vast majority of predictions and predict probas should match
    def pred_proba_match(pred, proba):
        if pred:
            return proba[1] >= proba[0]
        else:
            return proba[0] >= proba[1]

    irep_preds = irep.predict(test_x)
    irep_probas = irep.predict_proba(test_x)
    assert (
        np.mean(
            [
                pred_proba_match(pred, proba)
                for pred, proba in zip(irep_preds, irep_probas)
            ]
        )
        >= 0.80
    )
    rip_preds = rip.predict(test_x)
    rip_probas = irep.predict_proba(test_x)
    assert (
        np.mean(
            [
                pred_proba_match(pred, proba)
                for pred, proba in zip(rip_preds, irep_probas)
            ]
        )
        >= 0.80
    )


def test_score():
    assert irep.score(test_x, test_y) == pytest.approx(0.8382978723404255)
    assert rip.score(test_x, test_y) == pytest.approx(0.825531914893617)


def test_verbosity():
    irep_v5 = IREP(random_state=42, verbosity=5)
    irep_v5.fit(train, class_feat=CLASS_FEAT, pos_class=CREDIT_POS_CLASS)
    assert irep_v5.score(test_x, test_y) == pytest.approx(irep.score(test_x, test_y))

    rip_v5 = RIPPER(random_state=42, verbosity=5)
    rip_v5.fit(train, class_feat=CLASS_FEAT, pos_class=CREDIT_POS_CLASS)
    assert rip_v5.score(test_x, test_y) == pytest.approx(rip.score(test_x, test_y))


def test_missing_non_selected_features():
    missing_feat_df = test_x.copy()
    missing_feat_df.drop('A1', axis=1, inplace=True)

    irep_preds = irep.predict(missing_feat_df)
    assert all(p in (True, False) for p in irep_preds)
    assert not all(p == True for p in irep_preds)
    assert not all(p == False for p in irep_preds)
    assert sum(irep_preds) == 128

    rip_preds = rip.predict(missing_feat_df)
    assert all(p in (True, False) for p in rip_preds)
    assert not all(p == True for p in rip_preds)
    assert not all(p == False for p in rip_preds)
    assert sum(rip_preds) == 99


def test_missing_selected_features_raise_eror():
    missing_feat_df = test_x.copy()
    missing_feat_df.drop('A9', axis=1, inplace=True)

    try:
        irep_preds = irep.predict(missing_feat_df)
        exception_raised = False
    except:
        exception_raised = True
    assert exception_raised

    try:
        rip_preds = rip.predict(missing_feat_df)
        exception_raised = False
    except:
        exception_raised = True
    assert exception_raised
