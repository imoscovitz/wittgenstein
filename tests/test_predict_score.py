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

IREP_PREDICT_42 = [
    True,
    True,
    False,
    True,
    False,
    False,
    True,
    False,
    False,
    True,
    False,
    False,
    False,
    False,
    False,
    False,
    True,
    True,
    False,
    False
]
RIP_PREDICT_42 = [
    True,
    True,
    False,
    True,
    False,
    False,
    True,
    False,
    False,
    True,
    False,
    False,
    False,
    False,
    False,
    False,
    False,
    True,
    False,
    False
]
IREP_PREDICT_GIVE_REASONS_42 = ([
    True,
    True,
    False,
    True,
    False,
    False,
    True,
    False,
    False,
    True,
    False,
    False,
    False,
    False,
    False,
    False,
    True,
    True,
    False,
    False],
    [[Rule([Cond('A9','t')])],
    [Rule([Cond('A9','t')])],
    [],
    [Rule([Cond('A9','t')])],
    [],
    [],
    [Rule([Cond('A9','t')])],
    [],
    [],
    [Rule([Cond('A9','t')])],
    [],
    [],
    [],
    [],
    [],
    [],
    [Rule([Cond('A9','t')])],
    [Rule([Cond('A9','t')])],
    [],
    []
])
RIP_PREDICT_GIVE_REASONS_42 = ([
    True,
    True,
    False,
    True,
    False,
    False,
    True,
    False,
    False,
    True],
    [[Rule([Cond('A9','t'), Cond('A10','t'),Cond('A11','8-19')]),
        Rule([Cond('A9','t'), Cond('A6','q')])],
    [Rule([Cond('A9','t'),Cond('A10','t'),Cond('A8','0.5-0.88')]),
        Rule([Cond('A9','t'),Cond('A15','100-321')])],
    [],
    [Rule([Cond('A9','t'),Cond('A15','1058-4700')])],
    [],
    [],
    [Rule([Cond('A9','t'),Cond('A14','0')])],
    [],
    [],
    [Rule([Cond('A9','t'),Cond('A10','t'), Cond('A11','8-19')])]
])

IREP_PREDICT_PROBA_42 = np.array([
    [0.20920502, 0.79079498],
    [0.20920502, 0.79079498],
    [0.93055556, 0.06944444],
    [0.20920502, 0.79079498],
    [0.93055556, 0.06944444],
    [0.93055556, 0.06944444],
    [0.20920502, 0.79079498],
    [0.93055556, 0.06944444],
    [0.93055556, 0.06944444],
    [0.20920502, 0.79079498],
    [0.93055556, 0.06944444],
    [0.93055556, 0.06944444],
    [0.93055556, 0.06944444],
    [0.93055556, 0.06944444],
    [0.93055556, 0.06944444],
    [0.93055556, 0.06944444],
    [0.20920502, 0.79079498],
    [0.20920502, 0.79079498],
    [0.93055556, 0.06944444],
    [0.93055556, 0.06944444]
])
RIP_PREDICT_PROBA_42 = np.array([
    [0.12121212, 0.87878788],
    [0.13157895, 0.86842105],
    [0.84946237, 0.15053763],
    [0.02564103, 0.97435897],
    [0.84946237, 0.15053763],
    [0.84946237, 0.15053763],
    [0.10909091, 0.89090909],
    [0.84946237, 0.15053763],
    [0.84946237, 0.15053763],
    [0.        , 0.        ],
    [0.84946237, 0.15053763],
    [0.84946237, 0.15053763],
    [0.84946237, 0.15053763],
    [0.84946237, 0.15053763],
    [0.84946237, 0.15053763],
    [0.84946237, 0.15053763],
    [0.84946237, 0.15053763],
    [0.06730769, 0.93269231],
    [0.84946237, 0.15053763],
    [0.84946237, 0.15053763]
])

def test_predict():
    assert irep.predict(test_x[:20]) == IREP_PREDICT_42
    assert rip.predict(test_x[:20]) == RIP_PREDICT_42

def test_predict_give_reasons():
    assert irep.predict(test_x[:20], give_reasons=True) == IREP_PREDICT_GIVE_REASONS_42
    assert rip.predict(test_x[:10], give_reasons=True) == RIP_PREDICT_GIVE_REASONS_42

def test_predict_proba():
    np.testing.assert_array_almost_equal(irep.predict_proba(test_x[:20]), IREP_PREDICT_PROBA_42)
    np.testing.assert_array_almost_equal(rip.predict_proba(test_x[:20]), RIP_PREDICT_PROBA_42)

def test_score():
    assert(irep.score(test_x, test_y) == pytest.approx(0.851063829787234))
    assert(rip.score(test_x, test_y) == pytest.approx(0.8468085106382979))
