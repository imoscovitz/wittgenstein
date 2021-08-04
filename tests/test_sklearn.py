import numpy as np
import pandas as pd
import pytest
from sklearn.ensemble import StackingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV, cross_val_score
from sklearn.metrics import precision_score, recall_score
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC

from wittgenstein import IREP, RIPPER
from wittgenstein.base_functions import df_shuffled_split

DF = pd.read_csv("credit.csv")
DF = DF.head(len(DF) // 3)
CLASS_FEAT = "Class"
POS_CLASS = "+"
DF[CLASS_FEAT] = DF[CLASS_FEAT].map(lambda x: 1 if x == POS_CLASS else 0)
POS_CLASS = 1

train, test = df_shuffled_split(DF, random_state=42)
train_x, train_y = train.drop(CLASS_FEAT, axis=1), train[CLASS_FEAT]
test_x, test_y = test.drop(CLASS_FEAT, axis=1), test[CLASS_FEAT]


def test_score():
    irep = IREP(random_state=42)
    rip = RIPPER(random_state=42)

    irep.fit(train_x, train_y)
    assert irep.score(test_x, test_y, precision_score) >= 0.4
    assert irep.score(test_x, test_y, recall_score) >= 0.2

    rip.fit(train_x, train_y)
    assert rip.score(test_x, test_y, precision_score) >= 0.4
    assert rip.score(test_x, test_y, recall_score) >= 0.2


def test_cv():
    irep = IREP(random_state=42)
    rip = RIPPER(random_state=42)

    assert max(cross_val_score(irep, train_x, train_y, cv=3)) >= 0.4
    assert max(cross_val_score(rip, train_x, train_y, cv=3)) >= 0.4


def test_grid_search():
    irep = IREP(random_state=42)
    rip = RIPPER(random_state=42)

    param_grid = {"prune_size": [0.33, 0.5], "max_total_conds": [3, None]}
    grid = GridSearchCV(estimator=irep, param_grid=param_grid, cv=2)
    grid.fit(train_x, train_y)

    param_grid = {"prune_size": [0.33, 0.5], "k": [1, 2]}
    grid = GridSearchCV(estimator=rip, param_grid=param_grid, cv=2)
    grid.fit(train_x, train_y)


# This doesn't work. You first need to make dummies X_train, but it only generates a null ruleset
def test_stacking():
    irep = IREP(random_state=42)
    rip = RIPPER(random_state=42)

    df = DF.copy()
    numeric_cols = df.select_dtypes("number").columns
    categorical_cols = [
        col for col in df.columns if (col not in numeric_cols and not col == CLASS_FEAT)
    ]
    dum_df = pd.get_dummies(df[categorical_cols])
    for col in numeric_cols:
        dum_df[col] = df[col]
    dum_df[CLASS_FEAT] = df[CLASS_FEAT]
    sktrain, sktest = df_shuffled_split(dum_df, random_state=42)
    sktrain_x, sktrain_y = sktrain.drop(CLASS_FEAT, axis=1), train[CLASS_FEAT]
    sktest_x, sktest_y = sktest.drop(CLASS_FEAT, axis=1), test[CLASS_FEAT]

    lone_tree = DecisionTreeClassifier(random_state=42)
    lone_tree.fit(sktrain_x, sktrain_y)
    lone_tree_score = lone_tree.score(sktest_x, sktest_y)
    # print('lone_tree_score',lone_tree_score)

    irep_tree = SVC(random_state=42)
    irep_stack_estimators = [("irep", irep), ("tree", irep_tree)]
    irep_stack = StackingClassifier(
        estimators=irep_stack_estimators, final_estimator=LogisticRegression()
    )
    irep_stack.fit(sktrain_x, sktrain_y)
    irep_stack_score = irep_stack.score(sktest_x, sktest_y)
    # print('irep_stack_score', irep_stack_score)
    assert irep_stack_score != lone_tree_score

    rip_tree = DecisionTreeClassifier(random_state=42)
    rip_stack_estimators = [("rip", rip), ("tree", rip_tree)]
    rip_stack = StackingClassifier(
        estimators=rip_stack_estimators, final_estimator=LogisticRegression()
    )
    rip_stack.fit(sktrain_x, sktrain_y)
    rip_stack_score = rip_stack.score(sktest_x, sktest_y)
    # print('rip_stack_score',rip_stack_score)
    assert rip_stack_score != lone_tree_score
