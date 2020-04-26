import numpy as np
import pandas as pd
import pytest
from sklearn.ensemble import StackingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV
from sklearn.tree import DecisionTreeClassifier

from wittgenstein import IREP, RIPPER
from wittgenstein.base_functions import df_shuffled_split

DF = pd.read_csv("credit.csv")
DF = DF.head(len(DF) // 3)
CLASS_FEAT = "Class"
POS_CLASS = "+"

train, test = df_shuffled_split(DF, random_state=42)
train_x, train_y = train.drop(CLASS_FEAT, axis=1), train[CLASS_FEAT]
test_x, test_y = test.drop(CLASS_FEAT, axis=1), test[CLASS_FEAT]

irep = IREP(random_state=42, verbosity=1)
rip = RIPPER(random_state=42, verbosity=1)
tree = DecisionTreeClassifier(random_state=42)


def test_grid_search():
    param_grid = {"prune_size": [0.33, 0.5], "max_total_conds": [3, None]}
    grid = GridSearchCV(estimator=irep, param_grid=param_grid, cv=2)
    grid.fit(train_x, train_y, pos_class="+")

    param_grid = {"prune_size": [0.33, 0.5], "k": [1, 2]}
    grid = GridSearchCV(estimator=rip, param_grid=param_grid, cv=2)
    grid.fit(train_x, train_y, pos_class="+")


def test_stacking():
    skirep = IREP(random_state=42)
    sklearnable_df = DF.copy()
    sklearnable_df["Class"] = sklearnable_df["Class"].map(
        lambda x: 1 if x == "+" else 0
    )
    sktrain_y = sklearnable_df["Class"]
    sktrain_x = sklearnable_df.drop("Class", axis=1).select_dtypes(include=["float64"])
    estimators = [("irep", irep), ("tree", tree)]
    clf = StackingClassifier(
        estimators=estimators, final_estimator=LogisticRegression()
    )
    clf.fit(sktrain_x, sktrain_y)  # .score(test_x, test_y)
