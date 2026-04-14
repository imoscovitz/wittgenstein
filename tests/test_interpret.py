import numpy as np
from numpy.random import seed

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.metrics import precision_score, recall_score, f1_score
try:
    from tensorflow.keras import Sequential
    from keras.layers import Dense
except:
    pass

import torch
from torch import nn

from wittgenstein.irep import IREP
from wittgenstein.ripper import RIPPER
from wittgenstein.interpret import interpret_model, score_fidelity

inpath = ""
df = pd.read_csv(inpath + "credit.csv")
class_feat = "Class"

a2_mean = np.mean([float(val) for val in df["A2"].tolist() if val != "?"])
df["A2"] = df["A2"].map(lambda x: float(x) if x != "?" else a2_mean)
df["A14"] = df["A14"].map(lambda x: float(x) if x != "?" else a2_mean)
df["A15"] = df["A15"].map(lambda x: float(x) if x != "?" else a2_mean)

cat_feats = [
    col
    for col in df.columns
    if (col not in df.select_dtypes("number").columns and col != class_feat)
]
df = pd.get_dummies(df, columns=cat_feats)

X, y = df.drop("Class", axis=1), df["Class"]
y = y.map(lambda x: 1 if x == "+" else 0)

X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=42)


def test_interpret_keras():
    seed(1)
    wine_df = pd.read_csv(inpath + "wine.csv")
    X_wine, y_wine = wine_df.drop("Class", axis=1), wine_df["Class"]
    y_wine = y_wine.map(lambda x: True if x == 1 else False)
    X_train_wine, X_test_wine, y_train_wine, y_test_wine = train_test_split(
        X, y, random_state=42
    )

    model = Sequential()
    model.add(Dense(60, input_dim=13, activation="relu"))
    model.add(Dense(30, activation="relu"))
    model.add(Dense(1, activation="sigmoid"))
    model.compile(loss="binary_crossentropy", optimizer="adam", metrics=["accuracy"])
    model.fit(
        X_train_wine,
        y_train_wine,
        batch_size=1,
        epochs=10,
    )

    base_model = model
    irep = IREP(random_state=42)
    interpret_model(model=base_model, X=X_test_wine, interpreter=irep)
    assert (
        (irep.base_model)
        and (not irep.ruleset_.isuniversal())
        and (not irep.ruleset_.isnull())
    )
    irep.predict(X_test)

    rip = RIPPER(random_state=42)
    interpret_model(model=base_model, X=X_test_wine, interpreter=rip)
    assert (
        (rip.base_model)
        and (not rip.ruleset_.isuniversal())
        and (not rip.ruleset_.isnull())
    )
    rip.predict(X_test)


def test_interpret_pytorch():
    seed(1)
    torch.manual_seed(1)

    n_feats = X_train.shape[1]

    class CreditNet(nn.Module):
        def __init__(self):
            super().__init__()
            self.net = nn.Sequential(
                nn.Linear(n_feats, 32),
                nn.ReLU(),
                nn.Linear(32, 1),
                nn.Sigmoid(),
            )

        def forward(self, x):
            return self.net(x)

    model = CreditNet()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
    criterion = nn.BCELoss()

    X_t = torch.tensor(X_train.values.astype(float), dtype=torch.float32)
    y_t = torch.tensor(y_train.values.astype(float), dtype=torch.float32).unsqueeze(1)

    model.train()
    for _ in range(20):
        optimizer.zero_grad()
        loss = criterion(model(X_t), y_t)
        loss.backward()
        optimizer.step()

    model.eval()

    def torch_predict(X, model):
        with torch.no_grad():
            return (model(torch.tensor(X.values.astype(float), dtype=torch.float32)).squeeze() > 0.5).numpy()

    irep = IREP(random_state=42)
    interpret_model(model=model, X=X_test, interpreter=irep, model_predict_function=torch_predict)
    assert (
        irep.base_model
        and not irep.ruleset_.isuniversal()
        and not irep.ruleset_.isnull()
    )
    irep.predict(X_test)

    rip = RIPPER(random_state=42)
    interpret_model(model=model, X=X_test, interpreter=rip, model_predict_function=torch_predict)
    assert (
        rip.base_model
        and not rip.ruleset_.isuniversal()
        and not rip.ruleset_.isnull()
    )
    assert sum(rip.predict(X_test)) == 81


def test_interpret_svc():
    svc = SVC(random_state=42)
    svc.fit(X_train, y_train)

    irep = IREP(random_state=42)
    interpret_model(model=svc, X=X_test, interpreter=irep)
    assert sum(irep.predict(X_test)) == 18

    rip = RIPPER(random_state=42)
    interpret_model(model=svc, X=X_test, interpreter=rip)
    assert sum(rip.predict(X_test)) == 22


def test_score_fidelity():
    svc = SVC(kernel="rbf", random_state=42)
    svc.fit(X_train, y_train)

    rip = RIPPER(random_state=42)
    interpret_model(model=svc, X=X_test, interpreter=rip)
    score = score_fidelity(
        X_test, rip, model=svc, score_function=[precision_score, recall_score, f1_score]
    )
    assert len(score) == 3
    assert all(0.0 <= sc <= 1.0 for sc in score)
