# wittgenstein

_And is there not also the case where we play and--make up the rules as we go along?  
  -Ludwig Wittgenstein_

![the duck-rabbit](https://github.com/imoscovitz/wittgenstein/blob/master/duck-rabbit.jpg)

## Summary

This package implements two interpretable coverage-based ruleset algorithms: IREP and RIPPERk, plus additional features for model interpretation.

Performance is similar to sklearn's DecisionTree CART implementation (see [Performance Tests](https://github.com/imoscovitz/ruleset/blob/master/Performance%20Tests.ipynb)).

For explanation of the algorithms, see my article in _Towards Data Science_, or the papers below, under [Useful References](https://github.com/imoscovitz/wittgenstein#useful-references).

## Installation

To install, use
```bash
$ pip install wittgenstein
```

To uninstall, use
```bash
$ pip uninstall wittgenstein
```

## Requirements
- pandas
- numpy
- python version>=3.6

## Usage
Usage syntax is similar to sklearn's.

### Training

Once you have loaded and split your data...
```python
>>> import pandas as pd
>>> df = pd.read_csv(dataset_filename)
>>> from sklearn.model_selection import train_test_split # Or any other mechanism you want to use for data partitioning
>>> train, test = train_test_split(df, test_size=.33)
```
Use the `fit` method to train a `RIPPER` or `IREP` classifier:

```python
>>> import wittgenstein as lw
>>> ripper_clf = lw.RIPPER() # Or irep_clf = lw.IREP() to build a model using IREP
>>> ripper_clf.fit(df, class_feat='Poisonous/Edible', pos_class='p') # Or pass X and y data to .fit
>>> ripper_clf
<RIPPER(max_rules=None, random_state=2, max_rule_conds=None, verbosity=0, max_total_conds=None, k=2, prune_size=0.33, dl_allowance=64, n_discretize_bins=10) with fit ruleset> # Hyperparameter details available in the docstrings and TDS article below
```

Access the underlying trained model with the `ruleset_` attribute, or output it with `out_model()`. A ruleset is a disjunction of conjunctions -- 'V' represents 'or'; '^' represents 'and'.

In other words, the model predicts positive class if any of the inner-nested condition-combinations are all true:
```python
>>> ripper_clf.out_model() # or ripper_clf.ruleset_
[[Odor=f] V
[Gill-size=n ^ Gill-color=b] V
[Gill-size=n ^ Odor=p] V
[Odor=c] V
[Spore-print-color=r] V
[Stalk-surface-below-ring=y ^ Stalk-surface-above-ring=k] V
[Habitat=l ^ Cap-color=w] V
[Stalk-color-above-ring=y]]
```

`IREP` models tend be higher bias, `RIPPER`'s higher variance.

### Scoring
To score a trained model, use the `score` function:
```python
>>> X_test = test.drop('Poisonous/Edible', axis=1)
>>> y_test = test['Poisonous/Edible']
>>> ripper_clf.score(test_X, test_y)
0.9985686906328078
```

Default scoring metric is accuracy. You can pass in alternate scoring functions, including those available through sklearn:
```python
>>> from sklearn.metrics import precision_score, recall_score
>>> precision = clf.score(X_test, y_test, precision_score)
>>> recall = clf.score(X_test, y_test, recall_score)
>>> print(f'precision: {precision} recall: {recall}')
precision: 0.9914..., recall: 0.9953...
```

### Prediction
To perform predictions, use `predict`:
```python
>>> ripper_clf.predict(new_data)[:5]
[True, True, False, True, False]
```

Predict class probabilities with `predict_proba`:
```python
>>> ripper_clf.predict_proba(test)
# Pairs of negative and positive class probabilities
array([[0.01212121, 0.98787879],
       [0.01212121, 0.98787879],
       [0.77777778, 0.22222222],
       [0.2       , 0.8       ],
       ...
```

We can also ask our model to tell us why it made each positive prediction using `give_reasons`:
```python
>>> ripper_clf.predict(new_data[:5], give_reasons=True)
([True, True, False, True, True]
[<Rule [physician-fee-freeze=n]>],
[<Rule [physician-fee-freeze=n]>,
  <Rule [synfuels-corporation-cutback=y^adoption-of-the-budget-resolution=y^anti-satellite-test-ban=n]>], # This example met multiple sufficient conditions for a positive prediction
[],
[<Rule object: [physician-fee-freeze=n]>],
[])
```

### Model selection
wittgenstein is compatible with sklearn model_selection tools such as `cross_val_score` and `GridSearchCV`, as well
as ensemblers like `StackingClassifier`.

Cross validation:
```python
>>> # First dummify your categorical features and booleanize your class values to make sklearn happy
>>> X_train = pd.get_dummies(X_train, columns=X_train.select_dtypes('object').columns)
>>> y_train = y_train.map(lambda x: 1 if x=='p' else 0)
>>> cross_val_score(ripper_clf, X_train, y_train)
```

Grid search:
```python
>>> from sklearn.model_selection import GridSearchCV
>>> param_grid = {"prune_size": [0.33, 0.5], "k": [1, 2]}
>>> grid = GridSearchCV(estimator=ripper, param_grid=param_grid)
>>> grid.fit(X_train, y_train)
```

Ensemble:
```python
>>> from sklearn.ensemble import StackingClassifier
>>> from sklearn.tree import DecisionTreeClassifier
>>> from sklearn.naive_bayes import GaussianNB
>>> from sklearn.linear_model import LogisticRegression
>>> tree = DecisionTreeClassifier(random_state=42)
>>> nb = GaussianNB(random_state=42)
>>> estimators = [("rip", ripper_clf), ("tree", tree), ("nb", nb)]
>>> ensemble_clf = StackingClassifier(estimators=estimators, final_estimator=LogisticRegression())
>>> ensemble_clf.fit(X_train, y_train)
```

### Defining and altering models
You can directly specify a new model, modify a preexisting model, or train from a preexisting model -- whether to take into account subject matter expertise, to create a baseline for scoring, or for insight into what the model is doing.

To specify a new model, use `init_ruleset`:
```python
>>> ripper_clf.init_ruleset("[[Cap-shape=x^Cap-color=n] V [Odor=c] V ...]", class_feat=..., pos_class=...)
>>> ripper_clf.predict(df)
...
```
To modify a preexisting model, use `add_rule`, `replace_rule`, `remove_rule`, or `insert_rule`. To alter a model by index, use `replace_rule_at`, `remove_rule_at`, or `insert_rule_at`:
```python
>>> ripper_clf.replace_rule_at(1, '[Habitat=l]')
>>> ripper_clf.insert_rule(insert_before_rule='[Habitat=l]', new_rule='[Gill-size=n ^ Gill-color=b]')
>>> ripper_clf.out_model()
[[delicious=y^spooky-looking=y] V
[Gill-size=n ^ Gill-color=b] V
[Habitat=l]]
```
To specify a starting point for training, use `initial_model` when calling `fit`:
```python
>>> ripper_clf.fit(
>>> X_train,
>>> y_train,
>>> initial_model="[[delicious=y^spooky-looking=y] V [Odor=c]]")
```
Expected string syntax for a Ruleset is `[<Rule1> V <Rule2> V ...]`, for a Rule `[<Cond1>^<Cond2>^...], and for a Cond `feature=value`. '^' represents 'and'; 'V' represents 'or'. (See the [Training](https://github.com/imoscovitz/wittgenstein#training) section above).

### Interpreter models
Use the interpret module to interpret non-wittgenstein models. `interpret_model` fits a wittgenstein classifier to another model.
```python
# Train the model we want to interpret
>>> from tensorflow.keras import Sequential
>>> from tensorflow.keras.layers import Dense
>>> mlp = Sequential()
>>> mlp.add(Dense(60, input_dim=13, activation='relu'))
>>> mlp.add(Dense(30, activation='relu'))
>>> mlp.add(Dense(1, activation='sigmoid'))
>>> mlp.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
>>> mlp.fit(
>>>   X_train,
>>>   y_train,
>>>   batch_size=1,
>>>   epochs=10)

# Create and fit wittgenstein classifier to use as a model interpreter.
>>> from wittgenstein.interpret import interpret_model, score_fidelity
>>> interpreter = RIPPER(random_state=42)
>>> interpret_model(model=mlp, X=X_train, interpreter=interpreter).out_pretty()
[[Proline=>1227.0] V
[Proline=880.0-1048.0] V
[Proline=1048.0-1227.0] V
[Proline=736.0-880.0] V
[Alcalinityofash=16.8-17.72]]
```
We can also use the now-fitted interpreter to approximate the reasons behind the underlying model's positive predictions. (See [Prediction](https://github.com/imoscovitz/wittgenstein#prediction)).
```python
>>> preds = (mlp.predict(X_test.tail()) > .5).flatten()
>>> _, interpretation = interpreter.predict(X_test.tail(), give_reasons=True)
>>> print(f'tf preds: {preds}\n')
>>> interpretation
tf preds: [ True False False  True False]
[[<Rule [Proline=880.0-1048.0]>],
 [],
 [],
 [<Rule [Proline=736.0-880.0]>, <Rule [Alcalinityofash=16.8-17.72]>],
 []]
```
Score how faithfully the interpreter fits the underlying model with `score_fidelity`.
```python
>>> score_fidelity(
>>>    X_test,
>>>    interpreter,
>>>    model=mlp,
>>>    score_function=[precision_score, recall_score, f1_score])
[1.0, 0.7916666666666666, 0.8837209302325582]
```
## Issues
If you encounter any issues, or if you have feedback or improvement requests for how wittgenstein could be more helpful for you, please post them to [issues](https://github.com/imoscovitz/wittgenstein/issues), and I'll respond.

## Contributing
Contributions are welcome! If you are interested in contributing, let me know at ilan.moscovitz@gmail.com or on [linkedin](https://www.linkedin.com/in/ilan-moscovitz/).

## Useful references
- [My article in _Towards Data Science_ explaining IREP, RIPPER, and wittgenstein](https://towardsdatascience.com/how-to-perform-explainable-machine-learning-classification-without-any-trees-873db4192c68)
- [Furnkrantz-Widmer IREP paper](https://pdfs.semanticscholar.org/f67e/bb7b392f51076899f58c53bf57d5e71e36e9.pdf)
- [Cohen's RIPPER paper](http://citeseerx.ist.psu.edu/viewdoc/download?doi=10.1.1.107.2612&rep=rep1&type=pdf)
- [Partial decision trees](https://researchcommons.waikato.ac.nz/bitstream/handle/10289/1047/uow-cs-wp-1998-02.pdf?sequence=1&isAllowed=y)
- [Bayesian Rulesets](https://pdfs.semanticscholar.org/bb51/b3046f6ff607deb218792347cb0e9b0b621a.pdf)
- [C4.5 paper including all the gory details on MDL](https://pdfs.semanticscholar.org/cb94/e3d981a5e1901793c6bfedd93ce9cc07885d.pdf)
- [_Philosophical Investigations_](https://static1.squarespace.com/static/54889e73e4b0a2c1f9891289/t/564b61a4e4b04eca59c4d232/1447780772744/Ludwig.Wittgenstein.-.Philosophical.Investigations.pdf)

## Changelog

#### v0.3.0: 8/8/2021
- Speedup for binning continuous features (~several orders of magnitude)
- Add support for expert feedback: Ability to explicitly specify and alter models.
- Add surrogate interpreter
- Add support for non-pandas datasets (ex. numpy arrays)

#### v0.2.3: 5/21/2020
- Minor bugfixes and optimizations

#### v0.2.0: 5/4/2020
- Algorithmic optimizations to improve training speed (~10x - ~100x)
- Support for training on iterable datatypes besides DataFrames, such as numpy arrays and python lists
- Compatibility with sklearn ensembling metalearners and sklearn model_selection
- `.predict_proba` returns probas in neg, pos order
- Certain parameters (hyperparameters, random_state, etc.) should now be passed into IREP/RIPPER constructors rather than the .fit method.
- Sundry bugfixes
