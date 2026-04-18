# wittgenstein

_And is there not also the case where we play and--make up the rules as we go along?  
  -Ludwig Wittgenstein_

![the duck-rabbit](https://github.com/imoscovitz/wittgenstein/blob/master/duck-rabbit.jpg)

## Summary

This package implements two interpretable coverage-based ruleset algorithms: IREP and RIPPERk, as well as additional features for model interpretation.

Performance is similar to sklearn's DecisionTree CART implementation (see [Performance Tests](https://github.com/imoscovitz/wittgenstein/blob/master/examples/performance_tests.ipynb)).

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
>>> ripper_clf.score(X_test, y_test)
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
wittgenstein is compatible with sklearn model_selection tools such as `cross_val_score` and `GridSearchCV`, as well as ensemblers like `StackingClassifier`.

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
>>> nb = GaussianNB()
>>> estimators = [("rip", ripper_clf), ("tree", tree), ("nb", nb)]
>>> ensemble_clf = StackingClassifier(estimators=estimators, final_estimator=LogisticRegression())
>>> ensemble_clf.fit(X_train, y_train)
```

### Multiclass
For multiclass tasks, use in combination with sklearn `OneVsRestClassifier`:
```python
>>> from sklearn.multiclass import OneVsRestClassifier
>>> rip = RIPPER()
>>> clf = OneVsRestClassifier(rip)
>>> clf.fit(X, y)
```

### Defining and altering models
You can directly specify a new model, modify a preexisting model, or train from a preexisting model -- whether to take into account subject matter expertise, to create a baseline for scoring, or for insight into what the model is doing.

To specify a new model, use `init_ruleset`:
```python
>>> ripper_clf = RIPPER(random_state=42)
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
>>>   X_train,
>>>   y_train,
>>>   initial_model="[[delicious=y^spooky-looking=y] V [Odor=c]]")
```
Expected string syntax for a Ruleset is `[<Rule1> V <Rule2> V ...]`, for a Rule `[<Cond1>^<Cond2>^...], and for a Cond `feature=value`. '^' represents 'and'; 'V' represents 'or'. (See the [Training](https://github.com/imoscovitz/wittgenstein#training) section above).

### Interpreter models
Use the interpret module to interpret non-wittgenstein models. `interpret_model` generates a ruleset that approximates some black-box model by fitting a wittgenstein classifier to the predictions of the other model.
```python
>>> import torch
>>> from torch import nn
>>> from wittgenstein.interpret import interpret_model, score_fidelity

# Define and train a PyTorch model
>>> class WineNet(nn.Module):
...     def __init__(self):
...         super().__init__()
...         self.net = nn.Sequential(
...             nn.Linear(n_feats, 32), nn.ReLU(),
...             nn.Linear(32, 1), nn.Sigmoid())
...     def forward(self, x):
...         return self.net(x)

>>> model = WineNet()
>>> # ... train model ...

# Provide a predict function to handle DataFrame → tensor → binary array conversion
>>> def torch_predict(X, model):
...     with torch.no_grad():
...         t = torch.tensor(X.values.astype(float), dtype=torch.float32)
...         return (model(t).squeeze() > 0.5).numpy()

# Fit a RIPPER interpreter to approximate the model's predictions
>>> rip = RIPPER(random_state=1)
>>> interpret_model(model=model, X=X_test, interpreter=rip, model_predict_function=torch_predict)
>>> rip.out_model()
[[Colorintensity=5.68-7.24] V
[Proline=>1283.0] V
[Colorintensity=3.42-4.35]]
```
Use `give_reasons=True` to see which rules fired for each prediction.
```python
>>> preds, reasons = rip.predict(X_test, give_reasons=True)
>>> reasons[:5]
[[], [], [], [<Rule [Colorintensity=5.68-7.24]>, <Rule [Proline=>1283.0]>], []]
```
Score how faithfully the interpreter fits the underlying model with `score_fidelity`.
```python
>>> from sklearn.metrics import precision_score, recall_score, f1_score
>>> score_fidelity(
...    X_test,
...    rip,
...    model=model,
...    model_predict_function=torch_predict,
...    score_function=[precision_score, recall_score, f1_score])
[1.0, 0.75, 0.857]
```
### NLP
Building lexical models is straightforward.

```python
>>> # Extract texts and labels from the SMS Spam Collection dataset
>>> # ...
>>> from sklearn.feature_extraction.text import CountVectorizer
>>> vectorizer = CountVectorizer(binary=True, ngram_range=(1,3), min_df=5,
...   stop_words='english')
>>> X = vectorizer.fit_transform(texts).toarray() # convert from sparse to dense array
>>> rip = RIPPER()
>>> rip.fit(X, y=labels, pos_class='spam', feature_names=vectorizer.get_feature_names_out())
>>> rip.out_model() # Binary BoW model -- 1 indicates presence of token, 0 its absence
[[free=1 ^ txt=1] V
[claim=1] V
[mobile=1 ^ gt=0 ^ left=0 ^ free=1] V
[txt=1 ^ 150p=1] V
[txt=1 ^ win=1] V
[stop=1 ^ send=1] V
[mobile=1 ^ gt=0 ^ 50=1] V
[service=1 ^ dating=1] V
[reply=1 ^ video=1] V
[free=1 ^ nokia=1] V
[box=1 ^ po=1] V
...
```

You can also use `interpret_model` to distill an LM featurized with e.g. bag-of-words or TFIDF. The example below shows this approach with a neural net classifying whether recipes are Hungarian. (See [interpreter models](https://github.com/imoscovitz/wittgenstein/blob/master/README.md#interpreter-models) above).

```python
>>> from sklearn.feature_extraction.text import TfidfVectorizer
>>> vectorizer = TfidfVectorizer(stop_words='english', min_df=5, ngram_range=(1, 2), max_features=1000)
>>> X = vectorizer.fit_transform([example['ingredients'] for example in data])
>>> feature_names = vectorizer.get_feature_names_out()

>>> # Train a neural net on the TFIDF features>>> # Use the same torch_predict function and training pattern from the interpreter models section above
>>> rip = RIPPER(n_discretize_bins=5, random_state=42)
>>> X_df = pd.DataFrame(X, columns=feature_names)
>>> interpret_model(model=LM_model, X=X_df, interpreter=rip, model_predict_function=torch_predict)
>>> rip.out_model()
[[hungarian=0.11-0.21] V
[paprika=0.12-0.19 ^ cut=>0.076] V
[hungarian=0.21-0.32] V
[sweetpaprika=0.091-0.18 ^ fresh=<0.043] V
[paprika=0.12-0.19 ^ beef=>0.057] V
[eggnoodles=>0.31]
...
```

### wittgenstein-LLM collaboration
This demonstration combines LLM feature extraction with rule inference to produce both flexible predictions and explicit, interpretable rules. Results from this small-scale evaluation suggest this hybrid approach may improve classification performance while preserving the interpretability of rule-based models.

Using the CFPB consumer complaints dataset (400 train, 300 test, balanced classes), the task is to predict whether a consumer received monetary relief. The LLM extracts semantic features from complaint narratives — e.g., does the consumer describe a broken agreement, or cite specific legal statutes — which RIPPER then uses to learn interpretable rule combinations.

Accuracy and F1 scores on the held-out test set:

- LLM judge (zero-shot): 0.56 / 0.23
- LLM judge (10-shot): 0.69 / 0.58
- RIPPER on structured features only: 0.61 / 0.50
- RIPPER on structured + LLM-derived features (zero-shot): 0.79 / 0.75

The code can be adapted to other text classification tasks by defining new feature schemas and swapping in a different dataset.

The full example can be found in [this notebook](https://github.com/imoscovitz/wittgenstein/blob/master/examples/LLM_featurization_example.ipynb).

```python
>>> text_features = {
    "describes_specific_transaction": {
        "description": "Does the consumer describe a specific financial transaction with concrete details (dates, amounts, payment method, confirmation numbers)?",
        "values": ["yes", "no"]
    },
    "payment_made_not_credited": {
        "description": "Does the consumer claim they made a payment that was not properly applied or acknowledged by the company?",
        "values": ["yes", "no"]
    },
    "disputes_specific_fee": {
        "description": "Does the consumer dispute a specific fee such as a late fee, interest charge, or cancellation fee as unfair or incorrect?",
        "values": ["yes", "no"]
    },
    ...

>>> FEATURIZATION_PROMPT = \
  """You are a feature extractor for consumer financial complaints about debt collection.
  Read the complaint and return a JSON object with ONLY these keys and ONLY the allowed values.

  {feature_schema}

  Return ONLY valid JSON. No explanation or commentary.

  COMPLAINT:
  {narrative}""".replace('{feature_schema}', str(text_features))

>>> # Use an LLM to extract features (see notebook for implementation)
>>> featurized_texts_X_train = await featurize_X_texts(X_train, text_feat)
>>> featurized_X_train = pd.concat(
...   [X_train.reset_index(drop=True), pd.DataFrame(featurized_texts_X_train)], axis=1)
# repeat for X_test

>>> # Train and score the ruleset classifier
>>> rip = RIPPER(random_state=42)
>>> rip.fit(featurized_X_train.drop(text_feat, axis=1), y_train)
>>> rip.out_model()
[[describes_specific_transaction=yes ^ disputes_specific_fee=yes ^ Issue=Attemptstocollectdebtnotowed] V
[complaint_narrative_style=personal_story ^ describes_specific_transaction=yes ^ requests_refund_or_reversal=yes ^ Sub-issue=Attemptedtocollectwrongamount ^ disputes_specific_fee=yes] V
[complaint_narrative_style=personal_story ^ describes_specific_transaction=yes ^ Issue=Communicationtactics] V
[describes_specific_transaction=yes ^ Sub-issue=Debtisnotyours] V
[complaint_narrative_style=personal_story ^ disputes_specific_fee=yes] V
[complaint_narrative_style=personal_story ^ product_or_service_not_received=yes] V
[disputes_specific_fee=yes] V
[Issue=Attemptstocollectdebtnotowed ^ describes_specific_transaction=yes ^ Sub-issue=Debtwasresultofidentitytheft ^ prior_attempts_to_resolve=multiple]]

>>> rip.score(featurized_X_test.drop(text_feat, axis=1), y_test), rip.score(featurized_X_test.drop(text_feat, axis=1), y_test, score_function=f1_score)
(0.7866666666666666, np.float64(0.7480314960629921))
```

## Issues
If you encounter any issues, or if you have feedback or improvement requests for how wittgenstein could be more helpful for you, please post them to [issues](https://github.com/imoscovitz/wittgenstein/issues).

## Useful references
- [My article in _Towards Data Science_ explaining IREP, RIPPER, and wittgenstein](https://medium.com/towards-data-science/how-to-perform-explainable-machine-learning-classification-without-any-trees-873db4192c68)
- [Furnkrantz-Widmer IREP paper](https://pdfs.semanticscholar.org/f67e/bb7b392f51076899f58c53bf57d5e71e36e9.pdf)
- [Cohen's RIPPER paper](http://citeseerx.ist.psu.edu/viewdoc/download?doi=10.1.1.107.2612&rep=rep1&type=pdf)
- [Partial decision trees](https://researchcommons.waikato.ac.nz/bitstream/handle/10289/1047/uow-cs-wp-1998-02.pdf?sequence=1&isAllowed=y)
- [Bayesian Rulesets](https://pdfs.semanticscholar.org/bb51/b3046f6ff607deb218792347cb0e9b0b621a.pdf)
- [C4.5 paper including all the gory details on MDL](https://pdfs.semanticscholar.org/cb94/e3d981a5e1901793c6bfedd93ce9cc07885d.pdf)
- [_Philosophical Investigations_](https://static1.squarespace.com/static/54889e73e4b0a2c1f9891289/t/564b61a4e4b04eca59c4d232/1447780772744/Ludwig.Wittgenstein.-.Philosophical.Investigations.pdf)

## Changelog

#### v0.3.4: 4/3/2022
- Improvements to predict_proba calculation, including smoothing

#### v0.3.2: 8/8/2021
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
