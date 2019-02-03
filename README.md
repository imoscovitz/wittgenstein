# ruleset

This module implements two iterative coverage-based ruleset algorithms: IREP and RIPPERk.

Performance is similar to sklearn's DecisionTree CART implementation (see [performance]).

## Installation

To install, use
```bash
$ python setup.py install
```

To uninstall (e.g. when updating) you can use `pip`, _even if you are only working locally_
```bash
$ pip uninstall ruleset
```

## Usage

Usage syntax is similar to sklearn's. The current version, however, does require that data be passed in as a Pandas DataFrame.

Once you have loaded and split your data...
```python
>>> import pandas as pd
>>> df = pd.read_csv(dataset_filename)
>>> from sklearn.model_selection import train_test_split # or any other mechanism you want to use for data partitioning
>>> train, test = train_test_split(df, test_size=.33)
```
We can fit a ruleset classifier:
```
>>> import ruleset
>>> ripper_clf = ruleset.RIPPER(class_feat='class_feature_name')
>>> ripper_clf.fit(train)
>>> ripper_clf
```
Access the underlying trained model:
```
>>> ripper_clf.ruleset_
<Ruleset object: [physician-fee-freeze=n] V [synfuels-corporation-cutback=y^adoption-of-the-budget-resolution=y^anti-satellite-test-ban=n]>
>>>
```
To score our new model:
```
>>> test_X = test.drop(class_feat,axis=1)
>>> test_y = test[class_feat]
>>> ripper_clf.score(test_X, test_y)
0.9985686906328078
```
And perform predictions with it:
```
>>> ripper_clf.predict(new_data)[:10]
[True, True, True, True, False, False, True, False, True, False]
```

## Useful references
- [My medium post about the package](linky-link)
- [Furnkrantz-Widmer IREP paper](https://pdfs.semanticscholar.org/f67e/bb7b392f51076899f58c53bf57d5e71e36e9.pdf)
- [Cohen's RIPPER paper](https://www.let.rug.nl/nerbonne/teach/learning/cohen95fast.pdf)
- [C4.5 paper including all the gory details on MDL](https://pdfs.semanticscholar.org/cb94/e3d981a5e1901793c6bfedd93ce9cc07885d.pdf)
