# Author: Ilan Moscovitz <ilan.moscovitz@gmail.com>
# License: MIT

from wittgenstein.base_functions import score_accuracy
from wittgenstein.ripper import RIPPER
from wittgenstein import utils


def interpret_model(
    X,
    model,
    interpreter=RIPPER(),
    model_predict_function=None,
    score_function=score_accuracy,
):
    """Interpret another model using a wittgenstein interpreter as global surrogate.

        Parameters
        ----------
        model :
            trained classifier, e.g. sklearn, keras, pytorch, etc.
        X : DataFrame, numpy array, or other iterable
            Dataset upon which to interpret model's predictions.
        interpreter : IREP or RIPPER object, default=RIPPER()
            wittgenstein classifier to perform interpretation.
        model_predict_function : function, default=None
            if
        score_function : function, default=score_accuracy
            scoring function to evaluate how dutifully interpreter interpreted the model.

        Return
        ------
        tuple :
            interpreter fit to model,
            scoring of how dutifully interpreter interpreted the model on training data
    """
    model_preds = utils.try_np_tonum(
        model_predict(X, model, model_predict_function=model_predict_function)
    )
    interpreter.fit(X, model_preds)
    interpreter.base_model = model
    return interpreter.ruleset_


def score_fidelity(
    X,
    interpreter,
    model=None,
    model_preds=None,
    model_predict_function=None,
    score_function=score_accuracy,
):
    """Score how faithfully interpreter represents model.

    Parameters
    ----------
    X : DataFrame, numpy array, or other iterable
        Test dataset with which to score the model.
    interpreter : IREP or RIPPER object, default=RIPPER()
        wittgenstein classifier to perform interpretation.
    model : trained sklearn, keras, pytorch, or wittgenstein, etc. classifier, default=None
        either model or model_preds are required
    model_preds : iterable
        model predictions on X, default=None
    model_predict_function : function, default=None
        model's prediction function. If None, will attempt to figure it out.
    score_function : function or iterable of functions, default=score_accuracy
        criteria to use for scoring fidelity

    Returns
    -------
    score or list of scores"""

    if model is None and model_preds is None:
        raise ValueError(f"score_fidelity: You must pass a model or model predictions")
    elif model_preds is None:
        model_preds = utils.try_np_tonum(
            model_predict(X, model, model_predict_function=model_predict_function)
        )
    if not hasattr(score_function, "__iter__"):
        return interpreter.score(X, model_preds, score_function)
    else:
        return [interpreter.score(X, model_preds, func) for func in score_function]


def model_predict(X, model, model_predict_function=None):
    """Attempt to make predictions using model API"""

    if not model_predict_function:
        if _inpackage(model, "sklearn"):
            return _sklearn_predict(X, model)
        elif _inpackage(model, "tensorflow") or _inpackage(model, "keras"):
            return _keras_predict(X, model)
        elif inpackage(model, "torch"):
            return _torch_predict(X, model)
        elif inpackage(model, "wittgenstein"):
            return _wittgenstein_predict(X, model)
        else:
            return model.predict(X)
    else:
        return model_predict_function(X, model)


def _sklearn_predict(X, model):
    return model.predict(X)


def _keras_predict(X, model):
    return (model.predict(X) > 0.5).flatten()


def _torch_predict(X, model):
    return model(X)


def _wittgenstein_predict(X, model):
    return model.predict(X)


def _inpackage(model, str_):
    return str_ in str(type(model))


def _score_model(
    X, y, model, score_function=score_accuracy, model_predict_function=None
):
    model_preds = utils.try_np_tonum(
        model_predict(X, model=model, model_predict_function=model_predict_function)
    )
    return score_function(model_preds, y)
