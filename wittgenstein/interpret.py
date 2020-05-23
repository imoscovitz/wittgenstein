from wittgenstein.base_functions import score_accuracy
from wittgenstein.ripper import RIPPER
from wittgenstein import utils


def interpret_model(
    model,
    X,
    interpreter=RIPPER(),
    model_predict_function=None,
    score_function=score_accuracy,
):
    """Interpret a more complex model.

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
            scoring of how dutifully interpreter interpreted the model
    """

    if not model_predict_function:
        if _inpackage(model, "sklearn"):
            model_preds = _sklearn_predict(model, X)
        elif _inpackage(model, "tensorflow") or _inpackage(model, "keras"):
            model_preds = _keras_predict(model, X)
        elif inpackage(model, "torch"):
            model_preds = _torch_predict(model, X)
        elif inpackage(model, "wittgenstein"):
            model_preds = _wittgenstein_predict(model, X)
        else:
            model_preds = model.predict(X)
    else:
        model_preds = model_predict_function(model, X)
    model_preds = utils.try_np_tonum(model_preds)

    interpreter.fit(X, model_preds)
    resolution = interpreter.score(X, model_preds, score_function)
    interpreter.base_model = model
    return interpreter.ruleset_, resolution


def _sklearn_predict(model, X):
    return model.predict(X)


def _keras_predict(model, X):
    return [p[0] for p in model.predict_classes(X)]


def _torch_predict(model, X):
    return model(X)


def _wittgenstein_predict(model, X):
    return model.predict(X)


def _inpackage(model, str_):
    return str_ in str(type(model))
