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
    model_preds = utils.try_np_tonum(
        model_predict(X, model, model_predict_function=model_predict_function)
    )
    interpreter.fit(X, model_preds)
    resolution = score_resolution(X, interpreter=interpreter, model_preds=model_preds, score_function=score_function)
    interpreter.base_model = model
    return interpreter.ruleset_, resolution


def score_model(X, y, model, score_function=score_accuracy, model_predict_function=None):
    model_preds = utils.try_np_tonum(
        model_predict(X, model=model, model_predict_function=model_predict_function)
    )
    return score_function(model_preds, y)


def score_resolution(X, interpreter, model=None, model_preds=None, model_predict_function=None, score_function=score_accuracy):
    if model is None and model_preds is None:
        raise ValueError(f'score_resolution: You must pass a model or model predictions')
    elif model_preds is None:
        model_preds = utils.try_np_tonum(
            model_predict(X, model, model_predict_function=model_predict_function)
        )
    return interpreter.score(X, model_preds, score_function)

    
def model_predict(X, model, model_predict_function=None):
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
    return [p[0] for p in model.predict_classes(X)]


def _torch_predict(X, model):
    return model(X)


def _wittgenstein_predict(X, model):
    return model.predict(X)


def _inpackage(model, str_):
    return str_ in str(type(model))
