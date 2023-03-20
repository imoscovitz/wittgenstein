# Author: Ilan Moscovitz <ilan.moscovitz@gmail.com>
# License: MIT

import warnings


def _warn(message, category, filename, funcname, warnstack=[]):
    """Prettier version of warnings warnings.

    warnstack: (optional) list of tuples of filename and function(s) calling the function where warning occurs.
    """
    message = "\n" + message + "\n"
    filename += ".py"
    funcname = " ." + funcname
    if warnstack:
        filename = (
            str(
                [
                    stack_filename + ".py: ." + stack_funcname + " | "
                    for stack_filename, stack_funcname in warnstack
                ]
            )
            .strip("[")
            .strip("]")
            .strip(" ")
            .strip("'")
            + filename
        )
    warnings.showwarning(message, category, filename=filename, lineno=funcname)


def _check_any_datasets_not_empty(datasets):
    return any([len(dataset) > 0 for dataset in datasets])


def _check_is_model_fit(model):
    if not hasattr(model, "ruleset_"):
        raise AttributeError(
            "You should fit the ruleset classifier with .fit method before making predictions with it."
        )

def _check_all_of_type(iterable, type_):
    wrong_type_objects = []
    for object in iterable:
        if not isinstance(object, type_):
            wrong_type_objects.append(object)
    if wrong_type_objects:
        wrong_info = [(object, type(object).__name__) for object in wrong_type_objects]
        raise TypeError(f"Objects must be of type {type_}: {wrong_info}")

def _check_param_deprecation(kwargs, parameters):
    passed_parameters = []
    for param in kwargs.keys():
        if param in parameters:
            passed_parameters.append(param)
    if passed_parameters:
        _warn(
            f".fit: In the future, define parameters: {passed_parameters} during IREP/RIPPER object initialization instead of during model fitting.",
            DeprecationWarning,
            "irep/ripper",
            "fit",
        )

def _check_model_features_present(df, model_selected_features):
    df_feats = df.columns.tolist()
    missing_feats = [f for f in model_selected_features if f not in df_feats]
    if missing_feats:
        raise IndexError(
            f"The features selected by Ruleset model need to be present in prediction dataset. Provided features: {df_feats}.\nMissing features: {missing_feats}.\nEither ensure prediction dataset includes all Ruleset-selected features. If features are present under different names, you can use parameter 'feature_names'.\n"
        )

def _warn_only_single_class(only_value, pos_class, filename, funcname):
    missing_class = "positive" if only_value != pos_class else "negative"
    warning_str = f"No {missing_class} samples. All target labels={only_value}."
    _warn(
        warning_str, RuntimeWarning, filename=filename, funcname=funcname,
    )

def _check_df_allpos_allneg(df, class_feat, pos_class, filename=None, funcname=None):
    if not any([label==pos_class for label in df[class_feat]]):
        warning_str = f"No positive samples. Existing target labels={df[class_feat].unique().tolist()}."
        _warn(warning_str, RuntimeWarning, filename=filename, funcname=funcname)
    elif all([label==pos_class for label in df[class_feat]]):
        warning_str = f"No negative samples. Existing target labels={df[class_feat].unique().tolist()}."
        _warn(warning_str, RuntimeWarning, filename=filename, funcname=funcname)

def _check_valid_index(index, iterable, source_func):
    if index < 0 or index >= len(iterable):
        raise IndexError(
            f"{source_func}: {index} is out of range; {iterable} is of length {len(iterable)}"
        )

def _check_rule_exists(rule, ruleset, source_func):
    for r in ruleset:
        if r == rule:
            return
    raise ValueError(
        f"{source_func}: couldn't find Rule named '{rule}' in Ruleset: '{ruleset}'"
    )

def is_valid_decimal(s):
    try:
        float(s)
    except:
        return False
    return True
