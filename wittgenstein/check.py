import warnings

# def _get_non_default_params(param_values_defaults):

#    non_default_params = []
#    for param, (value, default) in param_values_defaults.items():
#        if value != default:
#            non_default_params.append(param)
#    return non_default_params

# def _check_fit_param_deprecation(param_values_defaults):
#    non_default_params = _get_non_default_params(param_values_defaults)
#    if non_default_params:
#        _warn(f'.fit: In the future, define parameters: {non_default_params} when initializating IREP or RIPPER object instead of during model fitting.',
#                DeprecationWarning,'irep/ripper','fit')


def _warn(message, category, filename, funcname, warnstack=[]):
    """ warnstack: (optional) list of tuples of filename and function(s) calling the function where warning occurs """
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


# def _get_missing_selected_features(df, model_selected_features):
#    df_feats = df.columns.tolist()
#    return [f for f in model_selected_features if f not in df_feats]


# TODO: Check in fit methods before fitting
def _check_any_pos(df, class_feat, pos_class):
    pass


def _check_any_neg(df, class_feat, pos_class):
    pass


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
