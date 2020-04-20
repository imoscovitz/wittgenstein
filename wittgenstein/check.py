from .base import _warn  # (message, category, filename, funcname, warnstack=[]):

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


def _check_any_datasets_not_empty(datasets):
    return any([len(dataset) > 0 for dataset in datasets])


def _check_is_model_fit(model):
    if not hasattr(model, "ruleset_"):
        raise AttributeError(
            "You should fit the ruleset classifier with .fit method before making predictions with it."
        )

# TODO: Check in fit methods before fitting
def _check_any_pos(df, class_feat, pos_class):
    pass


def _check_any_neg(df, class_feat, pos_class):
    pass


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
