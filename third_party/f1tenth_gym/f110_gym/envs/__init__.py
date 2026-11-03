"""Load the environment lazily so dynamics-only data tools work headlessly."""


def __getattr__(name):
    if name == "F110Env":
        from .f110_env import F110Env
        return F110Env
    raise AttributeError(name)
