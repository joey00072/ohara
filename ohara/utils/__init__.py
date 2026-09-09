from .tools import BetterCycle, auto_accelerator  # noqa
from .svd import svd_approx  # noqa
from .rand import random_name  # noqa


def model_summary(*args, **kwargs):
    """Load the optional Lightning summary integration only when requested."""
    from .info import model_summary as summarize

    return summarize(*args, **kwargs)
