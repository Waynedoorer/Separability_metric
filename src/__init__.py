from . import components
from . import models
from . import tao_index
from .api import tao_score

"""
tao_index: data separability framework training pipeline
components: core separability components & baselines
models: classifiers and performance of public datasets (SECOM, SPF)
utils: small helpers
api: expose high level api usage to easily calculate separability score for custom datasets
"""

__all__ = [
    "components",
    "models",
    "tao_index",
    "__version__",
    "tao_score"
]

__version__ = "0.1.0"

