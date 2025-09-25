from . import components
from . import models
from . import tao_index

"""
tao_index: data separability framework training pipeline
components: core separability components & baselines
models: classifiers and performance of public datasets (SECOM, SPF)
utils: small helpers
"""

__all__ = [
    "components",
    "models",
    "tao_index",
    "__version__",
]

__version__ = "0.1.0"

