"""Uncertainty quantification: estimators, equivalence judges, calibration.

Importing the package registers the shipped estimators and judges.
"""

from ucagent.core.uq import agreement as _agreement  # noqa: F401
from ucagent.core.uq import confidence_agreement as _confidence_agreement  # noqa: F401
from ucagent.core.uq import semantic_entropy as _semantic_entropy  # noqa: F401
from ucagent.core.uq import verbalized as _verbalized  # noqa: F401
