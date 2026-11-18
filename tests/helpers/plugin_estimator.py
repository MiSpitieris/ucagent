"""A test-only estimator, registered from a plugin module as a user's own would be."""

from __future__ import annotations

from ucagent.core.uq.agreement import DisagreementEstimator
from ucagent.core.uq.base import register_estimator


class PluginDisagreement(DisagreementEstimator):
    """Disagreement under another name: only the registration differs."""

    name = "plugin_disagreement"


register_estimator("plugin_disagreement", PluginDisagreement)
