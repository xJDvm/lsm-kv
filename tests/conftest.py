"""Shared pytest configuration and Hypothesis profiles."""

import os

from hypothesis import HealthCheck, settings

settings.register_profile("default", max_examples=50, deadline=None)
settings.register_profile(
    "ci",
    max_examples=500,
    deadline=None,
    suppress_health_check=[HealthCheck.too_slow],
)
settings.load_profile(os.environ.get("HYPOTHESIS_PROFILE", "default"))
