"""
Tests for the core solver functions.
"""

import numpy as np
import pytest
from bestbeta.solver import (
    loss_grad_betaincder,
    loss_grad_fd,
)


@pytest.mark.parametrize(
    "params, lower, upper, confidence",
    [
        ((2.0, 5.0), 0.1, 0.6, 0.8),
        ((1.5, 1.5), 0.2, 0.8, 0.6),
        ((10.0, 3.0), 0.5, 0.9, 0.95),
    ],
)
def test_gradient_correctness(params, lower, upper, confidence):
    """
    Tests that the analytical gradient from `betaincder` is close to the
    finite difference approximation.
    """
    # GIVEN an analytical gradient function and a finite difference gradient function
    # WHEN they are called with the same parameters
    grad_analytical = loss_grad_betaincder(params, lower, upper, confidence)
    grad_finite_diff = loss_grad_fd(params, lower, upper, confidence)

    # THEN the results should be very close
    assert np.allclose(grad_analytical, grad_finite_diff, rtol=1e-4, atol=1e-6), (
        f"Gradient mismatch:\nAnalytical: {grad_analytical}\nFinite Diff: {grad_finite_diff}"
    )
