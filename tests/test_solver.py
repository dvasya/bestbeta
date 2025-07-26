"""
Tests for the core solver functions.
"""

import numpy as np
import pytest
from scipy.stats import beta as beta_dist
from bestbeta.solver import (
    loss_grad_betaincder,
    loss_grad_fd,
    find_beta_distribution,
    beta_entropy,
    beta_entropy_grad,
)


@pytest.mark.parametrize(
    "params, lower, upper, confidence",
    [
        ((2.0, 5.0), 0.1, 0.6, 0.8),
        ((1.5, 1.5), 0.2, 0.8, 0.6),
        ((10.0, 3.0), 0.5, 0.9, 0.95),
    ],
)
def test_loss_gradient_correctness(params, lower, upper, confidence):
    """
    Tests that the analytical gradient from `betaincder` is close to the
    finite difference approximation for the loss function.
    """
    grad_analytical = loss_grad_betaincder(params, lower, upper, confidence)
    grad_finite_diff = loss_grad_fd(params, lower, upper, confidence)

    assert np.allclose(grad_analytical, grad_finite_diff, rtol=1e-4, atol=1e-6), (
        f"Loss Gradient mismatch:\nAnalytical: {grad_analytical}\nFinite Diff: {grad_finite_diff}"
    )


@pytest.mark.parametrize(
    "alpha, beta",
    [
        (1.0, 1.0),  # Uniform distribution
        (0.5, 0.5),  # Arcsin distribution
        (2.0, 5.0),
        (10.0, 3.0),
    ],
)
def test_beta_entropy_and_grad(alpha, beta):
    """
    Tests that the entropy gradient is close to its finite difference approximation.
    """
    params = np.array([alpha, beta])

    # Test entropy calculation (spot check against known values or properties if possible)
    # For (1,1) uniform, entropy is 0. (ln(B(1,1)) - 0 - 0 + 0) = ln(1) = 0
    if alpha == 1.0 and beta == 1.0:
        assert beta_entropy(params) == pytest.approx(0.0, abs=1e-9)

    # Finite difference approximation for entropy gradient
    eps = 1e-6
    grad_alpha_fd = (
        beta_entropy(params + np.array([eps, 0]))
        - beta_entropy(params - np.array([eps, 0]))
    ) / (2 * eps)
    grad_beta_fd = (
        beta_entropy(params + np.array([0, eps]))
        - beta_entropy(params - np.array([0, eps]))
    ) / (2 * eps)
    grad_fd = np.array([grad_alpha_fd, grad_beta_fd])

    grad_analytical = beta_entropy_grad(params)

    assert np.allclose(grad_analytical, grad_fd, rtol=1e-4, atol=1e-6), (
        f"Entropy Gradient mismatch for alpha={alpha}, beta={beta}:\n"
        f"Analytical: {grad_analytical}\nFinite Diff: {grad_fd}"
    )


@pytest.mark.parametrize(
    "lower, upper, confidence, alpha0, beta0, outside_odds, expected_alpha, expected_beta",
    [
        # fsolve mode (equal odds)
        (0.1, 0.9, 0.8, 1.0, 1.0, 1.0, 1.0, 1.0),  # Uniform distribution
        (0.05, 0.95, 0.9, 1.0, 1.0, 1.0, 1.0, 1.0),  # Uniform distribution
        # Closest solution mode (no outside_odds, should converge to a solution near alpha0, beta0)
        (0.1, 0.9, 0.95, 1.0, 1.0, None, 1.5, 1.5),  # Example from original script
        (0.2, 0.8, 0.9, 5.0, 5.0, None, 5.0, 5.0),  # Should stay close to initial guess
        # Maxent mode (hard to predict exact values, so we check for success and reasonable values)
        (0.1, 0.9, 0.95, 1.0, 1.0, "maxent", None, None),  # Maxent for wide CI
        (0.4, 0.6, 0.5, 2.0, 2.0, "maxent", None, None),  # Maxent for narrow CI
    ],
)
def test_find_beta_distribution(
    lower, upper, confidence, alpha0, beta0, outside_odds, expected_alpha, expected_beta
):
    """
    Tests the main find_beta_distribution function for different modes.
    """
    alpha, beta = find_beta_distribution(
        lower, upper, confidence, alpha0, beta0, outside_odds
    )

    # Basic validation for all modes
    assert alpha > 0
    assert beta > 0

    # Specific checks for fsolve mode (where we can predict exact results)
    if outside_odds == 1.0:
        assert alpha == pytest.approx(expected_alpha, rel=1e-3)
        assert beta == pytest.approx(expected_beta, rel=1e-3)
        # Verify CI coverage for fsolve mode
        actual_confidence = beta_dist.cdf(upper, alpha, beta) - beta_dist.cdf(
            lower, alpha, beta
        )
        assert actual_confidence == pytest.approx(confidence, rel=1e-3)

        # Verify probability split for fsolve mode
        total_outside_prob = 1 - confidence
        prob_below = total_outside_prob / (outside_odds + 1)
        prob_above = total_outside_prob - prob_below
        assert beta_dist.cdf(lower, alpha, beta) == pytest.approx(prob_below, rel=1e-3)
        assert (1 - beta_dist.cdf(upper, alpha, beta)) == pytest.approx(
            prob_above, rel=1e-3
        )

    # For closest solution and maxent, we primarily check for successful convergence
    # and that the CI constraint is met. Exact alpha/beta values are harder to predict
    # without running the solver and recording them.
    else:
        # Verify CI coverage for trust-constr modes
        actual_confidence = beta_dist.cdf(upper, alpha, beta) - beta_dist.cdf(
            lower, alpha, beta
        )
        assert actual_confidence == pytest.approx(confidence, rel=1e-3)

    # For maxent, we can also check that the entropy is maximized (relative to other solutions)
    # This would require more complex setup (e.g., find another valid solution and compare entropy)
    # For now, just ensuring it runs and satisfies the CI is sufficient.


@pytest.mark.parametrize(
    "lower, upper, confidence, alpha0, beta0, outside_odds",
    [
        (0.9, 0.1, 0.8, 1.0, 1.0, None),  # Invalid bounds
        (0.1, 0.9, 1.1, 1.0, 1.0, None),  # Invalid confidence
        (0.1, 0.9, 0.8, -1.0, 1.0, None),  # Invalid initial alpha
        (0.1, 0.9, 0.8, 1.0, -1.0, None),  # Invalid initial beta
        (0.1, 0.9, 0.8, 1.0, 1.0, "invalid_mode"),  # Invalid outside_odds string
    ],
)
def test_find_beta_distribution_invalid_inputs(
    lower, upper, confidence, alpha0, beta0, outside_odds
):
    """
    Tests that find_beta_distribution handles invalid inputs gracefully.
    """
    with pytest.raises((RuntimeError, ValueError, NotImplementedError)):
        find_beta_distribution(lower, upper, confidence, alpha0, beta0, outside_odds)


@pytest.mark.parametrize(
    "alpha, beta",
    [
        (1.0, 1.0),
        (2.0, 3.0),
        (5.0, 5.0),
        (0.5, 2.0),
        (10.0, 1.0),
    ],
)
def test_entropy_gradient_consistency(alpha, beta):
    """
    Tests that our entropy gradient is consistent with finite differences
    of SciPy's entropy function.
    """

    # Test that our entropy matches SciPy's entropy
    our_entropy = beta_entropy([alpha, beta])
    scipy_entropy = beta_dist.entropy(alpha, beta)
    assert np.allclose(our_entropy, scipy_entropy, rtol=1e-10, atol=1e-10), (
        f"Entropy mismatch for alpha={alpha}, beta={beta}: "
        f"our={our_entropy}, scipy={scipy_entropy}"
    )

    # Test that our gradient is consistent with finite differences of SciPy's entropy
    eps = 1e-6
    our_grad = beta_entropy_grad([alpha, beta])

    # Finite difference gradient using SciPy's entropy
    entropy_alpha_plus = beta_dist.entropy(alpha + eps, beta)
    entropy_alpha_minus = beta_dist.entropy(alpha - eps, beta)
    entropy_beta_plus = beta_dist.entropy(alpha, beta + eps)
    entropy_beta_minus = beta_dist.entropy(alpha, beta - eps)

    fd_grad = np.array(
        [
            (entropy_alpha_plus - entropy_alpha_minus) / (2 * eps),
            (entropy_beta_plus - entropy_beta_minus) / (2 * eps),
        ]
    )

    assert np.allclose(our_grad, fd_grad, rtol=1e-4, atol=1e-6), (
        f"Entropy gradient mismatch for alpha={alpha}, beta={beta}:\n"
        f"Our gradient: {our_grad}\n"
        f"Finite diff gradient: {fd_grad}"
    )
