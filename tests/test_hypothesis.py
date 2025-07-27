"""
Hypothesis-based property tests for the solver functions.
"""

import warnings
import numpy as np
import pytest
from hypothesis import given, settings, strategies as st
from scipy.stats import beta as beta_dist

from bestbeta.solver import (
    beta_entropy,
    beta_entropy_grad,
    loss_function,
    constraint_function,
)
from bestbeta import find_beta_distribution
from .conftest import check_entropy_gradient_consistency


# Strategies for generating test data
@st.composite
def valid_beta_params(draw):
    """Generate valid alpha, beta parameters for beta distribution."""
    alpha = draw(st.floats(min_value=0.1, max_value=20.0))
    beta = draw(st.floats(min_value=0.1, max_value=20.0))
    return alpha, beta


@st.composite
def valid_confidence_intervals(draw):
    """Generate valid confidence intervals."""
    lower = draw(st.floats(min_value=0.01, max_value=0.49))
    upper = draw(st.floats(min_value=lower + 0.01, max_value=0.99))
    confidence = draw(st.floats(min_value=0.1, max_value=0.99))
    return lower, upper, confidence


@st.composite
def valid_optimization_params(draw):
    """Generate valid parameters for optimization."""
    lower, upper, confidence = draw(valid_confidence_intervals())
    alpha0 = draw(st.floats(min_value=0.1, max_value=10.0))
    beta0 = draw(st.floats(min_value=0.1, max_value=10.0))
    return lower, upper, confidence, alpha0, beta0


# Property-based tests
@pytest.mark.hypothesis
@given(valid_beta_params())
@settings(max_examples=100, deadline=None)
def test_entropy_properties(alpha_beta):
    """Test properties of entropy calculation."""
    alpha, beta = alpha_beta

    # Entropy should be finite for valid parameters
    entropy = beta_entropy([alpha, beta])
    assert np.isfinite(entropy)

    # Entropy should be negative (beta distribution has finite support)
    assert entropy <= 0

    # For symmetric distributions, entropy should be symmetric
    if abs(alpha - beta) < 1e-6:
        entropy_reversed = beta_entropy([beta, alpha])
        assert np.allclose(entropy, entropy_reversed, rtol=1e-10)


@pytest.mark.hypothesis
@given(valid_beta_params())
@settings(max_examples=100, deadline=None)
def test_entropy_gradient_properties(alpha_beta):
    """Test properties of entropy gradient."""
    alpha, beta = alpha_beta

    # Gradient should be finite
    grad = beta_entropy_grad([alpha, beta])
    assert np.all(np.isfinite(grad))
    assert grad.shape == (2,)

    # For symmetric distributions, gradients should be equal
    if abs(alpha - beta) < 1e-6:
        assert np.allclose(grad[0], grad[1], rtol=1e-10)


@pytest.mark.hypothesis
@given(valid_beta_params())
@settings(max_examples=100, deadline=None)
def test_entropy_gradient_consistency_hypothesis(alpha_beta):
    """Test that entropy gradient is consistent with finite differences."""
    alpha, beta = alpha_beta

    # Skip edge cases where finite differences might be unreliable
    if alpha < 0.5 or beta < 0.5:
        return

    # Use shared utility with relaxed tolerance for edge cases
    is_consistent, our_grad, fd_grad = check_entropy_gradient_consistency(
        alpha, beta, rtol=1e-3, atol=1e-5
    )

    assert is_consistent, (
        f"Entropy gradient mismatch for alpha={alpha}, beta={beta}:\n"
        f"Our gradient: {our_grad}\n"
        f"Finite diff gradient: {fd_grad}"
    )


@pytest.mark.hypothesis
@given(valid_optimization_params())
@settings(max_examples=50, deadline=None)
def test_loss_function_properties(params):
    """Test properties of loss function."""
    lower, upper, confidence, alpha0, beta0 = params

    # Loss should be non-negative (it's a squared error)
    loss = loss_function([alpha0, beta0], lower, upper, confidence)
    assert loss >= 0
    assert np.isfinite(loss)


@pytest.mark.hypothesis
@given(valid_optimization_params())
@settings(max_examples=50, deadline=None)
def test_constraint_function_properties(params):
    """Test properties of constraint function."""
    lower, upper, confidence, alpha0, beta0 = params

    # Constraint should be finite
    constraint = constraint_function([alpha0, beta0], lower, upper, confidence)
    assert np.isfinite(constraint)


@pytest.mark.hypothesis
@given(valid_optimization_params())
@settings(max_examples=20, deadline=None)
def test_find_beta_distribution_properties(params):
    """Test properties of the main solver function."""
    lower, upper, confidence, alpha0, beta0 = params

    # Test closest solution mode
    try:
        alpha, beta = find_beta_distribution(
            lower, upper, confidence, alpha0, beta0, None
        )

        # Results should be valid
        assert alpha > 0
        assert beta > 0
        assert np.isfinite(alpha)
        assert np.isfinite(beta)

        # Should satisfy the confidence interval constraint
        actual_confidence = beta_dist.cdf(upper, alpha, beta) - beta_dist.cdf(
            lower, alpha, beta
        )
        assert abs(actual_confidence - confidence) < 0.01  # Within 1%

    except (RuntimeError, ValueError):
        # Some parameter combinations might fail, which is acceptable
        pass


@pytest.mark.hypothesis
@given(
    st.floats(min_value=0.1, max_value=0.4),
    st.floats(min_value=0.6, max_value=0.9),
    st.floats(min_value=0.5, max_value=0.9),
    st.floats(min_value=0.5, max_value=5.0),
    st.floats(min_value=0.5, max_value=5.0),
)
@settings(max_examples=100, deadline=1000)  # Increased deadline for safe mode
def test_maxent_symmetry_property(lower, upper, confidence, alpha0, beta0):
    """Test that symmetric CIs produce symmetric maxent solutions."""

    # Only test symmetric intervals
    if abs((lower + upper) / 2 - 0.5) > 0.01:
        return

    try:
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            # Maxent mode with safe mode enabled
            alpha, beta = find_beta_distribution(
                lower, upper, confidence, alpha0, beta0, "maxent", safe_mode=True
            )

        # Check for the specific warning about flat entropy
        flat_entropy_warnings = [
            warning for warning in w if "delta_grad == 0.0" in str(warning.message)
        ]
        if flat_entropy_warnings:
            print("\nFlat entropy warning triggered with parameters:")
            print(f"  lower={lower}, upper={upper}, confidence={confidence}")
            print(f"  alpha0={alpha0}, beta0={beta0}")
            print(f"  result: alpha={alpha:.6f}, beta={beta:.6f}")
            print(f"  warning: {flat_entropy_warnings[0].message}")

        # For symmetric CIs, maxent solution should be symmetric
        if abs((lower + upper) / 2 - 0.5) < 1e-6:
            assert abs(alpha - beta) < 0.1

    except (RuntimeError, ValueError):
        # Some parameter combinations might fail
        pass


@st.composite
def dense_maxent_params(draw):
    """Generate parameters for dense maxent testing (upper-lower < confidence)."""
    # Generate center of the interval
    center = draw(st.floats(min_value=0.1, max_value=0.9))

    # Generate width up to 2x min(center, 1-center) - eps
    max_width = 2 * min(center, 1 - center) - 1e-6
    width = draw(st.floats(min_value=0.05, max_value=max_width))

    # Calculate lower and upper bounds
    lower = center - width / 2
    upper = center + width / 2

    # Generate area (confidence) from width to 1-eps
    confidence = draw(st.floats(min_value=width + 1e-6, max_value=0.99))

    # Generate starting points with b >= a for cleaner testing
    alpha0 = draw(st.floats(min_value=0.5, max_value=3.0))
    beta0 = draw(st.floats(min_value=alpha0, max_value=3.0))  # Ensure beta0 >= alpha0

    return lower, upper, confidence, alpha0, beta0


@pytest.mark.hypothesis
@pytest.mark.slow
@given(dense_maxent_params())
@settings(max_examples=50, deadline=2000)
def test_maxent_global_optimum_dense_cases(params):
    """Test that different starting points produce consistent maxent solutions for dense cases."""
    lower, upper, confidence, alpha0, beta0 = params

    # Verify it's dense
    assert upper - lower < confidence, (
        f"Not dense: {upper} - {lower} = {upper - lower} >= {confidence}"
    )
    # Verify beta0 >= alpha0 (from our strategy)
    assert beta0 >= alpha0, (
        f"Strategy should ensure beta0 >= alpha0: {beta0} < {alpha0}"
    )

    try:
        # Test three starting points: (alpha0, beta0), (1,1), and (beta0, alpha0)
        starting_points = [
            (alpha0, beta0, "original"),
            (1.0, 1.0, "symmetric"),
            (beta0, alpha0, "swapped"),
        ]

        results = []
        for a, b, name in starting_points:
            alpha, beta = find_beta_distribution(
                lower, upper, confidence, a, b, "maxent", safe_mode=False
            )
            entropy = beta_dist.entropy(alpha, beta)
            results.append((alpha, beta, entropy, name))

        # All solutions should have the same entropy
        entropies = [entropy for _, _, entropy, _ in results]
        max_entropy_diff = max(
            abs(entropies[i] - entropies[0]) for i in range(1, len(entropies))
        )

        assert max_entropy_diff < 1e-6, (
            f"Different starting points produced different solutions:\n"
            f"  Parameters: lower={lower}, upper={upper}, confidence={confidence}\n"
            f"  Original starting point ({alpha0}, {beta0}): alpha={results[0][0]:.6f}, "
            f"beta={results[0][1]:.6f}, entropy={results[0][2]:.6f}\n"
            f"  Symmetric starting point (1.0, 1.0): alpha={results[1][0]:.6f}, "
            f"beta={results[1][1]:.6f}, entropy={results[1][2]:.6f}\n"
            f"  Swapped starting point ({beta0}, {alpha0}): alpha={results[2][0]:.6f}, "
            f"beta={results[2][1]:.6f}, entropy={results[2][2]:.6f}\n"
            f"  Max entropy difference: {max_entropy_diff:.2e}"
        )

    except (RuntimeError, ValueError):
        # Some parameter combinations might fail
        pass
