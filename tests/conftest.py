"""
Shared test utilities for BestBeta tests.
"""

import numpy as np
from scipy.stats import beta as beta_dist
from bestbeta.solver import beta_entropy_grad


def check_entropy_gradient_consistency(alpha, beta, eps=1e-6, rtol=1e-4, atol=1e-6):
    """
    Check that our entropy gradient is consistent with finite differences of SciPy's entropy.
    
    This is a shared utility used by multiple tests to avoid code duplication.
    """
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

    return np.allclose(our_grad, fd_grad, rtol=rtol, atol=atol), our_grad, fd_grad 