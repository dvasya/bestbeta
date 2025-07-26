"""
Core solver functions for finding beta distributions.
"""

from scipy.optimize import fsolve, minimize, NonlinearConstraint
from scipy.stats import beta as beta_dist
from scipy import special
import numpy as np
from betaincder import betaincderp, betaincderq  # type: ignore[import-untyped]

# ==============================================================================
# Objective and Constraint Functions
# ==============================================================================


def loss_function(params, lower, upper, confidence):
    """
    Loss function: squared error of the confidence interval coverage.
    This is the objective to minimize when finding the "closest" solution.
    """
    alpha, beta = params
    if alpha <= 0 or beta <= 0:
        return 1e6  # Return a large value for invalid parameters
    actual_confidence = beta_dist.cdf(upper, alpha, beta) - beta_dist.cdf(
        lower, alpha, beta
    )
    return (actual_confidence - confidence) ** 2


def constraint_function(params, lower, upper, confidence):
    """
    Equality constraint: the confidence interval coverage must equal the target.
    Returns 0 when the constraint is satisfied.
    """
    alpha, beta = params
    if alpha <= 0 or beta <= 0:
        return 1e6
    return (
        beta_dist.cdf(upper, alpha, beta)
        - beta_dist.cdf(lower, alpha, beta)
        - confidence
    )


# ==============================================================================
# Entropy Functions
# ==============================================================================


def beta_entropy(params):
    """Differential entropy of the beta distribution."""
    alpha, beta = params
    if alpha <= 0 or beta <= 0:
        return -1e6  # Return a large negative value for invalid parameters
    return beta_dist.entropy(alpha, beta)


def beta_entropy_grad(params):
    """Gradient of the beta distribution's entropy."""
    alpha, beta = params
    trigamma_alpha = special.polygamma(1, alpha)
    trigamma_beta = special.polygamma(1, beta)
    trigamma_alpha_beta = special.polygamma(1, alpha + beta)

    d_alpha = -(alpha - 1) * trigamma_alpha + (alpha + beta - 2) * trigamma_alpha_beta
    d_beta = -(beta - 1) * trigamma_beta + (alpha + beta - 2) * trigamma_alpha_beta
    return np.array([d_alpha, d_beta])


def beta_entropy_hess_fd(params):
    """Finite difference Hessian of the beta distribution's entropy."""
    eps = 1e-6
    alpha, beta = params
    grad_alpha_plus = beta_entropy_grad((alpha + eps, beta))
    grad = beta_entropy_grad(params)
    d2H_dalpha2 = (grad_alpha_plus[0] - grad[0]) / eps
    grad_beta_plus = beta_entropy_grad((alpha, beta + eps))
    d2H_dbeta2 = (grad_beta_plus[1] - grad[1]) / eps
    d2H_dalpha_dbeta = (grad_alpha_plus[1] - grad[1]) / eps
    return np.array([[d2H_dalpha2, d2H_dalpha_dbeta], [d2H_dalpha_dbeta, d2H_dbeta2]])


# ==============================================================================
# Gradient and Hessian Implementations
# ==============================================================================

# --- Using `betaincder` (Preferred) ---


def constraint_grad_betaincder(params, lower, upper):
    """Gradient (Jacobian) of the constraint function using the betaincder library."""
    alpha, beta = params
    dF_upper_dalpha = betaincderp(upper, alpha, beta)
    dF_lower_dalpha = betaincderp(lower, alpha, beta)
    dF_upper_dbeta = betaincderq(upper, alpha, beta)
    dF_lower_dbeta = betaincderq(lower, alpha, beta)
    return np.array(
        [dF_upper_dalpha - dF_lower_dalpha, dF_upper_dbeta - dF_lower_dbeta]
    )


def loss_grad_betaincder(params, lower, upper, confidence):
    """Gradient of the loss function using the betaincder library."""
    alpha, beta = params
    g = (
        beta_dist.cdf(upper, alpha, beta)
        - beta_dist.cdf(lower, alpha, beta)
        - confidence
    )
    constraint_grad = constraint_grad_betaincder(params, lower, upper)
    return 2 * g * constraint_grad


def loss_hessian_betaincder(params, lower, upper, confidence):
    """Hessian of the loss function using finite differences of the betaincder gradient."""
    eps = 1e-6
    alpha, beta = params
    grad_alpha_plus = loss_grad_betaincder(
        (alpha + eps, beta), lower, upper, confidence
    )
    grad = loss_grad_betaincder(params, lower, upper, confidence)
    d2L_dalpha2 = (grad_alpha_plus[0] - grad[0]) / eps
    grad_beta_plus = loss_grad_betaincder((alpha, beta + eps), lower, upper, confidence)
    d2L_dbeta2 = (grad_beta_plus[1] - grad[1]) / eps
    d2L_dalpha_dbeta = (grad_alpha_plus[1] - grad[1]) / eps
    return np.array([[d2L_dalpha2, d2L_dalpha_dbeta], [d2L_dalpha_dbeta, d2L_dbeta2]])


# --- Using Finite Differences (Fallback/Test) ---


def loss_grad_fd(params, lower, upper, confidence):
    """Gradient of the loss function using pure finite differences."""
    eps = 1e-6
    alpha, beta = params
    loss_alpha_plus = loss_function((alpha + eps, beta), lower, upper, confidence)
    loss_alpha_minus = loss_function((alpha - eps, beta), lower, upper, confidence)
    loss_beta_plus = loss_function((alpha, beta + eps), lower, upper, confidence)
    loss_beta_minus = loss_function((alpha, beta - eps), lower, upper, confidence)
    dL_dalpha = (loss_alpha_plus - loss_alpha_minus) / (2 * eps)
    dL_dbeta = (loss_beta_plus - loss_beta_minus) / (2 * eps)
    return np.array([dL_dalpha, dL_dbeta])


# ==============================================================================
# Solver Dispatcher
# ==============================================================================


def find_beta_distribution(lower, upper, confidence, alpha0, beta0, outside_odds=None):
    """
    Main solver function.
    """
    # Input validation
    if not 0 <= lower < upper <= 1:
        raise ValueError("Invalid bounds: Ensure 0 <= lower < upper <= 1.")
    if not 0 < confidence < 1:
        raise ValueError("Invalid confidence: Confidence must be between 0 and 1.")
    if alpha0 <= 0 or beta0 <= 0:
        raise ValueError(
            "Invalid initial guess: alpha0 and beta0 must be greater than 0."
        )

    initial_guess = np.array([alpha0, beta0])
    bounds = [(1e-6, None), (1e-6, None)]

    # Convert outside_odds to float if it's a number string
    if isinstance(outside_odds, str) and outside_odds.lower() not in ("maxent", "auto"):
        try:
            outside_odds = float(outside_odds)
        except ValueError as e:
            raise ValueError(
                f"Invalid outside_odds={outside_odds}. Must be a number or 'maxent' or 'auto'."
            ) from e

    if isinstance(outside_odds, (int, float)):
        # --- Method 1: fsolve for specified odds ---
        print("Using fsolve for specified odds")
        total_outside_prob = 1 - confidence
        prob_below = total_outside_prob / (outside_odds + 1)
        prob_above = total_outside_prob - prob_below

        def equations(params):
            a, b = params
            if a <= 0 or b <= 0:
                return (1e6, 1e6)
            eq1 = beta_dist.cdf(lower, a, b) - prob_below
            eq2 = beta_dist.cdf(upper, a, b) - (1 - prob_above)
            return (eq1, eq2)

        alpha, beta = fsolve(equations, initial_guess)
        return alpha, beta

    if outside_odds in ("maxent", "auto"):
        # --- Method 2: Max-Entropy (trust-constr) ---
        print("Using trust-constr for maximum entropy")

        # Progressive constraint tightening approach
        # Start with relaxed constraint and gradually tighten to exact constraint
        tolerance_levels = [1e-3, 1e-6, 1e-9, 1e-12]
        current_guess = initial_guess

        for tol in tolerance_levels:
            constraints = [
                NonlinearConstraint(
                    fun=lambda p: constraint_function(p, lower, upper, confidence),
                    lb=-tol,
                    ub=tol,
                    jac=lambda p: constraint_grad_betaincder(p, lower, upper),
                )
            ]

            res_maxent = minimize(
                fun=lambda p: -beta_entropy(p),  # Minimize the negative entropy
                x0=current_guess,
                method="trust-constr",
                jac=lambda p: -beta_entropy_grad(p),
                hess=lambda p: -beta_entropy_hess_fd(p),
                bounds=bounds,
                constraints=constraints,
                tol=tol,
            )

            if res_maxent.success:
                current_guess = res_maxent.x
            else:
                # If this tolerance level fails, try the next one
                continue

        # Final optimization with exact constraint
        final_constraints = [
            NonlinearConstraint(
                fun=lambda p: constraint_function(p, lower, upper, confidence),
                lb=0,
                ub=0,
                jac=lambda p: constraint_grad_betaincder(p, lower, upper),
            )
        ]

        final_res = minimize(
            fun=lambda p: -beta_entropy(p),
            x0=current_guess,
            method="trust-constr",
            jac=lambda p: -beta_entropy_grad(p),
            hess=lambda p: -beta_entropy_hess_fd(p),
            bounds=bounds,
            constraints=final_constraints,
            tol=1e-12,
        )

        if final_res.success:
            return final_res.x[0], final_res.x[1]
        raise RuntimeError(f"Maxent optimization failed: {final_res.message}")

    # --- Method 3: Closest Solution (trust-constr) ---
    print("Using trust-constr for closest solution")
    res = minimize(
        fun=loss_function,
        x0=initial_guess,
        args=(lower, upper, confidence),
        method="trust-constr",
        jac=loss_grad_betaincder,
        hess=loss_hessian_betaincder,
        bounds=bounds,
        tol=1e-12,
    )
    if res.success:
        return res.x[0], res.x[1]
    raise RuntimeError(f"Optimization failed: {res.message}")
