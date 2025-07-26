"""
Finds a beta distribution based on the given confidence interval and solver mode.
"""

import click
from bestbeta.solver import find_beta_distribution


@click.command()
@click.option(
    "-l",
    "--lower",
    type=float,
    required=True,
    help="Lower bound of the confidence interval.",
)
@click.option(
    "-u",
    "--upper",
    type=float,
    required=True,
    help="Upper bound of the confidence interval.",
)
@click.option(
    "-c",
    "--confidence",
    type=float,
    required=True,
    help="The desired confidence level (e.g., 0.95).",
)
@click.option(
    "-o",
    "--outside_odds",
    type=str,
    default="",
    help=(
        "Solver mode: a fixed probability split, 'auto' for auto-select (currently only maxent), "
        "'maxent' for max entropy, or empty for closest solution."
    ),
)
@click.option(
    "-a",
    "--alpha0",
    type=float,
    default=1.0,
    help="Initial guess for the alpha parameter.",
)
@click.option(
    "-b",
    "--beta0",
    type=float,
    default=1.0,
    help="Initial guess for the beta parameter.",
)
def main(lower, upper, confidence, outside_odds, alpha0, beta0):
    """Finds a beta distribution based on the given confidence interval and solver mode."""

    # Convert outside_odds to float if it's a number string
    try:
        if outside_odds and outside_odds.lower() != "maxent":
            outside_odds = float(outside_odds)
    except ValueError:
        click.echo(
            "Error: Invalid value for --outside_odds. Must be a number or 'maxent'. "
            f"Got: {outside_odds}",
            err=True,
        )
        return

    click.echo(
        f"Finding beta for {confidence * 100}% CI [{lower}, {upper}] with mode: {outside_odds}"
    )
    click.echo(f"Initial guess: alpha0={alpha0}, beta0={beta0}")

    try:
        alpha, beta = find_beta_distribution(
            lower, upper, confidence, alpha0, beta0, outside_odds
        )
        click.echo(f"\nResult: alpha = {alpha:.6f}, beta = {beta:.6f}")
    except Exception as e:
        click.echo(f"\nAn error occurred: {e}", err=True)


if __name__ == "__main__":
    main()
