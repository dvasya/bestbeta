"""
Finds a beta distribution based on the given confidence interval and solver mode.
"""

import click
from bestbeta import find_beta_distribution


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
    "--outer_odds",
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
def main(lower, upper, confidence, outer_odds, alpha0, beta0):
    """Finds a beta distribution based on the given confidence interval and solver mode."""

    # Convert outer_odds to float if it's a number string
    try:
        if outer_odds and outer_odds.lower() not in ("maxent", "auto"):
            outer_odds = float(outer_odds)
    except ValueError:
        click.echo(
            "Error: Invalid value for --outer_odds. Must be a number or 'maxent'. "
            f"Got: {outer_odds}",
            err=True,
        )
        return

    click.echo(
        f"Finding beta for {confidence * 100}% CI [{lower}, {upper}] with mode: {outer_odds}"
    )
    click.echo(f"Initial guess: alpha0={alpha0}, beta0={beta0}")

    try:
        alpha, beta = find_beta_distribution(
            lower, upper, confidence, alpha0, beta0, outer_odds
        )
        click.echo(f"\nResult: alpha = {alpha:.6f}, beta = {beta:.6f}")
    # pylint: disable-next=broad-exception-caught
    except Exception as e:
        click.echo(f"\nAn error occurred: {e}", err=True)


if __name__ == "__main__":
    main()
