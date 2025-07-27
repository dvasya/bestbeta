"""
BestBeta - A Python package for beta distribution calculations

This package provides tools to find beta distributions that match given confidence intervals.
"""

__version__ = "0.1.0b1"

from .solver import find_beta_distribution

__all__ = ["find_beta_distribution"]
