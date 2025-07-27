# Maximum-Entropy Beta Distribution Finder

This tool finds the beta distribution that corresponds to a given confidence interval (CI). It provides multiple solver modes:

1.  **Closest Solution**: Finds a valid beta distribution that is closest to a given starting point.
2.  **Specified Odds**: Finds a valid beta distribution where the probability outside the CI is split according to a specified ratio.
3.  **Maximum-Entropy Solver**: Finds the *unique* beta distribution that both matches the CI and has the maximum possible entropy.

For obvious reasons, this software is, and shall forever remain, in beta.

## Features

-   **Interactive UI**: A Streamlit-based web interface for easy input and visualization.
-   **Flexible Solver Modes**: Choose between different solver modes to explore the solution space.
-   **Efficient and Accurate**: Uses the `betaincder` library for fast and precise computation of beta function gradients.
-   **Modular and Extensible**: The core logic is structured as a Python library, making it easy to integrate into other projects.
-   **Robust Optimization**: Safe mode automatically detects problematic optimization landscapes and tries alternative starting points.
-   **Batch Processing**: Run multiple scenarios simultaneously and compare results.
-   **Property-Based Testing**: Comprehensive Hypothesis tests ensure mathematical correctness across parameter ranges.

## Installation

### Core Package (Library Only)
For users who only need the core `find_beta_distribution` function:

```bash
pip install bestbeta
```

### With Web Interface
For users who want the interactive Streamlit app:

```bash
pip install "bestbeta[ui]"
```

### For Developers
For developers who want to run tests and contribute:

```bash
pip install "bestbeta[dev]"
```

### Everything
For users who want everything (core + UI + dev tools):

```bash
pip install "bestbeta[all]"
```

## Usage

### Web Interface (Recommended)

If you installed with UI support:

```bash
streamlit run app.py
```

3.  Open the provided URL in your web browser to access the interactive UI.

### Command Line Interface

For programmatic use, you can also use the CLI script:

```bash
python scripts/find_beta.py -l 0.2 -u 0.8 -c 0.95 --outer_odds maxent
```

### As a Library

```python
# Import from the main package (recommended)
from bestbeta import find_beta_distribution

# Or import from the solver module directly
from bestbeta.solver import find_beta_distribution

# Find maximum entropy solution
alpha, beta = find_beta_distribution(0.2, 0.8, 0.95, 1.0, 1.0, "maxent")
print(f"alpha={alpha:.4f}, beta={beta:.4f}")
```

## Input Parameters

- **`lower`, `upper`**: Confidence interval bounds (must be between 0 and 1)
- **`confidence`**: Desired confidence level (between 0 and 1)
- **`outer_odds`**: Controls how probability mass is distributed outside the CI:
  - Empty/None: Find closest solution to starting point
  - Number (e.g., "1.0"): Equal probability below and above the interval
  - "maxent": Maximum entropy solution
  - "auto": Same as "maxent"
- **`alpha0`, `beta0`**: Starting point for optimization (defaults to 1.0, 1.0)

## Optimization Modes

The solver automatically detects the nature of your confidence interval:

- **Dense Intervals** (`upper - lower < confidence`): The optimization landscape is well-behaved with a unique global maximum. Different starting points converge to the same solution.

- **Sparse/Bimodal Intervals** (`upper - lower > confidence`): The landscape may contain saddle points or local optima. Safe mode (enabled by default) automatically detects problematic convergence and tries alternative starting points to find the global maximum entropy solution.

## Credits and Acknowledgements

-   Inspired by [@NunoSempere](https://twitter.com/NunoSempere)'s `fit-beta` ([GitHub](https://github.com/quantified-uncertainty/fit-beta), [blog](https://nunosempere.com/blog/2023/03/15/fit-beta/)).
-   Uses ["Derivatives of the Incomplete Beta Function"](https://www.jstatsoft.org/article/view/v003i01) (Journal of Statistical Software, Vol. 3, No. 1, 2000), as implemented in the `betaincder` library.
-   Built with [Gemini CLI](https://github.com/google/gemini-cli).

## Repository

The source code for this project is available on GitHub:
[https://github.com/dvasya/bestbeta](https://github.com/dvasya/bestbeta)
