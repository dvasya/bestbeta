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

## Usage

1.  Install the required dependencies:
    ```bash
    pip install -r requirements.txt
    ```

2.  Run the Streamlit application:
    ```bash
    streamlit run app.py
    ```

3.  Open the provided URL in your web browser to access the interactive UI.

## Credits and Acknowledgements

-   Inspired by (@NunoSempere)[https://twitter.com/NunoSempere]'s `fit-beta` ([GitHub](https://github.com/quantified-uncertainty/fit-beta), [blog](https://nunosempere.com/blog/2023/03/15/fit-beta/)).
-   Uses ["Derivatives of the Incomplete Beta Function"](https://www.jstatsoft.org/article/view/v003i01) (Journal of Statistical Software, Vol. 3, No. 1, 2000), as implemented in the `betaincder` library.
-   Built with [Gemini CLI](https://github.com/google/gemini-cli).

## Repository

The source code for this project is available on GitHub:
[https://github.com/dvasya/bestbeta](https://github.com/dvasya/bestbeta)
