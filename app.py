"""
Streamlit application for finding beta distributions.
"""

from math import isclose
from typing import Any

import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import numpy as np
from scipy.stats import beta as beta_dist

from bestbeta.solver import find_beta_distribution

st.set_page_config(layout="centered", page_icon="🎲", page_title="Best Beta")

st.title("Best Beta")

st.markdown("""
Finds the beta distribution that corresponds to a given confidence interval [[🔗 GitHub]](https://github.com/dvasya/bestbeta).
""")

# ==============================================================================
# Input Data Editor
# ==============================================================================

initial_df = pd.DataFrame(
    [
        {
            "label": "base example",
            "lower": 0.2,
            "upper": 0.8,
            "confidence": 0.95,
            "outer_odds": "",
            "alpha0": "1",
            "beta0": "1",
        },
        {
            "label": "custom start",
            "lower": 0.2,
            "upper": 0.8,
            "confidence": 0.95,
            "outer_odds": "",
            "alpha0": "2",
            "beta0": "3",
        },
        {
            "label": "enforce symmetry",
            "lower": 0.2,
            "upper": 0.8,
            "confidence": 0.95,
            "outer_odds": "1",
            "alpha0": "2",
            "beta0": "3",
        },
        {
            "label": "auto, custom start",
            "lower": 0.2,
            "upper": 0.8,
            "confidence": 0.95,
            "outer_odds": "maxent",
            "alpha0": "2",
            "beta0": "3",
        },
        {
            "label": "skewed, symmetric start",
            "lower": 0.01,
            "upper": 0.5,
            "confidence": 0.95,
            "outer_odds": "",
            "alpha0": "1",
            "beta0": "1",
        },
        {
            "label": "skewed, asymmetric start",
            "lower": 0.01,
            "upper": 0.5,
            "confidence": 0.95,
            "outer_odds": "",
            "alpha0": "3",
            "beta0": "2",
        },
        {
            "label": "skewed, equal outer_odds",
            "lower": 0.01,
            "upper": 0.5,
            "confidence": 0.95,
            "outer_odds": "1",
            "alpha0": "",
            "beta0": "",
        },
        {
            "label": "skewed, auto",
            "lower": 0.01,
            "upper": 0.5,
            "confidence": 0.95,
            "outer_odds": "maxent",
            "alpha0": "",
            "beta0": "",
        },
    ]
)

st.subheader(
    "Input Parameters",
    help=""" Inputs:
- (`lower`, `upper`) interval bounds
- `confidence` level inside the interval
- `outer_odds` ratio of probability mass below lower and above upper bound
  - `"auto"` for automatic, falls back to `"maxent"`
  - `"maxent"` for maximum entropy
  - number for custom ratio (`1.0` for equal probability below and above the interval)
  - leave blank to find closest value to the starting point
- (`alpha0`, `beta0`) starting point for the solver
  - defaults to `1.0`, `1.0`""",
)
edited_df = st.data_editor(initial_df, num_rows="dynamic")

col1, col2 = st.columns([1, 4])
with col1:
    auto_run = st.toggle(
        "Auto-run",
        value=True,
        help="Automatically run solver when page loads or data changes",
    )
with col2:
    manual_run = st.button("Run Solver")

if auto_run or manual_run:
    results = []
    fig = go.Figure()
    x = np.linspace(0, 1, 400)

    for i, row in edited_df.iterrows():
        try:
            # --- Input Validation ---
            if row["lower"] == row["upper"]:
                row["lower"], row["upper"] = row["upper"], row["lower"]
            if isclose(row["lower"], row["upper"]):
                st.error(
                    f"Row {i + 1}: Invalid bounds {row['lower']}, {row['upper']}. "
                    "Ensure lower != upper."
                )
                continue
            if not 0 <= row["lower"] < row["upper"] <= 1:
                st.error(
                    f"Row {i + 1}: Invalid bounds {row['lower']}, {row['upper']}. "
                    "Ensure 0 <= lower < upper <= 1."
                )
                continue
            if (
                not 0 < row["confidence"] < 1
                and not isclose(row["confidence"], 1)
                and not isclose(row["confidence"], 0)
            ):
                st.error(
                    f"Row {i + 1}: Confidence {row['confidence']} must be between 0 and 1."
                )
                continue

            # --- Parse outer_odds ---
            outer_odds_str = str(row["outer_odds"]).strip()
            if outer_odds_str == "":
                outer_odds: Any = None
            else:
                try:
                    outer_odds = float(outer_odds_str)
                except ValueError:
                    outer_odds = (
                        outer_odds_str  # Use as string (e.g., "maxent", "auto")
                    )

            # --- Parse label ---
            label = str(row["label"]).strip()
            display_label = label if label else f"row {i + 1}"

            # --- Parse alpha0 and beta0 ---
            alpha0_str = str(row["alpha0"]).strip()
            if not alpha0_str:
                alpha0 = 1.0
            else:
                try:
                    alpha0 = float(alpha0_str)
                except ValueError:
                    st.error(f"{display_label}: Invalid alpha0 {alpha0_str}")
                    continue

            beta0_str = str(row["beta0"]).strip()
            if not beta0_str:
                beta0 = 1.0
            else:
                try:
                    beta0 = float(beta0_str)
                except ValueError:
                    st.error(f"{display_label}: Invalid beta0 {beta0_str}")
                    continue

            # --- Run Solver ---
            alpha, beta = find_beta_distribution(
                row["lower"],
                row["upper"],
                row["confidence"],
                alpha0,
                beta0,
                outer_odds,
            )

            # --- Compute probability masses ---
            prob_below = beta_dist.cdf(row["lower"], alpha, beta)
            prob_above = 1 - beta_dist.cdf(row["upper"], alpha, beta)

            results.append(
                {
                    "label": display_label,
                    "alpha": f"{alpha:.4f}",
                    "beta": f"{beta:.4f}",
                    "P(<lower)": f"{prob_below:.4f}",
                    "P(>upper)": f"{prob_above:.4f}",
                    "outer_odds": f"{prob_above / prob_below:.4f}",
                }
            )

            # --- Add to Plot ---
            pdf = beta_dist.pdf(x, alpha, beta)
            fig.add_trace(
                go.Scatter(
                    x=x,
                    y=pdf,
                    mode="lines",
                    name=f"{display_label}: α={alpha:.2f}, β={beta:.2f}",
                )
            )

        except (RuntimeError, NotImplementedError) as e:
            st.error(f"{display_label}: {e}")
        # pylint: disable-next=broad-exception-caught
        except Exception as e:
            st.error(f"{display_label}: An unexpected error occurred: {e}")

    # ==============================================================================
    # Display Results
    # ==============================================================================

    if results:
        st.subheader("Results")
        fig.update_layout(
            xaxis_title="x",
            yaxis_title="Probability Density",
            legend_title="Parameters",
        )
        st.plotly_chart(fig, use_container_width=True)
        st.dataframe(pd.DataFrame(results), hide_index=True)
