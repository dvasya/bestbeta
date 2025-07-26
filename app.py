"""
Streamlit application for finding beta distributions.
"""

import warnings
from math import isclose
from typing import Any

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from scipy.stats import beta as beta_dist

from bestbeta.solver import beta_entropy, find_beta_distribution

st.set_page_config(layout="centered", page_icon="🎲", page_title="Best Beta")

st.title("Best Beta")

st.markdown(
    """
Finds the beta distribution that corresponds to a given confidence interval
[[🔗 GitHub]](https://github.com/dvasya/bestbeta).
"""
)

# ==============================================================================
# Input Data Editor
# ==============================================================================

DEFAULT_DATA = pd.DataFrame(
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
            "confidence": 0.49,
            "outer_odds": "",
            "alpha0": "1",
            "beta0": "1",
        },
        {
            "label": "skewed, asymmetric start",
            "lower": 0.01,
            "upper": 0.5,
            "confidence": 0.49,
            "outer_odds": "",
            "alpha0": "3",
            "beta0": "2",
        },
        {
            "label": "skewed, equal outer_odds",
            "lower": 0.01,
            "upper": 0.5,
            "confidence": 0.49,
            "outer_odds": "1",
            "alpha0": "2",
            "beta0": "3",
        },
        {
            "label": "skewed, auto",
            "lower": 0.01,
            "upper": 0.5,
            "confidence": 0.49,
            "outer_odds": "maxent",
            "alpha0": "",
            "beta0": "",
        },
    ]
)

# Initialize session state for the DataFrame if not already present
if "df" not in st.session_state:
    st.session_state.df = DEFAULT_DATA.copy()
# Initialize session state for user's edited data backup
if "user_edited_df" not in st.session_state:
    st.session_state.user_edited_df = None
# Initialize a key for the data editor to force re-render
if "data_editor_key" not in st.session_state:
    st.session_state.data_editor_key = 0
# Initialize auto-run state
if "auto_run_state" not in st.session_state:
    st.session_state.auto_run_state = True
# Initialize flag to track if current data is example data
if "is_example_data" not in st.session_state:
    st.session_state.is_example_data = True

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

# Data editor
edited_df = st.data_editor(
    st.session_state.df,
    num_rows="dynamic",
    key=f"data_editor_{st.session_state.data_editor_key}",
)

# Update session state with edited data
if not edited_df.equals(st.session_state.df):
    st.session_state.df = edited_df.copy()
    # Update the example data flag
    st.session_state.is_example_data = edited_df.equals(DEFAULT_DATA)

# Buttons for table manipulation and running the solver
col_buttons = st.columns(5)
with col_buttons[0]:
    st.session_state.auto_run_state = st.toggle(
        "Auto-run",
        value=st.session_state.auto_run_state,
        help="Automatically run solver when page loads or data changes",
        key="auto_run_toggle",
    )
with col_buttons[1]:
    manual_run = st.button("Run Solver", type="primary")
with col_buttons[2]:
    if st.button("Clear Table"):
        # If current df is not empty, save it as user_edited_df before clearing
        if not st.session_state.df.empty:
            st.session_state.user_edited_df = st.session_state.df.copy()
        else:
            st.session_state.user_edited_df = (
                None  # Ensure no restore option if starting from empty
            )

        st.session_state.df = pd.DataFrame(
            columns=DEFAULT_DATA.columns
        )  # Clear the table
        st.session_state.data_editor_key += 1  # Increment key to force re-render
        st.session_state.auto_run_state = False  # Disable auto-run
        st.session_state.is_example_data = False  # Clear table is not example data
        st.rerun()
with col_buttons[3]:
    # Use the tracked state instead of doing comparison every time
    is_currently_example = st.session_state.is_example_data

    if st.session_state.user_edited_df is not None and is_currently_example:
        # Case 3: Has backup, example data in widget = Restore button
        if st.button("Restore"):
            st.session_state.df = st.session_state.user_edited_df.copy()
            st.session_state.user_edited_df = None  # Clear backup after restoring
            st.session_state.data_editor_key += 1  # Increment key to force re-render
            st.session_state.is_example_data = False
            st.rerun()
    else:
        # Cases 1, 2, 4: Show Example button
        button_disabled = (
            is_currently_example and st.session_state.user_edited_df is None
        )
        if st.button("Example", disabled=button_disabled):
            # Cases 2, 4: Store current data as backup if it's not empty and different from example
            if not st.session_state.df.empty and not is_currently_example:
                st.session_state.user_edited_df = st.session_state.df.copy()
            # Now, load the default data
            st.session_state.df = DEFAULT_DATA.copy()
            st.session_state.data_editor_key += 1
            st.session_state.is_example_data = True
            st.rerun()

if st.session_state.auto_run_state or manual_run:
    results = []
    fig = go.Figure()
    x = np.linspace(0, 1, 400)

    for i, row in edited_df.iterrows():
        # Initialize display_label at the start of the loop iteration
        display_label = f"row {i + 1}"
        current_warnings = []
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")  # Always record warnings
            try:
                # --- Parse label ---
                label_val = row.get(
                    "label", ""
                )  # Use .get to safely retrieve, default to empty string
                if label_val:
                    display_label = str(label_val).strip()

                # --- Input Validation ---
                # Ensure all numeric inputs are floats
                try:
                    lower_val = float(row["lower"])
                    upper_val = float(row["upper"])
                    confidence_val = float(row["confidence"])
                except ValueError:
                    st.error(
                        f"{display_label}: Invalid numeric input in lower, upper, or confidence."
                    )
                    continue

                if lower_val == upper_val:
                    lower_val, upper_val = (
                        upper_val,
                        lower_val,
                    )  # This swap logic seems odd, should be handled by validation
                if isclose(lower_val, upper_val):
                    st.error(
                        f"{display_label}: Invalid bounds {lower_val}, {upper_val}. "
                        "Ensure lower != upper."
                    )
                    continue
                if not 0 <= lower_val < upper_val <= 1:
                    st.error(
                        f"{display_label}: Invalid bounds {lower_val}, {upper_val}. "
                        "Ensure 0 <= lower_val < upper_val <= 1."
                    )
                    continue
                if (
                    not 0 < confidence_val < 1
                    and not isclose(confidence_val, 1)
                    and not isclose(confidence_val, 0)
                ):
                    st.error(
                        f"{display_label}: Confidence {confidence_val} must be between 0 and 1."
                    )
                    continue

                # --- Parse outer_odds ---
                outer_odds_input = row["outer_odds"]
                outer_odds: Any = None  # Default to None

                if pd.isna(outer_odds_input) or (
                    isinstance(outer_odds_input, str) and outer_odds_input.strip() == ""
                ):
                    outer_odds = None  # Explicitly set to None for empty/NA
                elif isinstance(outer_odds_input, (int, float)):
                    outer_odds = float(outer_odds_input)  # Already a number
                else:  # It's a string
                    outer_odds_str = str(outer_odds_input).strip().lower()
                    if outer_odds_str in ("maxent", "auto"):
                        outer_odds = outer_odds_str  # Keep as string for these keywords
                    else:
                        try:
                            outer_odds = float(
                                outer_odds_input
                            )  # Try converting other strings to float
                        except ValueError:
                            st.error(
                                f"{display_label}: Invalid outer_odds={outer_odds_input}. "
                                "Must be a number, 'maxent', or 'auto'."
                            )
                            continue

                # --- Parse alpha0 and beta0 ---
                alpha0_val = row["alpha0"]
                if pd.isna(alpha0_val) or str(alpha0_val).strip() == "":
                    alpha0 = 1.0
                else:
                    try:
                        alpha0 = float(alpha0_val)
                    except ValueError:
                        st.error(f"{display_label}: Invalid alpha0 {alpha0_val}")
                        continue

                beta0_val = row["beta0"]
                if pd.isna(beta0_val) or str(beta0_val).strip() == "":
                    beta0 = 1.0
                else:
                    try:
                        beta0 = float(beta0_val)
                    except ValueError:
                        st.error(f"{display_label}: Invalid beta0 {beta0_val}")
                        continue

                # --- Run Solver ---
                alpha, beta = find_beta_distribution(
                    lower_val,
                    upper_val,
                    confidence_val,
                    alpha0,
                    beta0,
                    outer_odds,
                )

                # --- Compute probability masses and entropy ---
                prob_below = beta_dist.cdf(lower_val, alpha, beta)
                prob_above = 1 - beta_dist.cdf(upper_val, alpha, beta)
                entropy_val = beta_entropy(np.array([alpha, beta]))

                results.append(
                    {
                        "label": display_label,
                        "alpha": f"{alpha:.4f}",
                        "beta": f"{beta:.4f}",
                        "P(<lower)": f"{prob_below:.4f}",
                        "P(>upper)": f"{prob_above:.4f}",
                        "outer_odds": f"{prob_above / prob_below:.4f}",
                        "entropy": f"{entropy_val:.4f}",
                        "warnings": "; ".join([str(warn.message) for warn in w]),
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
