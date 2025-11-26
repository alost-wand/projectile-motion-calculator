"""
Streamlit Projectile Motion App
Converted from Flask -> Streamlit with improved UX, validation,
plot download, CSV export, session state, and tidy layout.
"""

import io
import math
from typing import Tuple

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import streamlit as st

g = 9.8  # m/s^2

st.set_page_config(page_title="Projectile Motion Lab", layout="wide", initial_sidebar_state="expanded")

# --- Helpers ---
def solve_time_to_ground(uy: float, h: float) -> float:
    """
    Positive root of 0.5*g*t^2 - uy*t - h = 0
    Returns time until projectile hits ground (t >= 0).
    """
    disc = uy**2 + 2 * g * h
    return (uy + math.sqrt(disc)) / g

def trajectory(u: float, angle_deg: float, h: float, motion_type: str, n_points: int = 300):
    angle_rad = math.radians(angle_deg)
    ux = u * math.cos(angle_rad)
    uy = u * math.sin(angle_rad)

    if motion_type == "Horizontal (initial velocity parallel to ground)":
        # horizontal launch from height h: ux = u, uy = 0
        uy = 0.0
        ux = u
        t_max = math.sqrt(2 * h / g) if h >= 0 else 0.0
        t = np.linspace(0, t_max, max(2, n_points))
        x = ux * t
        y = h - 0.5 * g * t**2

    elif motion_type == "Vertical (straight up/down)":
        # vertical motion: x = 0
        ux = 0.0
        uy = u
        t_max = solve_time_to_ground(uy, h)
        t = np.linspace(0, t_max, max(2, n_points))
        x = np.zeros_like(t)
        y = h + uy * t - 0.5 * g * t**2

    else:  # Angled launch
        t_max = solve_time_to_ground(uy, h)
        t = np.linspace(0, t_max, max(2, n_points))
        x = ux * t
        y = h + uy * t - 0.5 * g * t**2

    # clip tiny negative numerical values to zero
    y = np.where(y < 0, 0.0, y)
    return t, x, y

def compute_results(u: float, angle_deg: float, h: float, motion_type: str):
    angle_rad = math.radians(angle_deg)
    ux = u * math.cos(angle_rad)
    uy = u * math.sin(angle_rad)

    if motion_type == "Horizontal (initial velocity parallel to ground)":
        time = math.sqrt(2 * h / g) if h >= 0 else 0.0
        rng = ux * time
        max_height = h

    elif motion_type == "Vertical (straight up/down)":
        # total time until hits ground (up and down combined)
        time = solve_time_to_ground(uy, h)
        rng = 0.0
        max_height = h + (uy**2) / (2 * g)

    else:  # Angled
        time = solve_time_to_ground(uy, h)
        rng = ux * time
        max_height = h + (uy**2) / (2 * g)

    return {"time": time, "range": rng, "max_height": max_height}

# --- Sidebar controls ---
st.sidebar.header("Inputs")
motion_type = st.sidebar.selectbox(
    "Motion type",
    options=[
        "Angled (projectile)",
        "Horizontal (initial velocity parallel to ground)",
        "Vertical (straight up/down)",
    ],
)

col1, col2 = st.sidebar.columns(2)
with col1:
    u = st.number_input("Initial speed (m/s)", min_value=0.0, value=20.0, step=1.0, format="%.3f")
with col2:
    angle = st.number_input("Launch angle (deg)", min_value=0.0, max_value=90.0, value=45.0, step=1.0, format="%.3f")

h = st.sidebar.number_input("Initial height (m)", value=0.0, format="%.3f")

# Quick presets
st.sidebar.markdown("---")
if st.sidebar.button("Set: flat ground horizontal (u=10 m/s)"):
    u = 10.0
    angle = 0.0
    h = 0.0
    # set values into session_state so UI updates
    st.session_state.update({"u": u, "angle": angle, "h": h})

if st.sidebar.button("Set: typical angled (u=20 m/s, 45°)"):
    u = 20.0
    angle = 45.0
    h = 0.0
    st.session_state.update({"u": u, "angle": angle, "h": h})

# Ensure session state mirrors latest numeric inputs (useful when presets pressed)
st.session_state.setdefault("u", u)
st.session_state["u"] = u
st.session_state.setdefault("angle", angle)
st.session_state["angle"] = angle
st.session_state.setdefault("h", h)
st.session_state["h"] = h

st.sidebar.markdown("---")
st.sidebar.write("Gravity used: {:.2f} m/s²".format(g))

# --- Main layout ---
st.title("Projectile Motion — Interactive Lab")
st.write("""
Simple physics explorer for projectile motion.  
Enter initial speed, launch angle, and initial height. The app solves the motion and plots the trajectory.
""")

# Validate inputs
if u < 0:
    st.error("Initial speed must be non-negative.")
else:
    try:
        # Compute trajectory & results
        t_vals, x_vals, y_vals = trajectory(u=u, angle_deg=angle, h=h, motion_type=motion_type)
        res = compute_results(u=u, angle_deg=angle, h=h, motion_type=motion_type)

        # Metrics
        st.subheader("Results")
        c1, c2, c3 = st.columns(3)
        c1.metric("Flight time (s)", f"{res['time']:.4f}")
        c2.metric("Range (m)", f"{res['range']:.4f}")
        c3.metric("Max height (m)", f"{res['max_height']:.4f}")

        # Interactive plot
        st.subheader("Trajectory")
        fig, ax = plt.subplots(figsize=(8, 4.8))
        ax.plot(x_vals, y_vals, linewidth=2)
        ax.set_title("Projectile trajectory")
        ax.set_xlabel("Horizontal distance (m)")
        ax.set_ylabel("Vertical height (m)")
        ax.grid(True)

        # Adjust x-limits and y-limits to frame data nicely
        xpad = max(0.05 * (x_vals.max() - x_vals.min() if x_vals.size else 1), 0.5)
        ypad = max(0.05 * (y_vals.max() - y_vals.min() if y_vals.size else 1), 0.5)
        ax.set_xlim(left=min(0, x_vals.min()) - xpad, right=x_vals.max() + xpad)
        ax.set_ylim(bottom=0 - ypad, top=y_vals.max() + ypad)

        st.pyplot(fig)

        # Prepare CSV & PNG download
        df = pd.DataFrame({"t (s)": t_vals, "x (m)": x_vals, "y (m)": y_vals})

        buf = io.BytesIO()
        fig.savefig(buf, format="png", bbox_inches="tight")
        buf.seek(0)
        png_bytes = buf.getvalue()

        cold1, cold2 = st.columns([1, 1])
        with cold1:
            st.download_button("Download trajectory CSV", data=df.to_csv(index=False).encode("utf-8"),
                               file_name="trajectory.csv", mime="text/csv")
        with cold2:
            st.download_button("Download plot (PNG)", data=png_bytes, file_name="trajectory.png", mime="image/png")

        # Show data table collapsible
        with st.expander("Trajectory data (first 20 rows)"):
            st.dataframe(df.head(20))

        # Small notes / physics checks
        st.markdown("---")
        st.write("Notes:")
        st.write("- Flight time solved from the quadratic equation for vertical motion.")
        st.write("- For horizontal launches from height, initial vertical speed is 0 m/s.")
        st.write("- Tiny numerical negatives (y < 0) are clipped to 0 for display purposes.")

    except Exception as e:
        st.exception(f"An error occurred while computing the motion: {e}")
