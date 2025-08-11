import streamlit as st
import pandas as pd
import numpy as np
import altair as alt

st.set_page_config(page_title="Pump Curve Interpolator", layout="centered")

st.title("Pump Curve Interpolator")
st.caption("Enter pump curve data with columns: q (flow), h (head), rpm, efficiency (0–1). "
           "Then set a target RPM and flow to estimate head and efficiency.")

# ---------- Helpers ----------
def to_numeric(df, cols):
    for c in cols:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    return df

def piecewise_linear(x, xp, fp):
    """Linear interpolation with linear extrapolation on edges."""
    xp = np.asarray(xp, dtype=float)
    fp = np.asarray(fp, dtype=float)
    if len(xp) == 0:
        return np.nan
    # Sort by xp
    idx = np.argsort(xp)
    xp = xp[idx]
    fp = fp[idx]
    # Drop NaNs aligned
    mask = ~np.isnan(xp) & ~np.isnan(fp)
    xp = xp[mask]
    fp = fp[mask]
    if len(xp) == 0:
        return np.nan
    if len(xp) == 1:
        return float(fp[0])
    if x <= xp[0]:
        slope = (fp[1] - fp[0]) / (xp[1] - xp[0])
        return float(fp[0] + slope * (x - xp[0]))
    if x >= xp[-1]:
        slope = (fp[-1] - fp[-2]) / (xp[-1] - xp[-2])
        return float(fp[-1] + slope * (x - xp[-1]))
    return float(np.interp(x, xp, fp))

def sanitize_curve(df):
    """Clean and validate curve dataframe."""
    required_cols = ["q", "h", "rpm", "efficiency"]
    missing = [c for c in required_cols if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")
    df = df.copy()
    df = to_numeric(df, required_cols).dropna(subset=required_cols)
    # Keep only finite
    df = df[np.isfinite(df[required_cols]).all(axis=1)]
    # Optional: collapse duplicate (rpm, q) pairs by averaging
    df = (
        df.groupby(["rpm", "q"], as_index=False)
          .agg({"h": "mean", "efficiency": "mean"})
    )
    return df

def interpolate_across_rpm(df, target_rpm, target_flow):
    """
    Strategy:
    1) For each available rpm curve, interpolate h and eff at target_flow.
    2) Interpolate those values vs rpm to the target_rpm.
    3) If only one rpm curve exists, use affinity laws to translate:
       - Q_eq = Q_target * (rpm0 / target_rpm)
       - H_target = H(Q_eq @ rpm0) * (target_rpm / rpm0)^2
       - efficiency ~ efficiency(Q_eq @ rpm0)
    """
    rpms = np.sort(df["rpm"].unique())

    # Step 1: interpolate at target_flow on each rpm curve
    per_rpm = []
    for r in rpms:
        sub = df[df["rpm"] == r].sort_values("q")
        if len(sub) < 2:
            # With a single point we still allow linear extrapolation (flat slope=0) via piecewise logic
            pass
        h_at_flow = piecewise_linear(target_flow, sub["q"].values, sub["h"].values)
        e_at_flow = piecewise_linear(target_flow, sub["q"].values, sub["efficiency"].values)
        if np.isfinite(h_at_flow) and np.isfinite(e_at_flow):
            per_rpm.append((float(r), h_at_flow, e_at_flow))

    if len(per_rpm) >= 2:
        rp, hh, ee = map(np.array, zip(*per_rpm))
        head = piecewise_linear(target_rpm, rp, hh)
        eff = piecewise_linear(target_rpm, rp, ee)
        method = "2D interpolation across flow and RPM"
        details = {
            "rpm_used": rp.tolist(),
            "head_at_flow_per_rpm": hh.tolist(),
            "eff_at_flow_per_rpm": ee.tolist(),
        }
        return head, eff, method, details

    if len(per_rpm) == 1:
        rpm0, h_flow_r0, e_flow_r0 = per_rpm[0]
        if target_rpm <= 0 or rpm0 <= 0:
            return np.nan, np.nan, "invalid", {}
        scale = target_rpm / rpm0
        # Map target flow to equivalent flow on rpm0 curve
        q_eq = target_flow / scale
        sub = df[df["rpm"] == rpm0].sort_values("q")
        h_qeq = piecewise_linear(q_eq, sub["q"].values, sub["h"].values)
        e_qeq = piecewise_linear(q_eq, sub["q"].values, sub["efficiency"].values)
        head = h_qeq * (scale ** 2)
        eff = e_qeq  # efficiency approx unchanged across similar speeds
        method = "Affinity scaling from single RPM curve"
        details = {
            "rpm_ref": float(rpm0),
            "scale": float(scale),
            "q_equivalent_at_ref_rpm": float(q_eq),
            "head_at_qeq_ref": float(h_qeq),
            "eff_at_qeq_ref": float(e_qeq),
        }
        return head, eff, method, details

    return np.nan, np.nan, "no-data", {}

# ---------- Example data ----------
example = pd.DataFrame({
    "q":      [0, 50, 100, 150,  0, 42,  84, 126,  0, 33, 66, 99],
    "h":      [60, 58,  50,  35, 47, 45,  38,  26, 34, 32, 27, 19],
    "rpm":    [1800]*4 + [1500]*4 + [1200]*4,
    "efficiency": [0.60, 0.70, 0.78, 0.72,
                   0.58, 0.69, 0.77, 0.71,
                   0.55, 0.66, 0.74, 0.69],
})

# ---------- UI: Data entry ----------
st.subheader("Pump curve data")
use_example = st.toggle("Use example data", value=True, help="Switch off to paste/import your own data.")

if use_example:
    data = st.data_editor(
        example,
        num_rows="dynamic",
        use_container_width=True,
        height=300,
        key="editor_example",
    )
else:
    st.write("Paste or upload your data:")
    uploaded = st.file_uploader("Upload CSV with columns: q,h,rpm,efficiency", type=["csv"])
    if uploaded is not None:
        data = pd.read_csv(uploaded)
    else:
        # Provide an empty editable grid with headers
        data = st.data_editor(
            pd.DataFrame(columns=["q", "h", "rpm", "efficiency"]),
            num_rows="dynamic",
            use_container_width=True,
            height=300,
            key="editor_blank",
        )

# ---------- UI: Target inputs ----------
st.subheader("Target conditions")
c1, c2 = st.columns(2)
with c1:
    target_rpm = st.number_input("Target RPM", min_value=0.0, value=1500.0, step=50.0, format="%.2f")
with c2:
    target_flow = st.number_input("Target flow (same units as q)", min_value=0.0, value=90.0, step=1.0, format="%.3f")

calc = st.button("Calculate", type="primary")

# ---------- Compute ----------
if calc:
    try:
        curve = sanitize_curve(pd.DataFrame(data))
        if curve.empty:
            st.error("No valid data found. Please provide at least two points per RPM curve.")
        else:
            head, eff, method, details = interpolate_across_rpm(curve, target_rpm, target_flow)
            if not np.isfinite(head) or not np.isfinite(eff):
                st.error("Could not compute result. Ensure your data covers the target range or provides at least one RPM curve.")
            else:
                st.success("Calculation complete")
                m1, m2 = st.columns(2)
                with m1:
                    st.metric("Estimated head", f"{head:.3f}")
                with m2:
                    st.metric("Estimated efficiency", f"{eff:.4f}")

                with st.expander("Computation details"):
                    st.write(f"Method: {method}")
                    if details:
                        st.json(details)

                # ---------- Visualization ----------
                st.subheader("Curve visualization")
                chart_data = curve.copy()
                chart_data["rpm_str"] = chart_data["rpm"].astype(int).astype(str) + " rpm"

                base = alt.Chart(chart_data).properties(height=300, width=600)

                head_lines = base.mark_line(point=True).encode(
                    x=alt.X("q:Q", title="Flow (q)"),
                    y=alt.Y("h:Q", title="Head (h)"),
                    color=alt.Color("rpm_str:N", title="RPM"),
                    tooltip=["rpm", "q", "h", "efficiency"]
                )

                target_point = alt.Chart(pd.DataFrame({
                    "q": [target_flow],
                    "h": [head],
                    "rpm_str": [f"{int(round(target_rpm))} rpm"],
                })).mark_point(size=120, color="red").encode(
                    x="q:Q",
                    y="h:Q",
                    tooltip=[alt.Tooltip("q:Q", title="Target q"),
                             alt.Tooltip("h:Q", title="Est. head"),
                             alt.Tooltip("rpm_str:N", title="Target RPM")]
                )

                st.altair_chart(head_lines + target_point, use_container_width=True)

                st.caption("Note: Interpolation is linear in flow and RPM. With a single RPM curve, head is scaled using affinity laws (H ∝ RPM²) and efficiency is held from the equivalent-flow point.")
    except Exception as e:
        st.error(f"Error: {e}")

# ---------- Tips ----------
with st.expander("Data tips"):
    st.markdown(
        "- Provide multiple points per RPM curve for better accuracy.\n"
        "- Keep units consistent across q and h.\n"
        "- Efficiency should be between 0 and 1 (or convert % to fraction).\n"
        "- If you only have one RPM curve, the app uses pump affinity scaling to estimate at other speeds."
    )
