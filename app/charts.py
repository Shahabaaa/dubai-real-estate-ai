"""Plotly chart helpers for the Streamlit app."""
import plotly.express as px
import plotly.graph_objects as go

TEAL, GOLD, NAVY = "#0e7c86", "#d4a24c", "#0b3c49"
STATUS_COLORS = {"Off-plan": GOLD, "Existing": TEAL}


def _style(fig, height=340):
    fig.update_layout(height=height, margin=dict(l=10, r=10, t=50, b=10), template="plotly_white",
                      font=dict(size=13), legend=dict(orientation="h", y=1.12, x=0))
    return fig


def range_chart(price: float):
    fig = go.Figure()
    fig.add_shape(type="rect", x0=price * 0.8, x1=price * 1.2, y0=0.25, y1=0.75,
                  fillcolor="rgba(14,124,134,0.18)", line_width=0)
    fig.add_shape(type="rect", x0=price * 0.9, x1=price * 1.1, y0=0.25, y1=0.75,
                  fillcolor="rgba(14,124,134,0.45)", line_width=0)
    fig.add_trace(go.Scatter(x=[price], y=[0.5], mode="markers", showlegend=False,
                             marker=dict(size=18, color=GOLD, line=dict(color="white", width=2)),
                             hovertemplate="Estimate: AED %{x:,.0f}<extra></extra>"))
    for f, label in [(0.8, "-20%"), (0.9, "-10%"), (1.1, "+10%"), (1.2, "+20%")]:
        fig.add_annotation(x=price * f, y=0.88, text=f"{label}<br>{price * f:,.0f}", showarrow=False,
                           font=dict(size=11))
    fig.update_xaxes(range=[price * 0.74, price * 1.26], tickformat=",.0f", title="AED")
    fig.update_yaxes(visible=False, range=[0, 1.1])
    fig.update_layout(title="Estimate and likely range (dark band: within 10%, light band: within 20%)")
    return _style(fig, 260)


def ppsm_position(row, predicted_ppsm: float, area: str):
    p25, med, p75 = float(row["p25_ppsm"]), float(row["median_ppsm"]), float(row["p75_ppsm"])
    lo, hi = min(p25, predicted_ppsm) * 0.85, max(p75, predicted_ppsm) * 1.15
    fig = go.Figure()
    fig.add_shape(type="rect", x0=p25, x1=p75, y0=0.25, y1=0.75, fillcolor="rgba(11,60,73,0.25)", line_width=0)
    fig.add_shape(type="line", x0=med, x1=med, y0=0.2, y1=0.8, line=dict(color=NAVY, width=3))
    fig.add_trace(go.Scatter(x=[predicted_ppsm], y=[0.5], mode="markers", showlegend=False,
                             marker=dict(size=18, color=GOLD, line=dict(color="white", width=2)),
                             hovertemplate="This property: AED %{x:,.0f} per sq m<extra></extra>"))
    fig.add_annotation(x=med, y=0.92, text=f"Area median<br>{med:,.0f}", showarrow=False, font=dict(size=11))
    fig.update_xaxes(range=[lo, hi], tickformat=",.0f", title="AED per sq m")
    fig.update_yaxes(visible=False, range=[0, 1.1])
    fig.update_layout(title=f"Price per sq m vs typical sales in {area} (shaded: middle 50% of sales, last 12 months)")
    return _style(fig, 260)


def area_bar(stats, metric: str, top_n: int):
    col = "median_ppsm" if metric == "Median price per sq m" else "median_price_aed"
    d = stats.nlargest(top_n, col).sort_values(col)
    fig = px.bar(d, x=col, y="area", orientation="h", text=col, color=col,
                 color_continuous_scale=[[0, "#9fd3d8"], [1, NAVY]], hover_data={"n_sales": ":,"})
    fig.update_traces(texttemplate="%{text:,.0f}", textposition="outside")
    fig.update_layout(coloraxis_showscale=False, xaxis_title="AED", yaxis_title="",
                      title=f"Most expensive areas: {metric.lower()} (last 12 months)")
    return _style(fig, max(380, 26 * top_n + 100))


def bubble(stats):
    d = stats.copy()
    d["label"] = ""
    top = d.nlargest(8, "n_sales").index
    d.loc[top, "label"] = d.loc[top, "area"]
    fig = px.scatter(d, x="median_ppsm", y="offplan_share", size="n_sales", color="median_price_aed",
                     hover_name="area", text="label", size_max=45,
                     color_continuous_scale=[[0, TEAL], [1, GOLD]],
                     labels={"median_ppsm": "Median AED per sq m", "offplan_share": "Off-plan share of sales",
                             "median_price_aed": "Median price (AED)", "n_sales": "Sales"})
    fig.update_traces(textposition="top center")
    fig.update_yaxes(tickformat=".0%")
    fig.update_layout(title="Areas: price level vs off-plan share (bubble size = number of sales)")
    return _style(fig, 460)


def trend_line(trend, property_type: str):
    d = trend[trend["property_type"] == property_type]
    fig = px.line(d, x="quarter", y="median_ppsm", color="status", markers=True,
                  color_discrete_map=STATUS_COLORS,
                  labels={"median_ppsm": "Median AED per sq m", "quarter": "", "status": ""})
    fig.update_layout(title=f"Median price per sq m by quarter: {property_type.lower()}s")
    return _style(fig, 360)


def volume_bar(trend, property_type: str):
    d = trend[trend["property_type"] == property_type]
    fig = px.bar(d, x="quarter", y="n_sales", color="status", barmode="stack",
                 color_discrete_map=STATUS_COLORS, labels={"n_sales": "Sales", "quarter": "", "status": ""})
    fig.update_layout(title=f"Number of sales by quarter: {property_type.lower()}s")
    return _style(fig, 320)
