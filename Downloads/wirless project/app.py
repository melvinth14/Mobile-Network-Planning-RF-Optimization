"""Streamlit dashboard for mobile network planning in a crowded station."""

from __future__ import annotations

import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import seaborn as sns
import streamlit as st

from optimizer import OptimizationResult, optimize_network
from rf_engine import RFParameters, NetworkResult
from traffic_sim import StationLayout, generate_passengers, place_cells


st.set_page_config(
    page_title="Station RF Optimizer",
    page_icon="📡",
    layout="wide",
)


def _plot_network(
    positions: np.ndarray,
    macros: np.ndarray,
    micros: np.ndarray,
    result: NetworkResult,
    layout: StationLayout,
    title: str,
) -> plt.Figure:
    fig, ax = plt.subplots(figsize=(11, 5.5))
    palette = np.where(result.dropped, "crimson", np.where(result.sinr_db >= 10, "green", "gold"))
    ax.scatter(positions[:, 0], positions[:, 1], c=palette, s=7, alpha=0.45, linewidths=0)
    if len(macros):
        ax.scatter(macros[:, 0], macros[:, 1], marker="^", c="navy", s=180, label="Macro")
    if len(micros):
        ax.scatter(micros[:, 0], micros[:, 1], marker="s", c="purple", s=100, label="Micro/Pico")
    for index, cell in enumerate(np.vstack((macros, micros))):
        ax.text(cell[0], cell[1], str(index), color="black", fontsize=8, ha="center", va="center")
    ax.set(xlim=(0, layout.length_m), ylim=(0, layout.width_m), title=title, xlabel="Station length (m)", ylabel="Station width (m)")
    ax.grid(alpha=0.2)
    ax.legend(loc="upper right")
    fig.tight_layout()
    return fig


def _plot_sinr_distribution(before: NetworkResult, after: NetworkResult) -> plt.Figure:
    fig, ax = plt.subplots(figsize=(9, 4))
    sns.histplot(before.sinr_db, bins=40, stat="density", color="crimson", alpha=0.35, label="Before", ax=ax)
    sns.histplot(after.sinr_db, bins=40, stat="density", color="seagreen", alpha=0.35, label="After", ax=ax)
    ax.set(xlabel="SINR (dB)", ylabel="Density", title="SINR distribution")
    ax.legend()
    fig.tight_layout()
    return fig


def _plot_topology(
    positions: np.ndarray,
    cells: np.ndarray,
    result: NetworkResult,
    sample_size: int = 180,
) -> plt.Figure:
    rng = np.random.default_rng(42)
    selected = rng.choice(len(positions), min(sample_size, len(positions)), replace=False)
    graph = nx.Graph()
    graph.add_nodes_from([f"u{i}" for i in selected], bipartite=0)
    graph.add_nodes_from([f"c{i}" for i in range(len(cells))], bipartite=1)
    graph.add_edges_from((f"u{i}", f"c{result.serving_cell[i]}") for i in selected)
    coordinates = {f"u{i}": positions[i] for i in selected}
    coordinates.update({f"c{i}": cells[i] for i in range(len(cells))})
    fig, ax = plt.subplots(figsize=(9, 4.5))
    nx.draw_networkx_edges(graph, coordinates, ax=ax, alpha=0.12, width=0.5)
    nx.draw_networkx_nodes(graph, coordinates, nodelist=[f"u{i}" for i in selected], node_size=8, node_color="darkorange", ax=ax)
    nx.draw_networkx_nodes(graph, coordinates, nodelist=[f"c{i}" for i in range(len(cells))], node_size=160, node_color="royalblue", node_shape="^", ax=ax)
    ax.set_title("Sampled user-to-serving-cell topology")
    ax.set_axis_off()
    fig.tight_layout()
    return fig


def _metric_pair(label: str, before: float, after: float, suffix: str = "") -> None:
    left, right = st.columns(2)
    left.metric(f"{label} — Before", f"{before:.1f}{suffix}")
    right.metric(f"{label} — After", f"{after:.1f}{suffix}", delta=f"{after - before:+.1f}{suffix}")


def main() -> None:
    st.title("📡 Mobile Network Planning & RF Optimization")
    st.caption("Dense railway-station simulation for FRMCS / 4G / 5G planning fundamentals")
    with st.sidebar:
        st.header("Station layout")
        length = st.slider("Platform length (m)", 100, 1000, 500, 25)
        width = st.slider("Station width (m)", 40, 300, 120, 10)
        tracks = st.slider("Tracks / platform clusters", 1, 12, 4)
        concourse = st.slider("Main concourse area (m²)", 500, 30000, 6000, 500)
        st.header("Traffic and cells")
        users = st.slider("Active passengers", 1000, 10000, 4000, 500)
        macros_count = st.slider("Macro cells", 1, 8, 2)
        micros_count = st.slider("Micro / pico cells", 0, 16, 6)
        seed = st.number_input("Simulation seed", min_value=0, value=42, step=1)
        st.header("RF parameters")
        frequency = st.selectbox("Carrier frequency", [1900.0, 2600.0, 3500.0], index=2, format_func=lambda value: f"{value / 1000:g} GHz")
        tx_power = st.slider("Transmit power (dBm)", 20.0, 55.0, 43.0, 1.0)
        height = st.slider("Cell height (m)", 5.0, 40.0, 12.0, 1.0)
        gain = st.slider("Antenna gain (dBi)", 0.0, 24.0, 15.0, 1.0)
        noise = st.slider("Noise floor (dBm)", -120.0, -70.0, -96.0, 1.0)
        reuse = st.selectbox("Frequency reuse factor", [1, 3, 7], index=1)
        run = st.button("Run simulation", type="primary", use_container_width=True)

    if "simulation" not in st.session_state or run:
        layout = StationLayout(length, width, tracks, concourse)
        traffic = generate_passengers(users, layout, int(seed))
        macros, micros = place_cells(layout, macros_count, micros_count)
        cells = np.vstack((macros, micros))
        params = RFParameters(
            frequency_mhz=frequency,
            tx_power_dbm=tx_power,
            cell_height_m=height,
            antenna_gain_dbi=gain,
            noise_floor_dbm=noise,
            reuse_factor=reuse,
        )
        with st.spinner("Calculating propagation, CCI, SINR, and capacity..."):
            result = optimize_network(traffic.positions, traffic.demands_mbps, cells, params, int(seed))
        st.session_state.simulation = (layout, traffic, macros, micros, cells, result)

    layout, traffic, macros, micros, cells, optimization = st.session_state.simulation
    before, after = optimization.before, optimization.after
    st.subheader("Before vs after optimization")
    columns = st.columns(4)
    metrics = [
        ("Coverage rate", before.coverage_rate, after.coverage_rate, "%"),
        ("Mean SINR", before.mean_sinr_db, after.mean_sinr_db, " dB"),
        ("Congested cells", float(np.sum(before.overloaded)), float(np.sum(after.overloaded)), ""),
        ("Total throughput", before.total_throughput_mbps, after.total_throughput_mbps, " Mbps"),
    ]
    for column, (label, old, new, suffix) in zip(columns, metrics):
        with column:
            st.metric(f"{label} — Before", f"{old:.1f}{suffix}")
            st.metric(f"{label} — After", f"{new:.1f}{suffix}", delta=f"{new - old:+.1f}{suffix}")

    tab_map, tab_sinr, tab_topology, tab_cells = st.tabs(["Coverage map", "SINR analysis", "Topology", "Cell details"])
    with tab_map:
        st.pyplot(_plot_network(traffic.positions, macros, micros, after, layout, "Optimized station coverage and user status"), clear_figure=True)
    with tab_sinr:
        st.pyplot(_plot_sinr_distribution(before, after), clear_figure=True)
        st.dataframe(
            {
                "Metric": ["Dropped users", "Mean user demand", "Optimization iterations"],
                "Before": [f"{np.mean(before.dropped) * 100:.1f}%", f"{np.mean(traffic.demands_mbps):.2f} Mbps", "-"],
                "After": [f"{np.mean(after.dropped) * 100:.1f}%", f"{np.mean(traffic.demands_mbps):.2f} Mbps", str(optimization.iterations)],
            },
            hide_index=True,
            use_container_width=True,
        )
    with tab_topology:
        st.pyplot(_plot_topology(traffic.positions, cells, after), clear_figure=True)
        st.caption("Topology is sampled for readability; RF calculations use every active passenger.")
    with tab_cells:
        st.dataframe(
            {
                "Cell": np.arange(len(cells)),
                "Type": ["Macro"] * len(macros) + ["Micro/Pico"] * len(micros),
                "Users before": before.cell_users,
                "Users after": after.cell_users,
                "Utilization before": np.round(before.cell_utilization, 2),
                "Utilization after": np.round(after.cell_utilization, 2),
                "Channel after": optimization.optimized_channels,
                "Power after (dBm)": np.round(optimization.optimized_power_dbm, 1),
            },
        )


if __name__ == "__main__":
    main()
