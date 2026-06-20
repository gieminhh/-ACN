from __future__ import annotations

import time
from pathlib import Path

import pandas as pd
import plotly.graph_objects as go
import streamlit as st
import torch

from nazari_vrp import (
    NazariVRPModel,
    VRPConfig,
    generate_batch,
    learning_based_insertion,
    rollout,
    routes_from_actions,
    validate_actions,
)
from nazari_vrp.train import load_checkpoint


CHECKPOINT_PATH = Path("nazari_vrp_checkpoint.pt")


st.set_page_config(page_title="Learning-based Insertion for CVRP", layout="wide")


@st.cache_resource
def get_model(checkpoint_mtime: float | None) -> tuple[NazariVRPModel, str]:
    if CHECKPOINT_PATH.exists():
        model = load_checkpoint(CHECKPOINT_PATH, device="cpu")
        return model, f"Loaded checkpoint: {CHECKPOINT_PATH}"

    model = NazariVRPModel()
    model.eval()
    return model, "No checkpoint found. Showing an untrained model for code demo."


def plot_routes(batch, routes: list[list[int]]) -> go.Figure:
    coords = batch.coords_with_depot()[0].detach().cpu()
    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=[float(coords[0, 0])],
            y=[float(coords[0, 1])],
            mode="markers+text",
            marker=dict(size=16, color="#ef4444", symbol="star"),
            text=["Depot"],
            textposition="top center",
            name="Depot",
        )
    )
    fig.add_trace(
        go.Scatter(
            x=coords[1:, 0].tolist(),
            y=coords[1:, 1].tolist(),
            mode="markers+text",
            marker=dict(size=9, color="#2563eb"),
            text=[str(i) for i in range(1, coords.size(0))],
            textposition="top center",
            name="Customers",
        )
    )

    palette = [
        "#111827",
        "#16a34a",
        "#f97316",
        "#7c3aed",
        "#0891b2",
        "#be123c",
        "#4d7c0f",
    ]
    for idx, route in enumerate(routes, start=1):
        route_coords = coords[route]
        fig.add_trace(
            go.Scatter(
                x=route_coords[:, 0].tolist(),
                y=route_coords[:, 1].tolist(),
                mode="lines+markers",
                line=dict(width=3, color=palette[(idx - 1) % len(palette)]),
                marker=dict(size=6),
                name=f"Route {idx}",
            )
        )

    fig.update_layout(
        height=620,
        margin=dict(l=20, r=20, t=35, b=20),
        xaxis=dict(range=[-0.05, 1.05], title="x"),
        yaxis=dict(range=[-0.05, 1.05], title="y", scaleanchor="x", scaleratio=1),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0),
    )
    return fig


def route_loads(routes: list[list[int]], demand: torch.Tensor) -> list[float]:
    loads = []
    for route in routes:
        total = 0.0
        for node in route:
            if node > 0:
                total += float(demand[node - 1].item())
        loads.append(total)
    return loads


def flatten_routes(routes: list[list[int]]) -> list[int]:
    actions: list[int] = []
    for route in routes:
        actions.extend(route[1:])
    return actions


def direct_vehicle_count(actions: torch.Tensor) -> int:
    return len(routes_from_actions(actions.detach().cpu().tolist()))


def benchmark_instance(
    model: NazariVRPModel,
    config: VRPConfig,
    seed: int,
    map_idx: int,
    beam_width: int,
    sampling_samples: int,
) -> dict[str, object]:
    batch = generate_batch(1, config, seed=seed + map_idx)

    start = time.perf_counter()
    greedy = rollout(model, batch, decode_type="greedy")
    greedy_time = time.perf_counter() - start
    greedy_routes = routes_from_actions(greedy.actions[0].detach().cpu().tolist())
    greedy_valid, _ = validate_actions(
        greedy.actions[0].detach().cpu().tolist(),
        batch.demand[0].cpu(),
        config.capacity,
    )

    best_sampling_cost = float("inf")
    sampling_total_time = 0.0
    best_sampling_vehicles = 0
    best_sampling_valid = False
    for sample_idx in range(sampling_samples):
        torch.manual_seed(seed * 10_000 + map_idx * 1_000 + sample_idx)
        start = time.perf_counter()
        sampling = rollout(model, batch, decode_type="sampling")
        sample_time = time.perf_counter() - start
        sampling_total_time += sample_time
        sample_cost = float(sampling.cost.item())
        if sample_cost < best_sampling_cost:
            best_sampling_cost = sample_cost
            best_sampling_vehicles = direct_vehicle_count(sampling.actions[0])
            best_sampling_valid, _ = validate_actions(
                sampling.actions[0].detach().cpu().tolist(),
                batch.demand[0].cpu(),
                config.capacity,
            )

    start = time.perf_counter()
    proposed_greedy = learning_based_insertion(model, batch, decode_type="greedy")
    proposed_greedy_time = time.perf_counter() - start

    start = time.perf_counter()
    proposed_beam = learning_based_insertion(
        model,
        batch,
        decode_type="beam",
        beam_width=beam_width,
    )
    proposed_beam_time = time.perf_counter() - start

    if float(proposed_beam.cost.item()) < float(proposed_greedy.cost.item()):
        proposed = proposed_beam
        proposed_source = "beam"
    else:
        proposed = proposed_greedy
        proposed_source = "guided_best_insertion"
    proposed_time = proposed_greedy_time + proposed_beam_time

    proposed_valid, _ = validate_actions(
        flatten_routes(proposed.routes[0]),
        batch.demand[0].cpu(),
        config.capacity,
    )

    return {
        "map": f"MAP_{map_idx + 1:02d}",
        "greedy_cost": round(float(greedy.cost.item()), 4),
        "greedy_time_s": round(greedy_time, 4),
        "greedy_vehicles": len(greedy_routes),
        "greedy_valid": greedy_valid,
        "sampling_cost": round(best_sampling_cost, 4),
        "sampling_time_s": round(sampling_total_time, 4),
        "sampling_vehicles": best_sampling_vehicles,
        "sampling_valid": best_sampling_valid,
        "proposed_cost": round(float(proposed.cost.item()), 4),
        "proposed_time_s": round(proposed_time, 4),
        "proposed_vehicles": len(proposed.routes[0]),
        "proposed_source": proposed_source,
        "proposed_valid": proposed_valid,
    }


def run_comparison(
    model: NazariVRPModel,
    config: VRPConfig,
    seed: int,
    instances: int,
    beam_width: int,
    sampling_samples: int,
) -> pd.DataFrame:
    rows = [
        benchmark_instance(
            model=model,
            config=config,
            seed=seed,
            map_idx=idx,
            beam_width=beam_width,
            sampling_samples=sampling_samples,
        )
        for idx in range(instances)
    ]
    return pd.DataFrame(rows)


def best_average_label(df: pd.DataFrame) -> str:
    means = {
        "Greedy": float(df["greedy_cost"].mean()),
        "Sampling": float(df["sampling_cost"].mean()),
        "Proposed": float(df["proposed_cost"].mean()),
    }
    return min(means, key=means.get)


st.title("Learning-based Insertion Heuristic for CVRP")
st.caption(
    "Nazari-style RNN + attention learns a customer priority order, then Best Insertion places each customer at the lowest extra-distance position."
)

with st.sidebar:
    st.header("Tham số")
    num_customers = st.selectbox("Customers", [10, 20, 50, 100], index=1)
    config = VRPConfig(num_customers=num_customers)
    seed = st.number_input("Seed", min_value=0, max_value=999999, value=42, step=1)
    decode_label = st.radio("Priority decoder", ["Greedy", "Beam Search"], horizontal=False)
    beam_width = st.slider("Beam width", min_value=2, max_value=20, value=5, step=1)
    st.divider()
    comparison_instances = st.slider("Comparison maps", min_value=3, max_value=50, value=10, step=1)
    sampling_samples = st.slider("Sampling samples", min_value=1, max_value=50, value=8, step=1)

checkpoint_mtime = CHECKPOINT_PATH.stat().st_mtime if CHECKPOINT_PATH.exists() else None
model, model_status = get_model(checkpoint_mtime)

batch = generate_batch(1, config, seed=int(seed))
with torch.no_grad():
    result = learning_based_insertion(
        model,
        batch,
        decode_type="beam" if decode_label == "Beam Search" else "greedy",
        beam_width=beam_width,
    )

routes = result.routes[0]
priority_order = result.priority_orders[0]
actions_for_validation = []
for route in routes:
    actions_for_validation.extend(route[1:])
is_valid, valid_message = validate_actions(
    actions_for_validation,
    batch.demand[0].cpu(),
    config.capacity,
)
loads = route_loads(routes, batch.demand[0].cpu())

st.info(model_status)

comparison_tab, map_tab, explanation_tab = st.tabs(["Bảng so sánh", "Bản đồ", "Giải thích"])

with comparison_tab:
    run_clicked = st.button("Chạy / cập nhật bảng", type="primary")
    comparison_key = (
        num_customers,
        int(seed),
        beam_width,
        comparison_instances,
        sampling_samples,
        checkpoint_mtime,
    )

    if run_clicked:
        with st.spinner("Đang chạy so sánh..."):
            st.session_state["comparison_df"] = run_comparison(
                model=model,
                config=config,
                seed=int(seed),
                instances=comparison_instances,
                beam_width=beam_width,
                sampling_samples=sampling_samples,
            )
            st.session_state["comparison_key"] = comparison_key
            st.session_state["comparison_stamp"] = time.strftime("%Y-%m-%d %H:%M:%S")

    df = st.session_state.get("comparison_df")
    if df is not None and st.session_state.get("comparison_key") == comparison_key:
        st.caption(f"Lần chạy gần nhất: {st.session_state.get('comparison_stamp')}")
        st.dataframe(df, width="stretch", hide_index=True)

        summary = st.columns(4)
        summary[0].metric("Greedy cost TB", f"{df['greedy_cost'].mean():.4f}")
        summary[1].metric("Sampling cost TB", f"{df['sampling_cost'].mean():.4f}")
        summary[2].metric("Proposed cost TB", f"{df['proposed_cost'].mean():.4f}")
        summary[3].metric("Best avg", best_average_label(df))

        st.download_button(
            "Tải CSV",
            data=df.to_csv(index=False).encode("utf-8"),
            file_name="comparison_results.csv",
            mime="text/csv",
        )

with map_tab:
    top = st.columns(4)
    top[0].metric("Cost", f"{float(result.cost.item()):.4f}")
    top[1].metric("Routes", len(routes))
    top[2].metric("Capacity", f"{config.capacity:.0f}")
    top[3].metric("Validity", "OK" if is_valid else "Fail")

    if not is_valid:
        st.error(valid_message)

    st.plotly_chart(plot_routes(batch, routes), width="stretch")

    table = pd.DataFrame(
        {
            "route": list(range(1, len(routes) + 1)),
            "load": loads,
            "path": [" -> ".join(map(str, route)) for route in routes],
        }
    )
    st.dataframe(table, width="stretch", hide_index=True)

    st.subheader("Learned priority order")
    st.code(" -> ".join(map(str, priority_order)), language="text")

with explanation_tab:
    st.markdown(
        """
- Static input: depot and customer coordinates in the unit square.
- Dynamic input: remaining customer demand and current vehicle load.
- Decoder: recurrent state updated from the previously selected node.
- Attention: scores every feasible destination at each step.
- Masking: visited customers, customers exceeding remaining load, and invalid depot repeats are blocked.
- Learning part: the Nazari-style policy produces a customer priority order.
- Insertion part: each customer is inserted into the feasible route position with the smallest Delta(a,j,b).
- Reward: negative total distance after insertion, so maximizing reward improves the insertion-guided solution.
- Inference: greedy-guided insertion, sampling, and beam-guided insertion.
"""
    )
