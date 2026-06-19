import os
import time
from datetime import datetime

import torch
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from rl4co.envs import CVRPEnv
from rl4co.models.rl.ppo import PPO
from rl4co.models.zoo.am.policy import AttentionModelPolicy


# =========================
# CẤU HÌNH CHUNG
# =========================

NUM_NODES = 20
MAP_COUNT = 10
DEFAULT_BEAM_WIDTH = 5
DEFAULT_SAMPLING_SAMPLES = 50
DEFAULT_TEMPERATURE = 0.8

CHECKPOINT_CANDIDATES = [
    "Bo_nao_AI_CVRP/ppo_attention_cvrp_best.ckpt",
    "ppo_attention_cvrp_best.ckpt",
    "ppo_attention_cvrp.ckpt",
    "epoch=99-step=156800.ckpt",
]

st.set_page_config(
    page_title="CVRP - PPO Attention",
    layout="wide"
)


# =========================
# CSS GIAO DIỆN
# =========================

st.markdown(
    """
    <style>
    .main {
        background: linear-gradient(120deg, #f4f8ff, #ffffff);
    }

    .big-title {
        font-size: 42px;
        font-weight: 900;
        color: #111827;
        margin-bottom: 6px;
    }

    .subtitle {
        font-size: 17px;
        color: #64748b;
        margin-bottom: 26px;
    }

    .section-title {
        font-size: 28px;
        font-weight: 850;
        color: #111827;
        margin-top: 26px;
        margin-bottom: 18px;
    }

    .metric-card {
        padding: 22px;
        border-radius: 18px;
        background: white;
        box-shadow: 0 10px 28px rgba(15, 23, 42, 0.08);
        border: 1px solid #e5e7eb;
        margin-bottom: 14px;
    }

    .metric-label {
        font-size: 12px;
        letter-spacing: 2px;
        font-weight: 800;
        color: #64748b;
        text-transform: uppercase;
    }

    .metric-value {
        font-size: 28px;
        font-weight: 900;
        color: #111827;
        margin-top: 10px;
    }

    .route-box {
        padding: 16px;
        border-radius: 14px;
        background: white;
        border: 1px solid #e5e7eb;
        box-shadow: 0 6px 18px rgba(15, 23, 42, 0.06);
        margin-bottom: 12px;
        font-size: 14px;
    }

    div.stButton > button {
        height: 48px;
        width: 100%;
        border-radius: 12px;
        font-weight: 800;
        background: linear-gradient(90deg, #4f46e5, #0ea5e9);
        color: white;
        border: none;
    }

    div.stButton > button:hover {
        color: white;
        border: none;
        opacity: 0.92;
    }
    </style>
    """,
    unsafe_allow_html=True
)


# =========================
# HÀM PHỤ
# =========================

def metric_card(label, value):
    st.markdown(
        f"""
        <div class="metric-card">
            <div class="metric-label">{label}</div>
            <div class="metric-value">{value}</div>
        </div>
        """,
        unsafe_allow_html=True
    )


def route_box(title, route, load):
    route_text = " → ".join(str(x) for x in route)

    st.markdown(
        f"""
        <div class="route-box">
            <b>{title}</b><br>
            Load: {load:.4f}<br>
            {route_text}
        </div>
        """,
        unsafe_allow_html=True
    )


def find_checkpoint():
    for path in CHECKPOINT_CANDIDATES:
        if os.path.exists(path):
            return path
    return None


@st.cache_resource
def load_env_and_model():
    env = CVRPEnv(generator_params=dict(num_loc=NUM_NODES))
    policy = AttentionModelPolicy(env_name=env.name)

    checkpoint_path = find_checkpoint()

    if checkpoint_path is not None:
        model = PPO.load_from_checkpoint(
            checkpoint_path,
            env=env,
            policy=policy,
            map_location="cpu"
        )
        checkpoint_status = f"Đã load checkpoint: {checkpoint_path}"
    else:
        model = PPO(env, policy)
        checkpoint_status = "Không tìm thấy checkpoint, đang dùng model chưa train"

    model.to("cpu")
    model.eval()

    return env, model, checkpoint_status


def generate_td(env, seed):
    torch.manual_seed(seed)
    td = env.reset(batch_size=[1])
    return td


def get_cost(out):
    reward = out["reward"]

    if reward.numel() == 1:
        return -float(reward.item())

    return -float(reward.mean().item())


def actions_to_routes(actions):
    routes = []
    current_route = [0]

    for action in actions:
        node = int(action)

        if node == 0:
            if len(current_route) > 1:
                current_route.append(0)
                routes.append(current_route)
                current_route = [0]
        else:
            current_route.append(node)

    if len(current_route) > 1:
        current_route.append(0)
        routes.append(current_route)

    return routes


def get_coords_demands(td):
    locs = td["locs"][0].detach().cpu().numpy()
    demand_raw = td["demand"][0].detach().cpu().numpy()

    coords = [(float(x), float(y)) for x, y in locs]

    # demand trong RL4CO thường không gồm depot, nên thêm demand depot = 0
    demands = [0.0] + [float(x) for x in demand_raw]

    return coords, demands


def route_load(route, demands):
    total = 0.0

    for node in route:
        if node != 0 and node < len(demands):
            total += demands[node]

    return total


def solve_one(env, model, map_id, algorithm_key, beam_width=5, sampling_samples=50, temperature=0.8):
    seed = 42 + int(map_id)
    td = generate_td(env, seed)

    start = time.time()

    with torch.no_grad():
        if algorithm_key == "greedy":
            out = model(td.clone(), decode_type="greedy")

        elif algorithm_key == "sampling":
            out = model(
                td.clone(),
                decode_type="sampling",
                samples=sampling_samples,
                temperature=temperature
            )

        elif algorithm_key == "beam":
            out = model(
                td.clone(),
                decode_type="beam_search",
                beam_width=beam_width
            )

        else:
            raise ValueError("algorithm_key không hợp lệ")

    runtime = time.time() - start
    cost = get_cost(out)

    actions = out["actions"][0].detach().cpu().numpy().tolist()
    routes = actions_to_routes(actions)

    coords, demands = get_coords_demands(td)

    return {
        "map": f"MAP_{int(map_id):02d}",
        "seed": seed,
        "algorithm_key": algorithm_key,
        "cost": cost,
        "runtime": runtime,
        "vehicles": len(routes),
        "routes": routes,
        "coords": coords,
        "demands": demands,
    }


def run_all_maps(env, model, beam_width=5, sampling_samples=50, temperature=0.8):
    rows = []
    detail = {}

    progress = st.progress(0)
    total_jobs = MAP_COUNT * 3
    done = 0

    for map_id in range(1, MAP_COUNT + 1):
        map_name = f"MAP_{map_id:02d}"
        detail[map_name] = {}

        greedy = solve_one(
            env, model, map_id,
            algorithm_key="greedy",
            beam_width=beam_width,
            sampling_samples=sampling_samples,
            temperature=temperature
        )
        done += 1
        progress.progress(done / total_jobs)

        sampling = solve_one(
            env, model, map_id,
            algorithm_key="sampling",
            beam_width=beam_width,
            sampling_samples=sampling_samples,
            temperature=temperature
        )
        done += 1
        progress.progress(done / total_jobs)

        beam = solve_one(
            env, model, map_id,
            algorithm_key="beam",
            beam_width=beam_width,
            sampling_samples=sampling_samples,
            temperature=temperature
        )
        done += 1
        progress.progress(done / total_jobs)

        detail[map_name]["greedy"] = greedy
        detail[map_name]["sampling"] = sampling
        detail[map_name]["beam"] = beam

        costs = {
            "Greedy Decoder": greedy["cost"],
            "Sampling Decoder": sampling["cost"],
            "PPO + Attention + Beam Search": beam["cost"]
        }

        best_algorithm = min(costs.keys(), key=lambda k: costs[k])

        rows.append({
            "MAP": map_name,

            "GREEDY_COST": greedy["cost"],
            "GREEDY_TIME_S": greedy["runtime"],
            "GREEDY_VEHICLES": greedy["vehicles"],

            "SAMPLING_COST": sampling["cost"],
            "SAMPLING_TIME_S": sampling["runtime"],
            "SAMPLING_VEHICLES": sampling["vehicles"],

            "BEAM_COST": beam["cost"],
            "BEAM_TIME_S": beam["runtime"],
            "BEAM_VEHICLES": beam["vehicles"],

            "BEST_ALGORITHM": best_algorithm
        })

    progress.empty()

    df = pd.DataFrame(rows)
    return df, detail


def plot_routes(coords, routes, title):
    fig = go.Figure()

    customer_x = [coords[i][0] for i in range(1, len(coords))]
    customer_y = [coords[i][1] for i in range(1, len(coords))]
    customer_text = [str(i) for i in range(1, len(coords))]

    fig.add_trace(go.Scatter(
        x=customer_x,
        y=customer_y,
        mode="markers+text",
        text=customer_text,
        textposition="top center",
        marker=dict(size=8),
        name="Customers"
    ))

    fig.add_trace(go.Scatter(
        x=[coords[0][0]],
        y=[coords[0][1]],
        mode="markers+text",
        text=["Depot"],
        textposition="top center",
        marker=dict(size=18, symbol="diamond"),
        name="Depot"
    ))

    for idx, route in enumerate(routes, start=1):
        xs = [coords[node][0] for node in route]
        ys = [coords[node][1] for node in route]

        fig.add_trace(go.Scatter(
            x=xs,
            y=ys,
            mode="lines+markers",
            name=f"Xe {idx}: {len(route) - 2} khách",
            line=dict(width=3),
            marker=dict(size=6)
        ))

    fig.update_layout(
        title=title,
        height=680,
        xaxis_title="X",
        yaxis_title="Y",
        template="plotly_white",
        legend=dict(
            orientation="v",
            yanchor="top",
            y=1,
            xanchor="left",
            x=1.02
        ),
        margin=dict(l=20, r=20, t=60, b=20)
    )

    return fig


# =========================
# LOAD MODEL
# =========================

env, model, checkpoint_status = load_env_and_model()


# =========================
# HEADER
# =========================

st.markdown(
    '<div class="big-title">So sánh thuật toán cho bài toán CVRP</div>',
    unsafe_allow_html=True
)

st.markdown(
    '<div class="subtitle">RL4CO + PPO + Attention • So sánh Greedy, Sampling và Beam Search • Bản đồ lộ trình tương tác</div>',
    unsafe_allow_html=True
)

st.info(checkpoint_status)

tab1, tab2 = st.tabs(["Bảng so sánh", "Bản đồ lộ trình"])


# =========================
# SIDEBAR PARAMS
# =========================

with st.sidebar:
    st.header("Tùy chỉnh tham số")

    beam_width = st.slider(
        "Beam width",
        min_value=2,
        max_value=50,
        value=DEFAULT_BEAM_WIDTH,
        step=1
    )

    sampling_samples = st.slider(
        "Sampling samples",
        min_value=10,
        max_value=200,
        value=DEFAULT_SAMPLING_SAMPLES,
        step=10
    )

    temperature = st.slider(
        "Sampling temperature",
        min_value=0.1,
        max_value=2.0,
        value=DEFAULT_TEMPERATURE,
        step=0.1
    )


# =========================
# TAB 1: BẢNG SO SÁNH
# =========================

with tab1:
    st.markdown('<div class="section-title">Bảng kết quả so sánh</div>', unsafe_allow_html=True)

    col_run1, col_run2 = st.columns([1.5, 4])

    with col_run1:
        run_clicked = st.button("Chạy / cập nhật bảng")

    with col_run2:
        st.caption("Nút này chạy lại 10 bộ dữ liệu MAP_01 → MAP_10 với 3 cơ chế giải mã: Greedy, Sampling và Beam Search.")

    if run_clicked or "comparison_df" not in st.session_state:
        with st.spinner("Đang chạy thực nghiệm, chờ một chút..."):
            comparison_df, detail = run_all_maps(
                env=env,
                model=model,
                beam_width=beam_width,
                sampling_samples=sampling_samples,
                temperature=temperature
            )

            st.session_state["comparison_df"] = comparison_df
            st.session_state["detail"] = detail
            st.session_state["last_run"] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    df = st.session_state["comparison_df"]
    detail = st.session_state["detail"]

    st.caption(f"Lần chạy gần nhất: {st.session_state.get('last_run', '')}")

    df_show = df.copy()

    for col in [
        "GREEDY_COST", "GREEDY_TIME_S",
        "SAMPLING_COST", "SAMPLING_TIME_S",
        "BEAM_COST", "BEAM_TIME_S"
    ]:
        df_show[col] = df_show[col].round(4)

    st.dataframe(df_show, use_container_width=True, height=460)

    # =========================
    # THỐNG KÊ TỔNG QUAN
    # =========================

    st.markdown('<div class="section-title">Thống kê tổng quan</div>', unsafe_allow_html=True)

    total_map = len(df)
    greedy_best = int((df["BEST_ALGORITHM"] == "Greedy Decoder").sum())
    sampling_best = int((df["BEST_ALGORITHM"] == "Sampling Decoder").sum())
    beam_best = int((df["BEST_ALGORITHM"] == "PPO + Attention + Beam Search").sum())

    avg_greedy_cost = df["GREEDY_COST"].mean()
    avg_sampling_cost = df["SAMPLING_COST"].mean()
    avg_beam_cost = df["BEAM_COST"].mean()

    avg_greedy_time = df["GREEDY_TIME_S"].mean()
    avg_sampling_time = df["SAMPLING_TIME_S"].mean()
    avg_beam_time = df["BEAM_TIME_S"].mean()

    avg_cost_map = {
        "Greedy": avg_greedy_cost,
        "Sampling": avg_sampling_cost,
        "Beam Search": avg_beam_cost
    }

    best_avg = min(avg_cost_map.keys(), key=lambda k: avg_cost_map[k])

    c1, c2, c3, c4 = st.columns(4)

    with c1:
        metric_card("Tổng số map", total_map)

    with c2:
        metric_card("Greedy tốt hơn", greedy_best)

    with c3:
        metric_card("Sampling tốt hơn", sampling_best)

    with c4:
        metric_card("Beam tốt hơn", beam_best)

    c5, c6, c7, c8 = st.columns(4)

    with c5:
        metric_card("Best avg cost", best_avg)

    with c6:
        metric_card("Greedy cost TB", f"{avg_greedy_cost:.4f}")

    with c7:
        metric_card("Sampling cost TB", f"{avg_sampling_cost:.4f}")

    with c8:
        metric_card("Beam cost TB", f"{avg_beam_cost:.4f}")

    t1, t2, t3 = st.columns(3)

    with t1:
        metric_card("Greedy time TB", f"{avg_greedy_time:.4f}s")

    with t2:
        metric_card("Sampling time TB", f"{avg_sampling_time:.4f}s")

    with t3:
        metric_card("Beam time TB", f"{avg_beam_time:.4f}s")

    # =========================
    # BIỂU ĐỒ COST
    # =========================

    st.markdown('<div class="section-title">Biểu đồ so sánh cost</div>', unsafe_allow_html=True)

    cost_long = df[[
        "MAP",
        "GREEDY_COST",
        "SAMPLING_COST",
        "BEAM_COST"
    ]].melt(
        id_vars="MAP",
        var_name="ALGORITHM",
        value_name="COST"
    )

    cost_long["ALGORITHM"] = cost_long["ALGORITHM"].replace({
        "GREEDY_COST": "Greedy",
        "SAMPLING_COST": "Sampling",
        "BEAM_COST": "PPO + Beam"
    })

    fig_cost = px.bar(
        cost_long,
        x="MAP",
        y="COST",
        color="ALGORITHM",
        barmode="group",
        text="COST"
    )

    fig_cost.update_traces(
        texttemplate="%{text:.2f}",
        textposition="outside"
    )

    fig_cost.update_layout(
        height=520,
        xaxis_title="Map",
        yaxis_title="Cost",
        legend_title="Thuật toán",
        template="plotly_white",
        margin=dict(l=20, r=20, t=40, b=20)
    )

    st.plotly_chart(fig_cost, use_container_width=True)

    # =========================
    # BIỂU ĐỒ RUNTIME
    # =========================

    st.markdown('<div class="section-title">Biểu đồ so sánh thời gian chạy</div>', unsafe_allow_html=True)

    time_long = df[[
        "MAP",
        "GREEDY_TIME_S",
        "SAMPLING_TIME_S",
        "BEAM_TIME_S"
    ]].melt(
        id_vars="MAP",
        var_name="ALGORITHM",
        value_name="TIME_S"
    )

    time_long["ALGORITHM"] = time_long["ALGORITHM"].replace({
        "GREEDY_TIME_S": "Greedy",
        "SAMPLING_TIME_S": "Sampling",
        "BEAM_TIME_S": "PPO + Beam"
    })

    fig_time = px.bar(
        time_long,
        x="MAP",
        y="TIME_S",
        color="ALGORITHM",
        barmode="group",
        text="TIME_S"
    )

    fig_time.update_traces(
        texttemplate="%{text:.4f}",
        textposition="outside"
    )

    fig_time.update_layout(
        height=520,
        xaxis_title="Map",
        yaxis_title="Time (s)",
        legend_title="Thuật toán",
        template="plotly_white",
        margin=dict(l=20, r=20, t=40, b=20)
    )

    st.plotly_chart(fig_time, use_container_width=True)

    # =========================
    # BIỂU ĐỒ SỐ XE
    # =========================

    st.markdown('<div class="section-title">Biểu đồ so sánh số tuyến xe</div>', unsafe_allow_html=True)

    vehicle_long = df[[
        "MAP",
        "GREEDY_VEHICLES",
        "SAMPLING_VEHICLES",
        "BEAM_VEHICLES"
    ]].melt(
        id_vars="MAP",
        var_name="ALGORITHM",
        value_name="VEHICLES"
    )

    vehicle_long["ALGORITHM"] = vehicle_long["ALGORITHM"].replace({
        "GREEDY_VEHICLES": "Greedy",
        "SAMPLING_VEHICLES": "Sampling",
        "BEAM_VEHICLES": "PPO + Beam"
    })

    fig_vehicle = px.bar(
        vehicle_long,
        x="MAP",
        y="VEHICLES",
        color="ALGORITHM",
        barmode="group",
        text="VEHICLES"
    )

    fig_vehicle.update_traces(
        texttemplate="%{text}",
        textposition="outside"
    )

    fig_vehicle.update_layout(
        height=520,
        xaxis_title="Map",
        yaxis_title="Vehicles",
        legend_title="Thuật toán",
        template="plotly_white",
        margin=dict(l=20, r=20, t=40, b=20)
    )

    st.plotly_chart(fig_vehicle, use_container_width=True)


# =========================
# TAB 2: BẢN ĐỒ LỘ TRÌNH
# =========================

with tab2:
    st.markdown('<div class="section-title">Chọn dữ liệu và thuật toán để vẽ route</div>', unsafe_allow_html=True)

    if "detail" not in st.session_state:
        with st.spinner("Đang tạo dữ liệu ban đầu..."):
            comparison_df, detail = run_all_maps(
                env=env,
                model=model,
                beam_width=beam_width,
                sampling_samples=sampling_samples,
                temperature=temperature
            )

            st.session_state["comparison_df"] = comparison_df
            st.session_state["detail"] = detail

    detail = st.session_state["detail"]

    control_col1, control_col2, control_col3, control_col4 = st.columns([2, 2, 1.5, 2])

    with control_col1:
        selected_map = st.selectbox(
            "Chọn bộ dữ liệu",
            list(detail.keys())
        )

    with control_col2:
        algorithm_label = st.selectbox(
            "Chọn thuật toán",
            [
                "Greedy Decoder",
                "Sampling Decoder",
                "PPO + Attention + Beam Search"
            ]
        )

    algorithm_map = {
        "Greedy Decoder": "greedy",
        "Sampling Decoder": "sampling",
        "PPO + Attention + Beam Search": "beam"
    }

    with control_col3:
        draw_clicked = st.button("Vẽ lộ trình")

    with control_col4:
        st.write("")
        st.caption("Chọn MAP nào thì web vẽ đúng con đường của MAP đó.")

    algo_key = algorithm_map[algorithm_label]
    result = detail[selected_map][algo_key]

    coords = result["coords"]
    demands = result["demands"]
    routes = result["routes"]

    left_col, right_col = st.columns([2.4, 1.1])

    with left_col:
        title = (
            f"{algorithm_label} - {selected_map} | "
            f"cost={result['cost']:.4f} | vehicles={result['vehicles']}"
        )

        fig = plot_routes(
            coords=coords,
            routes=routes,
            title=title
        )

        st.plotly_chart(fig, use_container_width=True)

    with right_col:
        st.markdown('<div class="section-title">Thông tin</div>', unsafe_allow_html=True)

        card1, card2 = st.columns(2)

        with card1:
            metric_card("Customers", NUM_NODES)

        with card2:
            metric_card("Vehicles", result["vehicles"])

        card3, card4 = st.columns(2)

        with card3:
            metric_card("Cost", f"{result['cost']:.4f}")

        with card4:
            metric_card("Runtime", f"{result['runtime']:.4f}s")

        metric_card("Algorithm", algorithm_label)
        metric_card("Seed", result["seed"])

        st.markdown('<div class="section-title">Routes</div>', unsafe_allow_html=True)

        for idx, route in enumerate(routes, start=1):
            load = route_load(route, demands)
            route_box(f"Xe {idx}", route, load)