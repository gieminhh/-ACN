# -*- coding: utf-8 -*-
import os
import sys
import math
import json
import torch
from typing import cast
from datetime import datetime
from pathlib import Path
from rl4co.envs import CVRPEnv
from rl4co.models import PPO, AttentionModelPolicy

# Cấu hình encoding cho Windows
if sys.platform == "win32" and hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")  # type: ignore

def run_and_print_data():
    # 1. Khởi tạo môi trường 
    env = CVRPEnv(generator_params=dict(num_loc=20))
    policy = AttentionModelPolicy(env_name=env.name)

    
    checkpoint_path = "Bo_nao_AI_CVRP/ppo_attention_cvrp_best.ckpt"
    if not os.path.exists(checkpoint_path):
        checkpoint_path = "ppo_attention_cvrp_best.ckpt"

    print(f"--- ĐANG KIỂM TRA FILE CHECKPOINT: {checkpoint_path} ---")

    if os.path.exists(checkpoint_path):
        model = PPO.load_from_checkpoint(checkpoint_path, env=env, policy=policy)
        print("✓ ĐÃ GHÉP NÃO AI THÀNH CÔNG!")
    else:
        print("⚠ KHÔNG TÌM THẤY FILE .CKPT, ĐANG DÙNG MODEL CHƯA TRAIN!")
        model = PPO(env, policy)

    # 3. Tạo bản đồ và giải bài toán
    td = env.reset(batch_size=[1])
    model.to("cpu")
    model.eval()
    
    with torch.no_grad():
        # Dùng Beam Search cho kết quả tối ưu nhất
        out = model(td, decode_type="beam_search", beam_width=5)
    
    # Lấy dữ liệu để vẽ
    locs_tensor = cast(torch.Tensor, td["locs"])
    demand_tensor = cast(torch.Tensor, td["demand"])
    coords = locs_tensor[0].detach().cpu().numpy()
    demands = demand_tensor[0].detach().cpu().numpy()
    actions = out["actions"][0].cpu().numpy()
    reward = out["reward"][0].item()
    total_distance = -reward # Đổi dấu vì reward trong RL thường âm

    print(f"\n--- TỔNG CHI PHÍ LỘ TRÌNH: {total_distance:.4f} ---")

    # 4. Vẽ bản đồ tương tác bằng Plotly và xuất HTML
    try:
        import plotly.graph_objects as go
        
        fig = go.Figure()
        lats = coords[:, 1]
        lons = coords[:, 0]

        # Vẽ các điểm khách hàng
        fig.add_trace(go.Scatter(x=lons[1:], y=lats[1:], mode='markers', 
                                 marker=dict(size=10, color='blue'),
                                 name='Khách hàng'))
        
        # Vẽ kho (Depot)
        fig.add_trace(go.Scatter(x=[lons[0]], y=[lats[0]], mode='markers', 
                                 marker=dict(size=15, color='red', symbol='star'),
                                 name='KHO (Depot)'))

        # Vẽ đường đi từ Actions
        curr_x, curr_y = [lons[0]], [lats[0]]
        route_count = 1
        for a in actions:
            curr_x.append(lons[a])
            curr_y.append(lats[a])
            if a == 0: # Khi quay về kho
                fig.add_trace(go.Scatter(x=curr_x, y=curr_y, mode='lines+markers', 
                                         name=f'Tuyến {route_count}', line=dict(width=2)))
                curr_x, curr_y = [lons[0]], [lats[0]]
                route_count += 1
        
        # Nếu chưa về kho ở bước cuối
        if actions[-1] != 0:
            curr_x.append(lons[0])
            curr_y.append(lats[0])
            fig.add_trace(go.Scatter(x=curr_x, y=curr_y, mode='lines+markers', 
                                     name=f'Tuyến {route_count}', line=dict(width=2)))

        fig.update_layout(title=f"Bản đồ lộ trình CVRP (Tổng quãng đường: {total_distance:.4f})",
                          xaxis_title="Tọa độ X", yaxis_title="Tọa độ Y",
                          template="plotly_white")
        
        # Ghi file HTML
        output_file = "lo_trinh_tuong_tac.html"
        fig.write_html(output_file)
        print(f"✓ ĐÃ XUẤT FILE: {output_file}. Mở ngay bằng Chrome/Edge đi m!")

    except Exception as e:
        print(f"⚠ Lỗi khi vẽ HTML: {str(e)}")

if __name__ == "__main__":
    run_and_print_data()