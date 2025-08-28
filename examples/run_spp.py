#!/usr/bin/env python3
"""SPP実行スクリプト"""

from pyins.io.rinex import RinexNavReader, RinexObsReader
from pyins.gnss.spp import single_point_positioning
from pyins.coordinate import ecef2llh
import numpy as np

# RINEXファイルのパス
NAV_FILE = '/home/ubuntu/graph_ins/okujo_test/rover.nav'
OBS_FILE = '/home/ubuntu/graph_ins/okujo_test/main.obs'

# 初期位置（東京付近）
INITIAL_POS = np.array([-3954867.0, 3353972.0, 3701263.0])

def main():
    # ナビゲーションデータ読み込み
    print("Loading navigation data...")
    nav_reader = RinexNavReader(NAV_FILE)
    nav_data = nav_reader.read()
    
    # 観測データ読み込み
    print("Loading observation data...")
    obs_reader = RinexObsReader(OBS_FILE)
    obs_epochs = obs_reader.read()
    
    # 各エポックを処理
    for i, epoch in enumerate(obs_epochs[:10]):  # 最初の10エポックのみ
        print(f"\nEpoch {i+1}:")
        observations = epoch['observations']
        
        # GPS+GLONASS with RAIM
        solution, used_sats = single_point_positioning(
            observations,
            nav_data,
            initial_pos=INITIAL_POS,
            systems_to_use=['G', 'R'],  # GPS + GLONASS
            use_raim=True,               # RAIM有効
            raim_threshold=30.0          # 30m閾値
        )
        
        if solution:
            llh = ecef2llh(solution.rr)
            print(f"  位置: {np.rad2deg(llh[0]):.6f}°, {np.rad2deg(llh[1]):.6f}°, {llh[2]:.1f}m")
            print(f"  使用衛星数: {solution.ns}")
        else:
            print("  解なし")
    
    # GPS only比較
    print("\n" + "="*60)
    print("GPS only (比較用):")
    first_epoch = obs_epochs[0]
    solution_gps, _ = single_point_positioning(
        first_epoch['observations'],
        nav_data,
        initial_pos=INITIAL_POS,
        systems_to_use=['G'],  # GPS only
        use_raim=False
    )
    
    if solution_gps:
        llh_gps = ecef2llh(solution_gps.rr)
        print(f"  高度: {llh_gps[2]:.1f}m")

if __name__ == '__main__':
    main()