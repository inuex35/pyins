#!/usr/bin/env python3
"""
RTKLIB-style observation interpolation implementation
Based on RTKLIB's interpobs() and syncobs() functions
"""

import numpy as np
from typing import Dict, List, Tuple, Optional

# RTKLIBの定数
DTTOL = 0.005  # 5ms: 厳密な時刻同期許容値（高レート用）
DTTOL_LOWRATE = 0.025  # 25ms: 低レート（1Hz）用の許容値
MAXDTOE = 7200.0  # 最大軌道情報時刻差
FREQ_L1 = 1.57542e9  # L1周波数 (Hz)
CLIGHT = 299792458.0  # 光速 (m/s)
WAVELENGTH_L1 = CLIGHT / FREQ_L1  # L1波長


def timediff(t1: float, t2: float) -> float:
    """時刻差を計算（GPS週跨ぎを考慮）"""
    dt = t1 - t2
    if dt > 302400.0:  # 3.5日以上の差
        dt -= 604800.0  # 1週間引く
    elif dt < -302400.0:
        dt += 604800.0  # 1週間足す
    return dt


def interp_pseudorange(pr1: float, pr2: float, t1: float, t2: float, t: float) -> float:
    """
    擬似距離の線形補間（RTKLIBスタイル）
    
    Parameters:
    -----------
    pr1, pr2 : float
        前後エポックの擬似距離 [m]
    t1, t2 : float
        前後エポックの時刻 [s]
    t : float
        補間対象時刻 [s]
    
    Returns:
    --------
    float : 補間された擬似距離 [m]
    """
    if abs(t2 - t1) < 1e-9 or pr1 == 0 or pr2 == 0:
        return pr1
    
    # 線形補間係数
    alpha = (t - t1) / (t2 - t1)
    
    # 擬似距離の変化率を考慮
    # RTKLIBでは衛星の運動による変化を考慮
    pr_rate = (pr2 - pr1) / (t2 - t1)
    
    # 変化率が異常に大きい場合はクリップ（衛星速度の制限）
    MAX_RANGE_RATE = 1000.0  # m/s (保守的な値)
    if abs(pr_rate) > MAX_RANGE_RATE:
        pr_rate = np.sign(pr_rate) * MAX_RANGE_RATE
    
    # 補間
    pr_interp = pr1 + pr_rate * (t - t1)
    
    return pr_interp


def interp_carrier_phase(L1: float, L2: float, D1: float, D2: float, 
                        t1: float, t2: float, t: float, freq: float = FREQ_L1) -> float:
    """
    搬送波位相の補間（ドップラー周波数を使用）
    RTKLIBのinterpobs関数を模倣
    
    Parameters:
    -----------
    L1, L2 : float
        前後エポックの搬送波位相 [cycle]
    D1, D2 : float
        前後エポックのドップラー周波数 [Hz]
    t1, t2 : float
        前後エポックの時刻 [s]
    t : float
        補間対象時刻 [s]
    freq : float
        搬送波周波数 [Hz]
    
    Returns:
    --------
    float : 補間された搬送波位相 [cycle]
    """
    if abs(t2 - t1) < 1e-9 or L1 == 0 or L2 == 0:
        return L1
    
    dt = t - t1
    dt_total = t2 - t1
    
    # ドップラーから位相変化率を計算
    # ドップラー周波数 [Hz] = -範囲率 [m/s] / 波長 [m]
    if D1 != 0 and D2 != 0:
        # ドップラーの変化率（加速度に相当）
        dD = (D2 - D1) / dt_total
        
        # 2次の項まで考慮した補間（RTKLIBスタイル）
        L_interp = L1 + D1 * dt / (CLIGHT / freq) + 0.5 * dD * dt * dt / (CLIGHT / freq)
    else:
        # ドップラーが利用できない場合は線形補間
        L_interp = L1 + (L2 - L1) * dt / dt_total
    
    return L_interp


def interp_observation(obs1: dict, obs2: dict, t1: float, t2: float, t: float) -> dict:
    """
    観測値全体の補間（RTKLIBのinterpobs相当）
    
    Parameters:
    -----------
    obs1, obs2 : dict
        前後エポックの観測値
    t1, t2 : float
        前後エポックの時刻
    t : float
        補間対象時刻
    
    Returns:
    --------
    dict : 補間された観測値
    """
    if abs(t - t1) < DTTOL:
        return obs1
    if abs(t - t2) < DTTOL:
        return obs2
    
    # 補間された観測値を格納
    obs_interp = {}
    
    # 各周波数の擬似距離を補間
    if hasattr(obs1, 'P') and hasattr(obs2, 'P'):
        P_interp = []
        for i in range(min(len(obs1.P), len(obs2.P))):
            if obs1.P[i] != 0 and obs2.P[i] != 0:
                pr = interp_pseudorange(obs1.P[i], obs2.P[i], t1, t2, t)
                P_interp.append(pr)
            else:
                P_interp.append(0.0)
        obs_interp['P'] = np.array(P_interp)
    
    # 搬送波位相の補間（ドップラーを使用）
    if hasattr(obs1, 'L') and hasattr(obs2, 'L') and hasattr(obs1, 'D') and hasattr(obs2, 'D'):
        L_interp = []
        for i in range(min(len(obs1.L), len(obs2.L))):
            if obs1.L[i] != 0 and obs2.L[i] != 0:
                if i < len(obs1.D) and i < len(obs2.D):
                    L = interp_carrier_phase(obs1.L[i], obs2.L[i], 
                                           obs1.D[i], obs2.D[i], 
                                           t1, t2, t)
                else:
                    # ドップラーがない場合は線形補間
                    alpha = (t - t1) / (t2 - t1)
                    L = obs1.L[i] + alpha * (obs2.L[i] - obs1.L[i])
                L_interp.append(L)
            else:
                L_interp.append(0.0)
        obs_interp['L'] = np.array(L_interp)
    
    # その他の属性をコピー
    obs_interp['sat'] = obs1.sat if hasattr(obs1, 'sat') else obs1.get('sat')
    obs_interp['system'] = obs1.system if hasattr(obs1, 'system') else obs1.get('system')
    
    return obs_interp


def syncobs_rtklib(rover_obs_list: List[dict], base_obs_list: List[dict], 
                   use_lowrate: bool = False) -> List[Tuple[dict, dict, float]]:
    """
    RTKLIBのsyncobs関数を模倣した観測値同期
    
    Parameters:
    -----------
    rover_obs_list : List[dict]
        移動局観測値リスト
    base_obs_list : List[dict]
        基準局観測値リスト
    use_lowrate : bool
        低レート用の許容値を使用するか
    
    Returns:
    --------
    List[Tuple[dict, dict, float]] : 同期されたペアのリスト
    """
    dttol = DTTOL_LOWRATE if use_lowrate else DTTOL
    
    synchronized = []
    i, j = 0, 0  # インデックス
    
    while i < len(rover_obs_list) and j < len(base_obs_list):
        # 時刻を取得
        t_rover = rover_obs_list[i].get('gps_time', rover_obs_list[i].get('time'))
        t_base = base_obs_list[j].get('gps_time', base_obs_list[j].get('time'))
        
        if t_rover is None or t_base is None:
            break
        
        dt = timediff(t_rover, t_base)
        
        if abs(dt) < dttol:
            # 時刻が十分近い - ペアとして追加
            synchronized.append((rover_obs_list[i], base_obs_list[j], dt))
            i += 1
            j += 1
        elif dt > 0:
            # 移動局が進んでいる - 基準局を進める
            j += 1
            # 補間が可能か確認
            if j > 0 and j < len(base_obs_list):
                # 前後の基準局観測値で補間
                t_prev = base_obs_list[j-1].get('gps_time', base_obs_list[j-1].get('time'))
                t_next = base_obs_list[j].get('gps_time', base_obs_list[j].get('time'))
                
                if t_prev <= t_rover <= t_next and (t_next - t_prev) <= 2.0:
                    # 補間を実行
                    base_interp = interpolate_base_epoch(
                        base_obs_list[j-1], base_obs_list[j],
                        t_prev, t_next, t_rover
                    )
                    synchronized.append((rover_obs_list[i], base_interp, 0.0))
                    i += 1
        else:
            # 基準局が進んでいる - 移動局を進める
            i += 1
    
    return synchronized


def interpolate_base_epoch(base1: dict, base2: dict, t1: float, t2: float, t: float) -> dict:
    """
    基準局エポック全体の補間
    
    Parameters:
    -----------
    base1, base2 : dict
        前後の基準局エポック
    t1, t2 : float
        前後エポックの時刻
    t : float
        補間対象時刻
    
    Returns:
    --------
    dict : 補間されたエポック
    """
    # 補間されたエポックを作成
    base_interp = {
        'gps_time': t,
        'time': t,
        'interpolated': True
    }
    
    # 観測値を取得
    obs1 = base1.get('observations', {})
    obs2 = base2.get('observations', {})
    
    # 観測値を辞書形式に変換
    if isinstance(obs1, list):
        obs1_dict = {obs.sat: obs for obs in obs1}
    else:
        obs1_dict = obs1
    
    if isinstance(obs2, list):
        obs2_dict = {obs.sat: obs for obs in obs2}
    else:
        obs2_dict = obs2
    
    # 共通衛星の観測値を補間
    interp_obs = {}
    for sat in obs1_dict:
        if sat in obs2_dict:
            # この衛星の観測値を補間
            obs_interp = interp_observation(obs1_dict[sat], obs2_dict[sat], t1, t2, t)
            
            # 元のオブジェクトタイプを保持
            if hasattr(obs1_dict[sat], '__class__'):
                # オブジェクトとして再構築
                # Observationクラスには必須引数があるため、ダミー値で初期化
                try:
                    new_obs = obs1_dict[sat].__class__(
                        time=t,
                        sat=sat,
                        system=obs1_dict[sat].system if hasattr(obs1_dict[sat], 'system') else 1
                    )
                except:
                    # 引数なしで作成を試みる
                    new_obs = type('Observation', (), {})()
                    new_obs.time = t
                    new_obs.sat = sat
                
                if 'P' in obs_interp:
                    new_obs.P = obs_interp['P']
                if 'L' in obs_interp:
                    new_obs.L = obs_interp['L']
                if hasattr(obs1_dict[sat], 'D'):
                    # ドップラーは線形補間
                    alpha = (t - t1) / (t2 - t1)
                    if hasattr(obs2_dict[sat], 'D'):
                        new_obs.D = obs1_dict[sat].D + alpha * (obs2_dict[sat].D - obs1_dict[sat].D)
                    else:
                        new_obs.D = obs1_dict[sat].D
                if hasattr(obs1_dict[sat], 'system'):
                    new_obs.system = obs1_dict[sat].system
                if hasattr(obs1_dict[sat], 'SNR'):
                    new_obs.SNR = obs1_dict[sat].SNR
                if hasattr(obs1_dict[sat], 'LLI'):
                    new_obs.LLI = obs1_dict[sat].LLI
                if hasattr(obs1_dict[sat], 'code'):
                    new_obs.code = obs1_dict[sat].code
                    
                interp_obs[sat] = new_obs
            else:
                interp_obs[sat] = obs_interp
    
    base_interp['observations'] = interp_obs
    
    return base_interp