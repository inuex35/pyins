"""GNSS/IMU fusion example"""

import numpy as np
from datetime import datetime, timedelta
from pyins.core import GNSSTime, NavigationData
from pyins.sensors import IMUData, IMUConfig
from pyins.fusion import GNSSIMUFilter, NavigationState
from pyins.coordinate import llh2ecef, ecef2llh
import matplotlib.pyplot as plt

def generate_imu_data(true_state, imu_config, dt=0.01, duration=10):
    """Generate simulated IMU data"""
    imu_data_list = []
    timestamps = []
    
    # True trajectory parameters
    t = 0
    while t < duration:
        # Simple circular motion
        omega = 0.1  # rad/s
        radius = 100  # m
        
        # True acceleration and angular velocity
        acc_true = np.array([
            -radius * omega**2 * np.cos(omega * t),
            -radius * omega**2 * np.sin(omega * t),
            0
        ])
        gyro_true = np.array([0, 0, omega])
        
        # Add noise
        acc_noise = np.random.normal(0, imu_config.get_noise_std('acc_noise'), 3)
        gyro_noise = np.random.normal(0, imu_config.get_noise_std('gyro_noise'), 3)
        
        # Add gravity (simplified)
        gravity = np.array([0, 0, -9.81])
        
        # Create IMU measurement
        imu_meas = np.concatenate([
            acc_true + gravity + acc_noise,
            gyro_true + gyro_noise
        ])
        
        imu_data = IMUData(
            timestamp=t,
            sensor_id=imu_config.sensor_id,
            sensor_type=imu_config.sensor_type,
            data=imu_meas
        )
        
        imu_data_list.append(imu_data)
        timestamps.append(t)
        t += dt
        
    return imu_data_list, timestamps


def generate_gnss_observations(true_positions, timestamps, nav_data, noise_std=1.0):
    """Generate simulated GNSS observations"""
    from pygnss.core import Observation
    from pygnss.satellite import compute_satellite_position
    
    observations_list = []
    
    for i, (t, true_pos) in enumerate(zip(timestamps, true_positions)):
        observations = []
        
        # Generate observations from visible satellites
        for eph in nav_data.eph[:6]:  # Use first 6 satellites
            # Satellite position
            sat_pos, sat_clk, _ = compute_satellite_position(eph, t)
            
            # True range
            true_range = np.linalg.norm(sat_pos - true_pos)
            
            # Add noise
            pr_noise = np.random.normal(0, noise_std)
            pseudorange = true_range + pr_noise
            
            # Create observation
            obs = Observation(
                time=t,
                sat=eph.sat,
                P=np.array([pseudorange, 0, 0]),
                SNR=np.array([45.0, 0, 0])
            )
            observations.append(obs)
            
        observations_list.append(observations)
        
    return observations_list


def main():
    """Run GNSS/IMU fusion example"""
    
    # Setup
    print("GNSS/IMU Fusion Example")
    print("=" * 50)
    
    # Initial position (somewhere in Tokyo)
    initial_llh = np.array([np.deg2rad(35.6762), np.deg2rad(139.6503), 100.0])
    initial_ecef = llh2ecef(initial_llh)
    
    # IMU configuration
    imu_config = IMUConfig(
        sensor_id="imu_sim",
        sampling_rate=100.0,
        noise_params={
            'acc_noise': 0.1,
            'gyro_noise': 0.01,
            'acc_bias_walk': 0.001,
            'gyro_bias_walk': 0.0001
        },
        lever_arm=np.array([0.5, 0.0, 0.1])  # 50cm forward, 10cm up
    )
    
    # Initial state
    initial_state = NavigationState(
        time=0.0,
        position=initial_ecef,
        velocity=np.array([10.0, 0.0, 0.0]),  # 10 m/s east
        quaternion=np.array([1, 0, 0, 0]),
        acc_bias=np.array([0.01, -0.02, 0.03]),
        gyro_bias=np.array([0.001, -0.001, 0.002])
    )
    
    # Create filter
    ekf = GNSSIMUFilter(initial_state)
    
    # Generate trajectory
    duration = 60.0  # 60 seconds
    imu_dt = 0.01    # 100 Hz IMU
    gnss_dt = 1.0    # 1 Hz GNSS
    
    print(f"Simulating {duration}s trajectory...")
    print(f"  IMU rate: {1/imu_dt} Hz")
    print(f"  GNSS rate: {1/gnss_dt} Hz")
    
    # Generate IMU data
    imu_data_list, imu_timestamps = generate_imu_data(
        initial_state, imu_config, imu_dt, duration
    )
    
    # Generate true trajectory for GNSS simulation
    true_positions = []
    for t in np.arange(0, duration, gnss_dt):
        # Simple circular trajectory
        omega = 0.1
        radius = 100
        x = initial_ecef[0] + radius * (np.cos(omega * t) - 1)
        y = initial_ecef[1] + radius * np.sin(omega * t)
        z = initial_ecef[2]
        true_positions.append(np.array([x, y, z]))
    
    # Create navigation data with dummy ephemerides
    nav_data = NavigationData()
    current_time = GNSSTime.from_datetime(datetime.now())
    
    for i in range(8):  # 8 satellites
        eph = Ephemeris(
            sat=i+1,
            iode=0, iodc=0, sva=0, svh=0,
            week=current_time.week,
            toe=current_time.tow,
            toc=current_time.tow,
            ttr=current_time.tow,
            A=26559755.0**2,
            e=0.001,
            i0=0.98 + i * 0.1,
            OMG0=i * 0.785,
            omg=1.0,
            M0=i * 0.785,
            deln=0.0,
            OMGd=-8.0e-12,
            idot=0.0,
            crc=0.0, crs=0.0,
            cuc=0.0, cus=0.0,
            cic=0.0, cis=0.0,
            toes=current_time.tow,
            fit=4.0,
            f0=1e-10 * i,
            f1=0.0, f2=0.0
        )
        nav_data.eph.append(eph)
    
    # Generate GNSS observations
    gnss_timestamps = np.arange(0, duration, gnss_dt)
    gnss_observations = generate_gnss_observations(
        true_positions, gnss_timestamps, nav_data, noise_std=2.0
    )
    
    # Run fusion
    print("\nRunning GNSS/IMU fusion...")
    
    results = {
        'time': [],
        'position': [],
        'velocity': [],
        'attitude': [],
        'bias_acc': [],
        'bias_gyro': []
    }
    
    gnss_idx = 0
    last_gnss_time = -1
    
    for i, (imu_data, t) in enumerate(zip(imu_data_list, imu_timestamps)):
        # IMU prediction
        if i > 0:
            dt = imu_timestamps[i] - imu_timestamps[i-1]
            ekf.predict_imu(imu_data, dt)
        
        # GNSS update (if available)
        if gnss_idx < len(gnss_timestamps) and t >= gnss_timestamps[gnss_idx]:
            ekf.update_gnss(
                gnss_observations[gnss_idx],
                nav_data.eph,
                use_carrier=False  # Code only for simplicity
            )
            last_gnss_time = t
            gnss_idx += 1
            
        # Store results every 0.1s
        if i % 10 == 0:
            state = ekf.get_state()
            results['time'].append(t)
            results['position'].append(state.position.copy())
            results['velocity'].append(state.velocity.copy())
            results['attitude'].append(state.attitude_euler.copy())
            results['bias_acc'].append(state.acc_bias.copy())
            results['bias_gyro'].append(state.gyro_bias.copy())
    
    # Convert results to arrays
    for key in results:
        results[key] = np.array(results[key])
    
    # Plot results
    print("\nGenerating plots...")
    
    fig, axes = plt.subplots(3, 2, figsize=(12, 10))
    
    # Position
    ax = axes[0, 0]
    true_xy = np.array(true_positions)[:, :2]
    est_xy = results['position'][:, :2]
    ax.plot(true_xy[:, 0], true_xy[:, 1], 'b-', label='True', linewidth=2)
    ax.plot(est_xy[:, 0], est_xy[:, 1], 'r--', label='Estimated')
    ax.set_xlabel('East (m)')
    ax.set_ylabel('North (m)')
    ax.set_title('2D Trajectory')
    ax.legend()
    ax.axis('equal')
    
    # Height
    ax = axes[0, 1]
    ax.plot(results['time'], results['position'][:, 2], 'r-')
    ax.set_xlabel('Time (s)')
    ax.set_ylabel('Height (m)')
    ax.set_title('Height Profile')
    ax.grid(True)
    
    # Velocity
    ax = axes[1, 0]
    vel_mag = np.linalg.norm(results['velocity'], axis=1)
    ax.plot(results['time'], vel_mag, 'g-')
    ax.set_xlabel('Time (s)')
    ax.set_ylabel('Speed (m/s)')
    ax.set_title('Speed')
    ax.grid(True)
    
    # Attitude
    ax = axes[1, 1]
    ax.plot(results['time'], np.rad2deg(results['attitude'][:, 0]), 'r-', label='Roll')
    ax.plot(results['time'], np.rad2deg(results['attitude'][:, 1]), 'g-', label='Pitch')
    ax.plot(results['time'], np.rad2deg(results['attitude'][:, 2]), 'b-', label='Yaw')
    ax.set_xlabel('Time (s)')
    ax.set_ylabel('Angle (deg)')
    ax.set_title('Attitude')
    ax.legend()
    ax.grid(True)
    
    # Accelerometer bias
    ax = axes[2, 0]
    ax.plot(results['time'], results['bias_acc'][:, 0], 'r-', label='X')
    ax.plot(results['time'], results['bias_acc'][:, 1], 'g-', label='Y')
    ax.plot(results['time'], results['bias_acc'][:, 2], 'b-', label='Z')
    ax.set_xlabel('Time (s)')
    ax.set_ylabel('Bias (m/s²)')
    ax.set_title('Accelerometer Bias')
    ax.legend()
    ax.grid(True)
    
    # Gyroscope bias
    ax = axes[2, 1]
    ax.plot(results['time'], np.rad2deg(results['bias_gyro'][:, 0]), 'r-', label='X')
    ax.plot(results['time'], np.rad2deg(results['bias_gyro'][:, 1]), 'g-', label='Y')
    ax.plot(results['time'], np.rad2deg(results['bias_gyro'][:, 2]), 'b-', label='Z')
    ax.set_xlabel('Time (s)')
    ax.set_ylabel('Bias (deg/s)')
    ax.set_title('Gyroscope Bias')
    ax.legend()
    ax.grid(True)
    
    plt.tight_layout()
    plt.savefig('gnss_imu_fusion_results.png', dpi=150)
    print("Results saved to gnss_imu_fusion_results.png")
    
    # Print final statistics
    print("\nFinal Statistics:")
    final_state = ekf.get_state()
    final_llh = ecef2llh(final_state.position)
    print(f"  Position (LLH): {np.rad2deg(final_llh[0]):.6f}°, "
          f"{np.rad2deg(final_llh[1]):.6f}°, {final_llh[2]:.3f} m")
    print(f"  Velocity: {np.linalg.norm(final_state.velocity):.3f} m/s")
    print(f"  Attitude (RPY): {np.rad2deg(final_state.attitude_euler)} deg")
    
    # Position error
    if len(true_positions) > 0:
        pos_error = np.linalg.norm(final_state.position - true_positions[-1])
        print(f"  Final position error: {pos_error:.3f} m")


if __name__ == "__main__":
    # Add Ephemeris import
    from pygnss.core import Ephemeris
    main()