# PyINS - Comprehensive GNSS/IMU Processing Library

PyINS is a Python library for satellite positioning, pseudorange/carrier phase processing, coordinate transformations, and sensor fusion with IMU data. It is inspired by gnss-py, rtklib-py, and OB-GINS.

## Features

- **Satellite Processing**
  - Satellite position/velocity computation from broadcast ephemeris
  - Satellite clock correction with relativistic effects
  - Support for GPS, GLONASS, Galileo, BeiDou, QZSS

- **Observation Processing**
  - Pseudorange measurement processing
  - Carrier phase measurement with cycle slip detection
  - Dual-frequency ionosphere-free combinations
  - Troposphere and ionosphere models

- **Time Systems**
  - GPS time conversions (GPS seconds ↔ Week/TOW)
  - GNSS time representation and conversion
  - Time system conversions (GPS, Galileo, BeiDou, UTC)

- **Coordinate Systems**
  - ECEF/LLH/ENU/NED transformations
  - Geodetic computations (distance, azimuth)
  - Rotation representations (DCM, quaternion, Euler angles)

- **IMU Integration**
  - IMU data structures and bias modeling
  - IMU preintegration for factor graphs
  - Lever arm compensation

- **Sensor Fusion**
  - Tightly-coupled GNSS/IMU Extended Kalman Filter
  - Factor graph optimization
  - Multi-sensor support with lever arms

## Installation

```bash
pip install pyins
```

For development:
```bash
git clone https://github.com/yourusername/pyins.git
cd pyins
pip install -e .[dev]
```

## Quick Start

### Basic GNSS Processing

```python
import numpy as np
from pyins.core import GNSSTime, Observation, Ephemeris
from pyins.satellite import compute_satellite_position
from pyins.observation import compute_pseudorange_residual

# Create observation
obs = Observation(
    time=GNSSTime.from_datetime(datetime.now()).to_gps_seconds(),
    sat=1,  # GPS PRN 1
    P=np.array([22000000.0, 0, 0]),  # Pseudorange on L1
    L=np.array([115000000.0, 0, 0])  # Carrier phase on L1
)

# Compute satellite position
sat_pos, sat_clk, _ = compute_satellite_position(eph, obs.time)

# Compute measurement residual
residual, H, var = compute_pseudorange_residual(
    obs, eph, rcv_pos, rcv_clk
)
```

### GPS Time Conversion

```python
from pyins.core.time import (
    gps_seconds_to_week_tow,
    week_tow_to_gps_seconds,
    ensure_tow,
    is_gps_seconds
)

# Convert GPS seconds to week and time of week
gps_seconds = 1356393600.0
week, tow = gps_seconds_to_week_tow(gps_seconds)
print(f"GPS Week: {week}, TOW: {tow:.1f}s")

# Convert back to GPS seconds
gps_seconds_back = week_tow_to_gps_seconds(week, tow)

# Ensure time is in TOW format (for satellite position computation)
time_value = 1356393600.0  # Could be GPS seconds or TOW
tow = ensure_tow(time_value)  # Always returns TOW

# Check if time value is GPS seconds or TOW
if is_gps_seconds(time_value):
    print("Time is in GPS seconds format")
else:
    print("Time is in TOW format")
```

### Satellite Position Computation

```python
from pyins.satellite.satellite_position import compute_satellite_position
from pyins.core.time import ensure_tow

# Ensure time is in correct format for satellite position computation
obs_time = ensure_tow(observation_time)  # Convert GPS seconds to TOW if needed

# Compute satellite position using TOW
sat_pos, sat_clk, var = compute_satellite_position(ephemeris, obs_time)
```

### Multi-Frequency GNSS Processing

```python
from pyins.examples.basic_spp import single_point_positioning

# Process observations with multiple frequencies (L1, L2, L5)
solution, used_sats = single_point_positioning(
    observations=gnss_observations,
    nav_data=navigation_data,
    initial_pos=approx_position
)

print(f"Position: {solution.rr}")
print(f"Used satellites: {used_sats}")
```

### Factor Graph Implementation

```python
from src.gtsam_rtk_ins import GTSAMRTKINSEstimator

# Initialize GTSAM-based RTK+INS estimator
estimator = GTSAMRTKINSEstimator(initial_state)

# Add ephemeris data
for eph in ephemeris_list:
    estimator.add_ephemeris(eph)

# Process GNSS measurements (time conversion handled automatically)
estimator.add_gnss_measurements(observations, gps_time)

# Optimize solution
estimator.optimize()
```

## Time System Handling

PyINS automatically handles different time formats:

- **GPS Seconds**: Continuous seconds since GPS epoch (Jan 6, 1980)
- **GPS Week/TOW**: Week number + Time of Week (0-604800 seconds)
- **Automatic Detection**: Functions automatically detect format and convert as needed

```python
# These are equivalent for satellite position computation:
sat_pos1, _, _ = compute_satellite_position(eph, 1356393600.0)  # GPS seconds
sat_pos2, _, _ = compute_satellite_position(eph, 393600.0)      # TOW (if current week)

# Use ensure_tow() for guaranteed compatibility:
tow = ensure_tow(any_time_format)
sat_pos, _, _ = compute_satellite_position(eph, tow)
```

## Documentation

For detailed documentation and API reference, see:
- [API Documentation](docs/api.md)
- [Examples](examples/)
- [Time System Guide](docs/time_systems.md)

## Contributing

Contributions are welcome! Please see [CONTRIBUTING.md](CONTRIBUTING.md) for guidelines.

## License

MIT License - see [LICENSE](LICENSE) for details.