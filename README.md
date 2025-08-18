# PyINS - Python GNSS/INS Processing Library

PyINS is a comprehensive GNSS/IMU processing library for satellite positioning, RTK, sensor fusion, and coordinate transformations. Inspired by RTKLIB and gnss-py.

## Features

### GNSS Processing
- **Multi-constellation support**: GPS, Galileo, BeiDou, GLONASS, QZSS
- **RINEX file I/O**: Observation and navigation file parsing using gnsspy
- **Robust Single Point Positioning (SPP)**: With iterative least squares, troposphere modeling, and multi-GNSS support
- **RTK (Real-Time Kinematic)**: Double difference processing with data synchronization and interpolation
- **Ephemeris handling**: Satellite selection and position computation with time-safe ephemeris
- **SP3 Precise Ephemeris**: Support for IGS and MGEX precise orbit products with 40-80x accuracy improvement
- **Time management**: Unified TimeCore system for all GNSS time systems with automatic conversions
- **Carrier phase processing**: Cycle slip detection and ambiguity resolution (LAMBDA method)

### Sensor Fusion & IMU
- **IMU mechanization**: Preintegration and bias estimation
- **Extended Kalman Filter (EKF)**: For state estimation
- **Factor graph optimization**: Graph-based sensor fusion with weighted factors
- **Lever arm management**: Flexible robot sensor configuration system

### Coordinate Systems
- **Transformations**: ECEF/LLH/ENU/NED/ECI conversions
- **Height systems**: Ellipsoidal and orthometric (EGM96) height conversion
- **Rotation representations**: DCM, Euler angles, quaternions

## Installation

### Install from source
```bash
git clone https://github.com/inuex35/pyins.git
cd pyins
pip install -e .
```

### Install via pip (when published)
```bash
pip install pyins
```

### Dependencies
```bash
numpy>=1.19.0
scipy>=1.5.0
pandas>=1.1.0
matplotlib>=3.3.0
numba>=0.51.2
gnsspy
```

## Quick Start

### Single Point Positioning (SPP)
```python
from pyins.io.rinex import RinexObsReader, RinexNavReader
from pyins.gnss import robust_spp_solve
from pyins.coordinate import ecef2llh
import numpy as np

# Read RINEX files
obs_reader = RinexObsReader('rover.obs')
nav_reader = RinexNavReader('navigation.nav')

obs_data = obs_reader.read()
nav_data = nav_reader.read()

# Process first epoch
epoch = obs_data[0]
observations = epoch['observations']

# Solve for position using robust SPP
solution = robust_spp_solve(
    observations, 
    nav_data,
    systems_to_use=['G', 'E', 'C']  # GPS, Galileo, BeiDou
)

if solution and solution.valid:
    llh = ecef2llh(solution.rr)
    print(f"Position: lat={np.rad2deg(llh[0]):.6f}°, "
          f"lon={np.rad2deg(llh[1]):.6f}°, h={llh[2]:.1f}m")
```

### RTK Double Difference
```python
from pyins.rtk import DoubleDifferenceProcessor, DDLeastSquares, interpolate_epoch
from pyins.io.rinex import RinexObsReader, RinexNavReader
from pyins.coordinate import ecef2llh, ecef2enu

# Read rover and base data
rover_obs = RinexObsReader('rover.obs').read()
base_obs = RinexObsReader('base.obs').read()
nav_data = RinexNavReader('nav.nav').read()

# Initialize processors
dd_processor = DoubleDifferenceProcessor()
dd_solver = DDLeastSquares()

# Synchronize and interpolate data
rover_epoch = rover_obs[0]
base_epoch = interpolate_epoch(base_obs, rover_epoch['time'])

# Process double differences
dd_obs = dd_processor.process(
    rover_epoch['observations'],
    base_epoch['observations'],
    nav_data
)

# Solve for baseline
baseline = dd_solver.solve(dd_obs, base_position_ecef)
```

### SP3 Precise Ephemeris
```python
from pyins.gnss.sp3_ephemeris import SP3Ephemeris
from pyins.gnss.ephemeris import compute_satellite_position
from datetime import datetime

# Initialize SP3 handler (defaults to MGEX for multi-GNSS)
sp3 = SP3Ephemeris()

# Download SP3 automatically (MGEX COD by default)
sp3_file = sp3.download_sp3(datetime(2024, 1, 15))

# Or get best available product
sp3_file = sp3.get_auto_sp3(datetime.now(), prefer_mgex=True)

# Read SP3 data
sp3_data = sp3.read_sp3(sp3_file)

# Interpolate satellite position (RTKLIB-compatible Neville method)
from pyins.core.unified_time import TimeCore
time = TimeCore.from_gps(2200, 345600.0)
pos, clk = sp3.interpolate_position(sat_num=1, time=time, method='neville')

# Unified interface for broadcast/SP3 ephemeris
# Automatically uses SP3 if available, falls back to broadcast
pos, clk, var = compute_satellite_position(
    sat_num=1, 
    time=time,
    nav_or_sp3=sp3_data,  # Can be nav_data or sp3_data
    prefer_mgex=True       # Prefer MGEX for multi-GNSS
)
```

### Time Management with TimeCore
```python
from pyins.core.unified_time import TimeCore, TimeSystem

# Create from various formats
tc = TimeCore.from_gps(2200, 345600.0)      # GPS week/TOW
tc = TimeCore.from_unix(1700000000)         # Unix timestamp
tc = TimeCore.from_auto(gps_seconds)        # Auto-detect

# Convert between formats
gps_sec = tc.get_gps_seconds()
week, tow = tc.get_gps_week_tow()
unix = tc.get_unix()

# System-specific time
gps_tow = tc.get_tow(TimeSystem.GPS)
bds_tow = tc.get_tow(TimeSystem.BDS)  # BeiDou time

# Time arithmetic
tc2 = tc + 3600  # Add 1 hour
dt = tc2 - tc    # Difference in seconds
```

### Factor Graph Optimization
```python
from pyins.fusion.graph_optimizer import FactorGraph
from pyins.fusion.pseudorange_factor import PseudorangeFactor
from pyins.fusion.doppler_factor import DopplerFactor
from pyins.fusion.state import NavigationState

# Create factor graph
graph = FactorGraph()

# Create navigation state
state = NavigationState()

# Add pseudorange factor
pr_factor = PseudorangeFactor(
    observation=observation,
    satellite_info=sat_info,
    weight=1.0/pr_variance
)

# Add Doppler factor  
doppler_factor = DopplerFactor(
    measurement=doppler_obs,
    satellite_velocity=sat_vel,
    weight=1.0/doppler_variance
)

graph.add_factor(pr_factor)
graph.add_factor(doppler_factor)

# Optimize
result = graph.optimize()
```

## Module Structure

```
pyins/
├── core/           # Core utilities and constants
│   ├── constants.py
│   ├── unified_time.py
│   └── satellite_numbering.py
├── gnss/           # GNSS processing
│   ├── spp.py      # Single Point Positioning
│   ├── ephemeris.py
│   ├── sp3_ephemeris.py     # SP3 precise ephemeris
│   ├── sp3_downloader_ftp.py # SP3/CLK downloader
│   ├── sp3_interpolation.py  # Neville interpolation
│   └── frequency.py
├── rtk/            # RTK processing
│   ├── double_difference.py
│   ├── ambiguity_resolution.py
│   └── cycle_slip.py
├── fusion/         # Sensor fusion
│   ├── ekf.py
│   ├── graph_optimizer.py
│   └── pseudorange_factor.py
├── coordinate/     # Coordinate transformations
│   ├── transforms.py
│   └── geodetic.py
├── sensors/        # Sensor models
│   ├── imu.py
│   └── lever_arm.py
└── io/             # File I/O
    └── rinex.py
```

## Examples

### Basic Examples
```bash
# Multi-GNSS SPP processing
python pyins/examples/multi_gnss_spp.py

# SPP with TimeCore integration
python pyins/examples/spp_with_timecore.py

# SPP with SP3 precise ephemeris
python pyins/examples/spp_with_sp3.py

# RTK Double Difference
python examples/rtk_double_difference_example.py

# Robot lever arm configuration
python examples/robot_lever_arm_usage.py

# Factor graph with weighted measurements
python examples/weighted_factor_usage.py
```

## Satellite Numbering Convention

PyINS uses internal satellite numbering:
- GPS: 1-32 (G01-G32)
- GLONASS: 65-88 (R01-R24)
- Galileo: 97-132 (E01-E36)
- BeiDou: 141-177 (C01-C37)
- QZSS: 210-216 (J01-J07)

```python
from pyins.core.constants import sat2sys, sys2char, sat2prn

sat = 141  # Internal number for C01
sys = sat2sys(sat)        # Returns SYS_CMP (BeiDou)
sys_char = sys2char(sys)  # Returns 'C'
prn = sat2prn(sat)        # Returns 1
```

## SP3 Precise Ephemeris Support

### Accuracy Improvement
SP3 precise ephemeris provides 40-80x better orbit accuracy compared to broadcast ephemeris:
- **Broadcast ephemeris**: 1-2 meters orbit accuracy
- **IGS Final SP3**: 2.5 cm (80x improvement)
- **IGS Rapid SP3**: 5 cm (40x improvement)
- **Clock accuracy**: 5-7 ns → 0.1-0.2 ns

### Product Types
PyINS supports both GPS-only and Multi-GNSS SP3 products:

| Product | Systems | Satellites | Latency | Accuracy |
|---------|---------|------------|---------|----------|
| IGS Final | GPS | 32 | 12-18 days | ~2.5 cm |
| IGS Rapid | GPS | 32 | 17 hours | ~5 cm |
| COD MGEX | GPS+GLO+GAL+BDS+QZS | 100+ | 2-3 days | ~5 cm |
| GFZ MGEX | GPS+GLO+GAL+BDS+QZS | 100+ | 1-2 days | ~5 cm |
| WUM MGEX | GPS+GLO+GAL+BDS+QZS | 100+ | 2-3 days | ~5 cm |

### Default Configuration
PyINS defaults to MGEX products (COD) for multi-GNSS support. To use GPS-only products:
```python
sp3.download_sp3(date, product='igs')  # GPS-only IGS Final
```

### Interpolation Methods
- **Neville** (default): RTKLIB-compatible polynomial interpolation
- **Polyfit**: NumPy polynomial fitting
- **Lagrange**: SciPy Lagrange interpolation
- **Cubic Spline**: Smooth spline interpolation

## Multi-GNSS Considerations

### BeiDou Time System
BeiDou uses BDT with a 14-second offset from GPS time, handled automatically by TimeCore.

### GLONASS Notes
- Uses FDMA with inter-frequency bias (IFB)
- Excluded by default in SPP due to IFB issues
- Can be included with `systems_to_use=['G', 'R', 'E', 'C']`

## API Reference

### Key Classes

#### SP3Ephemeris
Precise ephemeris handler with automatic download and interpolation.
- Downloads from IGS/MGEX FTP servers
- Supports multiple interpolation methods
- RTKLIB-compatible Neville interpolation
- Automatic product selection based on data age

#### TimeCore
Unified time management for all GNSS systems.

#### FactorGraph
Factor graph optimization for sensor fusion with pseudorange and Doppler factors.

#### DoubleDifferenceProcessor
RTK double difference processing with per-system reference satellites.

#### RobotLeverArm
Flexible lever arm management for multi-sensor robots.

## Contributing

Contributions are welcome! Please submit pull requests or open issues on GitHub.

## License

This project is licensed under the Apache License 2.0 - see the [LICENSE](LICENSE) file for details.

## Acknowledgments and Third-Party Licenses

### RTKLIB Attribution

This software includes algorithms and concepts inspired by [RTKLIB](https://github.com/tomojitakasu/RTKLIB), an open-source GNSS positioning library.

```
RTKLIB: An Open Source Program Package for GNSS Positioning
Copyright (c) 2007-2013, T. Takasu, All rights reserved.

RTKLIB is distributed under the BSD 2-clause license.
```

We acknowledge the significant contributions of RTKLIB to the GNSS community and thank T. Takasu for making this valuable software available as open source.

### Other References

- **gnss-py**: GNSS processing in Python
- **rtklib-py**: Python wrapper for RTKLIB

The PyINS library reimplements and extends concepts from these projects while maintaining compatibility with standard GNSS processing workflows.

## Citation

If you use PyINS in your research, please cite:
```
@software{pyins,
  title = {PyINS: Python GNSS/INS Processing Library},
  url = {https://github.com/inuex35/pyins},
  year = {2024}
}
```