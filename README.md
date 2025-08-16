# PyINS - Python GNSS/INS Processing Library

PyINS is a comprehensive GNSS/IMU processing library for satellite positioning, RTK, sensor fusion, and coordinate transformations. Inspired by RTKLIB and gnss-py.

## Features

### GNSS Processing
- **Multi-constellation support**: GPS, Galileo, BeiDou, GLONASS, QZSS
- **RINEX file I/O**: Observation and navigation file parsing
- **Single Point Positioning (SPP)**: With iterative least squares and Sagnac correction
- **RTK (Real-Time Kinematic)**: Double difference processing with per-system reference satellites
- **Ephemeris handling**: Optimal satellite selection and position computation
- **Time management**: Unified TimeCore system for all GNSS time systems
- **Carrier phase processing**: Cycle slip detection and ambiguity resolution (LAMBDA)

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
from pyins.gnss.spp import single_point_positioning
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

# Solve for position
solution, used_sats = single_point_positioning(
    observations, 
    nav_data,
    systems_to_use=['G', 'E', 'C']  # GPS, Galileo, BeiDou
)

if solution:
    llh = ecef2llh(solution.rr)
    print(f"Position: lat={np.rad2deg(llh[0]):.6f}°, "
          f"lon={np.rad2deg(llh[1]):.6f}°, h={llh[2]:.1f}m")
```

### RTK Double Difference
```python
from pyins.rtk import DoubleDifferencePerSystem, DDLeastSquares
from pyins.io.rinex import RinexObsReader, RinexNavReader

# Read rover and base data
rover_obs = RinexObsReader('rover.obs').read()
base_obs = RinexObsReader('base.obs').read()
nav_data = RinexNavReader('nav.nav').read()

# Initialize processors
dd_processor = DoubleDifferencePerSystem()
dd_solver = DDLeastSquares()

# Process epoch
dd_obs = dd_processor.process(
    rover_obs[0]['observations'],
    base_obs[0]['observations'],
    nav_data
)

# Solve for baseline
baseline = dd_solver.solve(dd_obs, base_position)
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
from pyins.fusion import GraphOptimizer, PseudorangeFactor, DopplerFactor

# Create optimizer
optimizer = GraphOptimizer()

# Add factors
pr_factor = PseudorangeFactor(
    measurement=pseudorange,
    satellite_pos=sat_pos,
    weight=1.0/pr_variance
)

doppler_factor = DopplerFactor(
    measurement=doppler,
    satellite_vel=sat_vel,
    weight=1.0/doppler_variance
)

optimizer.add_factor(pr_factor)
optimizer.add_factor(doppler_factor)

# Optimize
result = optimizer.optimize(initial_state)
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
# Single Point Positioning
python examples/simple_example.py

# RTK Double Difference
python examples/rtk_double_difference_example.py

# Robot lever arm configuration
python examples/robot_lever_arm_usage.py

# Factor graph with weighted measurements
python examples/weighted_factor_usage.py
```

### Export to KML
```bash
# Convert trajectory to Google Earth format
python export_to_kml.py input.csv output.kml
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

## Multi-GNSS Considerations

### BeiDou Time System
BeiDou uses BDT with a 14-second offset from GPS time, handled automatically by TimeCore.

### GLONASS Notes
- Uses FDMA with inter-frequency bias (IFB)
- Excluded by default in SPP due to IFB issues
- Can be included with `systems_to_use=['G', 'R', 'E', 'C']`

## API Reference

### Key Classes

#### TimeCore
Unified time management for all GNSS systems.

#### GraphOptimizer
Factor graph optimization for sensor fusion.

#### DoubleDifferencePerSystem
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