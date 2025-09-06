# PyINS - Python GNSS/INS Processing Library

PyINS is a comprehensive GNSS/IMU processing library for satellite positioning, RTK, sensor fusion, and coordinate transformations. Inspired by RTKLIB and gnss-py.

## Features

### GNSS Processing
- **Multi-constellation support**: GPS, Galileo, BeiDou, GLONASS, QZSS
- **RINEX file I/O**: Observation and navigation file parsing using gnsspy
- **RTKLIB-compatible frequency mapping**: L[0]=L1/E1/B1, L[1]=L2/E5b/B2, L[2]=L5/E5a/B3
- **RTK (Real-Time Kinematic)**: Double difference processing with reference satellite selection
- **Carrier phase processing**: Cycle slip detection and ambiguity resolution (LAMBDA/MLAMBDA)
- **Observable combinations**: Ionosphere-free, wide-lane, narrow-lane combinations

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

### GNSS Frequency Calculations
```python
from pyins.io.rinex import RinexObsReader
from pyins.gnss.frequency import sat2freq
from pyins.core.constants import CLIGHT

# Load RINEX observation file
reader = RinexObsReader('data/main.obs')
epochs = reader.read()[:1]  # Read first epoch

observations = epochs[0]['observations']

# Process GPS satellites
gps_obs = [o for o in observations if o.system == 1]

for obs in gps_obs[:3]:
    if obs.L[0] != 0:  # L1 at index 0 (RTKLIB-compatible)
        # Calculate L1 frequency and wavelength
        freq_l1 = sat2freq(obs.sat, 0)
        wavelength_l1 = CLIGHT / freq_l1
        
        # Convert carrier phase to distance
        phase_cycles = obs.L[0]
        distance = phase_cycles * wavelength_l1
        
        print(f"PRN {obs.sat}:")
        print(f"  L1 phase: {phase_cycles:.3f} cycles")
        print(f"  Distance: {distance:.3f} m")
```

### Observable Combinations
```python
from pyins.io.rinex import RinexObsReader
from pyins.core.constants import CLIGHT, FREQ_L1, FREQ_L2

# Load RINEX file
reader = RinexObsReader('data/main.obs')
epochs = reader.read()[:1]
observations = epochs[0]['observations']

# Find dual-frequency GPS observations
gps_dual = [o for o in observations 
            if o.system == 1 and o.L[0] != 0 and o.L[1] != 0]

for obs in gps_dual[:1]:
    # Frequencies and wavelengths
    f1, f2 = FREQ_L1, FREQ_L2
    lambda1 = CLIGHT / f1
    lambda2 = CLIGHT / f2
    
    # Phase in meters
    L1_m = obs.L[0] * lambda1
    L2_m = obs.L[1] * lambda2
    
    # Ionosphere-free combination
    alpha = (f1**2) / (f1**2 - f2**2)
    beta = (f2**2) / (f1**2 - f2**2)
    IF_phase = alpha * L1_m - beta * L2_m
    
    # Wide-lane combination
    WL_phase = (f1 * L1_m - f2 * L2_m) / (f1 - f2)
    WL_wavelength = CLIGHT / (f1 - f2)
    
    print(f"PRN {obs.sat}:")
    print(f"  Ionosphere-free: {IF_phase:.3f} m")
    print(f"  Wide-lane: {WL_phase:.3f} m")
    print(f"  WL wavelength: {WL_wavelength:.3f} m")
```

### RTK Double Difference
```python
from pyins.rtk.double_difference import DoubleDifferenceProcessor
from pyins.io.rinex import RinexObsReader

# Load rover and base observations
rover_reader = RinexObsReader('data/main.obs')
base_reader = RinexObsReader('data/base.obs')

rover_epochs = rover_reader.read()[:1]
base_epochs = base_reader.read()[:1]

rover_obs = rover_epochs[0]['observations']
base_obs = base_epochs[0]['observations']

# Filter to GPS
gps_rover = [o for o in rover_obs if o.system == 1][:5]
gps_base = [o for o in base_obs if o.system == 1][:5]

# Find common satellites
rover_sats = {o.sat for o in gps_rover}
base_sats = {o.sat for o in gps_base}
common = rover_sats & base_sats

if len(common) >= 2:
    rover_common = [o for o in gps_rover if o.sat in common]
    base_common = [o for o in gps_base if o.sat in common]
    
    # Form double differences
    dd_processor = DoubleDifferenceProcessor()
    dd_data = dd_processor.form_double_differences(
        rover_common, base_common, frequency_idx=0  # L1
    )
    
    if dd_data:
        dd_pr, dd_cp, pairs, refs = dd_data
        print(f"Reference satellite: {refs}")
        print(f"DD observations: {len(dd_pr)}")
        print(f"First DD phase: {dd_cp[0]:.3f} cycles")
```

### LAMBDA Ambiguity Resolution
```python
from pyins.rtk.lambda_rtklib import mlambda
import numpy as np

# Float ambiguities (close to integers)
float_amb = np.array([10.3, -5.7, 15.1])

# Covariance matrix
Q = np.diag([0.01, 0.02, 0.015])

# Solve with LAMBDA
fixed_amb, residuals = mlambda(float_amb, Q, m=2)

print(f"Float ambiguities: {float_amb}")
print(f"Fixed ambiguities: {fixed_amb[:,0].astype(int)}")
print(f"Residuals: {residuals}")

# Check success
expected = np.array([10, -6, 15])
if np.allclose(fixed_amb[:,0], expected, atol=1):
    print("✓ LAMBDA resolution successful!")
```

## Examples

See the `examples/` directory for complete working examples:

- **GNSS Processing**
  - `gnss/example_gnss_processing.py` - Frequency calculations with real RINEX data
  - `gnss/example_observables.py` - Observable combinations (IF, WL, NL)
  - `gnss/example_satellite_positions.py` - Satellite position calculations

- **RTK Processing**
  - `rtk/example_double_difference.py` - Double difference with synthetic data
  - `rtk/example_lambda.py` - LAMBDA ambiguity resolution

Run all examples:
```bash
python examples/run_all_examples.py
```

## Project Structure

```
pyins/
├── core/          # Core data structures and constants
├── gnss/          # GNSS processing algorithms
├── rtk/           # RTK and ambiguity resolution
├── coordinate/    # Coordinate transformations
├── sensors/       # IMU and sensor fusion
├── io/            # RINEX and data I/O
└── examples/      # Working examples
```

## Contributing

Contributions are welcome! Please feel free to submit pull requests.

## License

MIT License

## Acknowledgments

- RTKLIB for reference implementations
- gnsspy for RINEX parsing capabilities