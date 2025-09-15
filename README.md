# PyINS

Python GNSS/INS processing library inspired by [RTKLIB](https://github.com/tomojitakasu/RTKLIB).

## Installation

```bash
pip install -e .
```

## Data Loading

```python
from pyins.io.rinex import RinexObsReader, RinexNavReader

# Load observation data
obs_reader = RinexObsReader('data/main.obs')
epochs = obs_reader.read()

# Load navigation data
nav_reader = RinexNavReader('data/main.nav')
ephemeris = nav_reader.read()

# Access observations
epoch = epochs[0]
observations = epoch['observations']
timestamp = epoch['time']
```

## SPP (Single Point Positioning)

```python
from pyins.gnss.spp import SPP
import numpy as np

# Initialize SPP solver
spp = SPP()

# Prepare observations (pseudoranges)
obs_data = []
for obs in observations:
    if obs.P[0] != 0:  # L1 pseudorange available
        obs_data.append({
            'sat': obs.sat,
            'system': obs.system,
            'pseudorange': obs.P[0],
            'snr': obs.S[0]
        })

# Compute position
initial_pos = np.array([0, 0, 0])  # ECEF coordinates
position, clock_bias = spp.compute(obs_data, ephemeris, initial_pos)

print(f"Position (ECEF): {position}")
print(f"Clock bias: {clock_bias} m")
```

## Double Difference Pseudo Range Factor

```python
from pyins.factors.dd_pseudorange_factor import DDPseudoRangeFactor
from pyins.rtk.double_difference import DoubleDifferenceProcessor
import numpy as np

# Create DD processor
dd_processor = DoubleDifferenceProcessor()

# Form double differences for pseudo ranges
rover_obs = epochs[0]['observations']
base_obs = base_epochs[0]['observations']

dd_data = dd_processor.form_double_differences(
    rover_obs, base_obs,
    frequency_idx=0  # L1
)

if dd_data:
    dd_pr, dd_cp, pairs, ref_sat = dd_data

    # Create DD pseudo range factor for graph optimization
    dd_pr_factor = DDPseudoRangeFactor(
        rover_pos=rover_position,
        base_pos=base_position,
        dd_measurements=dd_pr,  # Double differenced pseudo ranges
        sat_pairs=pairs,
        ref_satellite=ref_sat,
        weight=1.0 / 3.0  # 3m std dev for pseudo ranges
    )

    # Add to factor graph
    graph.add_factor(dd_pr_factor)
```

## Double Difference Carrier Phase Factor

```python
from pyins.factors.dd_carrierphase_factor import DDCarrierPhaseFactor
from pyins.rtk.double_difference import DoubleDifferenceProcessor
import numpy as np

# Create DD processor
dd_processor = DoubleDifferenceProcessor()

# Form double differences for carrier phases
dd_data = dd_processor.form_double_differences(
    rover_obs, base_obs,
    frequency_idx=0  # L1
)

if dd_data:
    dd_pr, dd_cp, pairs, ref_sat = dd_data

    # Create DD carrier phase factor for graph optimization
    dd_cp_factor = DDCarrierPhaseFactor(
        rover_pos=rover_position,
        base_pos=base_position,
        dd_measurements=dd_cp,  # Double differenced carrier phases
        sat_pairs=pairs,
        ref_satellite=ref_sat,
        ambiguities=ambiguity_vector,  # Integer ambiguities
        weight=1.0 / 0.003  # 3mm std dev for carrier phases
    )

    # Add to factor graph
    graph.add_factor(dd_cp_factor)
```

## License

This project is licensed under the Apache License 2.0 - see the [LICENSE](LICENSE) file for details.

## Acknowledgments

- [RTKLIB](https://github.com/tomojitakasu/RTKLIB) by T.Takasu - Reference implementation and algorithms
- Algorithm implementations are based on RTKLIB's BSD-2-Clause licensed code