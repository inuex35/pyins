# PyINS Examples

This directory contains working examples demonstrating PyINS functionality with real RINEX data.

## Directory Structure

```
examples/
├── gnss/                     # GNSS processing examples
│   ├── example_gnss_processing.py      # Frequency calculations with real data
│   ├── example_observables.py          # Observable combinations (IF, WL, NL)
│   └── example_satellite_positions.py  # Satellite position calculations
├── rtk/                      # RTK processing examples
│   ├── example_double_difference.py    # Double difference with synthetic data
│   └── example_lambda.py               # LAMBDA ambiguity resolution
└── run_all_examples.py       # Test runner for all examples
```

## Running Examples

### Run all examples:
```bash
python examples/run_all_examples.py
```

### Run individual example:
```bash
python examples/rtk/example_double_difference.py
python examples/gnss/example_gnss_processing.py
```

## Working Examples (All Tested ✓)

### GNSS Processing (3 examples)
1. **`gnss/example_gnss_processing.py`** - GNSS frequency calculations
   - Loads real RINEX observation files
   - Calculates frequencies and wavelengths for different GNSS systems
   - Converts carrier phase to distance

2. **`gnss/example_observables.py`** - Observable combinations
   - Ionosphere-free (IF) combination
   - Wide-lane (WL) combination
   - Narrow-lane (NL) combination
   - Uses dual-frequency GPS observations from real data

3. **`gnss/example_satellite_positions.py`** - Satellite position calculations
   - Attempts to load navigation data if available
   - Shows observation statistics by GNSS system
   - Computes satellite positions and velocities

### RTK Processing (2 examples)
4. **`rtk/example_double_difference.py`** - Double difference processing
   - Uses synthetic data to avoid timeout issues
   - Forms single and double differences
   - Demonstrates reference satellite selection
   - Shows DD phase and code observations

5. **`rtk/example_lambda.py`** - LAMBDA ambiguity resolution
   - Simple synthetic data example
   - Demonstrates integer ambiguity fixing
   - Shows best and second-best solutions
   - Validates resolution success

## Data Files

The examples use RINEX observation files from the `data/` directory (not committed to repository):
- `data/main.obs` - Rover observations (multi-GNSS, multi-frequency)
- `data/base.obs` - Base station observations
- `data/main.nav` - Navigation data (optional)

## Test Results

Running `python examples/run_all_examples.py`:

```
======================================================================
EXECUTION SUMMARY
======================================================================

Total examples: 5
  ✓ Success: 5
  ✗ Failed:  0

Total execution time: ~50s

----------------------------------------------------------------------
Example                                            Status     Time (s)  
----------------------------------------------------------------------
gnss/example_gnss_processing.py                    ✓ SUCCESS     ~24
gnss/example_observables.py                        ✓ SUCCESS     ~12
gnss/example_satellite_positions.py                ✓ SUCCESS     ~13
rtk/example_double_difference.py                   ✓ SUCCESS     <1
rtk/example_lambda.py                              ✓ SUCCESS     <1
```

## Key Features Demonstrated

### RTKLIB-Compatible Frequency Mapping
All examples use RTKLIB-standard frequency indexing:
- L[0] = L1/E1/B1/G1 (primary frequency)
- L[1] = L2/E5b/B2/G2 (secondary frequency)
- L[2] = L5/E5a/B3/G3 (tertiary frequency)

### Multi-GNSS Support
Examples work with all GNSS systems:
- GPS (L1, L2, L5)
- GLONASS (G1, G2)
- Galileo (E1, E5a, E5b)
- BeiDou (B1, B2, B3)
- QZSS (L1, L2, L5)

### Real Data Processing
- Examples load and process actual RINEX observation files
- Handles multi-frequency, multi-constellation data
- Demonstrates practical GNSS processing workflows

## Notes

- All examples are designed to complete quickly (<30s timeout)
- Synthetic data is used for RTK examples to ensure consistent execution
- Real RINEX data is used for GNSS examples when available
- Examples follow PyINS best practices and use actual library functions