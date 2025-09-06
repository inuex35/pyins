# PyINS Examples

This directory contains comprehensive examples for PyINS functions.

## Directory Structure

```
examples/
├── rtk/                      # RTK processing examples
│   ├── example_double_difference.py  # Double difference processing
│   ├── example_lambda.py            # LAMBDA ambiguity resolution
│   └── example_lambda_simple.py     # Simplified LAMBDA examples
├── gnss/                     # GNSS processing examples  
│   ├── example_gnss_processing.py   # General GNSS processing
│   └── example_observables.py       # Observable combinations
├── core/                     # Core functionality examples
├── sensors/                  # Sensor processing examples
├── io/                       # I/O operation examples
└── run_all_examples.py       # Test runner for all examples
```

## Running Examples

### Run all examples:
```bash
python run_all_examples.py
```

### Run specific module examples:
```bash
python run_all_examples.py --module rtk
```

### Run with verbose output:
```bash
python run_all_examples.py --verbose
```

### Run individual example:
```bash
python rtk/example_double_difference.py
```

## Working Examples

Currently working and tested:
- `rtk/example_double_difference.py` - RTK double difference processing with synthetic and real data

## Data Files

The following RINEX observation files in the `data/` directory are used for testing (not committed):
- `data/main.obs` - Rover observations
- `data/base.obs` - Base station observations
- `data/rover.nav` - Navigation data
- `data/main.pos` - Position solutions
- `data/main.pos.stat` - Solution statistics

## Key Features Demonstrated

### RTK Processing
- Double difference formation
- Multi-frequency processing (L1/L2/L5)
- Reference satellite selection
- Covariance propagation
- Real RINEX data processing

### LAMBDA Ambiguity Resolution
- Basic LAMBDA algorithm
- Multi-frequency cascaded resolution
- Bootstrapping method
- Partial ambiguity fixing
- Ratio test validation

### GNSS Observables
- Ionosphere-free combinations
- Geometry-free combinations
- Wide-lane/Narrow-lane combinations
- Melbourne-Wübbena combination
- Cycle slip detection

### Multi-GNSS Support
- GPS L1/L2/L5
- GLONASS G1/G2
- Galileo E1/E5a/E5b/E6
- BeiDou B1/B2/B3
- QZSS L1/L2/L5/L6

## Notes

- All examples follow multi-frequency, multi-GNSS policy
- Examples use actual pyins functions where available
- Synthetic data is used when real data is not required
- Examples are designed to be educational and demonstrate best practices