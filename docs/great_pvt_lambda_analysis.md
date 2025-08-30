# GREAT-PVT LAMBDA Implementation Analysis

## Key Features

### 1. **Simplified Decorrelation**
- Uses modified Cholesky decomposition
- Integer rounding for decorrelation factors
- Single-pass transformation with Z matrix
```python
mu = np.round(Q_work[i, j] / Q_work[i, i])
if mu != 0:
    Z[j, :] -= mu * Z[i, :]
    Q_work[j, :] -= mu * Q_work[i, :]
```

### 2. **Simple Integer Search**
- Primary candidate: Simple rounding
- Secondary candidate: Perturbation by ±1
- No complex tree search
- Direct residual computation

### 3. **Ratio Test**
- Uses square root of residual ratio
- `ratio = np.sqrt(residuals[1] / residuals[0])`
- Standard threshold of 3.0

### 4. **Partial Ambiguity Resolution**
- Elevation-based satellite selection
- Fixes only high-elevation satellites (>15°)
- Minimum satellite requirement (default: 4)

### 5. **Validation Function**
- Checks standardized residuals
- Maximum deviation limit (0.25 cycles)
- Both absolute and relative checks

## Comparison with Other Implementations

| Feature | GREAT-PVT | RTKLIB-py | pyins (current) |
|---------|-----------|-----------|-----------------|
| **Decorrelation** | Simple Cholesky | LD with swaps | LD with swaps |
| **Search** | Rounding + ±1 | Tree search (10000) | Tree search (10000) |
| **Complexity** | O(n²) | O(n³) worst | O(n³) worst |
| **Ratio test** | √(r₂/r₁) | r₂/r₁ | r₂/r₁ |
| **Partial AR** | Elevation-based | Variance-based | Variance-based |
| **Validation** | Built-in | External | External |

## Advantages of GREAT-PVT Approach

1. **Simplicity**
   - Easier to understand and debug
   - Lower computational cost
   - Suitable for real-time applications

2. **Robustness**
   - Elevation-based selection reduces multipath
   - Built-in validation checks
   - Conservative fixing strategy

3. **Practical**
   - Works well for short baselines
   - Good for dynamic applications
   - Fast convergence

## Limitations

1. **Search completeness**
   - May miss optimal solution
   - Limited to nearby integers
   - No guarantee of global minimum

2. **Decorrelation**
   - Less effective than full LAMBDA
   - May not handle high correlation well
   - Simplified transformation

## Implementation Strategy for pyins

### Option 1: Direct Port
- Copy GREAT-PVT's simple approach
- Maintain compatibility with existing interface
- Add elevation-based selection

### Option 2: Hybrid Approach
- Use GREAT-PVT's decorrelation (simpler)
- Keep RTKLIB's search (more thorough)
- Add validation from GREAT-PVT

### Option 3: Selectable Algorithms
- Implement both methods
- Allow user to choose based on requirements
- Default to GREAT-PVT for simplicity

## Recommended Implementation

Use **Option 2: Hybrid Approach**
- Combines simplicity with thoroughness
- Better ratio values from complete search
- Validation ensures reliability
- Elevation-based selection for robustness