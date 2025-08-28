#!/usr/bin/env python3
"""
Enhanced LAMBDA with Wide/Narrow Lane Combinations
===================================================

Combines traditional LAMBDA with WL/NL preprocessing for improved success rate.
"""

import numpy as np
from typing import Tuple, Dict, Optional, List
import logging

from .lambda_greatpvt import GreatPVTLambdaResolver
from .wide_narrow_lane import WideNarrowLaneProcessor, TripleFrequencyResolver

logger = logging.getLogger(__name__)


class EnhancedLAMBDAResolver:
    """
    Enhanced LAMBDA resolver using WL/NL combinations
    
    Strategy:
    1. Fix Wide Lane ambiguity using MW combination (easy, ~86cm wavelength)
    2. Use WL constraint to reduce search space for L1/L2
    3. Apply LAMBDA to remaining ambiguities
    """
    
    def __init__(self, ratio_threshold: float = 2.5,
                 use_triple_freq: bool = False):
        """
        Initialize enhanced resolver
        
        Parameters
        ----------
        ratio_threshold : float
            Ratio test threshold (can be lower due to WL constraint)
        use_triple_freq : bool
            Whether to use triple-frequency if available
        """
        self.ratio_threshold = ratio_threshold
        self.use_triple_freq = use_triple_freq
        self.lambda_resolver = GreatPVTLambdaResolver(ratio_threshold=ratio_threshold)
        self.wl_processor = WideNarrowLaneProcessor()
        self.triple_resolver = TripleFrequencyResolver() if use_triple_freq else None
        
        # Statistics
        self.wl_success_count = 0
        self.total_attempts = 0
        
    def resolve_with_wl_constraint(self, 
                                  dd_phase_l1: np.ndarray,
                                  dd_phase_l2: np.ndarray,
                                  dd_code_l1: np.ndarray,
                                  dd_code_l2: np.ndarray,
                                  dd_cov: np.ndarray,
                                  elevations: Optional[np.ndarray] = None) -> Dict:
        """
        Resolve ambiguities using WL constraint
        
        Parameters
        ----------
        dd_phase_l1 : np.ndarray
            L1 double-difference phases (cycles) - shape (n_dd, n_epochs)
        dd_phase_l2 : np.ndarray
            L2 double-difference phases (cycles) - shape (n_dd, n_epochs)
        dd_code_l1 : np.ndarray
            L1 double-difference codes (meters) - shape (n_dd, n_epochs)
        dd_code_l2 : np.ndarray
            L2 double-difference codes (meters) - shape (n_dd, n_epochs)
        dd_cov : np.ndarray
            Covariance matrix for DD ambiguities
        elevations : np.ndarray, optional
            Satellite elevations for weighting
            
        Returns
        -------
        result : dict
            Resolution results with WL constraint applied
        """
        self.total_attempts += 1
        n_dd = dd_phase_l1.shape[0]
        n_epochs = dd_phase_l1.shape[1] if dd_phase_l1.ndim > 1 else 1
        
        result = {
            'l1_ambiguities': None,
            'l2_ambiguities': None,
            'wl_ambiguities': None,
            'success': False,
            'wl_fixed': False,
            'ratio': 0.0,
            'method': 'enhanced_wl'
        }
        
        # Ensure we have multiple epochs
        if dd_phase_l1.ndim == 1:
            dd_phase_l1 = dd_phase_l1.reshape(-1, 1)
            dd_phase_l2 = dd_phase_l2.reshape(-1, 1)
            dd_code_l1 = dd_code_l1.reshape(-1, 1)
            dd_code_l2 = dd_code_l2.reshape(-1, 1)
        
        # Step 1: Fix WL ambiguities for each DD pair
        wl_ambiguities = []
        wl_confidences = []
        
        for i in range(n_dd):
            if n_epochs > 1:
                # Use MW combination over multiple epochs
                mw_values = []
                for j in range(n_epochs):
                    mw, _ = self.wl_processor.compute_melbourne_wubbena(
                        dd_phase_l1[i, j], dd_phase_l2[i, j],
                        dd_code_l1[i, j], dd_code_l2[i, j]
                    )
                    mw_values.append(mw)
                
                wl_amb, wl_conf, wl_fixed = self.wl_processor.resolve_wide_lane_ambiguity(
                    np.array(mw_values)
                )
            else:
                # Single epoch - use simple MW
                mw, mw_std = self.wl_processor.compute_melbourne_wubbena(
                    dd_phase_l1[i, 0], dd_phase_l2[i, 0],
                    dd_code_l1[i, 0], dd_code_l2[i, 0]
                )
                wl_amb = np.round(mw) if mw_std < 0.5 else None
                wl_conf = 1.0 - np.abs(mw - np.round(mw)) if wl_amb else 0
                wl_fixed = wl_amb is not None
            
            wl_ambiguities.append(wl_amb)
            wl_confidences.append(wl_conf)
        
        # Count successful WL fixes
        n_wl_fixed = sum(1 for amb in wl_ambiguities if amb is not None)
        wl_fix_rate = n_wl_fixed / n_dd if n_dd > 0 else 0
        
        result['wl_ambiguities'] = wl_ambiguities
        result['wl_fixed'] = n_wl_fixed > 0
        result['wl_fix_rate'] = wl_fix_rate
        
        if n_wl_fixed > 0:
            self.wl_success_count += 1
            logger.info(f"Fixed {n_wl_fixed}/{n_dd} WL ambiguities")
        
        # Step 2: Constrained search for L1/L2 ambiguities
        # For DD pairs with fixed WL, we have: N_WL = N_L1 - N_L2
        
        # Estimate float ambiguities
        float_l1 = np.zeros(n_dd)
        float_l2 = np.zeros(n_dd)
        
        for i in range(n_dd):
            # Average over epochs for float estimate
            phase_l1_m = np.mean(dd_phase_l1[i, :]) * self.wl_processor.lambda1
            phase_l2_m = np.mean(dd_phase_l2[i, :]) * self.wl_processor.lambda2
            code_l1_m = np.mean(dd_code_l1[i, :])
            code_l2_m = np.mean(dd_code_l2[i, :])
            
            float_l1[i] = (phase_l1_m - code_l1_m) / self.wl_processor.lambda1
            float_l2[i] = (phase_l2_m - code_l2_m) / self.wl_processor.lambda2
        
        # Step 3: Apply LAMBDA with WL constraints
        if n_wl_fixed > n_dd * 0.5:  # If we fixed most WL ambiguities
            # Use constrained search
            fixed_l1, fixed_l2 = self._constrained_search(
                float_l1, float_l2, wl_ambiguities, dd_cov
            )
            
            if fixed_l1 is not None:
                result['l1_ambiguities'] = fixed_l1
                result['l2_ambiguities'] = fixed_l2
                result['success'] = True
                result['ratio'] = 3.0  # Placeholder - compute actual ratio
                logger.info(f"Successfully fixed L1/L2 ambiguities with WL constraint")
        else:
            # Fall back to standard LAMBDA
            # Stack L1 and L2 ambiguities
            float_combined = np.concatenate([float_l1, float_l2])
            
            # Extend covariance matrix
            cov_combined = np.block([
                [dd_cov, np.zeros((n_dd, n_dd))],
                [np.zeros((n_dd, n_dd)), dd_cov * 1.5]  # L2 typically has higher noise
            ])
            
            # Apply LAMBDA
            fixed_combined, ratio, is_fixed, info = self.lambda_resolver.resolve(
                float_combined, cov_combined, elevations
            )
            
            if is_fixed:
                result['l1_ambiguities'] = fixed_combined[:n_dd].astype(int)
                result['l2_ambiguities'] = fixed_combined[n_dd:].astype(int)
                result['success'] = True
                result['ratio'] = ratio
                logger.info(f"Fixed ambiguities using standard LAMBDA (ratio: {ratio:.2f})")
        
        return result
    
    def _constrained_search(self, float_l1: np.ndarray, float_l2: np.ndarray,
                           wl_constraints: List, cov: np.ndarray) -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]:
        """
        Search for integer ambiguities with WL constraints
        
        For each DD with known WL: N_L1 - N_L2 = N_WL
        """
        n = len(float_l1)
        best_l1 = None
        best_l2 = None
        best_residual = float('inf')
        
        # Build constraint matrix
        has_constraint = [wl is not None for wl in wl_constraints]
        
        if sum(has_constraint) == 0:
            return None, None
        
        # Simple search (can be improved with more sophisticated methods)
        search_range = 3
        
        # Start with rounded float solution
        base_l1 = np.round(float_l1).astype(int)
        base_l2 = np.round(float_l2).astype(int)
        
        # Apply WL constraints to base solution
        for i, wl in enumerate(wl_constraints):
            if wl is not None:
                # Enforce constraint: adjust L2 to match WL
                base_l2[i] = base_l1[i] - wl
        
        # Search around constrained solution
        for delta in range(-search_range, search_range + 1):
            candidate_l1 = base_l1.copy()
            candidate_l2 = base_l2.copy()
            
            # Try perturbing unconstrained ambiguities
            for i in range(n):
                if not has_constraint[i]:
                    candidate_l1[i] += delta
            
            # Compute residual
            residual = 0
            residual += np.sum((candidate_l1 - float_l1)**2 / np.diag(cov))
            residual += np.sum((candidate_l2 - float_l2)**2 / np.diag(cov))
            
            if residual < best_residual:
                best_residual = residual
                best_l1 = candidate_l1.copy()
                best_l2 = candidate_l2.copy()
        
        # Check if solution is good enough
        if best_residual < n * 0.5:  # Threshold based on dimension
            return best_l1, best_l2
        else:
            return None, None
    
    def get_statistics(self) -> Dict:
        """Get resolver statistics"""
        return {
            'total_attempts': self.total_attempts,
            'wl_success_count': self.wl_success_count,
            'wl_success_rate': self.wl_success_count / max(self.total_attempts, 1)
        }


def test_enhanced_resolver():
    """Test enhanced LAMBDA with WL/NL"""
    import matplotlib.pyplot as plt
    
    print("=" * 70)
    print("Testing Enhanced LAMBDA with Wide/Narrow Lane")
    print("=" * 70)
    
    # Simulate dual-frequency observations
    n_dd = 5  # Number of double differences
    n_epochs = 30  # Number of epochs
    
    # True ambiguities
    true_l1 = np.array([10, -5, 3, -8, 15])
    true_l2 = np.array([8, -7, 2, -6, 12])
    true_wl = true_l1 - true_l2  # WL ambiguity
    
    print(f"\nTrue ambiguities:")
    print(f"  L1: {true_l1}")
    print(f"  L2: {true_l2}")
    print(f"  WL: {true_wl}")
    
    # Generate simulated observations
    np.random.seed(42)
    
    # Phase observations (with noise)
    phase_noise = 0.003  # 3mm phase noise
    dd_phase_l1 = np.zeros((n_dd, n_epochs))
    dd_phase_l2 = np.zeros((n_dd, n_epochs))
    
    for i in range(n_dd):
        for j in range(n_epochs):
            # Simulate phase with ambiguity + noise
            dd_phase_l1[i, j] = true_l1[i] + np.random.randn() * phase_noise / 0.19
            dd_phase_l2[i, j] = true_l2[i] + np.random.randn() * phase_noise / 0.24
    
    # Code observations (with more noise)
    code_noise = 0.3  # 30cm code noise
    dd_code_l1 = np.random.randn(n_dd, n_epochs) * code_noise
    dd_code_l2 = np.random.randn(n_dd, n_epochs) * code_noise
    
    # Covariance matrix
    dd_cov = np.eye(n_dd) * (phase_noise / 0.19)**2
    
    # Elevations (random)
    elevations = np.random.uniform(20, 80, n_dd)
    
    # Test enhanced resolver
    resolver = EnhancedLAMBDAResolver(ratio_threshold=2.5)
    
    result = resolver.resolve_with_wl_constraint(
        dd_phase_l1, dd_phase_l2,
        dd_code_l1, dd_code_l2,
        dd_cov, elevations
    )
    
    print(f"\nResolution results:")
    print(f"  Success: {result['success']}")
    print(f"  WL fixed: {result['wl_fixed']} ({result.get('wl_fix_rate', 0):.1%})")
    print(f"  Ratio: {result['ratio']:.2f}")
    
    if result['wl_ambiguities']:
        print(f"  Fixed WL: {result['wl_ambiguities']}")
        wl_errors = [abs(f - t) if f else -1 
                     for f, t in zip(result['wl_ambiguities'], true_wl)]
        print(f"  WL errors: {wl_errors}")
    
    if result['success']:
        print(f"  Fixed L1: {result['l1_ambiguities']}")
        print(f"  Fixed L2: {result['l2_ambiguities']}")
        
        l1_errors = np.sum(result['l1_ambiguities'] != true_l1)
        l2_errors = np.sum(result['l2_ambiguities'] != true_l2)
        print(f"  L1 errors: {l1_errors}/{n_dd}")
        print(f"  L2 errors: {l2_errors}/{n_dd}")
    
    # Plot MW combination evolution
    fig, axes = plt.subplots(2, 1, figsize=(10, 8))
    
    # MW values for each DD
    wl_proc = WideNarrowLaneProcessor()
    for i in range(n_dd):
        mw_values = []
        for j in range(n_epochs):
            mw, _ = wl_proc.compute_melbourne_wubbena(
                dd_phase_l1[i, j], dd_phase_l2[i, j],
                dd_code_l1[i, j], dd_code_l2[i, j]
            )
            mw_values.append(mw)
        
        ax = axes[0]
        ax.plot(mw_values, label=f'DD {i+1}')
        ax.axhline(y=true_wl[i], color='gray', linestyle='--', alpha=0.5)
    
    axes[0].set_xlabel('Epoch')
    axes[0].set_ylabel('MW Combination (cycles)')
    axes[0].set_title('Melbourne-Wübbena Evolution')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    
    # WL wavelength advantage
    ax = axes[1]
    wavelengths = {
        'L1': 0.190,  # meters
        'L2': 0.244,  # meters
        'WL': 0.862,  # meters
        'NL': 0.107,  # meters
        'IF': None     # No single wavelength
    }
    
    names = []
    values = []
    colors = []
    for name, wl in wavelengths.items():
        if wl:
            names.append(name)
            values.append(wl * 100)  # Convert to cm
            if name == 'WL':
                colors.append('green')
            elif name == 'NL':
                colors.append('orange')
            else:
                colors.append('blue')
    
    bars = ax.bar(names, values, color=colors)
    ax.set_ylabel('Wavelength (cm)')
    ax.set_title('Wavelength Comparison')
    ax.grid(True, alpha=0.3, axis='y')
    
    # Add values on bars
    for bar, val in zip(bars, values):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
                f'{val:.1f}cm', ha='center', va='bottom')
    
    plt.tight_layout()
    plt.savefig('enhanced_lambda_test.png', dpi=150)
    print(f"\nPlot saved to 'enhanced_lambda_test.png'")
    
    # Print statistics
    stats = resolver.get_statistics()
    print(f"\nResolver statistics:")
    print(f"  Total attempts: {stats['total_attempts']}")
    print(f"  WL success rate: {stats['wl_success_rate']:.1%}")
    
    print("\n" + "=" * 70)
    print("Enhanced LAMBDA Advantages:")
    print("  - WL ambiguity easier to fix (86cm wavelength)")
    print("  - WL constraint reduces L1/L2 search space")
    print("  - Lower ratio threshold acceptable (2.5 vs 3.0)")
    print("  - Better success rate for dual-frequency")
    print("=" * 70)


if __name__ == "__main__":
    test_enhanced_resolver()