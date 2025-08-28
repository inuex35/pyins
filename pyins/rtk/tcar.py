#!/usr/bin/env python3
"""
TCAR (Three-Carrier Ambiguity Resolution)
==========================================

TCAR uses geometry-free and ionosphere-free combinations of triple-frequency
observations to resolve ambiguities in a cascaded manner.

References:
    [1] Forssell et al. (1997) The use of three carriers for unambiguous 
        high precision kinematic GPS positioning
    [2] Vollath et al. (1999) Three or Four Carriers: How Many are Enough?
    [3] Feng (2008) GNSS three carrier ambiguity resolution using ionosphere-reduced
        virtual signals
"""

import numpy as np
from typing import Tuple, Optional, Dict, List
from dataclasses import dataclass
import logging

logger = logging.getLogger(__name__)

# GPS frequencies (Hz)
FREQ_L1 = 1575.42e6
FREQ_L2 = 1227.60e6
FREQ_L5 = 1176.45e6

# Wavelengths (m)
CLIGHT = 299792458.0
LAMBDA_L1 = CLIGHT / FREQ_L1  # ~0.19m
LAMBDA_L2 = CLIGHT / FREQ_L2  # ~0.24m
LAMBDA_L5 = CLIGHT / FREQ_L5  # ~0.25m


@dataclass
class TCARResult:
    """Result from TCAR processing"""
    ewl_ambiguities: np.ndarray  # Extra-wide lane
    wl_ambiguities: np.ndarray   # Wide lane
    nl_ambiguities: np.ndarray   # Narrow lane
    l1_ambiguities: np.ndarray   # L1 final
    l2_ambiguities: np.ndarray   # L2 final
    l5_ambiguities: np.ndarray   # L5 final
    success_flags: Dict[str, bool]
    statistics: Dict[str, float]


class TCARResolver:
    """
    Three-Carrier Ambiguity Resolution
    
    Resolves ambiguities using cascaded approach:
    1. Extra-wide lane (EWL) - easiest, longest wavelength
    2. Wide lane (WL) - intermediate
    3. Narrow lane (NL) - most precise
    """
    
    def __init__(self, 
                 ewl_threshold: float = 0.25,
                 wl_threshold: float = 0.25,
                 nl_threshold: float = 0.15):
        """
        Initialize TCAR resolver
        
        Parameters
        ----------
        ewl_threshold : float
            Threshold for EWL ambiguity fixing (cycles)
        wl_threshold : float
            Threshold for WL ambiguity fixing (cycles)
        nl_threshold : float
            Threshold for NL ambiguity fixing (cycles)
        """
        self.ewl_threshold = ewl_threshold
        self.wl_threshold = wl_threshold
        self.nl_threshold = nl_threshold
        
        # Compute combination wavelengths
        self.lambda_ewl = self._compute_ewl_wavelength()
        self.lambda_wl12 = self._compute_wl_wavelength(FREQ_L1, FREQ_L2)
        self.lambda_wl15 = self._compute_wl_wavelength(FREQ_L1, FREQ_L5)
    
    def resolve(self, 
                phase_l1: np.ndarray,
                phase_l2: np.ndarray,
                phase_l5: np.ndarray,
                code_l1: np.ndarray,
                code_l2: np.ndarray,
                code_l5: np.ndarray,
                elevations: Optional[np.ndarray] = None) -> TCARResult:
        """
        Resolve ambiguities using TCAR method
        
        Parameters
        ----------
        phase_l1, phase_l2, phase_l5 : np.ndarray
            Carrier phase observations (cycles)
        code_l1, code_l2, code_l5 : np.ndarray
            Code observations (meters)
        elevations : np.ndarray, optional
            Satellite elevation angles for weighting
            
        Returns
        -------
        result : TCARResult
            TCAR resolution result
        """
        n = len(phase_l1)
        
        # Step 1: Extra-Wide Lane (EWL) - combination of L2 and L5
        ewl_float, ewl_fixed, ewl_success = self._resolve_ewl(
            phase_l2, phase_l5, code_l2, code_l5
        )
        
        # Step 2: Wide Lane (WL) - multiple combinations
        wl12_float, wl12_fixed, wl12_success = self._resolve_wl(
            phase_l1, phase_l2, code_l1, code_l2, ewl_fixed
        )
        
        wl15_float, wl15_fixed, wl15_success = self._resolve_wl(
            phase_l1, phase_l5, code_l1, code_l5, ewl_fixed
        )
        
        # Use most reliable WL
        if wl12_success >= wl15_success:
            wl_fixed = wl12_fixed
            wl_success = wl12_success
        else:
            wl_fixed = wl15_fixed
            wl_success = wl15_success
        
        # Step 3: Narrow Lane (NL) using fixed WL
        nl_float, nl_fixed, nl_success = self._resolve_nl(
            phase_l1, phase_l2, phase_l5, wl_fixed, ewl_fixed
        )
        
        # Step 4: Recover original ambiguities
        l1_fixed, l2_fixed, l5_fixed = self._recover_original_ambiguities(
            ewl_fixed, wl_fixed, nl_fixed
        )
        
        # Compile results
        result = TCARResult(
            ewl_ambiguities=ewl_fixed,
            wl_ambiguities=wl_fixed,
            nl_ambiguities=nl_fixed,
            l1_ambiguities=l1_fixed,
            l2_ambiguities=l2_fixed,
            l5_ambiguities=l5_fixed,
            success_flags={
                'ewl': ewl_success > 0.95,
                'wl': wl_success > 0.95,
                'nl': nl_success > 0.95,
                'overall': ewl_success * wl_success * nl_success > 0.90
            },
            statistics={
                'ewl_success_rate': ewl_success,
                'wl_success_rate': wl_success,
                'nl_success_rate': nl_success,
                'ewl_wavelength': self.lambda_ewl,
                'wl_wavelength': self.lambda_wl12
            }
        )
        
        return result
    
    def _resolve_ewl(self, phase_l2: np.ndarray, phase_l5: np.ndarray,
                     code_l2: np.ndarray, code_l5: np.ndarray) -> Tuple[np.ndarray, np.ndarray, float]:
        """
        Resolve Extra-Wide Lane ambiguities
        
        EWL = L5 - L2 with wavelength ~5.86m
        """
        # Geometry-free combination
        ewl_phase = phase_l5 - phase_l2  # cycles
        
        # Use code to estimate float ambiguities
        ewl_code = (code_l5 - code_l2) / self.lambda_ewl
        
        # Float ambiguities
        ewl_float = ewl_code - ewl_phase
        
        # Round to nearest integer (easy due to long wavelength)
        ewl_fixed = np.round(ewl_float)
        
        # Compute success rate
        residuals = np.abs(ewl_float - ewl_fixed)
        success_rate = np.mean(residuals < self.ewl_threshold)
        
        return ewl_float, ewl_fixed.astype(int), success_rate
    
    def _resolve_wl(self, phase_l1: np.ndarray, phase_l2: np.ndarray,
                    code_l1: np.ndarray, code_l2: np.ndarray,
                    ewl_fixed: Optional[np.ndarray] = None) -> Tuple[np.ndarray, np.ndarray, float]:
        """
        Resolve Wide Lane ambiguities
        
        WL = L1 - L2 with wavelength ~86cm
        """
        # Melbourne-Wübbena combination
        f1, f2 = FREQ_L1, FREQ_L2
        alpha = f1 / (f1 + f2)
        beta = f2 / (f1 + f2)
        
        # Wide lane phase
        wl_phase = phase_l1 - phase_l2
        
        # Narrow lane code
        nl_code = alpha * code_l1 + beta * code_l2
        
        # MW combination
        lambda_wl = CLIGHT / (f1 - f2)
        mw = wl_phase - nl_code / lambda_wl
        
        # Float ambiguities
        wl_float = mw
        
        # Apply EWL constraint if available
        if ewl_fixed is not None:
            # Use EWL to constrain WL search space
            wl_constrained = wl_float - 0.1 * ewl_fixed  # Simplified constraint
            wl_fixed = np.round(wl_constrained)
        else:
            wl_fixed = np.round(wl_float)
        
        # Compute success rate
        residuals = np.abs(wl_float - wl_fixed)
        success_rate = np.mean(residuals < self.wl_threshold)
        
        return wl_float, wl_fixed.astype(int), success_rate
    
    def _resolve_nl(self, phase_l1: np.ndarray, phase_l2: np.ndarray,
                    phase_l5: np.ndarray, wl_fixed: np.ndarray,
                    ewl_fixed: np.ndarray) -> Tuple[np.ndarray, np.ndarray, float]:
        """
        Resolve Narrow Lane ambiguities
        
        NL = L1 + L2 with wavelength ~11cm
        """
        # Ionosphere-free combination
        f1, f2 = FREQ_L1, FREQ_L2
        alpha = f1**2 / (f1**2 - f2**2)
        beta = -f2**2 / (f1**2 - f2**2)
        
        # Narrow lane phase
        nl_phase = phase_l1 + phase_l2
        
        # Use fixed WL to constrain NL
        # N_NL = N_L1 + N_L2 = 2*N_L1 - N_WL
        nl_constrained = 2 * phase_l1 - wl_fixed
        
        # Round to nearest integer
        nl_fixed = np.round(nl_constrained)
        
        # Compute success rate
        residuals = np.abs(nl_constrained - nl_fixed)
        success_rate = np.mean(residuals < self.nl_threshold)
        
        return nl_constrained, nl_fixed.astype(int), success_rate
    
    def _recover_original_ambiguities(self, ewl: np.ndarray, wl: np.ndarray,
                                      nl: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Recover L1, L2, L5 ambiguities from combinations
        
        N_L1 = (N_NL + N_WL) / 2
        N_L2 = (N_NL - N_WL) / 2
        N_L5 = N_L2 + N_EWL
        """
        n_l1 = (nl + wl) / 2
        n_l2 = (nl - wl) / 2
        n_l5 = n_l2 + ewl
        
        return n_l1.astype(int), n_l2.astype(int), n_l5.astype(int)
    
    def _compute_ewl_wavelength(self) -> float:
        """Compute EWL wavelength"""
        return CLIGHT / (FREQ_L5 - FREQ_L2)  # ~5.86m
    
    def _compute_wl_wavelength(self, f1: float, f2: float) -> float:
        """Compute WL wavelength"""
        return CLIGHT / abs(f1 - f2)


class CascadedTCAR:
    """
    Cascaded TCAR with validation at each step
    """
    
    def __init__(self):
        self.tcar = TCARResolver()
        self.validation_thresholds = {
            'ewl_residual': 0.5,  # cycles
            'wl_residual': 0.3,   # cycles
            'nl_residual': 0.2    # cycles
        }
    
    def resolve_with_validation(self, observations: Dict) -> Dict:
        """
        Resolve with validation checks at each step
        
        Parameters
        ----------
        observations : dict
            Dictionary with L1, L2, L5 phase and code observations
            
        Returns
        -------
        result : dict
            Resolution results with validation status
        """
        # Extract observations
        phase_l1 = observations['L1']
        phase_l2 = observations['L2']
        phase_l5 = observations['L5']
        code_l1 = observations['P1']
        code_l2 = observations['P2']
        code_l5 = observations['P5']
        
        # Run TCAR
        tcar_result = self.tcar.resolve(
            phase_l1, phase_l2, phase_l5,
            code_l1, code_l2, code_l5
        )
        
        # Validate each step
        validation = {}
        
        # Validate EWL
        ewl_residual = self._compute_residual(
            phase_l5 - phase_l2,
            tcar_result.ewl_ambiguities,
            (code_l5 - code_l2) / self.tcar.lambda_ewl
        )
        validation['ewl_valid'] = ewl_residual < self.validation_thresholds['ewl_residual']
        
        # Validate WL
        wl_residual = self._compute_residual(
            phase_l1 - phase_l2,
            tcar_result.wl_ambiguities,
            (code_l1 - code_l2) / self.tcar.lambda_wl12
        )
        validation['wl_valid'] = wl_residual < self.validation_thresholds['wl_residual']
        
        # Overall validation
        validation['overall_valid'] = all([
            validation['ewl_valid'],
            validation['wl_valid'],
            tcar_result.success_flags['overall']
        ])
        
        return {
            'ambiguities': {
                'L1': tcar_result.l1_ambiguities,
                'L2': tcar_result.l2_ambiguities,
                'L5': tcar_result.l5_ambiguities
            },
            'combinations': {
                'EWL': tcar_result.ewl_ambiguities,
                'WL': tcar_result.wl_ambiguities,
                'NL': tcar_result.nl_ambiguities
            },
            'validation': validation,
            'statistics': tcar_result.statistics
        }
    
    def _compute_residual(self, phase: np.ndarray, fixed: np.ndarray,
                         code_derived: np.ndarray) -> float:
        """Compute residual for validation"""
        residual = phase + fixed - code_derived
        return np.std(residual)


def test_tcar():
    """Test TCAR resolution"""
    print("Testing TCAR (Three-Carrier Ambiguity Resolution)")
    print("=" * 60)
    
    # Generate test triple-frequency data
    np.random.seed(42)
    n = 8
    
    # True ambiguities
    true_l1 = np.random.randint(-20, 20, n)
    true_l2 = np.random.randint(-20, 20, n)
    true_l5 = np.random.randint(-20, 20, n)
    
    # Generate observations
    noise_phase = 0.01  # cycles
    noise_code = 0.3     # meters
    
    phase_l1 = true_l1 + np.random.normal(0, noise_phase, n)
    phase_l2 = true_l2 + np.random.normal(0, noise_phase, n)
    phase_l5 = true_l5 + np.random.normal(0, noise_phase, n)
    
    code_l1 = true_l1 * LAMBDA_L1 + np.random.normal(0, noise_code, n)
    code_l2 = true_l2 * LAMBDA_L2 + np.random.normal(0, noise_code, n)
    code_l5 = true_l5 * LAMBDA_L5 + np.random.normal(0, noise_code, n)
    
    # Resolve using TCAR
    resolver = TCARResolver()
    result = resolver.resolve(
        phase_l1, phase_l2, phase_l5,
        code_l1, code_l2, code_l5
    )
    
    print(f"EWL wavelength: {result.statistics['ewl_wavelength']:.2f} m")
    print(f"WL wavelength: {result.statistics['wl_wavelength']:.2f} m")
    print(f"")
    print(f"Success rates:")
    print(f"  EWL: {result.statistics['ewl_success_rate']:.1%}")
    print(f"  WL:  {result.statistics['wl_success_rate']:.1%}")
    print(f"  NL:  {result.statistics['nl_success_rate']:.1%}")
    print(f"")
    print(f"Ambiguity resolution results:")
    print(f"  True L1: {true_l1[:5]}")
    print(f"  Fixed L1: {result.l1_ambiguities[:5]}")
    print(f"  Errors: {np.abs(result.l1_ambiguities[:5] - true_l1[:5])}")
    
    # Test cascaded version with validation
    print("\nTesting Cascaded TCAR with Validation")
    print("-" * 60)
    
    cascaded = CascadedTCAR()
    observations = {
        'L1': phase_l1, 'L2': phase_l2, 'L5': phase_l5,
        'P1': code_l1, 'P2': code_l2, 'P5': code_l5
    }
    
    validated_result = cascaded.resolve_with_validation(observations)
    
    print(f"Validation status:")
    for key, value in validated_result['validation'].items():
        print(f"  {key}: {value}")
    
    print(f"\nFinal ambiguities:")
    print(f"  L1: {validated_result['ambiguities']['L1'][:5]}")
    print(f"  L2: {validated_result['ambiguities']['L2'][:5]}")
    print(f"  L5: {validated_result['ambiguities']['L5'][:5]}")


if __name__ == "__main__":
    test_tcar()