"""Clock Between Factor for GNSS clock continuity"""

import gtsam
import numpy as np


class ClockBetweenFactor:
    """
    Factory for creating clock between factors
    """
    
    @staticmethod
    def create(key_prev, key_curr, dt=1.0, clock_drift_noise=1.0, isb_drift_noise=0.1):
        """
        Create a clock between factor
        
        Parameters:
        -----------
        key_prev : gtsam.Symbol
            Previous epoch clock state key
        key_curr : gtsam.Symbol
            Current epoch clock state key
        dt : float
            Time interval between epochs (seconds)
        clock_drift_noise : float
            Expected GPS clock drift noise (m/s)
        isb_drift_noise : float
            Expected ISB drift noise (m/s), should be smaller than clock drift
            
        Returns:
        --------
        gtsam.CustomFactor
            Clock between factor
        """
        # Create error function with dt captured in closure
        def error_func(this: gtsam.CustomFactor, values: gtsam.Values, H: list[np.ndarray]):
            """
            Error function for clock between factor
            """
            key1 = this.keys()[0]  # Previous clock state
            key2 = this.keys()[1]  # Current clock state
            
            clock_prev = values.atVector(key1)
            clock_curr = values.atVector(key2)
            
            # Clock difference
            clock_diff = clock_curr - clock_prev
            
            # Expected drift is zero (clocks should be continuous)
            expected_drift = np.zeros_like(clock_diff)
            
            # Error is the difference from expected drift, scaled by time
            error = (clock_diff - expected_drift) / dt
            
            # Jacobians
            if H is not None:
                # Derivative w.r.t. previous clock state
                H[0] = -np.eye(len(clock_prev)) / dt
                # Derivative w.r.t. current clock state  
                H[1] = np.eye(len(clock_curr)) / dt
            
            return error
        
        # Create noise model
        # GPS clock can drift more than ISBs
        sigmas = np.array([
            clock_drift_noise,  # GPS clock drift
            isb_drift_noise,    # GLO ISB drift  
            isb_drift_noise,    # GAL ISB drift
            isb_drift_noise     # BDS ISB drift
        ])
        
        # Scale by time interval
        sigmas = sigmas * dt
        
        noise_model = gtsam.noiseModel.Diagonal.Sigmas(sigmas)
        
        # Create custom factor
        factor = gtsam.CustomFactor(
            noise_model,
            [key_prev, key_curr],
            error_func
        )
        
        return factor


class ClockPriorFactor:
    """
    Factory for creating clock prior factors based on SPP solution
    """
    
    @staticmethod
    def create_from_spp(key, spp_clocks, noise_sigmas=None):
        """
        Create a prior factor for clock states from SPP solution
        
        Parameters:
        -----------
        key : gtsam.Symbol
            Clock state key
        spp_clocks : np.ndarray
            Clock states from SPP solution [gps_clock, glo_isb, gal_isb, bds_isb]
        noise_sigmas : np.ndarray, optional
            Noise standard deviations for each clock component
            If None, uses default loose values
            
        Returns:
        --------
        gtsam.PriorFactorVector
            Clock prior factor
        """
        if noise_sigmas is None:
            # Loose prior - SPP can have ~10-100m error in clock
            noise_sigmas = np.array([
                100.0,  # GPS clock (m) - loose prior
                50.0,   # GLO ISB (m)
                50.0,   # GAL ISB (m)  
                50.0    # BDS ISB (m)
            ])
        
        noise_model = gtsam.noiseModel.Diagonal.Sigmas(noise_sigmas)
        
        # Create prior factor
        factor = gtsam.PriorFactorVector(key, spp_clocks, noise_model)
        
        return factor