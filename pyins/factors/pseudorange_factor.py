# Copyright 2024 inuex35
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Pseudorange Factor for GTSAM - Single Point Positioning"""

import numpy as np
import gtsam
from typing import List, Optional
from pyins.coordinate.dcm import enu2ecef_dcm
from pyins.coordinate.transforms import enu2ecef, ecef2llh
from pyins.core.constants import CLIGHT, sat2prn
from pyins.gnss.spp import tropmodel_simple as robust_troposphere_model
from pyins.gnss.spp import satazel


class PseudorangeFactor:
    """Pseudorange Factor for single point positioning
    
    Works in ENU coordinates and handles multi-constellation GNSS
    """
    
    @staticmethod
    def create(obs, measured_pr, sat_pos, sat_clk_bias, sys, noise_model, reference_ecef, 
               pose_key=None, clock_key=None, lever_arm=None):
        """Create a pseudorange factor
        
        Parameters:
        -----------
        obs : not used (for compatibility)
        measured_pr : float
            Measured pseudorange (meters)
        sat_pos : np.ndarray
            Satellite position in ECEF (meters)
        sat_clk_bias : float
            Satellite clock bias (meters)
        sys : int
            Satellite system (SYS_GPS, SYS_GAL, etc.)
        noise_model : gtsam.noiseModel
            Noise model for the measurement
        reference_ecef : np.ndarray
            Reference ECEF position for ENU conversion
        pose_key : gtsam.Symbol
            Key for the pose variable (if using pose)
        clock_key : gtsam.Symbol
            Key for the clock variable
        lever_arm : np.ndarray, optional
            Lever arm from IMU to GNSS antenna in body frame
            
        Returns:
        --------
        gtsam.CustomFactor
            The pseudorange factor
        """
        from pyins.core.constants import SYS_GPS, SYS_GLO, SYS_GAL, SYS_BDS
        from pyins.coordinate.transforms import ecef2llh
        
        # Get reference LLH for coordinate transformations
        reference_llh = ecef2llh(reference_ecef)
        
        # Determine clock index based on system
        clock_idx = 0  # GPS clock
        if sys == SYS_GLO:
            clock_idx = 1  # GLONASS ISB
        elif sys == SYS_GAL:
            clock_idx = 2  # Galileo ISB
        elif sys == SYS_BDS:
            clock_idx = 3  # BeiDou ISB
        
        # Pre-compute rotation matrix for Jacobian calculation only
        R_enu2ecef = enu2ecef_dcm(reference_llh)
        R_ecef2enu = R_enu2ecef.T
        
        # Convert satellite position to ENU for debugging
        from pyins.coordinate.transforms import ecef2enu
        sat_enu = ecef2enu(sat_pos, reference_llh)
        
        # Create error function
        def error_func(this, values, H=None):
            """Compute residual for pseudorange measurement
            
            The measurement equation is:
            PR = |r_sat - r_rcv| + c*dt_rcv - c*dt_sat + noise
            
            Where:
            - r_sat: satellite position (ECEF)
            - r_rcv: receiver position (ECEF)
            - dt_rcv: receiver clock bias (seconds, converted to meters)
            - dt_sat: satellite clock bias (meters)
            """
            try:
                # Get rover position
                if pose_key is not None:
                    # Determine if we have Pose3 or Vector by checking the type
                    rover_enu = None
                    is_pose3 = False
                    
                    # Try to get as Pose3 first (for INS integration)
                    try:
                        pose = values.atPose3(pose_key)
                        rover_enu = pose.translation()
                        is_pose3 = True

                        # Apply lever arm if provided
                        if lever_arm is not None and np.linalg.norm(lever_arm) > 0:
                            # Transform lever arm from body to navigation frame
                            R_body2nav = pose.rotation().matrix()
                            lever_arm_nav = R_body2nav @ lever_arm
                            rover_enu = rover_enu + lever_arm_nav
                    except Exception as e:
                        # If not a Pose3, try as Vector (position only)
                        try:
                            rover_enu = values.atVector(pose_key)
                            if len(rover_enu) != 3:
                                raise ValueError(f"Position vector must be 3D, got {len(rover_enu)}")
                            is_pose3 = False
                        except Exception as e2:
                            import logging
                            logging.getLogger(__name__).error(f"Failed to get position: Pose3 error: {e}, Vector error: {e2}")
                            raise ValueError(f"Could not retrieve position from values with key {pose_key}")
                else:
                    # Position-only factor (should not happen in current usage)
                    # But if it does, we need a position key instead
                    raise ValueError("Position-only mode requires a position key, not clock key")
                
                # Get clock states
                clocks = values.atVector(clock_key)
                receiver_clock_bias = clocks[clock_idx]  # In meters
                
                # Convert rover position from ENU to ECEF
                from pyins.coordinate.transforms import enu2ecef
                rover_ecef = enu2ecef(rover_enu, reference_llh)
                
                # Debug: Check if positions are reasonable
                if np.linalg.norm(rover_enu) > 10000:
                    import logging
                    logger = logging.getLogger(__name__)
                    logger.debug(f"Large ENU position: {rover_enu}")
                    logger.debug(f"Converted ECEF: {rover_ecef}")
                    logger.debug(f"Reference ECEF: {reference_ecef}")
                
                # Compute geometric range
                delta = sat_pos - rover_ecef
                computed_range = np.linalg.norm(delta)
                
                # Sanity check
                # Allow up to 45,000 km for BeiDou GEO/IGSO satellites
                if computed_range < 1e6 or computed_range > 45e6:
                    # Unrealistic range, return large error
                    import logging
                    logger = logging.getLogger(__name__)
                    logger.debug(f"Unrealistic range: {computed_range/1e3:.1f} km")
                    logger.debug(f"Sat pos: {sat_pos}")
                    logger.debug(f"Rover ECEF: {rover_ecef}")
                    logger.debug(f"Rover ENU: {rover_enu}")
                    logger.debug(f"Reference ECEF: {reference_ecef}")
                    return np.array([1e6])
                
                # Calculate elevation angle for troposphere model
                llh = ecef2llh(rover_ecef)
                el = elevation_angle(rover_ecef, sat_pos, llh)
                
                # Troposphere delay (following SPP implementation)
                trop = robust_troposphere_model(rover_ecef, el)
                
                # Compute expected pseudorange (following SPP exactly)
                # PR = range + receiver_clock - satellite_clock + troposphere
                # Note: sat_clk_bias is already in meters (converted in GNSSProcessor)
                expected_pr = computed_range + receiver_clock_bias - sat_clk_bias + trop
                
                # Residual (measurement - prediction)
                residual = measured_pr - expected_pr
                
                # Compute Jacobian if requested
                if H is not None:
                    # Unit vector from receiver to satellite in ECEF
                    e_ecef = delta / computed_range
                    
                    # Convert to ENU for position Jacobian
                    e_enu = R_ecef2enu @ e_ecef
                    
                    if pose_key is not None:
                        # Use the is_pose3 flag we determined earlier
                        if is_pose3:
                            # Jacobian w.r.t. pose (1x6: rotation, position)
                            H_pose = np.zeros((1, 6))
                            # Position part: derivative of range w.r.t. position
                            H_pose[0, 3:6] = -e_enu  # Negative because range decreases as we move toward satellite
                            
                            # Rotation part: only affects if lever arm is present
                            if lever_arm is not None and np.linalg.norm(lever_arm) > 0:
                                # The rotation affects the lever arm transformation
                                pose = values.atPose3(pose_key)
                                R_body2nav = pose.rotation().matrix()
                                lever_arm_nav = R_body2nav @ lever_arm
                                # Cross product gives the rotation sensitivity
                                lever_cross = np.cross(lever_arm_nav, e_enu)
                                H_pose[0, 0:3] = -lever_cross
                            
                            H[0] = H_pose
                        else:
                            # Vector case (position only)
                            H_position = np.zeros((1, 3))
                            H_position[0, :] = -e_enu  # Negative because range decreases as we move toward satellite
                            H[0] = H_position
                    
                    # Jacobian w.r.t. clock states (1x4 for multi-GNSS)
                    H_clock = np.zeros((1, 4))
                    H_clock[0, clock_idx] = 1.0  # Derivative of PR w.r.t. clock bias
                    H[1] = H_clock
                
                return np.array([residual])
                
            except Exception as e:
                # If any error occurs, return a large residual to avoid crashing
                import logging
                logging.getLogger(__name__).error(f"Error in pseudorange factor: {e}")
                return np.array([1e6])
        
        # Create factor with appropriate keys
        if pose_key is not None and clock_key is not None:
            keys = [pose_key, clock_key]
        else:
            raise ValueError("Both pose_key and clock_key must be provided")
        
        factor = gtsam.CustomFactor(
            noise_model,
            keys,
            error_func
        )
        
        return factor


class PseudorangePoseFactor(gtsam.CustomFactor):
    """
    Custom factor for GNSS pseudorange measurements with Pose3 states.
    
    This factor relates a Pose3 state and clock bias to a pseudorange measurement.
    The position is extracted from the Pose3 for range computation.
    """
    
    def __init__(self, pose_key: int, clock_key: int, measured_range: float,
                 sat_pos: np.ndarray, sat_clock: float, noise_model):
        """
        Initialize pseudorange factor for Pose3.
        
        Parameters:
        -----------
        pose_key : int
            Key for the Pose3 state
        clock_key : int
            Key for the receiver clock bias
        measured_range : float
            Measured pseudorange (m)
        sat_pos : np.ndarray
            Satellite position in ECEF (m)
        sat_clock : float
            Satellite clock bias (m)
        noise_model : gtsam.noiseModel
            Noise model for the measurement
        """
        # Store measurement data
        self.measured_range = measured_range
        self.sat_pos = sat_pos
        self.sat_clock = sat_clock
        
        # Initialize CustomFactor with keys and noise model
        super().__init__(noise_model, [pose_key, clock_key], self.error_func)
        
    def error_func(self, this: gtsam.CustomFactor, 
                   values: gtsam.Values, 
                   jacobians: Optional[List[np.ndarray]]) -> np.ndarray:
        """
        Compute error and Jacobians for pseudorange measurement.
        
        Parameters:
        -----------
        this : gtsam.CustomFactor
            Reference to this factor
        values : gtsam.Values
            Current values
        jacobians : List[np.ndarray], optional
            Output Jacobians
            
        Returns:
        --------
        error : np.ndarray
            Error vector (1x1)
        """
        # Get keys
        pose_key = this.keys()[0]
        clock_key = this.keys()[1]
        
        # Extract pose and then position
        pose = values.atPose3(pose_key)
        receiver_pos = pose.translation()  # Extract position from Pose3
        
        # Extract clock bias
        clock_bias = values.atDouble(clock_key)
        
        # Compute geometric range
        delta = self.sat_pos - receiver_pos
        geometric_range = np.linalg.norm(delta)
        
        # Predicted pseudorange = geometric range + receiver clock - satellite clock
        predicted_range = geometric_range + clock_bias - self.sat_clock
        
        # Error = measured - predicted
        error = np.array([self.measured_range - predicted_range])
        
        # Compute Jacobians if requested
        if jacobians is not None:
            # Jacobian w.r.t. Pose3 (1x6: rotation and position)
            # Derivative of range w.r.t. position
            if geometric_range > 0:
                drange_dpos = -delta / geometric_range  # 1x3
            else:
                drange_dpos = np.zeros(3)
            
            # Full Jacobian for Pose3: [rotation(3) | position(3)]
            # Range doesn't depend on rotation, so first 3 columns are zero
            J_pose = np.zeros((1, 6))
            J_pose[0, 3:6] = -drange_dpos  # Negative because error = measured - predicted
            
            # Jacobian w.r.t. clock bias
            J_clock = np.array([[-1.0]])  # Negative because error = measured - predicted
            
            jacobians[0] = J_pose
            jacobians[1] = J_clock
            
        return error


class PseudorangeWithISBPoseFactor(gtsam.CustomFactor):
    """
    Pseudorange factor with inter-system bias support for Pose3 states.
    
    Handles different GNSS systems (GPS, GLONASS, Galileo, BeiDou) with ISBs.
    """
    
    def __init__(self, pose_key: int, clock_key: int, measured_range: float,
                 sat_pos: np.ndarray, sat_clock: float, system: str, sat_num: int, noise_model):
        """
        Initialize pseudorange factor with ISB support.
        
        Parameters:
        -----------
        pose_key : int
            Key for the Pose3 state
        clock_key : int
            Key for the clock state vector [GPS_clock, GLO_ISB, GAL_ISB, BDS_ISB]
        measured_range : float
            Measured pseudorange (m)
        sat_pos : np.ndarray
            Satellite position in ECEF (m)
        sat_clock : float
            Satellite clock bias (m)
        system : str
            GNSS system ('G', 'R', 'E', 'C')
        sat_num : int
            Satellite number (internal pyins numbering)
        noise_model : gtsam.noiseModel
            Noise model for the measurement
        """
        self.measured_range = measured_range
        self.sat_pos = sat_pos
        self.sat_clock = sat_clock
        self.system = system
        self.sat_num = sat_num
        
        # System index for ISB
        self.system_idx = {'G': 0, 'R': 1, 'E': 2, 'C': 3}.get(system, 0)
        
        # Get PRN from satellite number using pyins function
        self.prn = sat2prn(sat_num)
        
        super().__init__(noise_model, [pose_key, clock_key], self.error_func)
        
    def error_func(self, this: gtsam.CustomFactor,
                   values: gtsam.Values,
                   jacobians: Optional[List[np.ndarray]]) -> np.ndarray:
        """
        Compute error and Jacobians for pseudorange with ISB.
        """
        # Get keys
        pose_key = this.keys()[0]
        clock_key = this.keys()[1]
        
        # Extract pose and position
        pose = values.atPose3(pose_key)
        receiver_pos = pose.translation()
        
        # Extract clock state vector [GPS_clock, GLO_ISB, GAL_ISB, BDS_ISB]
        clock_state = values.atVector(clock_key)
        
        # Get appropriate clock bias based on system
        if self.system == 'G':
            clock_bias = clock_state[0]  # GPS clock
        else:
            # Other systems: GPS clock + ISB
            clock_bias = clock_state[0] + clock_state[self.system_idx]
        
        # Compute geometric range
        delta = self.sat_pos - receiver_pos
        geometric_range = np.linalg.norm(delta)
        
        # Predicted pseudorange
        predicted_range = geometric_range + clock_bias - self.sat_clock
        
        # Error
        error = np.array([self.measured_range - predicted_range])
        
        # Compute Jacobians
        if jacobians is not None:
            # Jacobian w.r.t. Pose3
            if geometric_range > 0:
                drange_dpos = -delta / geometric_range
            else:
                drange_dpos = np.zeros(3)
            
            J_pose = np.zeros((1, 6))
            J_pose[0, 3:6] = -drange_dpos
            
            # Jacobian w.r.t. clock state
            J_clock = np.zeros((1, 4))  # [GPS, GLO, GAL, BDS]
            J_clock[0, 0] = -1.0  # GPS clock always contributes
            if self.system != 'G':
                J_clock[0, self.system_idx] = -1.0  # ISB for non-GPS
            
            jacobians[0] = J_pose
            jacobians[1] = J_clock
            
        return error
    
    def get_debug_info(self):
        """Get debug information for this factor."""
        return {
            'system': self.system,  # Already a single character: G/R/E/C/J
            'prn': self.prn,
            'sat_num': self.sat_num
        }