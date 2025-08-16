"""Coordinate transformation utilities for GNSS/INS processing"""

import numpy as np
from typing import Optional, Tuple
from .transforms import ecef2llh, llh2ecef, ecef2enu, enu2ecef
from .height_conversion import HeightSystem, convert_height


class CoordinateTransformer:
    """Handles coordinate transformations between ECEF, LLH, and local ENU frames
    
    This class encapsulates coordinate transformation logic and maintains
    reference positions for local coordinate systems.
    """
    
    def __init__(self, reference_ecef: Optional[np.ndarray] = None):
        """Initialize coordinate transformer
        
        Parameters:
        -----------
        reference_ecef : np.ndarray, optional
            Reference position in ECEF coordinates for local frame
        """
        self._reference_ecef: Optional[np.ndarray] = None
        self._reference_llh: Optional[np.ndarray] = None
        self._rotation_ecef_to_enu: Optional[np.ndarray] = None
        
        if reference_ecef is not None:
            self.set_reference(reference_ecef)
    
    def set_reference(self, reference_ecef: np.ndarray):
        """Set reference position for local coordinate frame
        
        Parameters:
        -----------
        reference_ecef : np.ndarray
            Reference position in ECEF coordinates [x, y, z] in meters
        """
        self._reference_ecef = reference_ecef.copy()
        self._reference_llh = ecef2llh(reference_ecef)
        self._compute_rotation_matrix()
    
    def _compute_rotation_matrix(self):
        """Compute rotation matrix from ECEF to ENU at reference position"""
        if self._reference_llh is None:
            return
            
        lat, lon = self._reference_llh[0], self._reference_llh[1]
        
        # Rotation matrix from ECEF to ENU
        sin_lat, cos_lat = np.sin(lat), np.cos(lat)
        sin_lon, cos_lon = np.sin(lon), np.cos(lon)
        
        self._rotation_ecef_to_enu = np.array([
            [-sin_lon, cos_lon, 0],
            [-sin_lat * cos_lon, -sin_lat * sin_lon, cos_lat],
            [cos_lat * cos_lon, cos_lat * sin_lon, sin_lat]
        ])
    
    @property
    def reference_ecef(self) -> Optional[np.ndarray]:
        """Get reference position in ECEF coordinates"""
        return self._reference_ecef.copy() if self._reference_ecef is not None else None
    
    @property
    def reference_llh(self) -> Optional[np.ndarray]:
        """Get reference position in LLH coordinates"""
        return self._reference_llh.copy() if self._reference_llh is not None else None
    
    @property
    def has_reference(self) -> bool:
        """Check if reference position is set"""
        return self._reference_ecef is not None
    
    def ecef_to_llh(self, position_ecef: np.ndarray) -> np.ndarray:
        """Convert ECEF position to geodetic coordinates
        
        Parameters:
        -----------
        position_ecef : np.ndarray
            Position in ECEF coordinates [x, y, z] in meters
            
        Returns:
        --------
        np.ndarray
            Position in geodetic coordinates [lat, lon, height]
            lat, lon in radians, height in meters
        """
        return ecef2llh(position_ecef)
    
    def llh_to_ecef(self, position_llh: np.ndarray) -> np.ndarray:
        """Convert geodetic coordinates to ECEF
        
        Parameters:
        -----------
        position_llh : np.ndarray
            Position in geodetic coordinates [lat, lon, height]
            lat, lon in radians, height in meters
            
        Returns:
        --------
        np.ndarray
            Position in ECEF coordinates [x, y, z] in meters
        """
        return llh2ecef(position_llh)
    
    def ecef_to_enu(self, position_ecef: np.ndarray) -> np.ndarray:
        """Convert ECEF position to local ENU coordinates
        
        Parameters:
        -----------
        position_ecef : np.ndarray
            Position in ECEF coordinates [x, y, z] in meters
            
        Returns:
        --------
        np.ndarray
            Position in local ENU coordinates [e, n, u] in meters
            
        Raises:
        -------
        ValueError
            If reference position is not set
        """
        if not self.has_reference:
            raise ValueError("Reference position not set")
        return ecef2enu(position_ecef, self._reference_llh)
    
    def enu_to_ecef(self, position_enu: np.ndarray) -> np.ndarray:
        """Convert local ENU coordinates to ECEF
        
        Parameters:
        -----------
        position_enu : np.ndarray
            Position in local ENU coordinates [e, n, u] in meters
            
        Returns:
        --------
        np.ndarray
            Position in ECEF coordinates [x, y, z] in meters
            
        Raises:
        -------
        ValueError
            If reference position is not set
        """
        if not self.has_reference:
            raise ValueError("Reference position not set")
        return enu2ecef(position_enu, self._reference_llh)
    
    def ecef_vector_to_enu(self, vector_ecef: np.ndarray) -> np.ndarray:
        """Convert vector from ECEF to ENU frame
        
        Parameters:
        -----------
        vector_ecef : np.ndarray
            Vector in ECEF frame (e.g., velocity, acceleration)
            
        Returns:
        --------
        np.ndarray
            Vector in ENU frame
            
        Raises:
        -------
        ValueError
            If reference position is not set
        """
        if not self.has_reference:
            raise ValueError("Reference position not set")
        return self._rotation_ecef_to_enu @ vector_ecef
    
    def enu_vector_to_ecef(self, vector_enu: np.ndarray) -> np.ndarray:
        """Convert vector from ENU to ECEF frame
        
        Parameters:
        -----------
        vector_enu : np.ndarray
            Vector in ENU frame (e.g., velocity, acceleration)
            
        Returns:
        --------
        np.ndarray
            Vector in ECEF frame
            
        Raises:
        -------
        ValueError
            If reference position is not set
        """
        if not self.has_reference:
            raise ValueError("Reference position not set")
        return self._rotation_ecef_to_enu.T @ vector_enu
    
    def convert_height(self, height: float, from_system: HeightSystem, 
                      to_system: HeightSystem, lat: float, lon: float) -> float:
        """Convert between different height systems
        
        Parameters:
        -----------
        height : float
            Height value to convert
        from_system : HeightSystem
            Source height system
        to_system : HeightSystem
            Target height system
        lat : float
            Latitude in radians
        lon : float
            Longitude in radians
            
        Returns:
        --------
        float
            Converted height value
        """
        return convert_height(height, from_system, to_system, lat, lon)
    
    def compute_baseline(self, pos1_ecef: np.ndarray, pos2_ecef: np.ndarray) -> Tuple[np.ndarray, float]:
        """Compute baseline vector and distance between two ECEF positions
        
        Parameters:
        -----------
        pos1_ecef : np.ndarray
            First position in ECEF coordinates
        pos2_ecef : np.ndarray
            Second position in ECEF coordinates
            
        Returns:
        --------
        tuple
            - Baseline vector in ENU coordinates (if reference set) or ECEF
            - Baseline distance in meters
        """
        baseline_ecef = pos2_ecef - pos1_ecef
        distance = np.linalg.norm(baseline_ecef)
        
        if self.has_reference:
            baseline_enu = self.ecef_vector_to_enu(baseline_ecef)
            return baseline_enu, distance
        else:
            return baseline_ecef, distance