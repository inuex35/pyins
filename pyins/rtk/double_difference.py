"""Double differenced GNSS measurement formation with frequency-specific reference satellites.

This module forms double-differenced (DD) pseudorange and carrier phase measurements
between rover and base receivers. Double differencing eliminates receiver clock
biases and satellite clock biases, reducing the number of unknowns in RTK positioning.

For carrier phase measurements, the integer ambiguity remains after double differencing,
but atmospheric delays are largely canceled for short baselines (<10km).

The module includes:
- Satellite selection and elevation computation
- Reference satellite selection per frequency (like RTKLIB)
- Single difference (SD) formation between receivers
- Double difference (DD) formation between satellites
- Carrier phase and pseudorange DD measurements
- Support for multi-frequency and multi-GNSS
"""

import numpy as np
from collections import defaultdict
import logging
from pyins.core.constants import CLIGHT, SYS_GPS, SYS_GLO, SYS_GAL, SYS_BDS, SYS_QZS
from pyins.core.constants import sat2sys
from pyins.gnss.ephemeris import satpos

logger = logging.getLogger(__name__)


def form_double_differences(rover_obs, base_obs, nav_data, gps_time,
                           reference_ecef=None, reference_llh=None,
                           rover_position=None,
                           use_systems=None, use_frequencies=None,
                           cutoff_angle=10.0,
                           base_obs_list=None, base_obs_index=None):
    """Form double-differenced GNSS measurements with frequency-specific reference satellites.

    Forms double-differenced pseudorange and carrier phase measurements
    for RTK positioning. Each frequency uses its own reference satellite
    selection, similar to RTKLIB implementation.

    Parameters
    ----------
    rover_obs : list[Observation]
        Rover receiver observations for the current epoch
    base_obs : list[Observation] or None
        Base receiver observations for the current epoch.
        If None, will use base_obs_list and base_obs_index or
        interpolate from base_obs_list
    nav_data : NavigationData
        Navigation data containing ephemerides
    gps_time : float
        GPS time of rover observations in seconds
    reference_ecef : np.ndarray, optional
        Reference (base) station position in ECEF (meters), shape (3,).
        Required if base_obs_list is provided for interpolation
    reference_llh : np.ndarray, optional
        Reference (base) station position in lat/lon/height (degrees, degrees, meters).
        Used for elevation angle calculation if provided
    rover_position : np.ndarray, optional
        Rover position in ECEF (meters), shape (3,).
        If provided, used for more accurate elevation angles
    use_systems : list[str], optional
        List of GNSS systems to use ('G', 'R', 'E', 'C', 'J').
        Default: ['G', 'E', 'R', 'C', 'J']
    use_frequencies : list[int], optional
        List of frequency indices to use (0=L1, 1=L2, etc.).
        Default: all available frequencies
    cutoff_angle : float, optional
        Elevation cutoff angle in degrees. Default: 10.0
    base_obs_list : list[dict], optional
        List of base observation epochs for interpolation.
        Each dict should have 'time' and 'observations' keys
    base_obs_index : int, optional
        Index into base_obs_list for current epoch.
        If provided, avoids searching through the list

    Returns
    -------
    list[dict]
        List of double-differenced measurements. Each dict contains:
        - 'sat': satellite number (non-reference satellite)
        - 'ref_sat': reference satellite number (per frequency)
        - 'sys': system identifier
        - 'freq_idx': frequency index (0=L1, 1=L2, etc.)
        - 'dd_obs': double-differenced pseudorange (meters)
        - 'dd_carrier': double-differenced carrier phase (cycles) or None
        - 'wavelength': carrier wavelength (meters) or None
        - 'elevation': satellite elevation angle (degrees)

    Notes
    -----
    The key improvement in this implementation is frequency-specific reference
    satellite selection. For each frequency:
    1. Find all satellites with valid observations at that frequency
    2. Select the highest elevation satellite as reference for that frequency
    3. Form DD measurements using frequency-specific reference

    This approach maximizes the number of DD measurements, especially for
    GLONASS where not all satellites transmit on all frequencies.
    """

    # Default systems
    if use_systems is None:
        use_systems = ['G', 'E', 'R', 'C', 'J']

    # Extract actual time from observations if available
    rover_time = gps_time
    base_time = gps_time

    # Check if observations have time information
    if rover_obs and hasattr(rover_obs[0], 'time'):
        rover_time = rover_obs[0].time
    elif rover_obs and isinstance(rover_obs, dict):
        first_obs = next(iter(rover_obs.values()), None)
        if first_obs and hasattr(first_obs, 'time'):
            rover_time = first_obs.time

    # Convert observations to dict if needed
    if isinstance(rover_obs, list):
        rover_obs_dict = {obs.sat: obs for obs in rover_obs}
    else:
        rover_obs_dict = rover_obs

    # Handle base observations
    if base_obs is not None:
        # Direct base observations provided
        if isinstance(base_obs, list):
            base_obs_dict = {obs.sat: obs for obs in base_obs}
        else:
            base_obs_dict = base_obs

        # Extract base time if available
        if isinstance(base_obs, list) and base_obs and hasattr(base_obs[0], 'time'):
            base_time = base_obs[0].time
    elif base_obs_list is not None and reference_ecef is not None:
        # Find matching base observation from list
        from ..gnss.interp import interpolate_base_epoch

        # Find the closest base observations for interpolation
        base_before = None
        base_after = None
        base_exact = None

        for i, base_epoch in enumerate(base_obs_list):
            base_epoch_time = base_epoch.get('gps_time', base_epoch.get('time', 0))

            if abs(base_epoch_time - rover_time) < 0.01:  # Within 10ms
                base_exact = base_epoch['observations']
                base_time = base_epoch_time
                break
            elif base_epoch_time < rover_time:
                base_before = (i, base_epoch, base_epoch_time)
            elif base_epoch_time > rover_time and base_after is None:
                base_after = (i, base_epoch, base_epoch_time)
                break

        if base_exact is not None:
            # Use exact match
            if isinstance(base_exact, list):
                base_obs_dict = {obs.sat: obs for obs in base_exact}
            else:
                base_obs_dict = base_exact
        elif base_before and base_after:
            # Interpolate between base observations
            idx_before, epoch_before, time_before = base_before
            idx_after, epoch_after, time_after = base_after

            # Simple linear interpolation for now
            base_interp = interpolate_base_epoch(
                epoch_before, epoch_after,
                time_before, time_after,
                rover_time
            )

            if isinstance(base_interp['observations'], list):
                base_obs_dict = {obs.sat: obs for obs in base_interp['observations']}
            else:
                base_obs_dict = base_interp['observations']
            base_time = rover_time
        else:
            logger.warning(f"Failed to find suitable base observations for time {rover_time}")
            return []
    else:
        logger.warning("No base observations provided")
        return []

    # Find common satellites
    common_sats = set(rover_obs_dict.keys()) & set(base_obs_dict.keys())
    if len(common_sats) < 2:
        return []

    # Group satellites by system
    system_sats = defaultdict(list)
    sys_map = {SYS_GPS: 'G', SYS_GLO: 'R', SYS_GAL: 'E', SYS_BDS: 'C', SYS_QZS: 'J'}

    for sat in common_sats:
        sys_id = sat2sys(sat)
        sys_char = sys_map.get(sys_id, 'U')
        if sys_char in use_systems:
            system_sats[sys_char].append(sat)

    # Compute satellite positions separately for rover and base
    # This is critical for accurate DD when observations are at different times
    rover_obs_list = list(rover_obs_dict.values())
    base_obs_list = list(base_obs_dict.values())

    # Update observation times to ensure correct satellite position/clock computation
    # This is necessary because observations may have incorrect time attributes
    for obs in rover_obs_list:
        obs.time = rover_time
    for obs in base_obs_list:
        obs.time = base_time

    # Debug: Check observation times
    if rover_obs_list and base_obs_list:
        rover_time = rover_obs_list[0].time if hasattr(rover_obs_list[0], 'time') else 0
        base_time = base_obs_list[0].time if hasattr(base_obs_list[0], 'time') else 0
        import logging
        logger = logging.getLogger(__name__)
        logger.debug(f"Rover obs time: {rover_time}, Base obs time: {base_time}, diff: {rover_time - base_time}")

    # Compute satellite positions/clocks for rover time
    rover_sat_positions, rover_sat_clocks, _, _ = satpos(rover_obs_list, nav_data)

    # Compute satellite positions/clocks for base time
    base_sat_positions, base_sat_clocks, _, _ = satpos(base_obs_list, nav_data)

    # Debug: Check if clocks are different
    if len(rover_sat_clocks) > 0 and len(base_sat_clocks) > 0:
        logger.debug(f"First rover sat clock: {rover_sat_clocks[0]*1e6:.3f} μs, base sat clock: {base_sat_clocks[0]*1e6:.3f} μs")

    # Create mapping from satellite to index for quick lookup
    rover_sat_index = {obs.sat: i for i, obs in enumerate(rover_obs_list)}
    base_sat_index = {obs.sat: i for i, obs in enumerate(base_obs_list)}

    dd_measurements = []

    # Process each system separately
    for system, sats in system_sats.items():
        if len(sats) < 2:
            continue

        # Prepare satellite data with positions and elevations
        sat_data = {}
        for sat in sats:
            # Get rover satellite position/clock
            if sat not in rover_sat_index:
                continue
            rover_idx = rover_sat_index[sat]
            rover_sat_pos = rover_sat_positions[rover_idx]
            rover_sat_clk = rover_sat_clocks[rover_idx]

            # Get base satellite position/clock
            if sat not in base_sat_index:
                continue
            base_idx = base_sat_index[sat]
            base_sat_pos = base_sat_positions[base_idx]
            base_sat_clk = base_sat_clocks[base_idx]

            # Skip if no valid position
            if np.all(rover_sat_pos == 0) or np.all(base_sat_pos == 0):
                continue

            # Calculate elevation angle (using rover position for consistency)
            if reference_llh is not None:
                # Use base position for elevation calculation
                from ..geometry.elevation import compute_elevation_angle
                elevation = compute_elevation_angle(rover_sat_pos, reference_ecef, reference_llh)
            else:
                # Approximate elevation from position
                elevation = 45.0  # Default if no reference position

            if elevation < cutoff_angle:
                continue

            sat_data[sat] = {
                'pos': rover_sat_pos,  # Rover satellite position
                'clk': rover_sat_clk,  # Rover satellite clock
                'base_pos': base_sat_pos,  # Base satellite position
                'base_clk': base_sat_clk,  # Base satellite clock
                'elevation': elevation,
                'rover_obs': rover_obs_dict[sat],
                'base_obs': base_obs_dict[sat]
            }

        if len(sat_data) < 2:
            continue

        # Determine maximum number of frequencies for this system
        max_system_freqs = 0
        for sat in sat_data:
            obs = sat_data[sat]['rover_obs']
            max_system_freqs = max(max_system_freqs, len(obs.P))

        # Process each frequency separately with its own reference satellite
        for freq_idx in range(max_system_freqs):
            # Skip if this frequency is not requested
            if use_frequencies is not None and freq_idx not in use_frequencies:
                continue

            # Find satellites that have valid observations for this frequency
            valid_sats_freq = []
            for sat in sat_data:
                rover_obs_sat = sat_data[sat]['rover_obs']
                base_obs_sat = sat_data[sat]['base_obs']

                # Check if this satellite has valid observations for this frequency
                if (len(rover_obs_sat.P) > freq_idx and rover_obs_sat.P[freq_idx] != 0 and
                    len(base_obs_sat.P) > freq_idx and base_obs_sat.P[freq_idx] != 0):
                    valid_sats_freq.append(sat)

            if len(valid_sats_freq) < 2:
                continue

            # Select reference satellite for this frequency
            # Choose the one with highest elevation among valid satellites
            ref_sat = max(valid_sats_freq, key=lambda s: sat_data[s]['elevation'])
            ref_data = sat_data[ref_sat]
            ref_rover_obs = ref_data['rover_obs']
            ref_base_obs = ref_data['base_obs']

            # Form double differences for each satellite vs reference
            for sat in valid_sats_freq:
                if sat == ref_sat:
                    continue

                data = sat_data[sat]
                rover_obs_sat = data['rover_obs']
                base_obs_sat = data['base_obs']

                # Extract pseudoranges
                rover_pr = rover_obs_sat.P[freq_idx]
                base_pr = base_obs_sat.P[freq_idx]
                ref_rover_pr = ref_rover_obs.P[freq_idx]
                ref_base_pr = ref_base_obs.P[freq_idx]

                # Form single differences from RAW observations
                # DO NOT apply satellite clock corrections to observations
                # Clock corrections are only applied to computed geometric ranges
                sd_rover = rover_pr - ref_rover_pr
                sd_base = base_pr - ref_base_pr

                # Form double difference
                dd_obs = sd_rover - sd_base

                # Debug: Print first DD for verification
                if len(dd_measurements) == 0 and freq_idx == 0:
                    import logging
                    logger = logging.getLogger(__name__)
                    logger.debug(f"First DD: G{ref_sat}-G{sat}")
                    logger.debug(f"  Rover: {rover_pr:.3f} - {ref_rover_pr:.3f} = {sd_rover:.3f}")
                    logger.debug(f"  Base:  {base_pr:.3f} - {ref_base_pr:.3f} = {sd_base:.3f}")
                    logger.debug(f"  DD: {dd_obs:.3f}")

                # Also extract carrier phase if available
                dd_carrier = None
                wavelength = None
                rover_cp_cycles = None
                base_cp_cycles = None
                ref_rover_cp_cycles = None
                ref_base_cp_cycles = None

                # Check carrier phase availability (L attribute)
                if (hasattr(rover_obs_sat, 'L') and hasattr(base_obs_sat, 'L') and
                    hasattr(ref_rover_obs, 'L') and hasattr(ref_base_obs, 'L')):

                    # Check if all have enough frequencies
                    if (len(rover_obs_sat.L) > freq_idx and len(base_obs_sat.L) > freq_idx and
                        len(ref_rover_obs.L) > freq_idx and len(ref_base_obs.L) > freq_idx):

                        # Get frequency first for conversion
                        from ..core.constants import FREQ_L1, FREQ_L2, FREQ_L5
                        from ..core.constants import FREQ_E1, FREQ_E5a, FREQ_E5b
                        from ..core.constants import FREQ_B1I, FREQ_B3

                        # Determine frequency based on system and freq_idx
                        sys_char = sys_map.get(sat2sys(sat), 'U')
                        freq = None

                        if sys_char == 'G':  # GPS
                            freq = [FREQ_L1, FREQ_L2, FREQ_L5][freq_idx] if freq_idx < 3 else None
                        elif sys_char == 'R':  # GLONASS
                            # GLONASS uses FDMA - calculate frequency based on FCN
                            from ..gnss.glonass_freq import get_glonass_frequency
                            freq = get_glonass_frequency(sat, freq_idx, nav_data)
                        elif sys_char == 'E':  # Galileo
                            freq = [FREQ_E1, FREQ_E5a, FREQ_E5b][freq_idx] if freq_idx < 3 else None
                        elif sys_char == 'C':  # BeiDou
                            freq = [FREQ_B1I, FREQ_B3][freq_idx] if freq_idx < 2 else None
                        elif sys_char == 'J':  # QZSS (same as GPS)
                            freq = [FREQ_L1, FREQ_L2, FREQ_L5][freq_idx] if freq_idx < 3 else None

                        if freq:
                            wavelength = CLIGHT / freq

                            # Extract carrier phase measurements (in cycles from RINEX)
                            rover_cp_cycles = rover_obs_sat.L[freq_idx] if rover_obs_sat.L[freq_idx] != 0 else None
                            base_cp_cycles = base_obs_sat.L[freq_idx] if base_obs_sat.L[freq_idx] != 0 else None
                            ref_rover_cp_cycles = ref_rover_obs.L[freq_idx] if ref_rover_obs.L[freq_idx] != 0 else None
                            ref_base_cp_cycles = ref_base_obs.L[freq_idx] if ref_base_obs.L[freq_idx] != 0 else None

                            if None not in [rover_cp_cycles, base_cp_cycles, ref_rover_cp_cycles, ref_base_cp_cycles]:
                                # Convert carrier phase to meters like RTKLIB
                                # RTKLIB: obs.L[ix,f] * _c / freq
                                rover_cp_meters = rover_cp_cycles * wavelength
                                base_cp_meters = base_cp_cycles * wavelength
                                ref_rover_cp_meters = ref_rover_cp_cycles * wavelength
                                ref_base_cp_meters = ref_base_cp_cycles * wavelength

                                # Form DD carrier phase in meters
                                sd_rover_meters = rover_cp_meters - ref_rover_cp_meters
                                sd_base_meters = base_cp_meters - ref_base_cp_meters
                                dd_carrier_meters = sd_rover_meters - sd_base_meters

                                # Convert back to cycles for output
                                dd_carrier = dd_carrier_meters / wavelength

                # Store the DD measurement
                dd_measurements.append({
                    'sat': sat,
                    'ref_sat': ref_sat,
                    'sys': system,
                    'freq_idx': freq_idx,
                    'dd_obs': dd_obs,
                    'dd_carrier': dd_carrier,
                    'wavelength': wavelength,
                    'elevation': data['elevation'],
                    'sat_pos': data['pos'],  # Rover satellite position
                    'ref_sat_pos': ref_data['pos'],  # Rover reference satellite position
                    'base_sat_pos': data['base_pos'],  # Base satellite position
                    'base_ref_sat_pos': ref_data['base_pos'],  # Base reference satellite position
                    'sat_clk': data['clk'],  # Rover satellite clock
                    'ref_sat_clk': ref_data['clk'],  # Rover reference satellite clock
                    'base_sat_clk': data['base_clk'],  # Base satellite clock
                    'base_ref_sat_clk': ref_data['base_clk'],  # Base reference satellite clock
                    'rover_carrier_ref': ref_rover_cp_cycles if dd_carrier is not None else None,
                    'rover_carrier_other': rover_cp_cycles if dd_carrier is not None else None,
                    'base_carrier_ref': ref_base_cp_cycles if dd_carrier is not None else None,
                    'base_carrier_other': base_cp_cycles if dd_carrier is not None else None
                })

    return dd_measurements