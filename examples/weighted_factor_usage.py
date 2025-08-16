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

"""Example usage of weighted pseudorange and Doppler factors"""

import numpy as np
from pyins.fusion.pseudorange_factor import (
    WeightedPseudorangeFactor, 
    PseudorangeMeasurementModel,
    MeasurementWeightConfig
)
from pyins.fusion.doppler_factor import DopplerFactor, DopplerMeasurementModel
from pyins.fusion.state import NavigationState
from pyins.fusion.graph_optimizer import FactorGraph
from pyins.core.data_structures import Observation, Ephemeris
from pyins.core.constants import FREQ_L1, SYS_GPS, SYS_BDS, CLIGHT
from pyins.coordinate.transforms import llh2ecef


def create_sample_ephemeris(prn, system=SYS_GPS):
    """Create sample ephemeris for testing"""
    return Ephemeris(
        sat=prn,
        iode=0,
        iodc=0,
        sva=0,
        svh=0,
        week=2000,
        code=0,
        flag=0,
        toe=0.0,
        toc=0.0,
        ttr=0.0,
        A=26559720.0,  # GPS semi-major axis
        e=0.001,
        i0=0.95,  # ~55 degrees inclination
        OMG0=(prn - 1) * np.pi / 6,  # Spread satellites
        omg=0.0,
        M0=0.0,
        deln=0.0,
        OMGd=-7.2921151467e-5,
        idot=0.0,
        crc=0.0,
        crs=0.0,
        cuc=0.0,
        cus=0.0,
        cic=0.0,
        cis=0.0,
        toes=0.0,
        fit=4.0,
        f0=0.0,
        f1=0.0,
        f2=0.0
    )


def create_sample_observations():
    """Create sample observations with varying quality"""
    observations = []
    
    # High quality GPS satellite
    obs1 = Observation(time=0.0, sat=1, system=SYS_GPS)
    obs1.P[0] = 20123456.789  # L1 pseudorange (m)
    obs1.D[0] = -234.567      # L1 Doppler (Hz)
    obs1.SNR[0] = 48.0        # High SNR
    observations.append(obs1)
    
    # Medium quality GPS satellite
    obs2 = Observation(time=0.0, sat=2, system=SYS_GPS)
    obs2.P[0] = 21234567.890
    obs2.D[0] = 123.456
    obs2.SNR[0] = 35.0        # Medium SNR
    observations.append(obs2)
    
    # Low quality BeiDou satellite
    obs3 = Observation(time=0.0, sat=6, system=SYS_BDS)
    obs3.P[0] = 22345678.901
    obs3.D[0] = -345.678
    obs3.SNR[0] = 25.0        # Low SNR
    observations.append(obs3)
    
    return observations


def demonstrate_weighted_factors():
    """Demonstrate weighted factor creation and usage"""
    print("=== Weighted Factor Demonstration ===\n")
    
    # Initial state (receiver at equator)
    state = NavigationState()
    state.position = llh2ecef(np.array([0.0, 0.0, 100.0]))  # lat, lon, height
    state.velocity = np.array([10.0, 0.0, 0.0])  # Moving east
    state.dcm = np.eye(3)
    state.clock_bias = np.array([0.0])
    
    # Create observations and ephemerides
    observations = create_sample_observations()
    ephemerides = {
        1: create_sample_ephemeris(1, SYS_GPS),
        2: create_sample_ephemeris(2, SYS_GPS),
        6: create_sample_ephemeris(6, SYS_BDS)
    }
    
    # Configure weighting
    weight_config = MeasurementWeightConfig(
        err_base=0.3,       # 30cm base error
        err_el=0.3,         # 30cm elevation-dependent error
        err_snr=0.2,        # SNR coefficient
        snr_max=50.0,       # Reference SNR
        eratio_code=100.0,  # Code is 100x noisier than phase
    )
    
    print("Weight Configuration:")
    print(f"  Base error: {weight_config.err_base} m")
    print(f"  Elevation error: {weight_config.err_el} m")
    print(f"  SNR coefficient: {weight_config.err_snr}")
    print(f"  Code/phase ratio: {weight_config.eratio_code}")
    print()
    
    # Create pseudorange measurement model
    pr_model = PseudorangeMeasurementModel(weight_config)
    
    # Create weighted pseudorange factors
    pr_factors = pr_model.create_factors(observations, ephemerides, state)
    
    print(f"Created {len(pr_factors)} pseudorange factors:")
    for i, factor in enumerate(pr_factors):
        obs = factor.observation
        variance = factor.compute_variance(state)
        weight = 1.0 / variance
        print(f"  Sat {obs.sat}: SNR={obs.SNR[0]:.1f} dB-Hz, "
              f"variance={variance:.1f} m², weight={weight:.6f}")
    print()
    
    # Create Doppler factors (if needed)
    doppler_model = DopplerMeasurementModel()
    
    # Prepare frequency information
    frequencies = {
        (1, 0): FREQ_L1,
        (2, 0): FREQ_L1,
        (6, 0): FREQ_L1
    }
    
    # Compute Doppler residuals
    doppler_res, doppler_H, doppler_R = doppler_model.compute_doppler_residuals(
        observations, ephemerides, state, frequencies
    )
    
    print(f"Doppler measurements:")
    for i, obs in enumerate(observations):
        if i < len(doppler_res):
            print(f"  Sat {obs.sat}: Doppler residual = {doppler_res[i]:.2f} Hz")
    print()
    
    # Demonstrate factor graph usage
    graph = FactorGraph()
    
    # Add state node
    node_id = graph.add_node(state)
    
    # Add weighted pseudorange factors
    for factor in pr_factors:
        # Update the factor's node IDs
        factor.node_ids = [node_id]
        # Update information matrix based on current state
        factor.update_information_matrix(state)
        # Add to graph
        graph.add_factor(factor)
    
    print(f"Factor graph contains {len(graph.factors)} factors")
    print("Ready for optimization with GTSAM or other solver")


def demonstrate_dynamic_weighting():
    """Show how weights change with conditions"""
    print("\n=== Dynamic Weighting Demonstration ===\n")
    
    weight_config = MeasurementWeightConfig(
        err_base=0.3,
        err_el=0.3,
        err_snr=0.3,
        snr_max=50.0,
        eratio_code=100.0
    )
    
    # Test observation
    obs = Observation(time=0.0, sat=1, system=SYS_GPS)
    obs.P[0] = 20000000.0
    
    # Test ephemeris
    eph = create_sample_ephemeris(1)
    
    # Test at different SNR levels
    print("Weight vs SNR (at 45° elevation):")
    print("SNR (dB-Hz) | Variance (m²) | Weight")
    print("-" * 40)
    
    for snr in [20, 30, 40, 50]:
        obs.SNR[0] = snr
        
        factor = WeightedPseudorangeFactor(
            node_ids=[0],
            residual=np.zeros(1),
            jacobian={},
            information=np.eye(1),
            observation=obs,
            ephemeris=eph,
            weight_config=weight_config
        )
        
        # Create a state (doesn't affect this simple test much)
        state = NavigationState()
        state.position = llh2ecef(np.array([0.0, 0.0, 100.0]))
        
        variance = factor.compute_variance(state)
        weight = 1.0 / variance
        
        print(f"    {snr:4.0f}    | {variance:10.1f}  | {weight:.6f}")
    
    print("\nNote: Lower SNR → Higher variance → Lower weight")


if __name__ == "__main__":
    demonstrate_weighted_factors()
    demonstrate_dynamic_weighting()