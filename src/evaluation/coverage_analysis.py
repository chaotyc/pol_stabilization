import os
import numpy as np
import scipy.io

def analyze_degree_coverage(s1, s2, s3):
    """
    Determine coverage of poincare sphere by percentage of 1 by 1 degree bins with at least 1 and at least 2 points.
    """
    # convert parameters to angles
    r = np.sqrt(s1**2 + s2**2 + s3**2)
    
    # ellipticity angle
    el_deg = np.degrees(np.arcsin(np.clip(s3 / r, -1.0, 1.0)))
    
    # polar angle
    pol_deg = np.degrees(np.arctan2(s2, s1))
    
    # define grid (1 degree steps, 360x180)
    range_limits = [[-180, 180], [-90, 90]]
    bins_pol = 360
    bins_el = 180
    
    # compute bins
    H, _, _ = np.histogram2d(
        pol_deg, el_deg, 
        bins=[bins_pol, bins_el], 
        range=range_limits
    )
    
    # calculate stats
    total_spaces = bins_pol * bins_el # 64,800 total bins
    
    pct_ge_1 = (np.sum(H >= 1) / total_spaces) * 100
    pct_ge_2 = (np.sum(H >= 2) / total_spaces) * 100
    
    return pct_ge_1, pct_ge_2

# load data
path = os.path.join('data', 'synthetic', '400k_samples_txp_1551.5_pax_1565.496_polcon_and_fiber_2_1Hz.mat')
data = scipy.io.loadmat(path)
data = {k: v.flatten()[:400000] for k, v in data.items() if isinstance(v, np.ndarray)}

# analyze PAX
pax_1, pax_2 = analyze_degree_coverage(
    data['s1_pax'].flatten(), data['s2_pax'].flatten(), data['s3_pax'].flatten()
)

# analyze TXP
txp_1, txp_2 = analyze_degree_coverage(
    data['s1_txp'].flatten(), data['s2_txp'].flatten(), data['s3_txp'].flatten()
)

print(f"PAX: >=1 pt: {pax_1:.2f}%, >=2 pts: {pax_2:.2f}%")
print(f"TXP: >=1 pt: {txp_1:.2f}%, >=2 pts: {txp_2:.2f}%")