import os
import glob
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

def process_dataset(path):
    print(f"\nAnalyzing dataset: {path}")
    try:
        data = scipy.io.loadmat(path)
        # Load all points
        data = {k: v.flatten() for k, v in data.items() if isinstance(v, np.ndarray)}
        
        if 's1_pax' in data and 's2_pax' in data and 's3_pax' in data:
            pax_1, pax_2 = analyze_degree_coverage(
                data['s1_pax'], data['s2_pax'], data['s3_pax']
            )
            print(f"PAX: >=1 pt: {pax_1:.2f}%, >=2 pts: {pax_2:.2f}%")
        else:
            print("PAX data not found in this dataset.")
            
        if 's1_txp' in data and 's2_txp' in data and 's3_txp' in data:
            txp_1, txp_2 = analyze_degree_coverage(
                data['s1_txp'], data['s2_txp'], data['s3_txp']
            )
            print(f"TXP: >=1 pt: {txp_1:.2f}%, >=2 pts: {txp_2:.2f}%")
        else:
            print("TXP data not found in this dataset.")
            
    except Exception as e:
        if "Unknown mat file type" in str(e):
            print(f"Error: {path} appears to be a Git LFS pointer file, not the actual .mat file.")
            print("Please run 'git lfs pull' to download the actual data files.")
        else:
            print(f"Error processing {path}: {e}")

# Find all .mat files in data directory
data_dir = 'data'
mat_files = []
for root, dirs, files in os.walk(data_dir):
    for file in files:
        if file.endswith('.mat'):
            mat_files.append(os.path.join(root, file))

if not mat_files:
    print(f"No .mat files found in {data_dir}")
else:
    for file_path in sorted(mat_files):
        process_dataset(file_path)
