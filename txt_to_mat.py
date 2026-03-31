import numpy as np
import scipy.io as sio
import os

# Define paths
paths = [r'C:\Users\timch\Downloads\Research\fio_pol_stabilization\Datasets\03_02_2026400k_samples_txp_1551.5_pax_1546.5_polcon_and_fiber_2_1Hz.txt']

for path in paths:
    # Generate matlab file path
    mat_file_path = path.replace('.txt', '.mat')

    # Check if file exists
    if not os.path.exists(path):
        print(f"Error: Could not find the file at {path}")
    else:
        print(f"Loading data from: {path}")
        
        # Load data
        # delimiter=',' splits the columns
        # skiprows=1 ignores the header line ('s1_pax, s2_pax, ...')
        # unpack=True transposes the data so we can unpack it directly into variables
        s1_pax, s2_pax, s3_pax, s1_txp, s2_txp, s3_txp = np.loadtxt(
            path, delimiter=',', skiprows=1, unpack=True
        )

        # Reshape arrays to 1 x N to match formatting
        s1_pax = s1_pax.reshape(1, -1)
        s2_pax = s2_pax.reshape(1, -1)
        s3_pax = s3_pax.reshape(1, -1)
        s1_txp = s1_txp.reshape(1, -1)
        s2_txp = s2_txp.reshape(1, -1)
        s3_txp = s3_txp.reshape(1, -1)

        # Create the dictionary mapping variable names to the arrays
        fileDic = {
            's1_pax': s1_pax, 
            's2_pax': s2_pax, 
            's3_pax': s3_pax,
            's1_txp': s1_txp, 
            's2_txp': s2_txp, 
            's3_txp': s3_txp
        }

        # Save the dictionary as a MATLAB .mat file
        sio.savemat(mat_file_path, fileDic)
        print(f"Saved MATLAB file: {mat_file_path}\n")