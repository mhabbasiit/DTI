import json
import os
import numpy as np
import subprocess
import nibabel as nib
from utilities import match_file_pattern, gen_qc_image, get_dimensions

from config import (
    FSL_HOME,
    B0_CORRECTION_QC_SLICES
)

# ==== Load JSON File ====
def load_json(json_file):
    with open(json_file, 'r') as f:
        return json.load(f)

# ==== Extract Readout Time ====
def get_readout_time(json_data):
    return json_data.get("TotalReadoutTime", 0.05)

def get_PE_direction(json_data):
    return json_data.get('PhaseEncodingDirection', None)

# ==== Extract Slice Timing Order ====
def get_slice_order(json_data):
    slice_timing = np.array(json_data.get("SliceTiming", []))
    slice_order = np.argsort(slice_timing)  # Get acquisition order
    return slice_order.tolist()

# ==== Write Acquisition Parameters (acqparams.txt) ====
def write_acqparams(readout_time, numAP, numPA, filename="acqparams.txt"):
    with open(filename, "w") as f:
        for i in range(numAP):
            f.write(f"0 1 0 {readout_time}\n")  # AP
        for i in range(numPA):
            f.write(f"0  -1 0 {readout_time}\n")  # PA

# ==== Write Slice Timing Order (slspec.txt) ====
def write_slspec(slice_order, filename="slspec.txt"):
    with open(filename, "w") as f:
        for slice_idx in slice_order:
            f.write(f"{slice_idx}\n")

# ==== Load BVAL File ====
def load_bvals(bval_file):
    return np.loadtxt(bval_file)

# ==== Get Indices of B≈0 Volumes ====
def get_b0_indices(bvals, threshold=50):
    return np.where(bvals < threshold)[0]  # Indices where bval < threshold

# ==== Write B0 Indices to File ====
def write_indices(indices, filename):
    np.savetxt(filename, indices, fmt="%d")

# ==== Extract B0 Volumes using FSL ====
def extract_b0_volumes(dwi_file, b0_indices_file, output_file):
    with open(b0_indices_file, "r") as f:
        indices = ",".join(f.read().split())  # Convert list to comma-separated string
    cmd = f"fslselectvols -i {dwi_file} -o {output_file} --vols={indices}"
    print(cmd)
    os.system(cmd)

def process_topup(subject_folder, correction_subject_folder, blip_up_patterns, blip_down_patterns, scan_num):

    json_AP_path = match_file_pattern(subject_folder, blip_up_patterns['json'])
    bvals_AP_path = match_file_pattern(subject_folder, blip_up_patterns['bval'])
    bvals_PA_path = match_file_pattern(subject_folder, blip_down_patterns['bval'])

    print(f'json_AP_path:{json_AP_path}')
    print(f'bvals_PA_path:{bvals_PA_path}')


    json_AP = load_json(json_AP_path)

    # Extract readout time & slice order
    readout_time = get_readout_time(json_AP)

    # Process BVAL files
    bvals_AP = load_bvals(bvals_AP_path)
    bvals_PA = load_bvals(bvals_PA_path)

    # Generate index files for b0s
    b0_indices_AP = get_b0_indices(bvals_AP)
    b0_indices_PA = get_b0_indices(bvals_PA)
    indices_AP_path = os.path.join(correction_subject_folder,blip_up_patterns['json'].replace('*','b0_indices').replace('.json','.txt'))
    indices_PA_path = os.path.join(correction_subject_folder,blip_down_patterns['json'].replace('*','b0_indices').replace('.json','.txt'))
    dwi_AP_path = match_file_pattern(subject_folder, blip_up_patterns['dwi'])
    dwi_PA_path = match_file_pattern(subject_folder, blip_down_patterns['dwi'])

    original_shape = get_dimensions(dwi_AP_path)
    if (original_shape[0]%2==0) and (original_shape[1]%2==0) and (original_shape[2]%2==0):
        print('All dimensions are even')
        config_cnf = 'b02b0_2.cnf'
    else:
        print('One dimension is odd')
        config_cnf = 'b02b0_1.cnf'

    b0_AP_path = os.path.join(correction_subject_folder,blip_up_patterns['dwi'].replace('*','b0_AP'))
    b0_PA_path = os.path.join(correction_subject_folder,blip_up_patterns['dwi'].replace('*','b0_PA'))
    write_indices(b0_indices_AP, indices_AP_path)
    write_indices(b0_indices_PA, indices_PA_path)
    print(f"B0 indices for AP: {b0_indices_AP}")
    print(f"B0 indices for PA: {b0_indices_PA}")

    # Extract b0 volumes
    extract_b0_volumes(dwi_AP_path, indices_AP_path, b0_AP_path)
    extract_b0_volumes(dwi_PA_path, indices_PA_path, b0_PA_path)

    # Save acquisition parameters & slice timing
    if readout_time:
        acq_path = os.path.join(correction_subject_folder,f'acq_scan_{scan_num}.txt')
        write_acqparams(readout_time, len(b0_indices_AP), len(b0_indices_PA), acq_path)
        print(f"Readout time ({readout_time}) saved")

    print("Merging")
    b0_all_path = os.path.join(correction_subject_folder, f'b0_all_scan_{scan_num}.nii.gz')
    # Merge b0 images for TOPUP
    os.system(f"fslmerge -t {b0_all_path} {b0_AP_path} {b0_PA_path}")

    print("Running topup")
    top_up_results_path = os.path.join(correction_subject_folder, f'topup_results_{scan_num}')
    unwarped_results_path = os.path.join(correction_subject_folder, f'b0_unwarped_{scan_num}')
    # Run TOPUP
    subprocess.run([
            "topup", f"--imain={b0_all_path}", f"--datain={acq_path}",
            f"--config={config_cnf}", f"--out={top_up_results_path}", f"--iout={unwarped_results_path}"])

    print("TOPUP preprocessing completed!")
    
def topup_qc(correction_subject_folder, scan_num):
    original_image_path = os.path.join(correction_subject_folder, f'b0_all_scan_{scan_num}.nii.gz')
    unwarped_results_path = os.path.join(correction_subject_folder, f'b0_unwarped_{scan_num}.nii.gz')
    original_image = nib.load(original_image_path).get_fdata()
    unwarped_image = nib.load(unwarped_results_path).get_fdata()

    image_series = [original_image, unwarped_image]
    slices_to_plot = B0_CORRECTION_QC_SLICES
    image_names = ['Original b0', 'Corrected b0']
    subject_name = os.path.basename(correction_subject_folder)
    volumes_to_plot = [0]
    suptitle = f'B0_correction_qc_scan{scan_num}'
    gen_qc_image(subject_name, correction_subject_folder, image_series, slices_to_plot, volumes_to_plot, suptitle, image_names, scan_num)
        
        
def run_topup(subject_folder, out_subject_folder, blip_up_patterns, blip_down_patterns):
    os.makedirs(out_subject_folder,exist_ok=True)
    num_scans = len(blip_up_patterns['dwi'])
    for n in range(num_scans):
        blip_up_patterns_n = {}
        blip_down_patterns_n = {}
        for k,v in blip_up_patterns.items():
            blip_up_patterns_n[k] = v[n]
        for k,v in blip_down_patterns.items():
            blip_down_patterns_n[k] = v[n]
        process_topup(subject_folder, out_subject_folder, blip_up_patterns_n, blip_down_patterns_n, n)
        topup_qc(out_subject_folder, n)
        
        
