import json
import os
import numpy as np
import subprocess
import sys
import glob
import os

from config import (
    INPUT_DIR,
    OUTPUT_DIR,
    LOG_DIR,
    FORCE_REPROCESS,
    ENABLE_DETAILED_LOGGING,
    LOG_LEVEL,
    LOG_FORMAT,
    REG_INPUT_DIR,
    REG_OUTPUT_DIR,
    REG_T1_FILE_PATTERN,
    REG_T2_FILE_PATTERN,
    MNI_TEMPLATE_VERSION,
    MNI_TEMPLATE_RESOLUTION,
    PERFORM_CROPPING,
    CROP_MARGIN
)


# ==== Load JSON File ====
def load_json(json_file):
    with open(json_file, 'r') as f:
        return json.load(f)

# ==== Extract Readout Time ====
def get_readout_time(json_data):
    return json_data.get("TotalReadoutTime", None)

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

# ==== Write eddy Indices to File ====
def write_eddy_indices(indices, filename):
    np.savetxt(filename, indices, fmt="%d")

def process_eddy(subject_folder, patterns, numdir):


    match_file_pattern

    json_AP = load_json(os.path.join(subject_folder,f"{subject}_dMRI_{numdir}_AP.json"))
    # Extract readout time & slice order
    readout_time = get_readout_time(json_AP)
    slice_order = get_slice_order(json_AP)  # Assume same slice timing for both
    slspec_path = os.path.join(subject_folder,f"slspec_{numdir}.txt")
    if slice_order:
        write_slspec(slice_order, slspec_path)
        print("Slice timing order saved to slspec.txt")

    # Process BVAL files
    bvals_AP = load_bvals(os.path.join(subject_folder,f"{subject}_dMRI_{numdir}_AP.bval"))
    bvals_PA = load_bvals(os.path.join(subject_folder,f"{subject}_dMRI_{numdir}_PA.bval"))

    b0_indices_AP = get_b0_indices(bvals_AP)
    b0_indices_PA = get_b0_indices(bvals_PA)

    eddy_indices = [1]*len(bvals_AP) + [len(b0_indices_AP)+1]*len(bvals_PA)
    eddy_indices_path = os.path.join(subject_folder,f"eddy_indices_{numdir}.txt")
    write_eddy_indices(np.array(eddy_indices),eddy_indices_path)

    # Merge the DWI images, bvals, and bvecs
    merged_dwi = os.path.join(subject_folder, f"dwi_merged_{numdir}.nii.gz")
    merged_bval = os.path.join(subject_folder, f"dwi_merged_{numdir}.bval")
    merged_bvec = os.path.join(subject_folder, f"dwi_merged_{numdir}.bvec")

    bvals_AP = os.path.join(subject_folder,f"{subject}_dMRI_{numdir}_AP.bval")
    bvals_PA = os.path.join(subject_folder,f"{subject}_dMRI_{numdir}_PA.bval")
    bvecs_AP = os.path.join(subject_folder,f"{subject}_dMRI_{numdir}_AP.bvec")
    bvecs_PA = os.path.join(subject_folder,f"{subject}_dMRI_{numdir}_PA.bvec")
    dwi_AP_path = os.path.join(subject_folder,f"{subject}_dMRI_{numdir}_AP.nii.gz")
    dwi_PA_path = os.path.join(subject_folder,f"{subject}_dMRI_{numdir}_PA.nii.gz")
    topup_path = os.path.join(subject_folder,f'topup_results_{numdir}')
    topup_image_path = os.path.join(subject_folder,f'b0_unwarped_{numdir}.nii.gz')
    acq_path = os.path.join(subject_folder,f"acqparams_{numdir}.txt")

    subprocess.call(f"fslmerge -t {merged_dwi} {dwi_AP_path} {dwi_PA_path}",shell=True)
    subprocess.call(f"paste -d ' ' {bvals_AP} {bvals_PA} > {merged_bval}",shell=True)
    subprocess.call(f"paste -d ' ' {bvecs_AP} {bvecs_PA} > {merged_bvec}",shell=True)

    #create mask
    subprocess.call(f"fslroi {topup_image_path} {os.path.join(subject_folder,f'b0_extract{numdir}')} 0 1",shell=True)
    subprocess.call(f"bet {os.path.join(subject_folder,f'b0_extract{numdir}')} {os.path.join(subject_folder,f'mask_bet{numdir}')} -m -f 0.4",shell=True)

    mask_path = os.path.join(subject_folder,f'mask_bet{numdir}_mask.nii.gz')
    out_path = os.path.join(subject_folder,f'eddy_aligned_{numdir}')
    eddy_cmd = f"eddy_cuda10.2 --topup={topup_path} --repol --ol_nstd=3.5 --ol_nvox=250 --imain={merged_dwi} --flm=quadratic --mask={mask_path} --out={out_path} --acqp={acq_path} --index={eddy_indices_path} --bvecs={merged_bvec} --bvals={merged_bval} --verbose --mporder=6 --slspec={slspec_path} --s2v_niter=5 --s2v_lambda=1 --s2v_interp=trilinear"
    subprocess.call(eddy_cmd,shell=True)
    print('Done {numdir}')


if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Usage: python preprocess_eddy.py <subject_id> <main_directory>")
        sys.exit(1)

    subject_id = sys.argv[1]
    main_directory = os.path.join(sys.argv[2],subject_id,'unprocessed/Diffusion')

    print('first set')
    process_eddy(main_directory, subject_id, 'dir98')
    print('second set')
    process_eddy(main_directory, subject_id, 'dir99')