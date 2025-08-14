import json
import os
import numpy as np
import subprocess
import sys
import os
import config
import nibabel as nib

from config import setup_fsl_env
from utilities import get_sessions, trim_odd_dimensions

from utilities import match_file_pattern, gen_qc_image


from config import (
    B0_CORRECTION_FOLDER,
    B0_CORRECTION,
    SLICE_TO_SLICE_CORRECTION,
    BASELINE_SLICE_ORDER_JSON,
    EDDY_CORRECTION_FOLDER,
    EDDY_CORRECTION_QC_SLICES,
    INPUT_SUBDIR,
    INPUT_DIR,
    NUM_SCANS_PER_SESSION
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

def eddy_qc(eddy_folder, scan_num):
    original_image_path = os.path.join(eddy_folder, f'dwi_merged_{scan_num}.nii.gz')
    unwarped_results_path = os.path.join(eddy_folder, f'eddy_aligned_{scan_num}.nii.gz')
    original_image = nib.load(original_image_path).get_fdata()
    unwarped_image = nib.load(unwarped_results_path).get_fdata()

    image_series = [original_image, unwarped_image]
    slices_to_plot = EDDY_CORRECTION_QC_SLICES
    image_names = ['Original image', 'Corrected image']
    subject_name = os.path.basename(eddy_folder)
    volumes_to_plot = [original_image.shape[-1]-1]
    suptitle = f'Eddy_correction_qc_scan{scan_num}\n'
    gen_qc_image(subject_name, eddy_folder, image_series, slices_to_plot, volumes_to_plot, suptitle, image_names, scan_num)
       

def run_eddy(subject_folder, correction_subject_folder, eddy_folder, blip_up_patterns, blip_down_patterns, scan_num):

    json_AP_path = match_file_pattern(subject_folder, blip_up_patterns['json'])
    bvals_AP_path = match_file_pattern(subject_folder, blip_up_patterns['bval'])
    bvals_PA_path = match_file_pattern(subject_folder, blip_down_patterns['bval'])

    print(f'json_AP_path:{json_AP_path}')
    print(f'bvals_PA_path:{bvals_PA_path}')

    json_AP = load_json(json_AP_path)

    if SLICE_TO_SLICE_CORRECTION:

        slspec_path = os.path.join(eddy_folder,f"slspec_{scan_num}.txt")

        if BASELINE_SLICE_ORDER_JSON:
            json_AP_backup = load_json(BASELINE_SLICE_ORDER_JSON)

        # Extract readout time & slice order
        slice_order = get_slice_order(json_AP)  # Assume same slice timing for both
        if slice_order:
            write_slspec(slice_order, slspec_path)
            print("Slice timing order saved to slspec.txt")
        else:
            slice_order = get_slice_order(json_AP_backup)

    # Process BVAL files
    bvals_AP = load_bvals(bvals_AP_path)
    bvals_PA = load_bvals(bvals_PA_path)            
        
    b0_indices_AP = get_b0_indices(bvals_AP)

    eddy_indices = [1]*len(bvals_AP) + [len(b0_indices_AP)+1]*len(bvals_PA)
    eddy_indices_path = os.path.join(eddy_folder,f"eddy_indices_{scan_num}.txt")
    write_eddy_indices(np.array(eddy_indices),eddy_indices_path)

    # Merge the DWI images, bvals, and bvecs
    merged_dwi = os.path.join(eddy_folder, f"dwi_merged_{scan_num}.nii.gz")
    merged_bval = os.path.join(eddy_folder, f"dwi_merged_{scan_num}.bval")
    merged_bvec = os.path.join(eddy_folder, f"dwi_merged_{scan_num}.bvec")

    dwi_AP_path = match_file_pattern(subject_folder, blip_up_patterns['dwi'])
    bvecs_AP_path = match_file_pattern(subject_folder, blip_up_patterns['bvec'])
    dwi_PA_path = match_file_pattern(subject_folder, blip_down_patterns['dwi'])
    bvecs_PA_path = match_file_pattern(subject_folder, blip_down_patterns['bvec'])
    topup_path = os.path.join(correction_subject_folder,f'topup_results_{scan_num}')
    topup_image_path = os.path.join(correction_subject_folder,f'b0_unwarped_{scan_num}.nii.gz')
    acq_path = os.path.join(correction_subject_folder,f"acq_scan_{scan_num}.txt")

    subprocess.call(f"fslmerge -t {merged_dwi} {dwi_AP_path} {dwi_PA_path}",shell=True)
    subprocess.call(f"paste -d ' ' {bvals_AP_path} {bvals_PA_path} > {merged_bval}",shell=True)
    subprocess.call(f"paste -d ' ' {bvecs_AP_path} {bvecs_PA_path} > {merged_bvec}",shell=True)

    #create mask
    subprocess.call(f"fslroi {topup_image_path} {os.path.join(eddy_folder,f'b0_extract{scan_num}')} 0 1",shell=True)
    subprocess.call(f"bet {os.path.join(eddy_folder,f'b0_extract{scan_num}')} {os.path.join(eddy_folder,f'mask_bet{scan_num}')} -m -f 0.4",shell=True)
    mask_path = os.path.join(eddy_folder,f'mask_bet{scan_num}_mask.nii.gz')
    out_path = os.path.join(eddy_folder,f'eddy_aligned_{scan_num}')

    if SLICE_TO_SLICE_CORRECTION:
        eddy_cmd = f"eddy_cuda10.2 --topup={topup_path} --repol --ol_nstd=3.5 --ol_nvox=250 --imain={merged_dwi} --flm=quadratic --mask={mask_path} --out={out_path} --acqp={acq_path} --index={eddy_indices_path} --bvecs={merged_bvec} --bvals={merged_bval} --verbose --mporder=6 --json={json_AP_path} --s2v_niter=5 --s2v_lambda=1 --s2v_interp=trilinear --data_is_shelled"
        #eddy_cmd = f"eddy_cuda10.2 --topup={topup_path} --repol --ol_nstd=3.5 --ol_nvox=250 --imain={merged_dwi} --flm=quadratic --mask={mask_path} --out={out_path} --acqp={acq_path} --index={eddy_indices_path} --bvecs={merged_bvec} --bvals={merged_bval} --verbose --mporder=6 --slspec={slspec_path} --s2v_niter=5 --s2v_lambda=1 --s2v_interp=trilinear"
    else:
        eddy_cmd = f"eddy_cuda10.2 --topup={topup_path} --repol --ol_nstd=3.5 --ol_nvox=250 --imain={merged_dwi} --flm=quadratic --mask={mask_path} --out={out_path} --acqp={acq_path} --index={eddy_indices_path} --bvecs={merged_bvec} --bvals={merged_bval} --verbose --mporder=6"
    
    subprocess.call(eddy_cmd,shell=True)
    print(f'Done {scan_num}')


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python process_eddy.py <subject_id>")
        sys.exit(1)
    else:
        subject_id = sys.argv[1]


    if not os.path.exists(EDDY_CORRECTION_FOLDER):
        os.makedirs(EDDY_CORRECTION_FOLDER, exist_ok=True)

    sessions = get_sessions(os.path.join(INPUT_DIR,subject_id))
    print(sessions)
    if not sessions:
        subject_folders = [os.path.join(INPUT_DIR,subject_id,INPUT_SUBDIR)]
        out_subject_folders = [os.path.join(EDDY_CORRECTION_FOLDER,subject_id)]
        b0_correction_folders = [os.path.join(B0_CORRECTION_FOLDER,subject_id)]
    else:
        subject_folders = [os.path.join(INPUT_DIR,subject_id,sess,INPUT_SUBDIR) for sess in sessions]
        out_subject_folders = [os.path.join(EDDY_CORRECTION_FOLDER,subject_id,sess) for sess in sessions]
        b0_correction_folders = [os.path.join(B0_CORRECTION_FOLDER,subject_id,sess) for sess in sessions]

    # subject_folder = os.path.join(INPUT_DIR,subject_id,INPUT_SUBDIR)
    # if not os.path.exists(EDDY_CORRECTION_FOLDER):
    #     os.mkdir(EDDY_CORRECTION_FOLDER)
    
    # out_subject_folder = os.path.join(EDDY_CORRECTION_FOLDER,subject_id)
    # if not os.path.exists(out_subject_folder):
    #     os.mkdir(out_subject_folder)

    setup_fsl_env()

    #b0_correction_folder = os.path.join(B0_CORRECTION_FOLDER,subject_id)

    blip_up_patterns = {}
    try:
        blip_up_patterns['dwi'] = config.DWI_FILE_PATTERNS
        blip_up_patterns['bval'] = config.BVAL_FILE_PATTERNS
        blip_up_patterns['bvec'] = config.BVEC_FILE_PATTERNS
        blip_up_patterns['json'] = config.JSON_FILE_PATTERNS
    except:
        print('Missing file pattern information in config')
    assert NUM_SCANS_PER_SESSION == len(blip_up_patterns['dwi'])
    assert NUM_SCANS_PER_SESSION == len(blip_up_patterns['bval'])
    assert NUM_SCANS_PER_SESSION == len(blip_up_patterns['bvec'])
    assert NUM_SCANS_PER_SESSION == len(blip_up_patterns['json'])

    blip_down_patterns = {}
    if B0_CORRECTION=='Topup':
        setup_fsl_env()
        try:
            blip_down_patterns['dwi'] = config.REVERSED_DWI_FILE_PATTERNS
            blip_down_patterns['bval'] = config.REVERSED_BVAL_FILE_PATTERNS
            blip_down_patterns['bvec'] = config.REVERSED_BVEC_FILE_PATTERNS
            blip_down_patterns['json'] = config.REVERSED_JSON_FILE_PATTERNS
        except:
            print('Missing file pattern information for reversed polarity in config')
        print(blip_up_patterns)
        print(blip_down_patterns)

    for n in range(NUM_SCANS_PER_SESSION):
        blip_up_patterns_n = {}
        blip_down_patterns_n = {}
        for k,v in blip_up_patterns.items():
            blip_up_patterns_n[k] = v[n]
        for k,v in blip_down_patterns.items():
            blip_down_patterns_n[k] = v[n]
        print(f"Processing scan # {n}")

        for subject_folder, out_subject_folder, b0_correction_folder in zip(subject_folders, out_subject_folders, b0_correction_folders):
            print(subject_folder)
            print(out_subject_folder)
            os.makedirs(out_subject_folder, exist_ok=True)
            run_eddy(subject_folder, b0_correction_folder, out_subject_folder, blip_up_patterns_n, blip_down_patterns_n, n)
            eddy_qc(out_subject_folder, n)