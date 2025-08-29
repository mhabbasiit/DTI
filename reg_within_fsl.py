"""
Merging of Acquisitions — reg_within_fsl.py
===========================================

Aligns and merges multiple diffusion MRI runs using FSL FLIRT. 
Rigid transformations are estimated between B0 reference images, 
applied to corresponding diffusion volumes, and propagated to 
b-vectors to ensure orientation consistency across runs.

Steps performed:
1. Register B0 images between runs using FLIRT rigid-body transform
2. Apply transforms to diffusion volumes (DWI)
3. Rotate b-vectors using polar decomposition of transformation matrices
4. Merge registered DWI volumes, b-vectors, and b-values
5. Save combined outputs for downstream processing

Authors:
- Mohammad H Abbasi (mabbasi [at] stanford.edu)
- Gustavo Chau (gchau [at] stanford.edu)

Stanford University
Created: 2025
Version: 1.0.0
"""
import os
import glob
import nibabel as nib
import numpy as np
import json
from datetime import datetime
import time
import argparse  # Added for command line arguments
import sys
import subprocess
from scipy.linalg import polar
from config import setup_fsl_env


# Import configuration
from config import (
    NUM_SCANS_PER_SESSION,
    REG_WITHIN_B0_INPUT_FOLDER,
    REG_WITHIN_INPUT_FOLDER,
    REG_WITHIN_OUTPUT_FOLDER,
    REG_WITHIN_B0_INPUT_NAMES,
    REG_WITHIN_INPUT_NAMES,
    REG_BVEC_INPUT_NAMES,
    REG_BVAL_INPUT_NAMES,
    REG_WITHIN_OUTPUT_PATTERN
)

# Add a results tracking dictionary
registration_results = {
    'successful_subjects': [],
    'failed_subjects': [],
    'missing_subjects': [],  # New category for subjects with missing files
    'errors': {},
    'processing_times': {},  
    'cropping_results': {},  
    'start_time': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
    'end_time': None,
    'total_subjects': 0,
    'success_rate': 0,
    'average_processing_time': 0  
}

def save_registration_report(results, output_dir):
    """Save registration results to a JSON file."""
    report_path = os.path.join(output_dir, 'registration_report.json')
    with open(report_path, 'w') as f:
        json.dump(results, f, indent=4)
    print(f"\nDetailed registration report saved to: {report_path}")

def format_time(seconds):
    """Convert seconds to human-readable format."""
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    seconds = int(seconds % 60)
    if hours > 0:
        return f"{hours}h {minutes}m {seconds}s"
    elif minutes > 0:
        return f"{minutes}m {seconds}s"
    else:
        return f"{seconds}s"

def print_registration_summary(results):
    """Print a summary of registration results."""
    print("\n=== Registration Summary ===")
    print(f"Total expected subjects: {results['total_subjects']}")
    print(f"Successfully processed: {len(results['successful_subjects'])}")
    print(f"Failed during processing: {len(results['failed_subjects'])}")
    print(f"Missing input files: {len(results['missing_subjects'])}")
    
    total_processed = len(results['successful_subjects']) + len(results['failed_subjects'])
    
    if total_processed > 0:
        success_rate = (len(results['successful_subjects']) / total_processed) * 100
        print(f"Success rate: {success_rate:.2f}%")
    elif results['total_subjects'] > 0:
        total_issues = len(results['failed_subjects']) + len(results['missing_subjects'])
        success_rate = (len(results['successful_subjects']) / results['total_subjects']) * 100
        fail_rate = (total_issues / results['total_subjects']) * 100
        print(f"Success rate: {success_rate:.2f}%")
        print(f"Total fail rate: {fail_rate:.2f}%")
    else:
        print("No subjects to process found.")
    
    if results['processing_times']:
        avg_time = results['average_processing_time']
        print(f"\nTiming Information:")
        print(f"Average processing time per subject: {format_time(avg_time)}")
        print("\nProcessing times per subject:")
        for subj, t in results['processing_times'].items():
            print(f"- Subject {subj}: {format_time(t)}")
    
    if results['failed_subjects']:
        print("\nFailed subjects (processing errors):")
        for subject, error in results['errors'].items():
            if subject in results['failed_subjects']:
                print(f"- {subject}: {error}")
    
    if results['missing_subjects']:
        print("\nMissing subjects (input files not found):")
        for subject in results['missing_subjects']:
            print(f"- {subject}")

def register_to(b0_path, mni_template_path, output_matrix, output_registered_path):
    flirt_cmd = [
        "flirt",
        "-in", b0_path,
        "-ref", mni_template_path,
        "-out", output_registered_path,
        "-omat", output_matrix,
        "-dof", "6",  # rigid (6) + affine (12)
    ]
    subprocess.run(flirt_cmd, check=True)

def apply_transform_to_dwi(dwi_path, mni_template_path, matrix_path, output_dwi_path, interp="trilinear"):
    flirt_cmd = [
        "flirt",
        "-in", dwi_path,
        "-ref", mni_template_path,
        "-applyxfm",
        "-init", matrix_path,
        "-out", output_dwi_path,
        "-interp", interp
    ]
    subprocess.run(flirt_cmd, check=True)

def rotate_bvecs(bvecs_path, output_bvecs_path, flirt_mat_path):
    # Load bvecs
    bvecs = np.loadtxt(bvecs_path)
    if bvecs.shape[0] != 3:
        bvecs = bvecs.T  # Ensure shape is (3, N)

    # Load 4x4 FLIRT matrix and extract the linear 3x3 component
    flirt_affine = np.loadtxt(flirt_mat_path)
    affine_3x3 = flirt_affine[:3, :3]

    # Use polar decomposition to extract the closest rotation matrix
    R, _ = polar(affine_3x3)

    # Apply rotation
    rotated_bvecs = R @ bvecs

    # Normalize vectors to unit length
    norms = np.linalg.norm(rotated_bvecs, axis=0)
    norms[norms == 0] = 1  # Avoid division by zero for b=0
    rotated_bvecs /= norms

    # Save corrected bvecs
    np.savetxt(output_bvecs_path, rotated_bvecs, fmt="%.6f")

def combine_matrices(paths, out_path):
    matrices = [np.loadtxt(p) for p in paths]
    # Print shapes for debugging
    for i, m in enumerate(matrices):
        print(f"DEBUG: Matrix {i+1} shape:", m.shape)
    # Determine if 1D or 2D based on the first matrix
    first = matrices[0]
    is_1d = isinstance(first[0], float) if first.ndim == 1 else False
    if is_1d:
        combined = np.concatenate(matrices)
    else:
        combined = np.concatenate(matrices, axis=1)
    np.savetxt(out_path, combined, fmt='%.6f')

def merge(out_path_dwi_comb, reg_scan_names):
    merge_cmd = f'fslmerge -t {out_path_dwi_comb}'
    for f in reg_scan_names:
        merge_cmd = merge_cmd + f' {f}'
    os.system(merge_cmd)


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python brain_extraction.py <subject_id>")
        sys.exit(1)
    else:
        subject_id = sys.argv[1]

    if not os.path.exists(REG_WITHIN_OUTPUT_FOLDER):
        os.mkdir(REG_WITHIN_OUTPUT_FOLDER)

    setup_fsl_env()


    input_b0_subject_folder = os.path.join(REG_WITHIN_B0_INPUT_FOLDER,subject_id)
    input_subject_folder = os.path.join(REG_WITHIN_INPUT_FOLDER,subject_id)
    out_folder = os.path.join(REG_WITHIN_OUTPUT_FOLDER,subject_id)
    if not os.path.exists(out_folder):
        os.mkdir(out_folder)

    # B0 names
    if REG_WITHIN_B0_INPUT_NAMES:
        b0_input_names = [os.path.join(input_b0_subject_folder,x) for x in REG_WITHIN_B0_INPUT_NAMES]
    else:
        b0_input_names = [os.path.join(input_b0_subject_folder,f'mask_bet_scan{x}.nii.gz') for x in range(NUM_SCANS_PER_SESSION)]

    # Whole diffusion MRI 
    if REG_WITHIN_INPUT_NAMES:
        input_names = [os.path.join(input_subject_folder,x) for x in REG_WITHIN_INPUT_NAMES]
    else:
        input_names = [os.path.join(input_subject_folder,f'eddy_aligned_{x}.nii.gz') for x in range(NUM_SCANS_PER_SESSION)]
    
    if REG_BVEC_INPUT_NAMES:
        bvec_input_names = [os.path.join(input_subject_folder,x) for x in REG_BVEC_INPUT_NAMES]
    else:
        bvec_input_names = [os.path.join(input_subject_folder,f'eddy_aligned_{x}.eddy_rotated_bvecs') for x in range(NUM_SCANS_PER_SESSION)]

    if REG_BVAL_INPUT_NAMES:
        bval_input_names = [os.path.join(input_subject_folder,x) for x in REG_BVAL_INPUT_NAMES]
    else:
        bval_input_names = [os.path.join(input_subject_folder,f'dwi_merged_{x}.bval') for x in range(NUM_SCANS_PER_SESSION)]

    # Scans will be registered to first one (#0)

    for i in range(1,NUM_SCANS_PER_SESSION):
        # Register b0's using rigid transformation
        print(f"DEBUG: REGISTERING B0 IMAGES")
        fixed = b0_input_names[0]
        moving = b0_input_names[i]
        output_matrix = os.path.join(out_folder,f'transf_{i}_to_0.mat')
        output_registered_path = os.path.join(out_folder,f'b0_reg_{i}_to_0.nii.gz')
        register_to(moving, fixed, output_matrix, output_registered_path)

        # Apply to DWI
        print(f"DEBUG: REGISTERING DWI IMAGES")
        fixed = input_names[0]
        moving = input_names[i]
        out_path_dwi_reg = os.path.join(out_folder,f'dwi_{i}_to_0.nii.gz')
        apply_transform_to_dwi(moving, fixed, output_matrix, out_path_dwi_reg)

        # Rotate bvecs
        print('DEBUG: REGISTERING BVECS')
        rotate_bvecs(bvec_input_names[i], bvec_input_names[i].replace('.bvec','_reg.bvec'), output_matrix)

    print('DEBUG: MERGING IMAGES')
    reg_names = []
    reg_names.append(input_names[0])
    for i in range(1,NUM_SCANS_PER_SESSION):
        reg_names.append(os.path.join(out_folder,f'dwi_{i}_to_0.nii.gz'))
    if REG_WITHIN_OUTPUT_PATTERN:
        out_path_dwi_comb = os.path.join(out_folder, f'{REG_WITHIN_OUTPUT_PATTERN}.nii.gz')
    else:
        out_path_dwi_comb = os.path.join(out_folder, f'dwi_all_combined.nii.gz')
    merge(out_path_dwi_comb, reg_names)

    print('DEBUG: MERGING BVECS')
    bvec_reg_names = []
    bvec_reg_names.append(bvec_input_names[0])
    for i in range(1,NUM_SCANS_PER_SESSION):
        bvec_reg_names.append(os.path.join(out_folder,bvec_input_names[i].replace('.bvec','_reg.bvec')))
    if REG_WITHIN_OUTPUT_PATTERN:
        out_path_bvec_comb = os.path.join(out_folder, f'{REG_WITHIN_OUTPUT_PATTERN}.bvec')
    else:
        out_path_bvec_comb = os.path.join(out_folder, f'dwi_all_combined.bvec')
    combine_matrices(bvec_reg_names, out_path_bvec_comb)

    print('DEBUG: MERGING BVALS')
    if REG_WITHIN_OUTPUT_PATTERN:
        out_path_bval_comb = os.path.join(out_folder, f'{REG_WITHIN_OUTPUT_PATTERN}.bval')
    else:
        out_path_bval_comb = os.path.join(out_folder, f'dwi_all_combined.bval')
    combine_matrices(bval_input_names, out_path_bval_comb)
