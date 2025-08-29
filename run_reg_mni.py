"""
Registration to MNI Space — run_reg_mni.py
==========================================

Registers diffusion images to the MNI152 template using a
two-step approach with FSL FLIRT (rigid + affine). Corresponding
b-vectors are rotated to preserve orientation
consistency. The pipeline also applies the transforms to the diffusion
volumes and masks, and prepares final outputs for downstream analysis.

Steps performed:
1. Register B0 to MNI (rigid, 6 DOF), save matrix and registered B0
3. Refine B0→MNI with affine (12 DOF), save matrix and registered B0
4. Apply rigid and then affine transforms to DWI
5. Rotate b-vectors using the previously computed transformations
6. Copy b-values (bval_final.bval) for consistency
7. Register mask with nearest-neighbour interpolation using the previously computed transformations

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
import shutil
from scipy.linalg import polar
from config import setup_fsl_env
from utilities import get_sessions

# Import configuration
from config import (
    TEMPLATE_PATH,
    REG_MNI_B0_INPUT_FOLDER,
    REG_MNI_INPUT_FOLDER,
    REG_MNI_OUTPUT_FOLDER,
    REG_MNI_INPUT_NAMES,
    REG_MNI_BVEC_INPUT_NAMES,
    REG_MNI_BVAL_INPUT_NAMES, 
    REG_MNI_B0_INPUT_NAMES,
    REG_MNI_OUTPUT_PATTERN,
    REG_MNI_MASK_INPUT_FOLDER,
    REG_MNI_MASK_NAMES
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

# def save_registration_report(results, output_dir):
#     """Save registration results to a JSON file."""
#     report_path = os.path.join(output_dir, 'registration_report.json')
#     with open(report_path, 'w') as f:
#         json.dump(results, f, indent=4)
#     print(f"\nDetailed registration report saved to: {report_path}")

# def format_time(seconds):
#     """Convert seconds to human-readable format."""
#     hours = int(seconds // 3600)
#     minutes = int((seconds % 3600) // 60)
#     seconds = int(seconds % 60)
#     if hours > 0:
#         return f"{hours}h {minutes}m {seconds}s"
#     elif minutes > 0:
#         return f"{minutes}m {seconds}s"
#     else:
#         return f"{seconds}s"

# def print_registration_summary(results):
#     """Print a summary of registration results."""
#     print("\n=== Registration Summary ===")
#     print(f"Total expected subjects: {results['total_subjects']}")
#     print(f"Successfully processed: {len(results['successful_subjects'])}")
#     print(f"Failed during processing: {len(results['failed_subjects'])}")
#     print(f"Missing input files: {len(results['missing_subjects'])}")
    
#     total_processed = len(results['successful_subjects']) + len(results['failed_subjects'])
    
#     if total_processed > 0:
#         success_rate = (len(results['successful_subjects']) / total_processed) * 100
#         print(f"Success rate: {success_rate:.2f}%")
#     elif results['total_subjects'] > 0:
#         total_issues = len(results['failed_subjects']) + len(results['missing_subjects'])
#         success_rate = (len(results['successful_subjects']) / results['total_subjects']) * 100
#         fail_rate = (total_issues / results['total_subjects']) * 100
#         print(f"Success rate: {success_rate:.2f}%")
#         print(f"Total fail rate: {fail_rate:.2f}%")
#     else:
#         print("No subjects to process found.")
    
#     if results['processing_times']:
#         avg_time = results['average_processing_time']
#         print(f"\nTiming Information:")
#         print(f"Average processing time per subject: {format_time(avg_time)}")
#         print("\nProcessing times per subject:")
#         for subj, t in results['processing_times'].items():
#             print(f"- Subject {subj}: {format_time(t)}")
    
#     if results['failed_subjects']:
#         print("\nFailed subjects (processing errors):")
#         for subject, error in results['errors'].items():
#             if subject in results['failed_subjects']:
#                 print(f"- {subject}: {error}")
    
#     if results['missing_subjects']:
#         print("\nMissing subjects (input files not found):")
#         for subject in results['missing_subjects']:
#             print(f"- {subject}")

def register_rigid(b0_path, mni_template_path, output_matrix, output_registered_path):
    flirt_cmd = [
        "flirt",
        "-in", b0_path,
        "-ref", mni_template_path,
        "-out", output_registered_path,
        "-omat", output_matrix,
        "-dof", "6",  # rigid (6)
    ]
    subprocess.run(flirt_cmd, check=True)

def register_affine(b0_path, mni_template_path, output_matrix, output_registered_path):
    flirt_cmd = [
        "flirt",
        "-in", b0_path,
        "-ref", mni_template_path,
        "-out", output_registered_path,
        "-omat", output_matrix,
        "-dof", "12",  # rigid (6)
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

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python run_reg_mni.py <subject_id>")
        sys.exit(1)
    else:
        subject_id = sys.argv[1]

    os.makedirs(REG_MNI_OUTPUT_FOLDER,exist_ok=True)

    setup_fsl_env()

    fixed = TEMPLATE_PATH

    sessions = get_sessions(os.path.join(REG_MNI_INPUT_FOLDER,subject_id))
    print(sessions)
    if not sessions:
        input_b0_subject_folders = [os.path.join(REG_MNI_B0_INPUT_FOLDER,subject_id)]
        input_subject_folders = [os.path.join(REG_MNI_INPUT_FOLDER,subject_id)]
        out_folders = [os.path.join(REG_MNI_OUTPUT_FOLDER,subject_id)]
        mask_folders = [os.path.join(REG_MNI_MASK_INPUT_FOLDER, subject_id)]
    else:
        input_b0_subject_folders = [os.path.join(REG_MNI_B0_INPUT_FOLDER,subject_id,sess) for sess in sessions]
        input_subject_folders = [os.path.join(REG_MNI_INPUT_FOLDER,subject_id,sess) for sess in sessions]
        out_folders = [os.path.join(REG_MNI_OUTPUT_FOLDER,subject_id,sess) for sess in sessions]
        mask_folders = [os.path.join(REG_MNI_MASK_INPUT_FOLDER, subject_id, sess) for sess in sessions]
    
    for input_b0_subject_folder, input_subject_folder, out_folder, input_mask_folder in zip(input_b0_subject_folders, input_subject_folders, out_folders, mask_folders):

        os.makedirs(out_folder,exist_ok=True)


        # B0 names
        if REG_MNI_B0_INPUT_NAMES:
            b0_input_name = os.path.join(input_b0_subject_folder,REG_MNI_B0_INPUT_NAMES)
        else:
            b0_input_name = os.path.join(input_b0_subject_folder,f'mask_bet_scan0.nii.gz')

        # Whole diffusion MRI 
        if REG_MNI_INPUT_NAMES:
            input_name = os.path.join(input_subject_folder,REG_MNI_INPUT_NAMES)
        else:
            input_name = os.path.join(input_subject_folder,f'dwi_all_combined.nii.gz')
        
        if REG_MNI_BVEC_INPUT_NAMES:
            bvec_input_names = os.path.join(input_subject_folder,REG_MNI_BVEC_INPUT_NAMES)
        else:
            bvec_input_names = os.path.join(input_subject_folder,f'dwi_all_combined.bvec')
        
        if REG_MNI_BVAL_INPUT_NAMES:
            bval_input_names = os.path.join(input_subject_folder, REG_MNI_BVAL_INPUT_NAMES)
        else:
            bval_input_names = os.path.join(input_subject_folder,f'dwi_all_combined.bval')


        mask_path =  os.path.join(input_mask_folder, REG_MNI_MASK_NAMES)
        

        # Register b0's using rigid transformation
        print(f"DEBUG: REGISTERING B0 IMAGES WITH RIGID TRANSFORMATION")
        moving = b0_input_name
        output_matrix_rigid = os.path.join(out_folder,'rigid_to_mni.mat')
        output_registered_path_rigid = os.path.join(out_folder,f'b0_reg_rigid.nii.gz')
        register_rigid(moving, fixed, output_matrix_rigid, output_registered_path_rigid)

        # Register b0's using affine transformation
        print(f"DEBUG: REGISTERING B0 IMAGES WITH AFFINE TRANSFORMATION")
        moving = output_registered_path_rigid
        output_matrix_affine = os.path.join(out_folder,'affine_to_mni.mat')
        output_registered_path_affine = os.path.join(out_folder,f'b0_reg_affine.nii.gz')
        register_affine(moving, fixed, output_matrix_affine, output_registered_path_affine)

        # Apply to DWI
        print(f"DEBUG: REGISTERING DWI IMAGES WITH RIGID TRANSFORMATION")
        moving = input_name
        out_path_dwi_reg_rigid = os.path.join(out_folder,f'dwi_reg_rigid.nii.gz')
        apply_transform_to_dwi(moving, fixed, output_matrix_rigid, out_path_dwi_reg_rigid)

        print(f"DEBUG: REGISTERING DWI IMAGES WITH AFFINE TRANSFORMATION")
        moving = out_path_dwi_reg_rigid
        out_path_dwi_reg_affine = os.path.join(out_folder,f'dwi_reg_affine.nii.gz')
        apply_transform_to_dwi(moving, fixed, output_matrix_affine, out_path_dwi_reg_affine)

        # Rotate bvecs
        print('DEBUG: REGISTERING BVECS WITH RIGID TRANSFORMATION')
        out_path_bvec_reg_rigid = os.path.join(out_folder,f'bvec_reg_rigid.bvec')
        rotate_bvecs(bvec_input_names, out_path_bvec_reg_rigid , output_matrix_rigid)

        print('DEBUG: REGISTERING BVECS WITH AFFINE TRANSFORMATION')
        out_path_bvec_reg_affine = os.path.join(out_folder,f'bvec_reg_affine.bvec')
        rotate_bvecs(out_path_bvec_reg_rigid, out_path_bvec_reg_affine , output_matrix_affine)
        # rotate_bvecs(bvec_input_names[i], bvec_input_names[i].replace('.bvec','_reg.bvec'), output_matrix)

        # Copy bval
        print('DEBUG: COPYING BVAL')
        out_path_bval = os.path.join(out_folder,f'bval_final.bval')
        shutil.copyfile(bval_input_names, out_path_bval)

        print('DEBUG: REGISTERING MASK')
        moving = mask_path
        out_path_mask_reg_rigid = os.path.join(out_folder,f'mask_reg_rigid.nii.gz')
        apply_transform_to_dwi(moving, fixed, output_matrix_rigid, out_path_mask_reg_rigid,'nearestneighbour')
        moving = out_path_mask_reg_rigid
        out_path_mask_reg_affine = os.path.join(out_folder,f'mask_reg_affine.nii.gz')
        apply_transform_to_dwi(moving, fixed, output_matrix_affine, out_path_mask_reg_affine,'nearestneighbour')
