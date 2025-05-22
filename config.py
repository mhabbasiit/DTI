#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Configuration file for neuroimaging pipeline
Contains settings for:
1. Common configurations (paths, logging, etc.)
2. Topup and Eddy correction
3. Skull stripping
4. DTI fit
5. Registration

Author: Gustavo Chau (gchau@stanford.edu)
"""

import os
import multiprocessing
import subprocess

###############################################################################
#                         FSL Settings                                #
###############################################################################
FSL_HOME = "/simurgh/u/gustavochau/fsl"
FSL_BIN = os.path.join(FSL_HOME,'share/fsl/bin')
# FreeSurfer environment variables
FSL_ENV = {
    "FSLOUTPUTTYPE": "NIFTI_GZ",
    "FSLDIR": FSL_HOME
}

def setup_fsl_env():
    """
    Set up FreeSurfer environment variables in the current Python process
    """
    os.environ.update(FSL_ENV)
    os.system(f". {FSL_HOME}/etc/fslconf/fsl.sh")



###############################################################################
#                         Common/Shared Settings                                #
###############################################################################

# ------------------------------
# Common/Shared Settings
# ------------------------------
# Input/Output directories
INPUT_DIR = "/simurgh/group/BWM/DTISherlock/hcpdev/imagingcollection01"
INPUT_SUBDIR = 'unprocessed/Diffusion'
OUTPUT_DIR = "/simurgh/group/gustavochau/HCP-Dev-newprocessed"
TEMP_DIR = "tmp"
QC_DIR = os.path.join(OUTPUT_DIR, "QC")
NUM_SCANS_PER_SESSION = 2

# Processing settings
DICOM_TO_NIFTY = False
FORCE_REPROCESS = False

###############################################################################
#                         B0 field correction settings                        #
###############################################################################

B0_CORRECTION_FOLDER = os.path.join(OUTPUT_DIR, "B0_correction")
B0_CORRECTION = 'Topup' # Topup, Fieldmap or None

# File patterns for input files. Length of lists should match NUM_SCANS_PER_SESSION
DWI_FILE_PATTERNS = ['*_dMRI_dir98_AP.nii.gz', '*_dMRI_dir99_AP.nii.gz']
BVAL_FILE_PATTERNS = ['*_dMRI_dir98_AP.bval', '*_dMRI_dir99_AP.bval']
BVEC_FILE_PATTERNS = ['*_dMRI_dir98_AP.bvec', '*_dMRI_dir99_AP.bvec']
JSON_FILE_PATTERNS = ['*_dMRI_dir98_AP.json', '*_dMRI_dir99_AP.json']

# File patterns for Reversed polarity input files. Length of lists should match NUM_SCANS_PER_SESSION
REVERSED_DWI_FILE_PATTERNS = ['*_dMRI_dir98_PA.nii.gz', '*_dMRI_dir99_PA.nii.gz']
REVERSED_BVAL_FILE_PATTERNS = ['*_dMRI_dir98_PA.bval', '*_dMRI_dir99_PA.bval']
REVERSED_BVEC_FILE_PATTERNS = ['*_dMRI_dir98_PA.bvec', '*_dMRI_dir99_PA.bvec']
REVERSED_JSON_FILE_PATTERNS = ['*_dMRI_dir98_PA.json', '*_dMRI_dir99_PA.json']
B0_CORRECTION_QC_SLICES = [17,40]

###############################################################################
#                         Eddy correction settings                            #
###############################################################################

EDDY_CORRECTION = True
SLICE_TO_SLICE_CORRECTION = True
BASELINE_SLICE_ORDER = None # Subject to use as template for correction if json file has no info


# ###############################################################################
# #                      Skull Stripping Settings                                 #
# ###############################################################################

# # ------------------------------
# # FreeSurfer Configuration
# # ------------------------------
# FREESURFER_HOME = "/scr/mabbasi/freesurfer_7.4.1/freesurfer"
# FREESURFER_SUBJECTS_DIR = os.path.join(OUTPUT_DIR, "freesurfer_subjects")
# FREESURFER_LICENSE_FILE = os.path.join(FREESURFER_HOME, "license.txt")
# FREESURFER_THREADS = 8  # Number of threads for FreeSurfer processes
# USE_FREESURFER = True   # Whether to use FreeSurfer for brain extraction

# # FreeSurfer binary paths
# FS_BIN_DIR = os.path.join(FREESURFER_HOME, "bin")
# MRICONV_BIN = os.path.join(FS_BIN_DIR, "mri_convert")
# RECON_ALL_BIN = os.path.join(FS_BIN_DIR, "recon-all")
# SYNTHSTRIP_BIN = os.path.join(FS_BIN_DIR, "mri_synthstrip")

# # FreeSurfer environment variables
# FREESURFER_ENV = {
#     "FREESURFER_HOME": FREESURFER_HOME,
#     "SUBJECTS_DIR": FREESURFER_SUBJECTS_DIR,
#     "PATH": f"{FS_BIN_DIR}:$PATH",
#     "FS_LICENSE": FREESURFER_LICENSE_FILE,
#     "FSFAST_HOME": os.path.join(FREESURFER_HOME, "fsfast"),
#     "FSF_OUTPUT_FORMAT": "nii.gz",
#     "MNI_DIR": os.path.join(FREESURFER_HOME, "mni"),
#     "FSL_DIR": os.path.join(FREESURFER_HOME, "fsl"),
#     "MINC_BIN_DIR": os.path.join(FREESURFER_HOME, "mni", "bin"),
#     "MINC_LIB_DIR": os.path.join(FREESURFER_HOME, "mni", "lib"),
#     "MNI_DATAPATH": os.path.join(FREESURFER_HOME, "mni", "data"),
#     "LOCAL_DIR": os.path.join(FREESURFER_HOME, "local"),
#     "PERL5LIB": os.path.join(FREESURFER_HOME, "perl", "lib")
# }

# def setup_freesurfer_env():
#     """
#     Set up FreeSurfer environment variables in the current Python process
#     """
#     os.environ.update(FREESURFER_ENV)
    
# def generate_freesurfer_setup_script(output_path="setup_freesurfer_auto.sh"):
#     """
#     Generate a bash script to set up FreeSurfer environment
#     """
#     script_content = [
#         "#!/bin/bash",
#         "",
#         "# This file is automatically generated by config.py",
#         "# Do not edit manually",
#         "",
#         f"# FreeSurfer {os.path.basename(FREESURFER_HOME)} Setup Script",
#         "",
#     ]
    
#     # Add environment variables
#     for key, value in FREESURFER_ENV.items():
#         script_content.append(f"export {key}={value}")
    
#     # Add verification steps
#     script_content.extend([
#         "",
#         "# Verify FreeSurfer setup",
#         "if [ ! -d \"$FREESURFER_HOME\" ]; then",
#         "    echo 'ERROR: FREESURFER_HOME directory not found'",
#         "    exit 1",
#         "fi",
#         "",
#         "if [ ! -f \"$FS_LICENSE\" ]; then",
#         "    echo 'ERROR: FreeSurfer license file not found'",
#         "    exit 1",
#         "fi",
#         "",
#         "# Test FreeSurfer installation",
#         "if command -v mri_convert >/dev/null 2>&1; then",
#         "    echo 'FreeSurfer environment has been set up successfully'",
#         "    echo 'FREESURFER_HOME: $FREESURFER_HOME'",
#         "    echo 'SUBJECTS_DIR: $SUBJECTS_DIR'",
#         "else",
#         "    echo 'ERROR: FreeSurfer commands not found in PATH'",
#         "    exit 1",
#         "fi"
#     ])
    
#     # Write script to file
#     with open(output_path, "w") as f:
#         f.write("\n".join(script_content))
    
#     # Make script executable
#     os.chmod(output_path, 0o755)
#     return output_path

# # ------------------------------
# # SynthStrip Configuration
# # ------------------------------
# SYNTHSTRIP_DOCKER_IMAGE = "freesurfer/synthstrip:latest"
# SYNTHSTRIP_PARAMS = {
#     "mask": True,        # Generate binary mask
#     "no-csf": True,      # Exclude CSF from the mask
#     "border": 1          # Border around the brain in mm
# }

# # ------------------------------
# # Skull Stripping QC Settings
# # ------------------------------
# # General QC settings
# ENABLE_QC = True              # Generate quality control metrics and images
# QC_GENERATE_IMAGES = True     # Generate QC images for visual inspection
# QC_SUMMARY_FILE = "qc_summary.csv"  # Summary file for QC results

# # Brain volume thresholds
# QC_BRAIN_VOLUME_MIN_ML = 800   # Minimum brain volume in milliliters
# QC_BRAIN_VOLUME_MAX_ML = 2000  # Maximum brain volume in milliliters

# # QC visualization settings
# QC_IMAGE_DPI = 150           # DPI for QC images
# QC_IMAGE_FORMAT = "png"      # Format for QC images (png, jpg, pdf)
# QC_SLICE_VIEWS = ["axial", "sagittal", "coronal"]  # Views to generate
# QC_SLICE_NUMBERS = {         # Slice numbers for each view (None = middle slice)
#     "axial": None,
#     "sagittal": None,
#     "coronal": None
# }