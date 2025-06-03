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

# Logging settings
ENABLE_DETAILED_LOGGING = True
LOG_DIR = "/simurgh/group/gustavochau/HCP-Dev-newprocessed/logs"

LOG_LEVEL = "DEBUG"             # DEBUG, INFO, WARNING, ERROR, CRITICAL
LOG_FORMAT = '%(asctime)s - %(name)s - %(levelname)s - %(message)s' if ENABLE_DETAILED_LOGGING else '%(message)s'

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


# ###############################################################################
# #                      Skull Stripping Settings                               #
# ###############################################################################

SKULL_STRIP_INPUT_FOLDER = B0_CORRECTION_FOLDER
SKULL_STRIP_OUTPUT_FOLDER = os.path.join(OUTPUT_DIR, "Skull_stripping")
SKULL_STRIP_INPUT_NAMES = None # List of names or if None, will assume output of topup
SKULL_STRIP_OUTPUT_PATTERN =  None # List of names or if None, will use "mask_bet_scan{scan_num}"
FRACTIONAL_INTENSITY = 0.4 # Parameter between 0 and 1 used by BET, smaller values give larger brain outline

# ------------------------------
# Skull Stripping QC Settings
# ------------------------------
# General QC settings
ENABLE_QC = True              # Generate quality control metrics and images
QC_GENERATE_IMAGES = True     # Generate QC images for visual inspection
QC_SUMMARY_FILE = "qc_summary.csv"  # Summary file for QC results

# Brain volume thresholds
QC_BRAIN_VOLUME_MIN_ML = 800   # Minimum brain volume in milliliters
QC_BRAIN_VOLUME_MAX_ML = 2000  # Maximum brain volume in milliliters

# QC visualization settings
QC_IMAGE_DPI = 150           # DPI for QC images
QC_IMAGE_FORMAT = "png"      # Format for QC images (png, jpg, pdf)
QC_SLICE_VIEWS = ["axial", "sagittal", "coronal"]  # Views to generate
QC_SLICE_NUMBERS = {         # Slice numbers for each view (None = middle slice)
    "axial": None,
    "sagittal": None,
    "coronal": None
}

###############################################################################
#                         Eddy correction settings                            #
###############################################################################
EDDY_CORRECTION_FOLDER = os.path.join(OUTPUT_DIR, "Eddy_correction")
SLICE_TO_SLICE_CORRECTION = True
BASELINE_SLICE_ORDER_JSON = None # Subject to use as template for correction if json file has no info
EDDY_CORRECTION_QC_SLICES = [17,40]

###############################################################################
#                         Registration within scans                           #
###############################################################################

REG_WITHIN_B0_INPUT_FOLDER = B0_CORRECTION_FOLDER
REG_WITHIN_INPUT_FOLDER = EDDY_CORRECTION_FOLDER
REG_WITHIN_OUTPUT_FOLDER = os.path.join(OUTPUT_DIR, "Reg_within_and_merged")
REG_WITHIN_INPUT_NAMES = None # List of names or if None, will assume output of eddy
REG_WITHIN_B0_INPUT_NAMES = None # List of names or if None, will assume output of Skull stripping step
REG_WITHIN_OUTPUT_PATTERN =  None # List of names or if None, will use "merged_dwi"