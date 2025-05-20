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
#                         Common/Shared Settings                                #
###############################################################################

# ------------------------------
# Common/Shared Settings
# ------------------------------
# Input/Output directories
INPUT_DIR = "/simurgh/group/BWM/DataSets/OpenNeuro/raw/ds004215-download"
INPUT_SUBDIR = 'ses-01/dwi'
OUTPUT_DIR = "/simurgh/group/BWM/DataSets/OpenNeuro/processed/Structural/"
TEMP_DIR = "tmp"
LOG_DIR = "/simurgh/group/BWM/DataSets/OpenNeuro/processed/Structural/logs"
QC_DIR = os.path.join(OUTPUT_DIR, "QC")
NUM_SCANS_PER_SESSION = 2

# Processing settings
DICOM_TO_NIFTY = False
FORCE_REPROCESS = False


###############################################################################
#                         B0 field correction settings                        #
###############################################################################

B0_CORRECTION = 'Topup' # Topup, Fieldmap or None
SLICE_TO_SLICE_CORRECTION = True
BASELINE_SLICE_ORDER = None # Subject to use as template for correction if json file has no info
B0_CORRECTION_FOLDER = ''

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


EDDY_CORRECTION = True


