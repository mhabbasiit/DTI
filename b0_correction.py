#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Wrapper for B0 offfield correction

Author: Gustavo Chau (gchau@stanford.edu)
"""


import config
import sys
import os
from config import (
    INPUT_DIR,
    B0_CORRECTION_FOLDER,
    INPUT_SUBDIR,
    B0_CORRECTION,
    NUM_SCANS_PER_SESSION
)
from process_topup import run_topup
from config import setup_fsl_env
from utilities import get_sessions


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python b0_correction.py <subject_id>")
        sys.exit(1)
    else:
        subject_id = sys.argv[1]

    if B0_CORRECTION is None:
        print("configured not to do b0 correction")
        sys.exit(1)
    if B0_CORRECTION not in ['Topup','Fieldmap']:
        print("B0 correction methods not supported")
        sys.exit(1)
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

    sessions = get_sessions(os.path.join(INPUT_DIR,subject_id))
    print(sessions)
    if not sessions:
        subject_folders = [os.path.join(INPUT_DIR,subject_id,INPUT_SUBDIR)]
    else:
        subject_folders = [os.path.join(INPUT_DIR,subject_id,sess,INPUT_SUBDIR) for sess in sessions]

    if not os.path.exists(B0_CORRECTION_FOLDER):
        os.makedirs(B0_CORRECTION_FOLDER, exist_ok=True)
    if not sessions:
        out_subject_folders = [os.path.join(B0_CORRECTION_FOLDER,subject_id)]
    else:
        out_subject_folders = [os.path.join(B0_CORRECTION_FOLDER,subject_id,sess) for sess in sessions]

    if B0_CORRECTION=='Topup':
        setup_fsl_env()
        
        blip_down_patterns = {}
        try:
            blip_down_patterns['dwi'] = config.REVERSED_DWI_FILE_PATTERNS
            blip_down_patterns['bval'] = config.REVERSED_BVAL_FILE_PATTERNS
            blip_down_patterns['bvec'] = config.REVERSED_BVEC_FILE_PATTERNS
            blip_down_patterns['json'] = config.REVERSED_JSON_FILE_PATTERNS
        except:
            print('Missing file pattern information for reversed polarity in config')
        print(blip_up_patterns)
        print(blip_down_patterns)
        for subject_folder, out_subject_folder in zip(subject_folders, out_subject_folders):
            print(subject_folder)
            print(out_subject_folder)
            os.makedirs(out_subject_folder, exist_ok=True)
            run_topup(subject_folder, out_subject_folder, blip_up_patterns, blip_down_patterns)
    elif B0_CORRECTION=='Fieldmap':
        # Not implemented yet
        pass
