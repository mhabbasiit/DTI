import os
import logging
from datetime import datetime
import argparse  # Add for command line argument parsing
import multiprocessing  # For parallel processing
from concurrent.futures import ProcessPoolExecutor  # For parallel processing
import psutil  # For system resource detection
import fnmatch
import pandas as pd
import re
import nibabel as nib
import glob
import numpy as np
import nibabel as nib
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import sys
from scipy.spatial.transform import Rotation
from scipy import linalg
from utilities import find_file
from utilities import get_sessions

# Import configuration
from config import (
    QC_DIR,
    LOG_DIR,
    TEMPLATE_PATH,
    SKULL_STRIP_OUTPUT_FOLDER,
    REG_MNI_OUTPUT_FOLDER,
    MODALITY_PATTERNS,
    REG_MNI_OUTPUT_FOLDER,
    NUM_SCANS_PER_SESSION
)

if NUM_SCANS_PER_SESSION>1:
    from config import (
    REG_WITHIN_B0_INPUT_FOLDER,
    REG_WITHIN_B0_INPUT_NAMES,
    REG_WITHIN_OUTPUT_FOLDER)

# Logging setup
log_dir = LOG_DIR  # Use LOG_DIR from config.py
os.makedirs(log_dir, exist_ok=True)
log_file = os.path.join(log_dir, f"qc_log_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log")

# Set up logging for both main process and worker processes
def setup_logging(log_file):
    # Remove all handlers associated with the root logger object
    for handler in logging.root.handlers[:]:
        logging.root.removeHandler(handler)

    # Create a lock for file access
    file_handler = logging.FileHandler(log_file)
    file_handler.setLevel(logging.INFO)
    
    # Create a stream handler for console output
    stream_handler = logging.StreamHandler()
    stream_handler.setLevel(logging.INFO)
    
    # Set the format
    formatter = logging.Formatter('[%(asctime)s] %(levelname)s - %(message)s', datefmt='%Y-%m-%d %H:%M:%S')
    file_handler.setFormatter(formatter)
    stream_handler.setFormatter(formatter)
    
    # Configure the root logger
    logging.root.setLevel(logging.INFO)
    logging.root.addHandler(file_handler)
    logging.root.addHandler(stream_handler)

# Initialize logging for the main process
setup_logging(log_file)
logging.info("QC script started")
logging.info(f"Logging to: {log_file}")

# Author: Mohammad Abbasi (mabbasi@stanford.edu)
# Created: 2025

# ---------------
# Configuration
# ---------------
qc_output_dir   = QC_DIR
skullstrip_dir  = SKULL_STRIP_OUTPUT_FOLDER
os.makedirs(qc_output_dir, exist_ok=True)

# Now that directories are defined, log them
logging.info(f"Working directories:")
logging.info(f"  QC output directory: {qc_output_dir}")
logging.info(f"  Skullstrip directory: {skullstrip_dir}")
logging.info(f"  Using REG_PATH: {REG_MNI_OUTPUT_FOLDER}")

# QC thresholds
TRANSLATION_PASS = 3.0    # ≤3 mm ⇒ Passed
TRANSLATION_WARN = 7.0    # 3–7 mm ⇒ Warning
                          # >7 mm ⇒ Failed
ROTATION_PASS = 1.5       # ≤1.5° ⇒ Passed
ROTATION_WARN = 3.0       # 1.5–3° ⇒ Warning
                          # >3° ⇒ Failed

DICE_PASS = 0.8
DICE_WARN = 0.7

# -----------------------------------------------------------------------------
# Helper: find all sessions for a subject - FIXED for AIBL dataset structure
# -----------------------------------------------------------------------------
def find_subject_sessions(subj_dir, derivatives_dir, modality='Diffusion'):
    """
    Find all available sessions for a given subject.
    For AIBL dataset: session = timestamp (e.g., 2008-07-07_14_10_46.0)
    Uses MODALITY_PATTERNS from config to find ALL T1 modality directories.
    
    Parameters:
    -----------
    subj_dir : str
        Subject directory path
    modality : str
        'T1' or 'T2'
        
    Returns:
    --------
    list of str
        List of session names (timestamps) for the subject
    """
    import fnmatch
    
    def matches_modality_pattern(dir_name, modality):
        """Check if directory name matches any pattern for the given modality"""
        patterns = MODALITY_PATTERNS.get(modality, [])
        return any(fnmatch.fnmatch(dir_name.upper(), pattern.upper()) for pattern in patterns)
    
    sessions = set()  # Use set to avoid duplicates
    
    # Extract subject ID from directory path
    #subj_id = os.path.basename(subj_dir)
    subj_id = subj_dir
    
    # STEP 1: Check skullstrip directory for ALL T1 modality directories
    aibl_skullstrip_path = os.path.join(skullstrip_dir, subj_id)
    if os.path.exists(aibl_skullstrip_path):
        # List all directories in subject directory
        all_dirs = [d for d in os.listdir(aibl_skullstrip_path) 
                   if os.path.isdir(os.path.join(aibl_skullstrip_path, d))]
        
        # Filter directories that match T1 modality patterns
        t1_modality_dirs = [d for d in all_dirs if matches_modality_pattern(d, 'T1')]
        
        logging.info(f"[DEBUG] Found T1 modality directories in skullstrip for {subj_id}: {t1_modality_dirs}")
        
        # Check each T1 modality directory for timestamp sessions
        for mod_dir in t1_modality_dirs:
            mod_path = os.path.join(aibl_skullstrip_path, mod_dir)
            if os.path.exists(mod_path):
                # List all timestamp directories in this modality
                timestamp_dirs = [d for d in os.listdir(mod_path) 
                                 if os.path.isdir(os.path.join(mod_path, d)) and 
                                 ('_' in d and '.' in d)]  # timestamp pattern: YYYY-MM-DD_HH_MM_SS.S
                
                # Add timestamp sessions
                for timestamp in timestamp_dirs:
                    sessions.add(timestamp)
                    logging.info(f"[DEBUG] Found session {timestamp} in T1 modality {mod_dir}")
    
    # STEP 2: Check registration directory for ALL T1 modality directories
    reg_path = os.path.join(derivatives_dir, subj_id)
    if os.path.exists(reg_path):
        # List all directories that could be modality-based
        all_reg_dirs = [d for d in os.listdir(reg_path) 
                       if os.path.isdir(os.path.join(reg_path, d))]
        
        # Filter directories that match T1 modality patterns  
        t1_reg_modality_dirs = [d for d in all_reg_dirs if matches_modality_pattern(d, 'T1')]
        
        logging.info(f"[DEBUG] Found T1 modality directories in registration for {subj_id}: {t1_reg_modality_dirs}")
        
        # For each T1 modality directory, look for timestamp subdirectories
        for mod_dir in t1_reg_modality_dirs:
            mod_reg_path = os.path.join(reg_path, mod_dir)
            if os.path.exists(mod_reg_path):
                timestamp_dirs = [d for d in os.listdir(mod_reg_path) 
                                 if os.path.isdir(os.path.join(mod_reg_path, d)) and 
                                 ('_' in d and '.' in d)]  # timestamp pattern
                
                # Add any timestamp sessions not already found
                for timestamp in timestamp_dirs:
                    if timestamp not in sessions:
                        sessions.add(timestamp)
                        logging.info(f"[DEBUG] Found additional session {timestamp} in registration T1 modality {mod_dir}")
    
    # Convert set back to sorted list
    sessions = sorted(list(sessions))
    
    if not sessions:
        logging.warning(f"No sessions found for subject {subj_id}")
        return []
    
    logging.info(f"Total sessions found for {subj_id}: {len(sessions)}")
    for session in sessions:
        logging.info(f"  - {session}")
    
    return sessions



# -----------------------------------------------------------------------------
# 1) File existence checks with improved pattern matching - UPDATED WITH MODALITY INFO
# -----------------------------------------------------------------------------
def check_file_existence(subject_dirs, derivatives_dir):
    records = []
    for subj in subject_dirs:
        sid = subj #os.path.basename(subj)
        logging.info(f"[QC] Checking existence for subject {sid}")
        
        # Find all sessions (timestamps) for this subject
        sessions = find_subject_sessions(subj, derivatives_dir)
        
        # If no sessions found, log a warning and skip
        if not sessions:
            logging.warning(f"No sessions found for subject {sid}, skipping")
            sessions = ['']
            
        
        # Process each session (timestamp)
        for session_timestamp in sessions:
            session_id = session_timestamp  # Use timestamp as session_id
            logging.info(f"[QC] Checking session {session_id} for subject {sid}")
            
            # FOR AIBL: Find ALL T1 modality directories for this session
            # instead of just one modality directory
            base_subject_path = os.path.join(skullstrip_dir, sid)
            t1_modality_dirs = []
            
            if os.path.exists(base_subject_path):
                all_dirs = [d for d in os.listdir(base_subject_path) 
                           if os.path.isdir(os.path.join(base_subject_path, d))]
                
                # Filter directories that match T1 modality patterns and have this session
                import fnmatch
                def matches_modality_pattern(dir_name, modality):
                    """Check if directory name matches any pattern for the given modality"""
                    patterns = MODALITY_PATTERNS.get(modality, [])
                    return any(fnmatch.fnmatch(dir_name.upper(), pattern.upper()) for pattern in patterns)
                
                for mod_dir in all_dirs:
                    if matches_modality_pattern(mod_dir, 'T1'):
                        session_path = os.path.join(base_subject_path, mod_dir, session_timestamp)
                        if os.path.exists(session_path):
                            t1_modality_dirs.append(mod_dir)
                            logging.info(f"[QC] Found T1 modality {mod_dir} for session {session_timestamp}")
            
            modality_name = 'Diffusion'

            # Create a separate record for EACH T1 modality directory
            logging.info(f"[QC] Processing modality {modality_name} for subject {sid} session {session_id}")
            
            # Create unique session_id that includes modality to avoid conflicts
            unique_session_id = f"{session_id}_{modality_name}"
            
            # Use the helper function to determine correct paths for this specific modality
            skullstrip_t1_dir = os.path.join(skullstrip_dir, sid, session_timestamp)

            # Find T2 directory (should be the same for all T1 modalities in a session)
            skullstrip_t2_dir = None
            reg_base = os.path.join(derivatives_dir, sid)
            if os.path.exists(base_subject_path):
                all_dirs = [d for d in os.listdir(base_subject_path) 
                            if os.path.isdir(os.path.join(base_subject_path, d))]
                for mod_dir in all_dirs:
                    if matches_modality_pattern(mod_dir, 'T2'):
                        session_path = os.path.join(base_subject_path, mod_dir, session_timestamp)
                        if os.path.exists(session_path):
                            skullstrip_t2_dir = session_path
                            break
            
            # Registration directory for this specific modality
            registration_dir = os.path.join(derivatives_dir, sid, session_timestamp)
            
            # Create session-specific record with unique identifier
            record = {
                'subject_id': sid,
                'session_id': unique_session_id,  # Use unique ID to separate modalities
                'session': session_timestamp,
                'modality_name': modality_name,  # Add modality directory name
                'skull_ok': False  # Initialize skull_ok
            }
            
            # Use improved T2 detection
            record['t2_expected'] = False
            
            # More flexible patterns with fallbacks
            pats = {
                # Main registration outputs - required
                'T1_rigid':  ["*dwi_reg_rigid.nii.gz", "*dwi*reg*rigid.nii.gz"],
                'T1_affine': ["*dwi_reg_affine.nii.gz", "*dwi*reg*affine.nii.gz"],
                
                # Transformation matrices - required
                'T1_rigid_mat':     ["*rigid*.mat"],
                'T1_affine_mat':    ["*affine*.mat"],
                
            }
            
            # Try to find files using the patterns
            ex = {}
            for k, patterns in pats.items():
                # Choose appropriate directory based on file type
                if k.startswith('T1_') and 'rigid' not in k and 'affine' not in k:
                    search_dir = skullstrip_t1_dir
                elif k.startswith('T2_') and 'rigid' not in k and 'affine' not in k:
                    search_dir = skullstrip_t2_dir
                else:
                    search_dir = registration_dir
                
                # Special case for warped and cropped files - check both registration and skullstrip dirs
                if '_warped' in k or '_cropped' in k:
                    # Search in multiple directories based on modality
                    if k.startswith('T1_'):
                        search_dirs = [registration_dir, skullstrip_t1_dir]
                    else:  # T2
                        search_dirs = [registration_dir, skullstrip_t2_dir] if skullstrip_t2_dir else [registration_dir]
                    
                    found = find_files_multi_dirs(search_dirs, patterns, verbose=False)
                    if found:
                        ex[k] = True
                        logging.info(f"  Found {k}: {os.path.basename(found)} for {modality_name} session {session_id}")
                    else:
                        ex[k] = False
                        logging.warning(f"  Could not find {k} for subject {sid} {modality_name} session {session_id}")
                else:
                    # Regular search for non-warped/cropped files
                    found = None
                    for p in patterns:
                        found = find_file(search_dir, p, session=None, verbose=False)
                        if found:
                            logging.info(f"  Found {k}: {os.path.basename(found)} for {modality_name} session {session_id}")
                            break
                
                    ex[k] = bool(found)

            # Check skull strip QC images - search in session-specific directories
            t1_qc_files = []
            
            # Search for T1 QC files in T1 modality directory
            if os.path.exists(skullstrip_t1_dir):
                t1_qc_files = glob.glob(os.path.join(skullstrip_t1_dir, "*desc-qc.png"))
                t1_qc_files += glob.glob(os.path.join(skullstrip_t1_dir, "*brain*desc-qc.png"))
            
            logging.info(t1_qc_files)
            skull_t1_qc = t1_qc_files[0] if t1_qc_files else None
            
            if skull_t1_qc:
                logging.info(f"  Found T1 skull QC image: {os.path.basename(skull_t1_qc)} for {modality_name}")

            ex['T1_skull_qc'] = bool(skull_t1_qc)

            # Improved criteria for determining if required files exist
            # Basic T1 requirements - always needed
            t1_rigid_ok = ex.get('T1_rigid_exists', False) and ex.get('T1_rigid_mat_exists', False)
            t1_affine_ok = ex.get('T1_affine_exists', False) and ex.get('T1_affine_mat_exists', False)
            t1_cropped_ok = ex.get('T1_cropped', False) or ex.get('T1_rigid_cropped', False)
            
            # Check if the skull strip QC images exist
            t1_skull_ok = ex.get('T1_skull_qc', False)

            skull_ok = t1_skull_ok
            all_rigid = t1_rigid_ok
            all_affine = t1_affine_ok
            all_cropped = t1_cropped_ok
                
            # Count missing files based on what's expected (only T1 files are counted as critical)
            t1_items = [k for k in ex.keys() if k.startswith('T1_')]
            missing = sum(not ex.get(k, False) for k in t1_items)

            # Add existence flags to record
            record.update({f"{k}_exists": v for k, v in ex.items()})
            record.update({
                'all_rigid': all_rigid,
                'all_affine': all_affine,
                'all_cropped': all_cropped,
                'skull_ok': skull_ok,
                'missing_count': missing
            })
            
            records.append(record)
    
    return pd.DataFrame(records)

# -----------------------------------------------------------------------------
# 2.5) Analyze Registration Matrices - CORRECTED VERSION
# -----------------------------------------------------------------------------
def analyze_registration_matrices(subject_dirs, derivatives_dir, modality, mni_template, list_patterns):
    registration_results = []
    for subj in subject_dirs:
        sid = subj
        logging.info(f"[QC] Analyzing registration matrices for subject {sid}")
        
        # Find all sessions for this subject
        sessions = None #find_subject_sessions(subj)
        
        # If no sessions found, log a warning and create a single record
        if not sessions:
            logging.warning(f"No sessions found for subject {sid}, creating a single record")
            sessions = [None]  # Use None to indicate "no specific session"
        
        # Process each session - but now we need to handle multiple modalities per session
        for session in sessions:
            session_id = session.replace('ses-', '') if session else 'unknown'
            logging.info(f"[QC] Analyzing session {session_id} for subject {sid}")
            
            # FOR AIBL: Find ALL T1 modality directories for this session
            base_subject_path = os.path.join(skullstrip_dir, sid)
            t1_modality_dirs = []
            
            if os.path.exists(base_subject_path):
                all_dirs = [d for d in os.listdir(base_subject_path) 
                           if os.path.isdir(os.path.join(base_subject_path, d))]
                
                # Filter directories that match T1 modality patterns and have this session
                import fnmatch
                def matches_modality_pattern(dir_name, modality):
                    """Check if directory name matches any pattern for the given modality"""
                    patterns = MODALITY_PATTERNS.get(modality, [])
                    return any(fnmatch.fnmatch(dir_name.upper(), pattern.upper()) for pattern in patterns)
                
                for mod_dir in all_dirs:
                    if matches_modality_pattern(mod_dir, 'T1'):
                        session_path = os.path.join(base_subject_path, mod_dir, session)
                        if os.path.exists(session_path):
                            t1_modality_dirs.append(mod_dir)
                            logging.info(f"[QC] Found T1 modality {mod_dir} for session {session}")
            
            modality_name = 'Diffusion'
            # Process each T1 modality separately
            logging.info(f"[QC] Analyzing registration for modality {modality_name} subject {sid} session {session_id}")
            
            # Create unique session_id that matches the one used in check_file_existence
            unique_session_id = f"{session_id}_{modality_name}"
            
            # Use the helper function to determine correct paths for this specific modality
            if session:
                registration_dir = os.path.join(derivatives_dir, sid, session)
            else:
                registration_dir = os.path.join(derivatives_dir, sid)
            
            # Create session-specific record
            result = {
                'subject_id': sid,
                'session_id': unique_session_id,  # Use same unique ID as in file existence check
                'session': session,
                'T1w_rigid_status': 'N/A',
                'T1w_rigid_dice': None,
                # 'T1w_rigid_rotation': None,
                # 'T1w_rigid_rotation_status': 'N/A',
                # 'T1w_rigid_rotation_needed': False,
                'T1w_affine_status': 'N/A',
                'T1w_affine_dice': None,
                # 'T1w_affine_rotation': None,
                # 'T1w_affine_rotation_needed': False
            }
            
            # Debug directory contents
            if os.path.exists(registration_dir):
                logging.info(f"[QC] Registration directory exists: {registration_dir}")
                try:
                    files = os.listdir(registration_dir)
                    logging.info(f"[QC] Files in registration directory: {len(files)} files")
                    for f in files[:10]:  # Show only first 10 files
                        logging.info(f"  - {f}")
                    if len(files) > 10:
                        logging.info(f"  ... and {len(files) - 10} more files")
                except Exception as e:
                    logging.error(f"[QC] Error listing files: {e}")
            else:
                logging.warning(f"[QC] Registration directory does not exist: {registration_dir}")
            
            # Load MNI template for comparison - THIS IS THE KEY FIX
            mni_template_path = None
            if mni_template:
                mni_template_path = str(mni_template)
                logging.info(f"[QC] Using MNI template: {mni_template_path}")
            else:
                logging.warning(f"[QC] No MNI template found, skipping template comparison for {sid} {modality_name}")
            
            # Check both rigid and affine registration
            for tag, warped_patterns in list_patterns:
                logging.info(f"[QC] {tag.capitalize()} for {sid} {modality_name} session {session_id}")
                
                # Try multiple patterns for warped image
                img_warped = None
                for pattern in warped_patterns:
                    found = find_file(registration_dir, pattern)
                    if found:
                        img_warped = found
                        logging.info(f"[QC] Found warped image using pattern '{pattern}': {os.path.basename(img_warped)}")
                        break
                
                # Print paths for debugging
                logging.info(f"[QC] MNI template file: {mni_template_path}")
                logging.info(f"[QC] {tag.capitalize()} warped file: {img_warped}")
                
                # CORRECTED COMPARISON: Compare warped image with MNI template
                if mni_template_path and img_warped and os.path.exists(mni_template_path) and os.path.exists(img_warped):
                    try:
                        # Load MNI template and warped image
                        img_template = nib.load(mni_template_path)
                        img_warped_nib = nib.load(img_warped)
                        
                        data_template = img_template.get_fdata()
                        data_warped = img_warped_nib.get_fdata()
                        
                        logging.info(f"[QC] Template shape: {data_template.shape}")
                        logging.info(f"[QC] Warped image shape: {data_warped.shape}")
                        
                        # Check if images have compatible shapes for comparison
                        if data_template.shape != data_warped.shape:
                            logging.warning(f"[QC] Shape mismatch between template {data_template.shape} and warped {data_warped.shape}")
                            # Try to resample or crop to match
                            # For now, we'll use affine comparison only
                            
                            # Compare affine matrices - both should be in MNI space
                            affine_template = img_template.affine
                            affine_warped = img_warped_nib.affine
                            
                            # Extract translation from difference in affine origins
                            trans_vec = affine_warped[:3, 3] - affine_template[:3, 3]
                            trans = np.linalg.norm(trans_vec)
                            
                            logging.info(f"[QC] Template affine origin: {affine_template[:3, 3]}")
                            logging.info(f"[QC] Warped affine origin: {affine_warped[:3, 3]}")
                            logging.info(f"[QC] Translation difference: {trans:.2f}mm")
                            
                        else:
                            # Images have same shape - can do full comparison
                            # Create masks - try multiple thresholds if needed
                            mask_template = data_template > (data_template.mean() * 0.1)  # Lower threshold for template
                            mask_warped = data_warped > (data_warped.mean() * 0.1)
                            
                            # If masks are too small, try even lower thresholds
                            if mask_template.sum() < 1000:
                                logging.info(f"[QC] Template mask too small, using very low threshold")
                                mask_template = data_template > (data_template.max() * 0.01)
                            if mask_warped.sum() < 1000:
                                logging.info(f"[QC] Warped mask too small, using very low threshold")
                                mask_warped = data_warped > (data_warped.max() * 0.01)
                            
                            # Debug: Print mask sizes
                            logging.info(f"[QC] Template mask: {mask_template.sum()} voxels ({mask_template.sum()/mask_template.size*100:.1f}%)")
                            logging.info(f"[QC] Warped mask: {mask_warped.sum()} voxels ({mask_warped.sum()/mask_warped.size*100:.1f}%)")
                            
                            if mask_template.sum() > 100 and mask_warped.sum() > 100:
                                # Both images should be in MNI space - use centroid comparison
                                ijk_template = np.array(np.where(mask_template)).T.mean(axis=0)
                                ijk_warped = np.array(np.where(mask_warped)).T.mean(axis=0)
                                
                                # Convert to mm coordinates
                                mm_template = nib.affines.apply_affine(img_template.affine, ijk_template)
                                mm_warped = nib.affines.apply_affine(img_warped_nib.affine, ijk_warped)
                                
                                # Calculate translation distance
                                trans = np.linalg.norm(mm_warped - mm_template)
                                
                                # Also calculate overlap metrics for additional validation
                                intersection = np.logical_and(mask_template, mask_warped)
                                union = np.logical_or(mask_template, mask_warped)
                                dice = 2.0 * intersection.sum() / (mask_template.sum() + mask_warped.sum()) if (mask_template.sum() + mask_warped.sum()) > 0 else 0
                                jaccard = intersection.sum() / union.sum() if union.sum() > 0 else 0
                                
                                logging.info(f"[QC] Template centroid: {mm_template}")
                                logging.info(f"[QC] Warped centroid: {mm_warped}")
                                logging.info(f"[QC] Translation distance: {trans:.2f}mm")
                                logging.info(f"[QC] Dice coefficient: {dice:.3f}")
                                logging.info(f"[QC] Jaccard index: {jaccard:.3f}")
                                
                                # If overlap is very poor, the centroids might not be meaningful
                                if dice < 0.3:
                                    logging.warning(f"[QC] Poor overlap (Dice={dice:.3f}), translation metric may be unreliable")
                                    trans = 15.0  # Assign poor quality score
                                
                            else:
                                logging.warning(f"[QC] Insufficient mask coverage for meaningful comparison")
                                # Fall back to affine matrix comparison
                                affine_template = img_template.affine
                                affine_warped = img_warped_nib.affine
                                trans_vec = affine_warped[:3, 3] - affine_template[:3, 3]
                                trans = np.linalg.norm(trans_vec)
                                logging.info(f"[QC] Using affine-based translation: {trans:.2f}mm")
                        
                        # Extract rotation from the warped image's affine matrix
                        R = img_warped_nib.affine[:3,:3]
                        scale = np.cbrt(abs(np.linalg.det(R)))
                        Rn = R/scale if scale!=0 else R
                        
                        # For rigid registration, expect minimal rotation from canonical orientation
                        # For affine, check the rotation component after polar decomposition
                        U = Rn if tag=='rigid' else linalg.polar(Rn)[0]
                        
                        try:
                            # Calculate angles relative to identity (canonical orientation)
                            identity_diff = U @ np.linalg.inv(np.eye(3))
                            angles = Rotation.from_matrix(identity_diff).as_euler('xyz', degrees=True)
                            rot = np.linalg.norm(angles)
                        except Exception as e:
                            logging.warning(f"[QC] Error calculating rotation, using simpler method: {e}")
                            # Simpler rotation calculation
                            try:
                                angles = Rotation.from_matrix(U).as_euler('xyz', degrees=True)
                                rot = np.linalg.norm(angles)
                            except:
                                rot = np.nan
                        
                        if dice >= DICE_PASS:
                            status = 'Passed'
                        elif dice>= DICE_WARN:
                            status = 'Warning'
                        else:
                            status = 'Failed'
                        # # Apply QC thresholds
                        # if trans <= TRANSLATION_PASS:
                        #     status = 'Passed'
                        # elif trans <= TRANSLATION_WARN:
                        #     status = 'Warning'
                        # else:
                        #     status = 'Failed'
                        
                        # # Using thresholds for rotation
                        # if not np.isnan(rot):
                        #     if rot <= ROTATION_PASS:
                        #         rot_status = 'Passed'
                        #     elif rot <= ROTATION_WARN:
                        #         rot_status = 'Warning'
                        #     else:
                        #         rot_status = 'Failed'
                        # else:
                        #     rot_status = 'N/A'
                        
                        # need = rot > ROTATION_PASS if not np.isnan(rot) else False
                        
                        # result[f'T1w_{tag}_translation'] = float(trans)
                        # result[f'T1w_{tag}_rotation']    = float(rot) if not np.isnan(rot) else None
                        # result[f'T1w_{tag}_rotation_status'] = rot_status
                        # result[f'T1w_{tag}_rotation_needed'] = need
                        result[f'T1w_{tag}_dice'] = float(dice)
                        result[f'T1w_{tag}_status']      = status
                        
                        # Print results
                        logging.info(f"[QC] {tag.capitalize()} registration quality (vs MNI template):")
                        logging.info(f"[QC] {tag.capitalize()} translation: {trans:.2f}mm ({status})")
                        # if not np.isnan(rot):
                        #     logging.info(f"[QC] {tag.capitalize()} rotation: {rot:.2f}° ({rot_status}, needed: {need})")
                        # else:
                        #     logging.info(f"[QC] {tag.capitalize()} rotation: N/A (could not calculate)")
                        
                    except Exception as e:
                        logging.error(f"[QC] Error processing template comparison for {sid} {modality_name} session {session_id}: {e}")
                        #result[f'T1w_{tag}_translation'] = None
                        result[f'T1w_{tag}_dice']    = None
                        result[f'T1w_{tag}_status']      = f'Error: {str(e)[:50]}...'
                        
                elif not mni_template_path:
                    logging.warning(f"[QC] No MNI template available for comparison")
                    #result[f'T1w_{tag}_translation'] = None
                    result[f'T1w_{tag}_dice']    = None
                    result[f'T1w_{tag}_status']      = 'N/A (No template)'
                    
                elif not img_warped:
                    logging.warning(f"[QC] No {tag} warped image found for {sid} {modality_name} session {session_id}")
                    #result[f'T1w_{tag}_translation'] = None
                    result[f'T1w_{tag}_dice']    = None
                    result[f'T1w_{tag}_status']      = 'N/A (Missing warped file)'
                    
                else:
                    logging.error(f"[QC] Template or warped file does not exist")
                    #result[f'T1w_{tag}_translation'] = None
                    result[f'T1w_{tag}_dice']    = None
                    result[f'T1w_{tag}_status']      = 'N/A (File not found)'
            
            registration_results.append(result)
    
    return pd.DataFrame(registration_results)


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python run_final_qc.py <subject_id>")
        sys.exit(1)
    else:
        subject_id = sys.argv[1]
    os.makedirs(os.path.join(QC_DIR,subject_id),exist_ok=True)

    ## Check file existence
    #######################################
    sessions = get_sessions(os.path.join(REG_MNI_OUTPUT_FOLDER,subject_id))
    print(sessions)
    if not sessions:
        subject_paths = [[os.path.join(REG_MNI_OUTPUT_FOLDER,subject_id)]]
        csv_out_paths = [os.path.join(QC_DIR,subject_id,'file_existance.csv')]
    else:
        subject_paths = [[os.path.join(REG_MNI_OUTPUT_FOLDER,subject_id,sess)] for sess in sessions]
        csv_out_paths = [os.path.join(QC_DIR,subject_id,sess,'file_existance.csv') for sess in sessions]

    for subject_path, csv_out in zip(subject_paths,csv_out_paths):
        print(subject_path)
        os.makedirs(os.path.dirname(csv_out),exist_ok=True)
        df_files = check_file_existence(subject_path, REG_MNI_OUTPUT_FOLDER)
        df_files.to_csv(csv_out, index=False)

    ## Check within subject registration
    #######################################
    if NUM_SCANS_PER_SESSION>1:
        if not sessions:
            subject_paths = [[os.path.join(REG_MNI_OUTPUT_FOLDER,subject_id)]]
            input_b0_subject_folders = [os.path.join(REG_WITHIN_B0_INPUT_FOLDER,subject_id)]
            csv_out_paths = [os.path.join(QC_DIR,subject_id,'within_subject_registraction_qc.csv')]
        else:
            subject_paths = [[os.path.join(REG_MNI_OUTPUT_FOLDER,subject_id,sess)] for sess in sessions]
            input_b0_subject_folders = [os.path.join(REG_WITHIN_B0_INPUT_FOLDER,subject_id,sess) for sess in sessions]
            csv_out_paths = [os.path.join(QC_DIR,subject_id,sess,'within_subject_registraction_qc.csv') for sess in sessions]

        for subject_path, input_b0_subject_folder, csv_out in zip(subject_paths,input_b0_subject_folders, csv_out_paths):    
            if REG_WITHIN_B0_INPUT_NAMES:
                base_b0 = REG_WITHIN_B0_INPUT_NAMES[0]
            else:
                base_b0 = os.path.join(input_b0_subject_folder,'mask_bet_scan0.nii.gz') 
            patterns_within = [
                ('rigid', ["*b0_reg_*_to_0.nii.gz"]),
                ('affine', ["None",
                            "None"])
            ]
            df_within = analyze_registration_matrices(subject_path, REG_WITHIN_OUTPUT_FOLDER, 'Diffusion',base_b0,patterns_within)
            df_within.to_csv(csv_out, index=False)

    ## Check MNI registration
    ####################################
    if not sessions:
        subject_paths = [[os.path.join(REG_MNI_OUTPUT_FOLDER,subject_id)]]
        csv_out_paths = [os.path.join(QC_DIR,subject_id,'mni_registraction_qc.csv')]
    else:
        subject_paths = [[os.path.join(REG_MNI_OUTPUT_FOLDER,subject_id,sess)] for sess in sessions]
        csv_out_paths = [os.path.join(QC_DIR,subject_id,sess,'mni_registraction_qc.csv') for sess in sessions]
    for subject_path, csv_out in zip(subject_paths,csv_out_paths):    
    #subject_paths = [os.path.join(REG_MNI_OUTPUT_FOLDER,subject_id)]
        patterns_mni = [
            ('rigid', ["*b0_reg_rigid.nii.gz", 
                        "*b0*reg*rigid.nii.gz"]),
            ('affine', ["*b0_reg_affine.nii.gz",
                        "*b0*reg*affine.nii.gz"])
        ]
        df_mni = analyze_registration_matrices(subject_path, REG_MNI_OUTPUT_FOLDER, 'Diffusion',TEMPLATE_PATH,patterns_mni)
        df_mni.to_csv(csv_out, index=False)

