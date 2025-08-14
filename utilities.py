import os
import glob
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import logging
from datetime import datetime
import re
from pathlib import Path

import nibabel as nib
import numpy as np
import os

def get_dimensions(nifti_path):
    # Load the MRI volume
    img = nib.load(nifti_path)
    data = img.get_fdata()
    affine = img.affine
    header = img.header

    original_shape = data.shape
    if len(original_shape) != 4:
        raise ValueError(f"Expected 4D volume, got shape: {original_shape}")

    return original_shape

def trim_odd_dimensions(nifti_path):
    # Load the MRI volume
    img = nib.load(nifti_path)
    data = img.get_fdata()
    affine = img.affine
    header = img.header

    original_shape = data.shape
    if len(original_shape) != 4:
        raise ValueError(f"Expected 4D volume, got shape: {original_shape}")

    # Track which dimensions are modified
    trimmed = False
    new_shape = list(original_shape)

    # Slice indices
    slicer = [slice(None)] * 4

    for dim in range(3):  # Only check x, y, z (not time)
        if original_shape[dim] % 2 == 1:
            new_shape[dim] -= 1
            slicer[dim] = slice(0, new_shape[dim])
            trimmed = True

    if trimmed:
        print(f"Original shape: {original_shape}, trimming to: {tuple(new_shape)}")
        trimmed_data = data[tuple(slicer)]
        new_img = nib.Nifti1Image(trimmed_data, affine, header)
        nib.save(new_img, nifti_path)
        print(f"Saved trimmed image back to: {nifti_path}")
    else:
        print(f"No trimming needed for shape: {original_shape}")
        

def is_session_folder(folder_name):
    # Check if the folder name matches the date format YYYY-MM-DD
    return re.fullmatch(r"\d{4}-\d{2}-\d{2}", folder_name) is not None

def get_sessions(main_folder):
    session_folders = [
        d for d in os.listdir(main_folder)
        if os.path.isdir(os.path.join(main_folder, d)) and is_session_folder(d)
    ]
    return session_folders

def match_file_pattern(subject_folder,pattern):
    pattern_path = os.path.join(subject_folder, pattern)
    matching_files = glob.glob(pattern_path)
    return matching_files[0]

def gen_qc_image(subject_name, out_path, image_series, slices_to_plot, volumes_to_plot, suptitle=None, image_names=None, scan_num = 0):

    num_images = len(image_series)
    num_slices = len(slices_to_plot)
    
    for vol in volumes_to_plot:
        
        fig, axes = plt.subplots(num_slices, num_images, figsize=(num_images*5, num_slices*5))
        
        # Normalizes images for display
        for i, im in enumerate(image_series):
            maximum = np.percentile(im[im > 0], 98)
            minimum = np.percentile(im[im > 0], 2)
            image_series[i] = (im - minimum)/(maximum - minimum)

        for j, im in enumerate(image_series):
            for i, z in enumerate(slices_to_plot):
                plot_slice = np.rot90(im[:,:,z,vol])
                im0 = axes[i, j].imshow(plot_slice, cmap='gray', vmin=0, vmax=1)
                if image_names:
                    axes[i, j].set_title(f'{image_names[j]} - Slice {z}')
                axes[i, j].axis('off')
    
    # Extract filename parts for title
    file_path = os.path.join(out_path, f'QC-{subject_name}-scan#{scan_num}-volume-{vol}.png')
    filename = os.path.basename(file_path)
    plt.suptitle(f"{suptitle} subject {subject_name} \n \n",
                fontsize=16, y=0.95)
    
    plt.tight_layout()
    try:
        plt.savefig(file_path, dpi=300, bbox_inches='tight', pad_inches=0.2)
        print(f"Successfully saved QC image at: {file_path}")
    except Exception as e:
        print(f"Failed to save QC image: {str(e)}")
    plt.close()


def init_logger(step_name, LOG_DIR, LOG_LEVEL, LOG_FORMAT):

    # Set up logging
    try:
        os.makedirs(LOG_DIR, exist_ok=True)
        log_file = os.path.join(LOG_DIR, f"{step_name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log")
        
        # First set up console logging
        console_handler = logging.StreamHandler()
        console_handler.setLevel(LOG_LEVEL)
        console_handler.setFormatter(logging.Formatter(LOG_FORMAT))
        
        # Create logger
        logger = logging.getLogger(step_name)
        logger.setLevel(LOG_LEVEL)
        logger.addHandler(console_handler)
        
        # Try to add file handler
        try:
            # Check if we can write to the directory
            test_file = os.path.join(LOG_DIR, "test_write_access.tmp")
            with open(test_file, 'w') as f:
                f.write("Test write access")
            os.remove(test_file)
            
            # If we get here, we have write access, so add the file handler
            file_handler = logging.FileHandler(log_file)
            file_handler.setLevel(LOG_LEVEL)
            file_handler.setFormatter(logging.Formatter(LOG_FORMAT))
            logger.addHandler(file_handler)
            logger.info(f"Logs will be saved to: {log_file}")
        except (IOError, PermissionError) as e:
            print(f"Warning: Could not create log file in {LOG_DIR}. Error: {str(e)}")
            print(f"Logs will only be displayed in the console.")
    except Exception as e:
        # Fall back to basic logging if the above fails
        logging.basicConfig(
            level=LOG_LEVEL,
            format=LOG_FORMAT,
            handlers=[logging.StreamHandler()]
        )
        logger = logging.getLogger('synthstrip')
        logger.error(f"Failed to set up logging properly: {str(e)}")
        logger.warning("Logs will only be displayed in the console.")
    return logger

def find_file(root, pattern, session=None, verbose=False):
    """
    Find a file matching pattern under root directory, with improved debugging.
    Uses configuration variables to handle AIBL-specific directory structure.
    
    Parameters:
    -----------
    root : str
        Root directory to search in
    pattern : str
        File pattern to match
    session : str or None
        If provided, search only in the specified session directory
    verbose : bool
        Enable verbose output for debugging
        
    Returns:
    --------
    str or None
        Path to the first matching file, or None if no matches found
    """
    if verbose:
        logging.info(f"Searching for '{pattern}' in '{root}' (session: {session})")
    
    # Check if root exists
    if not os.path.exists(root):
        if verbose:
            logging.warning(f"Root directory does not exist: {root}")
        return None
    
    # Search paths to check
    search_paths = []
    
    # For AIBL registration structure, check numbered subdirectories
    if 'registration' in root:
        # Look for numbered subdirectories (e.g., "1", "2", etc.)
        if os.path.exists(root):
            subdirs = [d for d in os.listdir(root) 
                      if os.path.isdir(os.path.join(root, d)) and d.isdigit()]
            
            for subdir in subdirs:
                subdir_path = os.path.join(root, subdir)
                search_paths.append(os.path.join(subdir_path, '**'))
                
                # Also check the "other" subdirectory for transformation files
                other_path = os.path.join(subdir_path, 'other')
                if os.path.exists(other_path):
                    search_paths.append(os.path.join(other_path, '**'))
    
    # If no specific registration paths were added, or for skullstrip directories,
    # search the entire root directory
    if not search_paths:
        search_paths.append(os.path.join(root, '**'))
        if verbose:
            logging.info(f"Searching in all directories under root: {root}")
    
    # Search in all identified paths
    matches = []
    for search_path in search_paths:
        # First try the exact pattern
        path_matches = glob.glob(os.path.join(search_path, pattern), recursive=True)
        if path_matches:
            matches.extend(path_matches)
            if verbose:
                logging.info(f"Found {len(path_matches)} matches in {search_path}")
        
        # If no matches, try a more relaxed pattern by removing underscores
        if not path_matches and '_' in pattern:
            relaxed_pattern = pattern.replace('_', '*')
            if verbose:
                logging.info(f"No matches, trying relaxed pattern: {relaxed_pattern}")
            relaxed_matches = glob.glob(os.path.join(search_path, relaxed_pattern), recursive=True)
            if relaxed_matches:
                matches.extend(relaxed_matches)
        
        # If still no matches, try an even more relaxed pattern
        if not path_matches and not matches:
            # Extract key parts from the pattern
            if 'T1w' in pattern or 'T1-' in pattern:
                key = 'T1'
            elif 'T2w' in pattern or 'T2-' in pattern or 'T2_' in pattern:
                key = 'T2'
            else:
                key = pattern.split('_')[0] if '_' in pattern else pattern.split('.')[0]
                
            if 'rigid' in pattern:
                second_key = 'rigid'
            elif 'cropped' in pattern:
                second_key = 'crop'
            elif 'zscore' in pattern:
                second_key = 'z'
            elif 'warped' in pattern:
                second_key = 'warp'
            elif 'affine' in pattern:
                second_key = 'affine'
            elif '.mat' in pattern:
                second_key = 'mat'
            else:
                second_key = ''
                
            if second_key:
                very_relaxed = f"*{key}*{second_key}*.nii.gz" if '.nii' in pattern else f"*{key}*{second_key}*.mat"
            else:
                very_relaxed = f"*{key}*.nii.gz" if '.nii' in pattern else f"*{key}*.mat"
                
            if verbose:
                logging.info(f"Still no matches, trying very relaxed pattern: {very_relaxed}")
            very_relaxed_matches = glob.glob(os.path.join(search_path, very_relaxed), recursive=True)
            if very_relaxed_matches:
                matches.extend(very_relaxed_matches)
    
    if verbose:
        if matches:
            logging.info(f"Found {len(matches)} total matches: {matches[:3]}")
        else:
            logging.warning(f"No matches found for pattern: {pattern} in any search path")
    
    return matches[0] if matches else None