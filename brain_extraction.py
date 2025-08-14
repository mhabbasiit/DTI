import json
import os
import numpy as np
import subprocess
import sys
import os
import config
import nibabel as nib
import logging
import csv
from matplotlib import pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable

from config import setup_fsl_env

from utilities import init_logger, get_sessions


from config import (
    SKULL_STRIP_INPUT_FOLDER,
    SKULL_STRIP_OUTPUT_FOLDER,
    SKULL_STRIP_INPUT_NAMES,
    SKULL_STRIP_OUTPUT_PATTERN,
    NUM_SCANS_PER_SESSION,
    FRACTIONAL_INTENSITY,
    ENABLE_QC,
    QC_GENERATE_IMAGES,
    QC_SUMMARY_FILE,
    QC_BRAIN_VOLUME_MIN_ML,
    QC_BRAIN_VOLUME_MAX_ML,
    QC_IMAGE_DPI,
    QC_IMAGE_FORMAT,
    QC_SLICE_VIEWS,
    QC_SLICE_NUMBERS
)

from config import (
    LOG_DIR,
    LOG_LEVEL,
    LOG_FORMAT  
)

def run_bet(input_name, output_name, f):
    subprocess.call(f"bet {input_name} {output_name} -m -f {f}",shell=True)
    print(f'Done masking {input_name}')

def perform_quality_check(output_brain_file, input_file, modality, logger):
    """Perform quality check on extracted brain.
    
    Parameters:
    -----------
    output_brain_file : str
        Path to the brain-extracted output file
    input_file : str
        Path to the original input file
    modality : str
        Imaging modality (T1w or T2w)
        
    Returns:
    --------
    dict
        QC results including brain volume and pass/fail status
    """
    # Get the output directory where the output file is located
    output_dir = os.path.dirname(output_brain_file)
    
    qc_result = {
        'passed_qc': True,
        'brain_volume_ml': 0,
        'qc_image_path': '',
        'error': None
    }
    
    try:
        logger.info(f"Starting QC for {output_brain_file}")
        brain_img = nib.load(output_brain_file)
        brain_data = brain_img.get_fdata()
        
        if not os.path.exists(input_file):
            logger.warning(f"Original input file not found at {input_file}")
            raise FileNotFoundError(f"Input file not found: {input_file}")
        
        raw_img = nib.load(input_file)
        raw_data = raw_img.get_fdata()
        
        mask_file = output_brain_file.replace('.nii.gz', '_mask.nii.gz')
        if not os.path.exists(mask_file):
            logger.error(f"Mask file not found: {mask_file}")
            raise FileNotFoundError(f"No such file or no access: '{mask_file}'")
            
        mask_img = nib.load(mask_file)
        mask_data = mask_img.get_fdata()
        
        voxel_volume_mm3 = np.prod(brain_img.header.get_zooms())
        brain_volume_mm3 = np.sum(mask_data > 0) * voxel_volume_mm3
        brain_volume_ml = brain_volume_mm3 / 1000
        
        qc_result['brain_volume_ml'] = brain_volume_ml
        
        if ENABLE_QC and (brain_volume_ml < QC_BRAIN_VOLUME_MIN_ML or brain_volume_ml > QC_BRAIN_VOLUME_MAX_ML):
            qc_result['passed_qc'] = False
            qc_result['error'] = f"Brain volume ({brain_volume_ml:.2f} ml) outside acceptable range ({QC_BRAIN_VOLUME_MIN_ML}-{QC_BRAIN_VOLUME_MAX_ML} ml)"
            logger.warning(f"QC failed for {output_brain_file}: {qc_result['error']}")
        
        if QC_GENERATE_IMAGES:
            # Create QC image filename in the same directory as the output file
            qc_basename = os.path.basename(output_brain_file).replace('.nii.gz', '_desc-qc.png')
            qc_image_file = os.path.join(output_dir, qc_basename)
            qc_result['qc_image_path'] = qc_image_file
            
            logger.info(f"Generating QC image at: {qc_image_file}")
            
            fig, axes = plt.subplots(4, 3, figsize=(15, 20))
            
            x_mid = brain_data.shape[0] // 2
            y_mid = brain_data.shape[1] // 2
            z_mid = brain_data.shape[2] // 2
            
            raw_data_norm = raw_data / np.percentile(raw_data[raw_data > 0], 99)
            raw_data_norm = np.clip(raw_data_norm, 0, 1)
            
            brain_data_norm = brain_data / np.percentile(brain_data[brain_data > 0], 99)
            brain_data_norm = np.clip(brain_data_norm, 0, 1)
            
            def add_colorbar(im, ax):
                divider = make_axes_locatable(ax)
                cax = divider.append_axes("right", size="5%", pad=0.05)
                plt.colorbar(im, cax=cax)
            
            views = [
                (x_mid, ':', ':', 'Sagittal'),
                (':', y_mid, ':', 'Coronal'),
                (':', ':', z_mid, 'Axial')
            ]
            
            for row, (x, y, z, view_name) in enumerate(views):
                x_slice = slice(None) if x == ':' else x
                y_slice = slice(None) if y == ':' else y
                z_slice = slice(None) if z == ':' else z
                
                raw_slice = np.rot90(eval(f"raw_data_norm[{x}, {y}, {z}]"))
                brain_slice = np.rot90(eval(f"brain_data_norm[{x}, {y}, {z}]"))
                mask_slice = np.rot90(eval(f"mask_data[{x}, {y}, {z}]"))
                
                im0 = axes[row, 0].imshow(raw_slice, cmap='gray')
                axes[row, 0].set_title(f'{view_name}\nRaw Image')
                add_colorbar(im0, axes[row, 0])
                
                im1 = axes[row, 1].imshow(brain_slice, cmap='gray')
                axes[row, 1].set_title(f'{view_name}\nBrain Extracted')
                add_colorbar(im1, axes[row, 1])
                
                axes[row, 2].imshow(raw_slice, cmap='gray')
                im2 = axes[row, 2].imshow(mask_slice, cmap='Reds', alpha=0.3)
                axes[row, 2].set_title(f'{view_name}\nRaw + Mask')
                add_colorbar(im2, axes[row, 2])
                
                for ax in axes[row]:
                    ax.axis('off')
            
            for col in range(3):
                if col == 0:
                    im = axes[3, col].imshow(mask_slice, cmap='Reds')
                    axes[3, col].set_title('Brain Mask')
                elif col == 1:
                    diff = raw_slice - brain_slice
                    im = axes[3, col].imshow(diff, cmap='RdBu_r')
                    axes[3, col].set_title('Raw - Brain (Difference)')
                else:
                    axes[3, col].text(0.5, 0.5, 
                                    f"Brain Volume: {brain_volume_ml:.2f} ml\n" +
                                    f"Status: {'PASS' if qc_result['passed_qc'] else 'FAIL'}\n" +
                                    f"Valid Range: {QC_BRAIN_VOLUME_MIN_ML}-{QC_BRAIN_VOLUME_MAX_ML} ml",
                                    horizontalalignment='center',
                                    verticalalignment='center',
                                    transform=axes[3, col].transAxes,
                                    fontsize=12)
                    axes[3, col].axis('off')
                    continue
                
                add_colorbar(im, axes[3, col])
                axes[3, col].axis('off')
            
            # Extract filename parts for title
            filename = os.path.basename(input_file)
            
            plt.suptitle(f"Brain Extraction QC Report\n" +
                        f"{filename}",
                        fontsize=16, y=0.95)
            
            plt.tight_layout()
            try:
                plt.savefig(qc_image_file, dpi=150, bbox_inches='tight', pad_inches=0.2)
                logger.info(f"Successfully saved QC image at: {qc_image_file}")
            except Exception as e:
                logger.error(f"Failed to save QC image: {str(e)}")
            plt.close()
        
        if QC_SUMMARY_FILE:
            # Create QC summary file in the output directory
            qc_summary_path = os.path.join(output_dir, QC_SUMMARY_FILE)
            file_exists = os.path.isfile(qc_summary_path)
            logger.info(f"Writing QC summary to: {qc_summary_path}")
            
            try:
                with open(qc_summary_path, mode='a') as csvfile:
                    fieldnames = ['input_file', 'output_file', 'modality', 'qc_status', 'brain_volume_ml', 'qc_image_path', 'error_message']
                    writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
                    
                    if not file_exists:
                        writer.writeheader()
                    
                    writer.writerow({
                        'input_file': input_file,
                        'output_file': output_brain_file,
                        'modality': modality,
                        'qc_status': 'PASS' if qc_result['passed_qc'] else 'FAIL',
                        'brain_volume_ml': f"{qc_result['brain_volume_ml']:.2f}",
                        'qc_image_path': qc_result['qc_image_path'],
                        'error_message': qc_result['error'] if 'error' in qc_result and qc_result['error'] else 'None'
                    })
                logger.info(f"Successfully updated QC summary file: {qc_summary_path}")
            except Exception as e:
                logger.error(f"Failed to write to QC summary file: {str(e)}")
    
    except Exception as e:
        error_msg = f"Error in quality check: {str(e)}"
        logger.error(error_msg)
        qc_result['passed_qc'] = False
        qc_result['error'] = error_msg
    
    return qc_result



if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python brain_extraction.py <subject_id>")
        sys.exit(1)
    else:
        subject_id = sys.argv[1]

    setup_fsl_env()
    logger = init_logger('skullstripping', LOG_DIR, LOG_LEVEL, LOG_FORMAT)

    os.makedirs(SKULL_STRIP_OUTPUT_FOLDER,exist_ok=True)

    sessions = get_sessions(os.path.join(SKULL_STRIP_INPUT_FOLDER,subject_id))
    print(sessions)
    if not sessions:
        subject_folders = [os.path.join(SKULL_STRIP_INPUT_FOLDER,subject_id)]
        out_subject_folders = [os.path.join(SKULL_STRIP_OUTPUT_FOLDER,subject_id)]
    else:
        subject_folders = [os.path.join(SKULL_STRIP_INPUT_FOLDER,subject_id,sess) for sess in sessions]
        out_subject_folders = [os.path.join(SKULL_STRIP_OUTPUT_FOLDER,subject_id,sess) for sess in sessions]

    for input_subject_folder, out_folder in zip(subject_folders,out_subject_folders):


        os.makedirs(out_folder,exist_ok=True)

        if SKULL_STRIP_INPUT_NAMES:
            topup_image_paths = [os.path.join(input_subject_folder,x) for x in SKULL_STRIP_INPUT_NAMES]
        else:
            topup_image_paths = [os.path.join(input_subject_folder,f'b0_unwarped_{x}_mean.nii.gz') for x in range(NUM_SCANS_PER_SESSION)]
        
        if SKULL_STRIP_OUTPUT_PATTERN:
            out_paths = [os.path.join(input_subject_folder,f'{SKULL_STRIP_OUTPUT_PATTERN}_scan{x}') for x in range(NUM_SCANS_PER_SESSION)]
        else:
            out_paths = [os.path.join(out_folder,f'mask_bet_scan{x}') for x in range(NUM_SCANS_PER_SESSION)]
    
        for in_path, out_path in zip(topup_image_paths,out_paths):
            run_bet(in_path, out_path, FRACTIONAL_INTENSITY)
            perform_quality_check(f"{out_path}.nii.gz", in_path, 'Diffusion', logger)