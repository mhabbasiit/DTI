import os
import glob
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import logging
from datetime import datetime

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