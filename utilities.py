import os
import glob
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

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