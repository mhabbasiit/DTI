"""
Diffusion Tensor Model Fitting — run_dtifit_dipy.py
===================================================

Fits a diffusion tensor model using DIPY (Garyfallidis et al., 2014).
This script takes preprocessed diffusion MRI data (DWI, bvec, bval, mask)
and outputs standard DTI-derived measures including fractional anisotropy (FA),
mean diffusivity (MD), radial diffusivity (RD), and axial diffusivity (AD).
Eigenvectors of the tensor are also saved for visualization.
Steps performed:
1. Load diffusion MRI data, b-values, b-vectors, and brain mask
2. Construct gradient table using DIPY
3. Fit the diffusion tensor model voxel-wise
4. Extract tensor components (Dxx, Dxy, Dxz, Dyy, Dyz, Dzz)
5. Save DTI-derived tensor components and maps: FA, MD, RD, AD, and eigenvectors (V1, V2, V3)
6. Generate QC images showing a color FA map

References:
- Basser, P.J., Mattiello, J., & LeBihan, D. (1994). MR diffusion tensor spectroscopy 
  and imaging. Biophysical Journal, 66(1), 259–267. doi:10.1016/S0006-3495(94)80775-1

  - https://pubmed.ncbi.nlm.nih.gov/8130344/
  
- Garyfallidis, E., Brett, M., Correia, M.M., Williams, G.B., & Nimmo-Smith, I. (2014). 
  DIPY, a library for the analysis of diffusion MRI data. Frontiers in Neuroinformatics, 8, 8. 
  
  - https://pmc.ncbi.nlm.nih.gov/articles/PMC3931231/
  
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
from dipy.reconst.dti import TensorModel
from dipy.io import read_bvals_bvecs
from dipy.core.gradients import gradient_table
import nibabel as nib
import matplotlib.pyplot as plt
from utilities import get_sessions


# Import configuration
from config import (
    MASK_PATH,
    MASK_NAME,
    DTIFIT_INPUT_FOLDER,
    DTIFIT_DWI_INPUT_NAME,
    DTIFIT_BVEC_INPUT_NAME,
    DTIFIT_BVAL_INPUT_NAME,
    DTIFIT_OUT_FOLDER,
    DTIFIT_QC_SLICES
)

def load_nifti(file_path):
    """
    Load a NIfTI file and return the image data as a numpy array.
    """
    nifti_img = nib.load(file_path)
    return nifti_img.get_fdata(), nifti_img.affine

def dipy_dtifit(data_path, bval_path, bvec_path, mask_path, out_dir):
    print(f"DEBUG: LOADING DATA")

    diff_file, affine = load_nifti(data_path)
    mask_file, _ = load_nifti(mask_path)
    bvals = np.loadtxt(bval_path)
    bvecs= np.loadtxt(bvec_path)
    gtab = gradient_table(bvals, bvecs)

    print(f"DEBUG: FITTING DTI MODEL")

    # Fitting the Diffusion Tensor Model
    tensor_model = TensorModel(gtab)
    tensor_fit = tensor_model.fit(diff_file, mask=mask_file)

    # Extract the diffusion tensor (3x3 matrix per voxel)
    dt_matrix = tensor_fit.quadratic_form  # Shape: (x, y, z, 3, 3)

    # Prepare FSL order: Dxx, Dxy, Dxz, Dyy, Dyz, Dzz
    Dxx = dt_matrix[..., 0, 0]
    Dxy = dt_matrix[..., 0, 1]
    Dxz = dt_matrix[..., 0, 2]
    Dyy = dt_matrix[..., 1, 1]
    Dyz = dt_matrix[..., 1, 2]
    Dzz = dt_matrix[..., 2, 2]

    # Stack in FSL order
    fsl_tensors = np.stack([Dxx, Dxy, Dxz, Dyy, Dyz, Dzz], axis=-1)  # Shape: (x, y, z, 6)

    print(f"DEBUG: EXPORTING DATA")

    # Extract eigenvectors
    eigenvectors = tensor_fit.evecs  # Shape: (x, y, z, 3, 3)
    V1, V2, V3 = eigenvectors[..., 0], eigenvectors[..., 1], eigenvectors[..., 2]  # First, second, third eigenvector

    # Save outputs of dti model
    tensor_img = nib.Nifti1Image(fsl_tensors, affine)
    nib.save(tensor_img, os.path.join(out_dir,'dipy_tensor.nii.gz'))
    fa_img = nib.Nifti1Image(tensor_fit.fa, affine)
    nib.save(fa_img, os.path.join(out_dir,'dipy_fa.nii.gz'))
    md_img = nib.Nifti1Image(tensor_fit.md, affine)
    nib.save(md_img, os.path.join(out_dir,'dipy_md.nii.gz'))
    nib.save(nib.Nifti1Image(V1, affine), os.path.join(out_dir,'V1.nii.gz'))
    nib.save(nib.Nifti1Image(V2, affine), os.path.join(out_dir,'V2.nii.gz'))
    nib.save(nib.Nifti1Image(V3, affine), os.path.join(out_dir,'V3.nii.gz'))
    nib.save(nib.Nifti1Image(tensor_fit.rd, affine), os.path.join(out_dir,'RD.nii.gz'))
    nib.save(nib.Nifti1Image(tensor_fit.ad, affine), os.path.join(out_dir,'AD.nii.gz'))

    return tensor_fit.fa, tensor_fit.color_fa

def dtifit_qc_image(subject_name, out_path, image_series, slices_to_plot, suptitle=None):

    num_images = len(image_series)
    num_slices = len(slices_to_plot)
    image_names = ['FA','V1']
        
    fig, axes = plt.subplots(num_slices, num_images, figsize=(num_images*5, num_slices*5))
    
    # # Normalizes images for display
    # for i, im in enumerate(image_series):
    #     maximum = np.percentile(im[im > 0], 98)
    #     minimum = np.percentile(im[im > 0], 2)
    #     image_series[i] = (im - minimum)/(maximum - minimum)

    for j, im in enumerate(image_series):
        for i, z in enumerate(slices_to_plot):
            
            if j==0:
                plot_slice = np.rot90(im[:,:,z])
                im0 = axes[i, j].imshow(plot_slice, cmap='gray', vmin=0, vmax=1)
            else:
                plot_slice = np.rot90(im[:,:,z])
                im0 = axes[i, j].imshow(plot_slice)
            if image_names:
                axes[i, j].set_title(f'{image_names[j]} - Slice {z}')
            axes[i, j].axis('off')
    
    # Extract filename parts for title
    file_path = os.path.join(out_path, f'QC-Dtifit-{subject_name}.png')
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

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python run_dtifit_dipy.py <subject_id>")
        sys.exit(1)
    else:
        subject_id = sys.argv[1]

    os.makedirs(DTIFIT_OUT_FOLDER, exist_ok=True)

    sessions = get_sessions(os.path.join(DTIFIT_INPUT_FOLDER,subject_id))
    print(sessions)
    if not sessions:
        input_subject_folders = [os.path.join(DTIFIT_INPUT_FOLDER,subject_id)]
        out_folders = [os.path.join(DTIFIT_OUT_FOLDER,subject_id)]
    else:
        input_subject_folders = [os.path.join(DTIFIT_INPUT_FOLDER,subject_id,sess) for sess in sessions]
        out_folders = [os.path.join(DTIFIT_OUT_FOLDER,subject_id,sess) for sess in sessions]


    for input_subject_folder, out_folder in zip(input_subject_folders,out_folders):
        os.makedirs(out_folder, exist_ok=True)
        # B0 names
        if DTIFIT_DWI_INPUT_NAME:
            dwi_input_name = os.path.join(input_subject_folder,DTIFIT_DWI_INPUT_NAME)
        else:
            dwi_input_name = os.path.join(input_subject_folder,f'dwi_reg_affine.nii.gz')

        if DTIFIT_BVEC_INPUT_NAME:
            bvec_input_name = os.path.join(input_subject_folder,DTIFIT_BVEC_INPUT_NAME)
        else:
            bvec_input_name = os.path.join(input_subject_folder,f'bvec_reg_affine.bvec')

        if DTIFIT_BVAL_INPUT_NAME:
            bval_input_name = os.path.join(input_subject_folder,DTIFIT_BVAL_INPUT_NAME)
        else:
            bval_input_name = os.path.join(input_subject_folder,f'bval_final.bval')

        if MASK_NAME:
            mask_input_name = os.path.join(input_subject_folder,MASK_NAME)
        else:
            mask_input_name = os.path.join(input_subject_folder,'mask_reg_affine.nii.gz')

        fa_img, V1 = dipy_dtifit(dwi_input_name, bval_input_name, bvec_input_name, mask_input_name, out_folder)
        image_series = [fa_img, V1]
        dtifit_qc_image(subject_id, out_folder, image_series, DTIFIT_QC_SLICES, suptitle=f'Dtifit QC')    
    
