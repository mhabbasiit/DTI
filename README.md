# Diffusion MRI Preprocessing Pipeline

This repository contains Python code to perform preprocessing of diffusion MRI (dMRI) data.  
At present, the only distortion correction supported is **top-up correction**, which requires an additional reversed polarity acquisition.  

---

## Prerequisites

- [FSL](https://fsl.fmrib.ox.ac.uk/fsl/)  
- [ANTs](https://github.com/ANTsX/ANTs)  
- [DIPY](https://docs.dipy.org/stable/user_guide/installation)  

---

## Configuration

Paths and file naming are controlled by `config.py`.  

All scripts work on a **subject-by-subject** basis.  

---

## Processing Steps

### 1. Topup Correction — `b0_correction.py`
Correct susceptibility-induced distortions using FSL TOPUP (Andersson et al., 2003).  
This step estimates a field map from images with opposite phase-encoding directions and applies it to unwarp distorted diffusion volumes.  

**Reference:** Andersson, J.L.R., Skare, S., & Ashburner, J. (2003). *NeuroImage, 20(2), 870–888.*

- https://pubmed.ncbi.nlm.nih.gov/14568458/

---

### 2. Eddy Current and Motion Correction — `process_eddy.py`
Correct distortions caused by eddy currents and subject motion using FSL EDDY (Andersson & Sotiropoulos, 2016).  
Includes slice-to-volume correction and outlier replacement.  

Steps performed:
1. Load and merge diffusion-weighted images (AP/PA or reversed polarity)
2. Prepare acquisition parameters and B0 indices
3. Run FSL EDDY with slice-to-volume correction (optional)
4. Generate QC images comparing original vs corrected volumes

**Reference:** Andersson, J.L.R., & Sotiropoulos, S.N. (2016). *NeuroImage, 125, 1063–1078.*

- https://pubmed.ncbi.nlm.nih.gov/26481672/

---

### 3. Brain Extraction — `brain_extraction.py`
Removes non-brain tissue from diffusion MRI data using FSL BET 
(Smith, 2002). This step performs skull stripping and generates 
quality control (QC) images and summary files to validate 
the extracted brain.

Steps performed:
1. Apply FSL BET to remove non-brain tissue
2. Generate brain mask and masked output
3. Compute brain volume (ml) and validate against the expected range
4. Create QC images (raw, brain-extracted, raw+mask, differences)
5. Write QC summary (CSV) for downstream review

**Reference:** Smith, S.M. (2002). *Human Brain Mapping, 17(3), 143–155.*

- https://pubmed.ncbi.nlm.nih.gov/12391568/

---

### 4. Merging of Acquisitions — `reg_within_fsl.py`

Aligns and merges multiple diffusion MRI runs using FSL FLIRT in case a session is broken down into multiple acquisitions.
Rigid transformations are estimated between B0 reference images,
applied to corresponding diffusion volumes, and propagated to
b-vectors to ensure orientation consistency across runs.
Steps performed:
1. Register B0 images between runs using FLIRT rigid-body transform
2. Apply transforms to diffusion volumes (DWI)
3. Rotate b-vectors using the computed transformation matrix
4. Merge registered DWI volumes, b-vectors, and b-values
5. Save combined outputs for downstream processing

---

### 5. Registration to MNI Space — `run_reg_mni.py`
Registers diffusion images to the MNI152 template using a
two-step approach with FSL FLIRT (rigid + affine). Corresponding
b-vectors are rotated to preserve orientation
consistency. The pipeline also applies the transforms to the diffusion
volumes and masks, and prepares final outputs for downstream analysis.

Steps performed:
1. Register B0 to MNI (rigid, 6 DOF), save the matrix and registered B0
2. Refine B0→MNI with affine (12 DOF), save matrix, and registered B0
3. Apply rigid and then affine transforms to DWI
4. Rotate b-vectors using the previously computed transformations
5. Copy b-values (bval_final.bval) for consistency
6. Register mask with nearest-neighbour interpolation using the previously computed transformations 

---

### 6. Diffusion Tensor Model Fitting — `run_dtifit_dipy.py`
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

**Reference:** 
- Basser, P.J., Mattiello, J., & LeBihan, D. (1994). MR diffusion tensor spectroscopy 
  and imaging. Biophysical Journal, 66(1), 259–267.

  - https://pubmed.ncbi.nlm.nih.gov/8130344/
  
- Garyfallidis, E., Brett, M., Correia, M.M., Williams, G.B., & Nimmo-Smith, I. (2014). 
  DIPY, a library for the analysis of diffusion MRI data. Frontiers in Neuroinformatics, 8, 8. 
  
  - https://pmc.ncbi.nlm.nih.gov/articles/PMC3931231/

---

### 7. Quality Control of Final Outputs — `run_final_qc.py`
Perform automated QC checks, including:  
- File existence validation  
- Dice coefficient calculation for registration accuracy (Zou et al., 2004)  

**Reference:** Zou, K.H., et al. (2004). *Academic Radiology, 11(2), 178–189.*  

---

### 8. Generation of HTML QC Reports — `dti_qc.py`
Generate automated QC reports summarizing:  
- Raw vs. corrected b0 volumes  
- Eddy-corrected vs. uncorrected volumes  
- Brain extraction evaluation  
- FA and color FA maps  
- Registration metrics  
