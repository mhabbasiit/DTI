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
See `config_hcpag.py` for an example configuration for the HCP-Aging dataset.  

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
3. Create mask for eddy correction
4. Run FSL EDDY with slice-to-volume correction (optional)
5. Generate QC images comparing original vs corrected volumes

**Reference:** Andersson, J.L.R., & Sotiropoulos, S.N. (2016). *NeuroImage, 125, 1063–1078.*

- https://pubmed.ncbi.nlm.nih.gov/26481672/

---

### 3. Brain Extraction — `brain_extraction.py`
Remove non-brain tissue using FSL BET (Smith, 2002).  

**Reference:** Smith, S.M. (2002). *Human Brain Mapping, 17(3), 143–155.*

---

### 4. Merging of Acquisitions — `reg_within_fsl.py`
Align and merge multiple runs using FSL FLIRT.  
Transformations are also applied to b-vectors.  

---

### 5. Registration to MNI Space — `run_reg_mni.py`
Register diffusion-derived maps to the MNI152 template using a two-step approach (rigid + affine) with FSL FLIRT.  
Corresponding b-vectors are transformed as well.  

---

### 6. Diffusion Tensor Model Fitting — `run_dtifit_dipy.py`
Fit a diffusion tensor model using DIPY (Garyfallidis et al., 2014).  
Outputs include FA, MD, RD, and AD maps.  

**References:**  
- Basser, P.J., et al. (1994). *Biophysical Journal, 66(1), 259–267.*  
- Alexander, A.L., et al. (2007). *Neurotherapeutics, 4(3), 316–329.*  
- Garyfallidis, E., et al. (2014). *Frontiers in Neuroinformatics, 8, 8.*  

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
