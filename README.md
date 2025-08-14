# Description
This repository contains python code to perform processing of Diffusion MRI data. At present, the only distortion correction supported is topup correction which require an extra reversed polarity acquisition

# Prerequisites

* FSL (https://fsl.fmrib.ox.ac.uk/fsl/)
* ANTs (https://github.com/ANTsX/ANTs)
* Dipy (https://docs.dipy.org/stable/user_guide/installation) 

# Steps

Paths and naming of files is controlled by the config.py file.
Please see config_hcpag.py for an example of configuration for the HCP-Aging dataset.
The steps this repository considers are:

1. Topup Correction: b0_correction.py
2. Eddy Correction: process_eddy.py
3. Brain Extraction: brain_extraction.py
4. Merging of Acquisitions (Optional, if the session is split into multiple scans): reg_within_fsl.py
5. DTI Model Fitting: run_dtifit_dipy.py
6. Registration to MNI Space using Rigid + Affine transformation: run_reg_mni.py
7. Quality control of final output:  run_final_qc.py

All the files consider as input the subject identifier (They work on a subject by subject basis). An example of how to parallelize this in a slurm environment is shown in run_topup.slurm  (https://github.com/mhabbasiit/DTI/blob/master/run_topup.slurm) 
