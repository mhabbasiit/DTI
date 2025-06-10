import os
import glob
import nibabel as nib
import numpy as np
import json
from datetime import datetime
import time
import SimpleITK as sitk
import argparse  # Added for command line arguments
import sys

# Import configuration
from config import (
    NUM_SCANS_PER_SESSION,
    REG_WITHIN_B0_INPUT_FOLDER,
    REG_WITHIN_INPUT_FOLDER,
    REG_WITHIN_OUTPUT_FOLDER,
    REG_WITHIN_B0_INPUT_NAMES,
    REG_WITHIN_INPUT_NAMES,
    REG_WITHIN_OUTPUT_PATTERN
)

# Add a results tracking dictionary
registration_results = {
    'successful_subjects': [],
    'failed_subjects': [],
    'missing_subjects': [],  # New category for subjects with missing files
    'errors': {},
    'processing_times': {},  
    'cropping_results': {},  
    'start_time': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
    'end_time': None,
    'total_subjects': 0,
    'success_rate': 0,
    'average_processing_time': 0  
}


# Author: Mohammad Abbasi (mabbasi@stanford.edu)

# ----------------------------
# 1. Settings and Directories
# ----------------------------

def parse_arguments():
    """Parse command line arguments for the registration script."""
    parser = argparse.ArgumentParser(description='Register T1 and T2 brain images to MNI space.')
    parser.add_argument('--input_dir', type=str, default=REG_INPUT_DIR,
                        help='Input directory with brain-extracted images from HD-BET.')
    parser.add_argument('--output_dir', type=str, default=REG_OUTPUT_DIR,
                        help='Output directory for registered images.')
    parser.add_argument('--crop_output', action='store_true', default=PERFORM_CROPPING,
                        help='Crop output images to reduce file size and focus on brain.')
    parser.add_argument('--crop_margin', type=int, default=CROP_MARGIN,
                        help='Margin in voxels to add around the brain when cropping.')
    parser.add_argument('--subjects', type=str, nargs='+', default=None,
                        help='Specific subjects to process (e.g., "ON39384")')
    parser.add_argument('--debug', action='store_true', help='Enable debug output')
    parser.add_argument('--force', action='store_true', default=FORCE_REPROCESS,
                        help='Force reprocessing even if output exists')
    return parser.parse_args()

def save_registration_report(results, output_dir):
    """Save registration results to a JSON file."""
    report_path = os.path.join(output_dir, 'registration_report.json')
    with open(report_path, 'w') as f:
        json.dump(results, f, indent=4)
    print(f"\nDetailed registration report saved to: {report_path}")

def format_time(seconds):
    """Convert seconds to human-readable format."""
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    seconds = int(seconds % 60)
    if hours > 0:
        return f"{hours}h {minutes}m {seconds}s"
    elif minutes > 0:
        return f"{minutes}m {seconds}s"
    else:
        return f"{seconds}s"

def print_registration_summary(results):
    """Print a summary of registration results."""
    print("\n=== Registration Summary ===")
    print(f"Total expected subjects: {results['total_subjects']}")
    print(f"Successfully processed: {len(results['successful_subjects'])}")
    print(f"Failed during processing: {len(results['failed_subjects'])}")
    print(f"Missing input files: {len(results['missing_subjects'])}")
    
    total_processed = len(results['successful_subjects']) + len(results['failed_subjects'])
    
    if total_processed > 0:
        success_rate = (len(results['successful_subjects']) / total_processed) * 100
        print(f"Success rate: {success_rate:.2f}%")
    elif results['total_subjects'] > 0:
        total_issues = len(results['failed_subjects']) + len(results['missing_subjects'])
        success_rate = (len(results['successful_subjects']) / results['total_subjects']) * 100
        fail_rate = (total_issues / results['total_subjects']) * 100
        print(f"Success rate: {success_rate:.2f}%")
        print(f"Total fail rate: {fail_rate:.2f}%")
    else:
        print("No subjects to process found.")
    
    if results['processing_times']:
        avg_time = results['average_processing_time']
        print(f"\nTiming Information:")
        print(f"Average processing time per subject: {format_time(avg_time)}")
        print("\nProcessing times per subject:")
        for subj, t in results['processing_times'].items():
            print(f"- Subject {subj}: {format_time(t)}")
    
    if results['failed_subjects']:
        print("\nFailed subjects (processing errors):")
        for subject, error in results['errors'].items():
            if subject in results['failed_subjects']:
                print(f"- {subject}: {error}")
    
    if results['missing_subjects']:
        print("\nMissing subjects (input files not found):")
        for subject in results['missing_subjects']:
            print(f"- {subject}")

def register_image(fixed, moving, output_prefix, subject_id=None, rigid_precalculated=None):
    """
    Run a two-stage registration with SimpleITK:
    1. Rigid registration (rotation + translation)
    2. Affine registration (using rigid result as initial position)
    """
    try:
        output_dir = os.path.dirname(output_prefix)
        if not os.path.exists(output_dir):
            print(f"DEBUG: Creating output directory: {output_dir}")
            os.makedirs(output_dir, exist_ok=True)
        
        print(f"DEBUG: Loading images for registration")
        # Load images using SimpleITK
        fixed_image = sitk.ReadImage(fixed, sitk.sitkFloat32)
        moving_image = sitk.ReadImage(moving, sitk.sitkFloat32)
        
        print(f"DEBUG: Fixed image size: {fixed_image.GetSize()}, dimension: {fixed_image.GetDimension()}")
        print(f"DEBUG: Moving image size: {moving_image.GetSize()}, dimension: {moving_image.GetDimension()}")
             
        # Get the basename without extension for output naming
        basename = os.path.basename(moving).replace(".nii.gz", "")
        
        # STAGE 1: Rigid Registration
        print("DEBUG: Starting Rigid registration (Stage 1)")
        
        if not rigid_precalculated:

            # Initialize registration method for Rigid
            rigid_registration = sitk.ImageRegistrationMethod()
            
            # Set up similarity metric for Rigid
            rigid_registration.SetMetricAsMattesMutualInformation(numberOfHistogramBins=50)
            rigid_registration.SetMetricSamplingStrategy(rigid_registration.RANDOM)
            rigid_registration.SetMetricSamplingPercentage(0.25)
            
            # Set up interpolator for Rigid
            rigid_registration.SetInterpolator(sitk.sitkLinear)
            
            # Set up optimizer for Rigid
            rigid_registration.SetOptimizerAsGradientDescent(learningRate=0.1,
                                                        numberOfIterations=1000,
                                                        convergenceMinimumValue=1e-6,
                                                        convergenceWindowSize=10)
            rigid_registration.SetOptimizerScalesFromPhysicalShift()
            
            # Set up Rigid transform
            rigid_transform = sitk.CenteredTransformInitializer(fixed_image, 
                                                            moving_image,
                                                            sitk.Euler3DTransform(),
                                                            sitk.CenteredTransformInitializerFilter.GEOMETRY)
            rigid_registration.SetInitialTransform(rigid_transform)
            
            # Multi-resolution framework for Rigid
            rigid_registration.SetShrinkFactorsPerLevel([4, 2, 1])
            rigid_registration.SetSmoothingSigmasPerLevel([2, 1, 0])
        
        try:
            # Execute Rigid registration
            if rigid_precalculated:
                final_rigid_transform = sitk.ReadTransform(rigid_precalculated)
                print(f"DEBUG: Loaded rigid transformation: {rigid_precalculated}")
            else:
                final_rigid_transform = rigid_registration.Execute(fixed_image, moving_image)
                print("DEBUG: Rigid registration completed successfully")
                # Save Rigid transform for reference
                rigid_transform_path = os.path.join(output_dir, f"{basename}_to0_rigid.mat")
                sitk.WriteTransform(final_rigid_transform, rigid_transform_path)
            
            # Apply Rigid transform to get intermediate result
            resampler = sitk.ResampleImageFilter()
            resampler.SetReferenceImage(fixed_image)
            resampler.SetInterpolator(sitk.sitkLinear)
            resampler.SetTransform(final_rigid_transform)
            
            rigid_moved_image = resampler.Execute(moving_image)
            
            # Save intermediate Rigid-registered image
            rigid_warped_path = os.path.join(output_dir, f"{basename}_to0_rigid_warped.nii.gz")
            sitk.WriteImage(rigid_moved_image, rigid_warped_path)
            print(f"DEBUG: Saved Rigid-registered intermediate image to {rigid_warped_path}")
            
            # Create a mock result object to maintain compatibility
            class MockResult:
                class Outputs:
                    def __init__(self, prefix, basename):
                        # Final results (two-stage pipeline)
                        self.warped_image = os.path.join(os.path.dirname(prefix), f"{basename}_to0_warped.nii.gz")
                        # Both transforms needed for complete transformation
                        self.forward_transforms = [
                            os.path.join(os.path.dirname(prefix), f"{basename}_to0_rigid.mat"),
                        ]
                        
                        # Individual stage results for reference
                        self.rigid_warped_image = os.path.join(os.path.dirname(prefix), f"{basename}_to0_rigid_warped.nii.gz")
                        self.rigid_transform = os.path.join(os.path.dirname(prefix), f"{basename}_to0_rigid.mat")
                
                def __init__(self, prefix, basename):
                    self.outputs = self.Outputs(prefix, basename)
            
            return MockResult(output_prefix, basename)
            
        except Exception as e:
            print(f"DEBUG: Registration failed with error: {str(e)}")
            raise
        
    except Exception as e:
        error_msg = str(e)
        print(f"ERROR during registration for subject {subject_id}: {error_msg}")
        if subject_id:
            registration_results['failed_subjects'].append(subject_id)
            registration_results['errors'][subject_id] = error_msg
        raise

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python brain_extraction.py <subject_id>")
        sys.exit(1)
    else:
        subject_id = sys.argv[1]

    if not os.path.exists(REG_WITHIN_OUTPUT_FOLDER):
        os.mkdir(REG_WITHIN_OUTPUT_FOLDER)

    input_b0_subject_folder = os.path.join(REG_WITHIN_B0_INPUT_FOLDER,subject_id)
    input_subject_folder = os.path.join(REG_WITHIN_INPUT_FOLDER,subject_id)
    out_folder = os.path.join(REG_WITHIN_OUTPUT_FOLDER,subject_id)
    if not os.path.exists(out_folder):
        os.mkdir(out_folder)

    # B0 names
    if REG_WITHIN_B0_INPUT_NAMES:
        b0_input_names = [os.path.join(input_b0_subject_folder,x) for x in REG_WITHIN_B0_INPUT_NAMES]
    else:
        b0_input_names = [os.path.join(input_b0_subject_folder,f'mask_bet_scan{x}.nii.gz') for x in range(NUM_SCANS_PER_SESSION)]

    # Whole diffusion MRI 
    if REG_WITHIN_INPUT_NAMES:
        input_names = [os.path.join(input_subject_folder,x) for x in REG_WITHIN_INPUT_NAMES]
    else:
        input_names = [os.path.join(input_subject_folder,f'eddy_aligned_{x}.nii.gz') for x in range(NUM_SCANS_PER_SESSION)]
    

    # Scans will be registered to first one (#0)

    for i in range(1,NUM_SCANS_PER_SESSION):
        # Register b0's using rigid transformation
        fixed = b0_input_names[0]
        moving = b0_input_names[i]
        output_prefix = os.path.join(out_folder,'reg')
        print(f"DEBUG: REGISTERING B0 IMAGES")

        reg_info = register_image(fixed, moving, output_prefix, subject_id=subject_id)
        # Apply transformation to whole diffusion image
        obj_reg = register_image(fixed, moving, output_prefix, subject_id=subject_id)

        print(f"DEBUG: REGISTERING FULL DWI")
        fixed = input_names[0]
        moving = input_names[i]
        output_prefix = os.path.join(out_folder,'reg')
        # Apply transformation to bvecs
        register_image(fixed, moving, output_prefix, subject_id=subject_id, rigid_precalculated=obj_reg.outputs.rigid_transform)

    # Merge scans


