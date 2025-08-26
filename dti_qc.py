#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
DTI Preprocessing Quality Control (QC) Script
==============================================

This script provides comprehensive quality control for DTI preprocessing pipeline.
It generates QC images and reports for each step of the DTI processing:

1. Top up correction (B0 field correction)
2. Skull stripping  
3. Eddy correction
4. Registration within session (if partitioned scans)
5. Registration to MNI
6. DTI fit

Based on the processing structure used in HCP-Dev processing pipeline.

Author: Mohammad Abbasi (mabbasi@stanford.edu)
Created: 2025
"""

import os
import sys
import glob
import json
import argparse
import numpy as np
import nibabel as nib
import pandas as pd
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
import logging
from pathlib import Path

# Import configuration
try:
    from config import NUM_SCANS_PER_SESSION
except ImportError:
    NUM_SCANS_PER_SESSION = 1  # Default fallback
import shutil
from collections import defaultdict

# Import nilearn for brain visualization
try:
    from nilearn import plotting, image
    NILEARN_AVAILABLE = True
except ImportError:
    print("Warning: nilearn not available. Some visualizations will be skipped.")
    NILEARN_AVAILABLE = False

# Import configuration
from config import *

# ==========================================
# Configuration and Setup
# ==========================================

def setup_logging(output_dir, subject_id):
    """Setup simple logging for QC process"""
    try:
        # Simple structure: DTI_QC/logs/
        log_dir = os.path.join(output_dir, "QC")
        os.makedirs(log_dir, exist_ok=True)
        
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        log_file = os.path.join(log_dir, f"{subject_id}_qc.log")
        
        # Configure logging
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(log_file),
                logging.StreamHandler(sys.stdout)
            ]
        )
        
        logging.info(f"DTI QC started for subject: {subject_id}")
        return log_file
        
    except PermissionError:
        # Fallback to temp directory
        log_file = f"/tmp/{subject_id}_qc.log"
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(log_file),
                logging.StreamHandler(sys.stdout)
            ]
        )
        
        logging.warning(f"Using fallback log: {log_file}")
        logging.info(f"DTI QC started for subject: {subject_id}")
        return log_file

# ==========================================
# QC Functions for Each Processing Step
# ==========================================

class DTIQualityControl:
    """Main DTI Quality Control class"""
    
    def __init__(self, subject_id, output_base_dir):
        self.subject_id = subject_id
        self.output_base_dir = output_base_dir
        
        # Create simple QC structure
        self.qc_base_dir = os.path.join(output_base_dir, "QC")
        
        # Subject directory - directly in base folder
        self.subject_dir = os.path.join(self.qc_base_dir, subject_id)
        self.qc_dir = self.subject_dir  # Everything in subject folder
        
        # Create directories automatically
        try:
            os.makedirs(self.subject_dir, exist_ok=True)
            logging.info(f"Created subject directory: {self.subject_dir}")
        except Exception as e:
            logging.warning(f"Could not create directory {self.subject_dir}: {e}")
        
        # Initialize QC results storage
        self.qc_results = {
            'subject_id': subject_id,
            'timestamp': datetime.now().isoformat(),
            'session': None,  # Will be detected later
            'topup_qc': {},
            'skull_stripping_qc': {},
            'eddy_qc': {},
            'registration_within_qc': {},
            'registration_mni_qc': {},
            'dtifit_qc': {},
            'overall_status': 'UNKNOWN'
        }
        
        # Set up directories for each step
        self.directories = {
            'topup': os.path.join(output_base_dir, "B0_correction", subject_id),
            'skull_stripping': os.path.join(output_base_dir, "Skull_stripping", subject_id),
            'eddy': os.path.join(output_base_dir, "Eddy_correction", subject_id),
            'registration_within': os.path.join(output_base_dir, "Reg_within_and_merged", subject_id),
            'registration_mni': os.path.join(output_base_dir, "Reg_MNI", subject_id),
            'dtifit': os.path.join(output_base_dir, "Dtifit", subject_id)
        }
        
        logging.info(f"DTI QC initialized for subject {subject_id}")
        for step, directory in self.directories.items():
            if os.path.exists(directory):
                logging.info(f"  {step}: {directory} ✓")
            else:
                logging.warning(f"  {step}: {directory} ✗ (not found)")



    # Removed old QC check functions - now using read_existing_qc_files() instead

    def find_session_directory(self, base_path):
        """Find session directory if it exists (e.g., 2024-02-13)"""
        if not os.path.exists(base_path):
            return base_path
        
        # List directories in the base path
        try:
            subdirs = [d for d in os.listdir(base_path) 
                      if os.path.isdir(os.path.join(base_path, d))]
            
            # Look for date-like directories (YYYY-MM-DD format)
            for subdir in subdirs:
                if len(subdir) == 10 and subdir.count('-') == 2:
                    try:
                        year, month, day = subdir.split('-')
                        if (len(year) == 4 and len(month) == 2 and len(day) == 2 and
                            year.isdigit() and month.isdigit() and day.isdigit()):
                            session_path = os.path.join(base_path, subdir)
                            logging.info(f"Found session directory: {session_path}")
                            return session_path
                    except:
                        continue
            
            # If no session directory found, return original path
            return base_path
        except Exception as e:
            logging.warning(f"Error checking for session directory in {base_path}: {e}")
            return base_path

    def get_session_name(self, base_path):
        """Get session name if it exists, otherwise return empty string"""
        if not os.path.exists(base_path):
            return ""
        
        try:
            subdirs = [d for d in os.listdir(base_path) 
                      if os.path.isdir(os.path.join(base_path, d))]
            
            # Look for date-like directories (YYYY-MM-DD format)
            for subdir in subdirs:
                if len(subdir) == 10 and subdir.count('-') == 2:
                    try:
                        year, month, day = subdir.split('-')
                        if (len(year) == 4 and len(month) == 2 and len(day) == 2 and
                            year.isdigit() and month.isdigit() and day.isdigit()):
                            return subdir
                    except:
                        continue
            return ""
        except:
            return ""

    def read_existing_qc_files(self):
        """Read existing QC CSV files and extract information"""
        qc_dir = os.path.join(self.output_base_dir, 'QC', self.subject_id)
        
        # Check if session directory exists inside QC folder and save session info
        session_name = self.get_session_name(qc_dir)
        if session_name:
            self.qc_results['session'] = session_name
        qc_dir = self.find_session_directory(qc_dir)
        
        # Read file existence CSV
        file_exist_csv = os.path.join(qc_dir, 'file_existance.csv')
        if os.path.exists(file_exist_csv):
            try:
                import pandas as pd
                df = pd.read_csv(file_exist_csv)
                if not df.empty:
                    row = df.iloc[0]
                    self.qc_results['file_existence_qc'] = {
                        'status': 'PASS',
                        'csv_exists': True,
                        'files_checked': len(df.columns) - 5,  # Excluding basic columns
                        'skull_ok': row.get('skull_ok', False),
                        'missing_count': row.get('missing_count', 0)
                    }
                    logging.info(f"File existence: {row.get('missing_count', 0)} missing files")
            except Exception as e:
                logging.error(f"Error reading file existence CSV: {e}")
                self.qc_results['file_existence_qc'] = {'status': 'ERROR', 'csv_exists': False}
        
        # Check within subject registration CSV - only if NUM_SCANS_PER_SESSION > 1
        within_csv = os.path.join(qc_dir, 'within_subject_registraction_qc.csv')
        if NUM_SCANS_PER_SESSION > 1 and os.path.exists(within_csv):
            try:
                df = pd.read_csv(within_csv)
                if not df.empty:
                    row = df.iloc[0]
                    self.qc_results['registration_within_qc'] = {
                        'status': 'PASS',
                        'csv_exists': True,
                        'rigid_dice': row.get('T1w_rigid_dice', 0),
                        'affine_dice': row.get('T1w_affine_dice', 0),
                        'rigid_status': row.get('T1w_rigid_status', 'Unknown'),
                        'affine_status': row.get('T1w_affine_status', 'Unknown')
                    }
                    logging.info(f"Within registration: {row.get('T1w_rigid_status', 'Unknown')}")
            except Exception as e:
                logging.error(f"Error reading within registration CSV: {e}")
                self.qc_results['registration_within_qc'] = {'status': 'ERROR', 'csv_exists': False}
        elif NUM_SCANS_PER_SESSION <= 1:
            # Skip within subject registration if only one scan per session
            self.qc_results['registration_within_qc'] = {
                'status': 'SKIP',
                'csv_exists': False,
                'reason': f'Single scan per session (NUM_SCANS_PER_SESSION={NUM_SCANS_PER_SESSION})'
            }
            logging.info(f"Skipping within registration: Single scan per session (NUM_SCANS_PER_SESSION={NUM_SCANS_PER_SESSION})")
        
        # Read MNI registration CSV
        mni_csv = os.path.join(qc_dir, 'mni_registraction_qc.csv')
        if os.path.exists(mni_csv):
            try:
                df = pd.read_csv(mni_csv)
                if not df.empty:
                    row = df.iloc[0]
                    self.qc_results['registration_mni_qc'] = {
                        'status': 'PASS',
                        'csv_exists': True,
                        'rigid_dice': row.get('T1w_rigid_dice', 0),
                        'affine_dice': row.get('T1w_affine_dice', 0),
                        'rigid_status': row.get('T1w_rigid_status', 'Unknown'),
                        'affine_status': row.get('T1w_affine_status', 'Unknown')
                    }
                    logging.info(f"MNI registration: Rigid={row.get('T1w_rigid_status', 'Unknown')}, Affine={row.get('T1w_affine_status', 'Unknown')}")
            except Exception as e:
                logging.error(f"Error reading MNI registration CSV: {e}")
                self.qc_results['registration_mni_qc'] = {'status': 'ERROR', 'csv_exists': False}

    def check_qc_images_only(self):
        """Check for QC images existence only"""
        # Check topup images
        topup_dir = os.path.join(self.output_base_dir, 'B0_correction', self.subject_id)
        topup_dir = self.find_session_directory(topup_dir)
        topup_images = 0
        if os.path.exists(topup_dir):
            topup_images = len(glob.glob(os.path.join(topup_dir, "QC-*.png")))
        self.qc_results['topup_qc'] = {'status': 'PASS' if topup_images > 0 else 'WARNING', 'qc_images_found': topup_images}
        
        # Check skull stripping images and volumes
        skull_dir = os.path.join(self.output_base_dir, 'Skull_stripping', self.subject_id)
        skull_dir = self.find_session_directory(skull_dir)
        skull_images = 0
        volumes = []
        if os.path.exists(skull_dir):
            skull_images = len(glob.glob(os.path.join(skull_dir, "*desc-qc.png")))
            # Try to read qc_summary.csv for volumes
            qc_summary_path = os.path.join(skull_dir, 'qc_summary.csv')
            if os.path.exists(qc_summary_path):
                try:
                    import pandas as pd
                    df = pd.read_csv(qc_summary_path)
                    for _, row in df.iterrows():
                        volumes.append({
                            'scan': row.get('scan', 'unknown'),
                            'brain_volume_ml': row.get('brain_volume_ml', 0)
                        })
                except Exception as e:
                    logging.warning(f"Could not read skull stripping volumes: {e}")
        
        self.qc_results['skull_stripping_qc'] = {
            'status': 'PASS' if skull_images > 0 else 'WARNING', 
            'qc_images_found': skull_images,
            'volume_measurements': volumes
        }
        
        # Check eddy images
        eddy_dir = os.path.join(self.output_base_dir, 'Eddy_correction', self.subject_id)
        eddy_dir = self.find_session_directory(eddy_dir)
        eddy_images = 0
        if os.path.exists(eddy_dir):
            eddy_images = len(glob.glob(os.path.join(eddy_dir, "QC-*.png")))
        self.qc_results['eddy_qc'] = {'status': 'PASS' if eddy_images > 0 else 'WARNING', 'qc_images_found': eddy_images}
        
        # Check DTI fit image
        dtifit_dir = os.path.join(self.output_base_dir, 'Dtifit', self.subject_id)
        dtifit_dir = self.find_session_directory(dtifit_dir)
        dtifit_image = False
        if os.path.exists(dtifit_dir):
            dtifit_image = os.path.exists(os.path.join(dtifit_dir, f"QC-Dtifit-{self.subject_id}.png"))
        self.qc_results['dtifit_qc'] = {'status': 'PASS' if dtifit_image else 'WARNING', 'qc_image_exists': dtifit_image}

    def check_basic_statistics(self):
        """Check basic DTI statistics if FA/MD maps exist"""
        dtifit_dir = os.path.join(self.output_base_dir, 'Dtifit', self.subject_id)
        dtifit_dir = self.find_session_directory(dtifit_dir)
        
        if 'dtifit_qc' not in self.qc_results:
            self.qc_results['dtifit_qc'] = {}
        if 'map_statistics' not in self.qc_results['dtifit_qc']:
            self.qc_results['dtifit_qc']['map_statistics'] = {}
        
        # Look for FA map
        fa_file = os.path.join(dtifit_dir, 'dipy_fa.nii.gz')
        if os.path.exists(fa_file):
            try:
                import nibabel as nib
                import numpy as np
                fa_img = nib.load(fa_file)
                fa_data = fa_img.get_fdata()
                fa_data = fa_data[fa_data > 0]  # Remove background
                
                self.qc_results['dtifit_qc']['map_statistics']['dipy_fa.nii.gz'] = {
                    'mean': float(np.mean(fa_data)),
                    'std': float(np.std(fa_data)),
                    'min': float(np.min(fa_data)),
                    'max': float(np.max(fa_data))
                }
                logging.info(f"FA statistics: mean={np.mean(fa_data):.3f}")
            except Exception as e:
                logging.warning(f"Could not read FA statistics: {e}")
        
        # Look for MD map
        md_file = os.path.join(dtifit_dir, 'dipy_md.nii.gz')
        if os.path.exists(md_file):
            try:
                import nibabel as nib
                import numpy as np
                md_img = nib.load(md_file)
                md_data = md_img.get_fdata()
                md_data = md_data[md_data > 0]  # Remove background
                
                self.qc_results['dtifit_qc']['map_statistics']['dipy_md.nii.gz'] = {
                    'mean': float(np.mean(md_data)),
                    'std': float(np.std(md_data)),
                    'min': float(np.min(md_data)),
                    'max': float(np.max(md_data))
                }
                logging.info(f"MD statistics: mean={np.mean(md_data):.6f}")
            except Exception as e:
                logging.warning(f"Could not read MD statistics: {e}")

    def generate_summary_report(self):
        """Generate comprehensive summary report from existing QC files"""
        logging.info("Generating summary report from existing QC files...")
        
        # Read existing QC CSV files instead of running QC checks
        self.read_existing_qc_files()
        
        # Check for QC images and basic statistics only
        self.check_qc_images_only()
        self.check_basic_statistics()
        
        # Determine overall status
        statuses = []
        for qc_type in ['topup_qc', 'skull_stripping_qc', 'eddy_qc', 'registration_within_qc', 
                       'registration_mni_qc', 'dtifit_qc', 'file_existence_qc']:
            if qc_type in self.qc_results and 'status' in self.qc_results[qc_type]:
                status = self.qc_results[qc_type]['status']
                # Ignore SKIP status in overall calculation
                if status != 'SKIP':
                    statuses.append(status)
        
        # Overall status logic
        if 'ERROR' in statuses or 'FAIL' in statuses:
            overall_status = 'FAIL'
        elif 'WARNING' in statuses:
            overall_status = 'WARNING'
        else:
            overall_status = 'PASS'
        
        self.qc_results['overall_status'] = overall_status
        
        # Save JSON in subject folder
        try:
            summary_json = os.path.join(self.qc_dir, f"{self.subject_id}_qc_results.json")
            with open(summary_json, 'w') as f:
                json.dump(self.qc_results, f, indent=2, default=str)
            logging.info(f"QC results saved to: {summary_json}")
                
        except PermissionError:
            # Fallback to temp directory
            summary_json = f"/tmp/{self.subject_id}_qc_results.json"
            with open(summary_json, 'w') as f:
                json.dump(self.qc_results, f, indent=2, default=str)
            logging.warning(f"Using fallback location: {summary_json}")
        
        # Generate summary CSV
        self.generate_summary_csv()
        
        # Generate HTML report
        self.generate_html_report()
        
        logging.info(f"Overall QC Status: {overall_status}")
        logging.info(f"Summary saved to: {summary_json}")
        
        return self.qc_results

    def generate_summary_csv(self):
        """Generate summary CSV with key metrics"""
        summary_data = {
            'subject_id': self.subject_id,
            'session': self.qc_results.get('session', 'N/A'),
            'timestamp': self.qc_results['timestamp'],
            'overall_status': self.qc_results['overall_status'],
            'topup_status': self.qc_results.get('topup_qc', {}).get('status', 'N/A'),
            'skull_stripping_status': self.qc_results.get('skull_stripping_qc', {}).get('status', 'N/A'),
            'eddy_status': self.qc_results.get('eddy_qc', {}).get('status', 'N/A'),
            'registration_within_status': self.qc_results.get('registration_within_qc', {}).get('status', 'N/A'),
            'registration_mni_status': self.qc_results.get('registration_mni_qc', {}).get('status', 'N/A'),
            'dtifit_status': self.qc_results.get('dtifit_qc', {}).get('status', 'N/A'),
        }
        
        # Add brain volume if available
        skull_qc = self.qc_results.get('skull_stripping_qc', {})
        if 'volume_measurements' in skull_qc and skull_qc['volume_measurements']:
            for i, vol_measure in enumerate(skull_qc['volume_measurements']):
                summary_data[f'brain_volume_scan{i}_ml'] = vol_measure['brain_volume_ml']
        
        # Add DTI statistics if available
        dtifit_qc = self.qc_results.get('dtifit_qc', {})
        if 'map_statistics' in dtifit_qc:
            if 'dipy_fa.nii.gz' in dtifit_qc['map_statistics']:
                fa_stats = dtifit_qc['map_statistics']['dipy_fa.nii.gz']
                summary_data['fa_mean'] = fa_stats['mean']
                summary_data['fa_std'] = fa_stats['std']
            
            if 'dipy_md.nii.gz' in dtifit_qc['map_statistics']:
                md_stats = dtifit_qc['map_statistics']['dipy_md.nii.gz']
                summary_data['md_mean'] = md_stats['mean']
                summary_data['md_std'] = md_stats['std']
        
        # Save CSV in subject folder
        try:
            summary_csv = os.path.join(self.qc_dir, f"{self.subject_id}_qc_summary.csv")
            df = pd.DataFrame([summary_data])
            df.to_csv(summary_csv, index=False)
            logging.info(f"Summary CSV saved to: {summary_csv}")
            
            # Also append to combined summary for all subjects (avoid duplicates)
            all_subjects_csv = os.path.join(self.qc_base_dir, "all_subjects_summary.csv")
            if os.path.exists(all_subjects_csv):
                # Read existing data
                existing_df = pd.read_csv(all_subjects_csv)
                # Remove any existing entries for this subject
                existing_df = existing_df[existing_df['subject_id'] != self.subject_id]
                # Append new entry
                combined_df = pd.concat([existing_df, df], ignore_index=True)
                combined_df.to_csv(all_subjects_csv, index=False)
            else:
                df.to_csv(all_subjects_csv, index=False)
            logging.info(f"Updated combined summary: {all_subjects_csv}")
            
        except PermissionError:
            summary_csv = f"/tmp/{self.subject_id}_qc_summary.csv"
            df = pd.DataFrame([summary_data])
            df.to_csv(summary_csv, index=False)
            logging.warning(f"Using fallback location: {summary_csv}")

    def generate_html_report(self):
        """Generate enhanced HTML report with images and tables"""
        logging.info("Generating enhanced HTML report...")
        
        html_content = f"""
        <!DOCTYPE html>
        <html>
        <head>
            <title>DTI QC Report - {self.subject_id}</title>
            <meta charset="UTF-8">
            <meta name="viewport" content="width=device-width, initial-scale=1.0">
            <style>
                body {{ font-family: Arial, sans-serif; margin: 20px; }}

                h1 {{ color: #2c3e50; }}
                h2 {{ color: #2c3e50; }}
                
                .pass {{ color: green; font-weight: bold; }}
                .warning {{ color: orange; font-weight: bold; }}
                .fail {{ color: red; font-weight: bold; }}
                .error {{ color: red; font-weight: bold; }}
                
                .stats {{ background: #f9f9f9; padding: 15px; margin: 20px 0; }}
                .section {{ margin: 20px 0; }}
                
                table {{ border-collapse: collapse; width: 100%; margin: 20px 0; }}
                th, td {{ border: 1px solid #ddd; padding: 8px; text-align: left; }}
                th {{ background-color: #f2f2f2; }}
                
                .images-grid {{
                    display: grid;
                    grid-template-columns: repeat(auto-fit, minmax(300px, 1fr));
                    gap: 20px;
                    margin: 20px 0;
                }}
                .image-card {{
                    border: 1px solid #dee2e6;
                    border-radius: 8px;
                    overflow: hidden;
                    box-shadow: 0 2px 8px rgba(0,0,0,0.1);
                    background: white;
                }}
                .image-card img {{ 
                    width: 100%; 
                    height: auto; 
                    display: block;
                }}
                .image-caption {{
                    padding: 15px;
                    background: #f8f9fa;
                    font-weight: 600;
                    color: #495057;
                    text-align: center;
                }}
                
                .section {{ 
                    margin-bottom: 40px; 
                    padding: 20px;
                    border: 1px solid #dee2e6;
                    border-radius: 8px;
                    background: white;
                }}
                
                .stats-highlight {{
                    background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
                    color: white;
                    padding: 20px;
                    border-radius: 8px;
                    margin: 20px 0;
                }}
                
                .footer {{
                    text-align: center;
                    margin-top: 40px;
                    padding-top: 20px;
                    border-top: 2px solid #dee2e6;
                    color: #6c757d;
                }}
            </style>
        </head>
        <body>
            <h1>DTI Quality Control Report</h1>
            <h2>Subject: {self.subject_id}</h2>
            <p>Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</p>
            <p><a href="../../DTI_QC_Summary.html">← Back to Summary Report</a></p>
            
            <table>
                <tr><th colspan="2" style="background-color: #e6f3ff; text-align: center;">Overall Summary</th></tr>
                <tr><th>Overall Status</th><td><span class="{self.qc_results['overall_status'].lower()}">{self.qc_results['overall_status']}</span></td></tr>
                <tr><th>Processing Steps</th><td>6</td></tr>"""
        
        # Add summary statistics
        passed_count = sum(1 for qc_type in ['topup_qc', 'skull_stripping_qc', 'eddy_qc', 'registration_within_qc', 'registration_mni_qc', 'dtifit_qc'] 
                          if self.qc_results.get(qc_type, {}).get('status') == 'PASS')
        
        html_content += f"""
                <tr><th>Passed Steps</th><td class="pass">{passed_count}</td></tr>"""
        
        # Add brain volume if available
        skull_qc = self.qc_results.get('skull_stripping_qc', {})
        if 'volume_measurements' in skull_qc and skull_qc['volume_measurements']:
            avg_volume = sum(vm['brain_volume_ml'] for vm in skull_qc['volume_measurements']) / len(skull_qc['volume_measurements'])
            html_content += f"""
                <tr><th>Brain Volume (avg)</th><td>{avg_volume:.0f} mL</td></tr>"""
        
        # Add FA statistics if available
        dtifit_qc = self.qc_results.get('dtifit_qc', {})
        if 'map_statistics' in dtifit_qc and 'dipy_fa.nii.gz' in dtifit_qc['map_statistics']:
            fa_mean = dtifit_qc['map_statistics']['dipy_fa.nii.gz']['mean']
            html_content += f"""
                <tr><th>FA Mean</th><td>{fa_mean:.3f}</td></tr>"""
        

        
        # Add separator row
        html_content += """
                <tr><th colspan="2" style="background-color: #f0f8ff; text-align: center;">QC Steps Details</th></tr>"""
        
        # Add File Existence QC first
        if 'file_existence_qc' in self.qc_results:
            file_qc = self.qc_results['file_existence_qc']
            status = file_qc.get('status', 'UNKNOWN')
            html_content += f"""
                <tr><th>File Existence Check</th><td class="{status.lower()}">{status}</td></tr>"""
            if 'files_checked' in file_qc:
                html_content += f"""
                <tr><td>&nbsp;&nbsp;Files Checked</td><td>{file_qc['files_checked']}</td></tr>
                <tr><td>&nbsp;&nbsp;Files Found</td><td>{file_qc.get('files_found', 'N/A')}</td></tr>
                <tr><td>&nbsp;&nbsp;Files Missing</td><td>{file_qc.get('files_missing', 'N/A')}</td></tr>"""
        
        # Add each QC section to the same table
        qc_sections = [
            ('topup_qc', 'Top up Correction'),
            ('skull_stripping_qc', 'Skull Stripping'),
            ('eddy_qc', 'Eddy Correction'), 
            ('registration_within_qc', 'Within-Session Registration'),
            ('registration_mni_qc', 'MNI Registration'),
            ('dtifit_qc', 'DTI Fit')
        ]
        
        for qc_key, section_name in qc_sections:
            if qc_key in self.qc_results:
                qc_data = self.qc_results[qc_key]
                status = qc_data.get('status', 'UNKNOWN')
                
                html_content += f"""
                <tr><th>{section_name}</th><td class="{status.lower()}">{status}</td></tr>"""
                
                # Add specific details for each QC type
                if qc_key == 'topup_qc':
                    # Add topup QC image links - specific files only
                    topup_dir = os.path.join(self.output_base_dir, 'B0_correction', self.subject_id)
                    topup_session = self.get_session_name(topup_dir)
                    topup_dir = self.find_session_directory(topup_dir)
                    if os.path.exists(topup_dir):
                        # Look for QC images with flexible pattern matching
                        image_files = glob.glob(os.path.join(topup_dir, "QC-*.png"))
                        image_links = []
                        for img_path in image_files:
                            img_name = os.path.basename(img_path)
                            if topup_session:
                                rel_path = f"../../../B0_correction/{self.subject_id}/{topup_session}/{img_name}"
                            else:
                                rel_path = f"../../../B0_correction/{self.subject_id}/{img_name}"
                            image_links.append(f'<a href="{rel_path}" target="_blank">{img_name}</a>')
                        if image_links:
                            html_content += f"""
                            <tr><td>&nbsp;&nbsp;QC Images</td><td>{' | '.join(image_links)}</td></tr>"""
                    
                elif qc_key == 'skull_stripping_qc':
                    # Show brain volumes from CSV
                    if 'volume_measurements' in qc_data:
                        for vm in qc_data['volume_measurements']:
                            html_content += f"""
                            <tr><td>&nbsp;&nbsp;Brain Vol Scan {vm['scan']}</td><td>{vm['brain_volume_ml']:.1f} mL</td></tr>"""
                    
                    # Add skull stripping QC image links - specific files only
                    skull_dir = os.path.join(self.output_base_dir, 'Skull_stripping', self.subject_id)
                    skull_session = self.get_session_name(skull_dir)
                    skull_dir = self.find_session_directory(skull_dir)
                    if os.path.exists(skull_dir):
                        # Look for QC images with flexible pattern matching
                        image_files = glob.glob(os.path.join(skull_dir, "*desc-qc.png"))
                        image_links = []
                        for img_path in image_files:
                            img_name = os.path.basename(img_path)
                            if skull_session:
                                rel_path = f"../../../Skull_stripping/{self.subject_id}/{skull_session}/{img_name}"
                            else:
                                rel_path = f"../../../Skull_stripping/{self.subject_id}/{img_name}"
                            image_links.append(f'<a href="{rel_path}" target="_blank">{img_name}</a>')
                        if image_links:
                            html_content += f"""
                            <tr><td>&nbsp;&nbsp;QC Images</td><td>{' | '.join(image_links)}</td></tr>"""
                    
                elif qc_key == 'eddy_qc':
                    # Add eddy QC image links - specific files only
                    eddy_dir = os.path.join(self.output_base_dir, 'Eddy_correction', self.subject_id)
                    eddy_session = self.get_session_name(eddy_dir)
                    eddy_dir = self.find_session_directory(eddy_dir)
                    if os.path.exists(eddy_dir):
                        # Look for QC images with flexible pattern matching
                        image_files = glob.glob(os.path.join(eddy_dir, "QC-*.png"))
                        image_links = []
                        for img_path in image_files:
                            img_name = os.path.basename(img_path)
                            if eddy_session:
                                rel_path = f"../../../Eddy_correction/{self.subject_id}/{eddy_session}/{img_name}"
                            else:
                                rel_path = f"../../../Eddy_correction/{self.subject_id}/{img_name}"
                            image_links.append(f'<a href="{rel_path}" target="_blank">{img_name}</a>')
                        if image_links:
                            html_content += f"""
                            <tr><td>&nbsp;&nbsp;QC Images</td><td>{' | '.join(image_links)}</td></tr>"""
                    
                elif qc_key == 'registration_within_qc':
                    # Check if within registration is skipped or available
                    if qc_data.get('status') == 'SKIP':
                        html_content += f"""
                        <tr><td>&nbsp;&nbsp;Reason</td><td>{qc_data.get('reason', 'Skipped')}</td></tr>"""
                    else:
                        # Show CSV data from existing files
                        csv_dir = os.path.join(self.output_base_dir, 'QC', self.subject_id)
                        csv_session = self.get_session_name(csv_dir)
                        csv_dir = self.find_session_directory(csv_dir)
                        csv_path = os.path.join(csv_dir, 'within_subject_registraction_qc.csv')
                        if os.path.exists(csv_path):
                            if csv_session:
                                rel_csv_path = f"../../../QC/{self.subject_id}/{csv_session}/within_subject_registraction_qc.csv"
                            else:
                                rel_csv_path = f"../../../QC/{self.subject_id}/within_subject_registraction_qc.csv"
                            html_content += f"""
                            <tr><td>&nbsp;&nbsp;CSV Report</td><td><a href="{rel_csv_path}" target="_blank">within_subject_registraction_qc.csv</a></td></tr>"""
                        # Add Dice coefficient metrics from existing CSV
                        if 'rigid_dice' in qc_data:
                            html_content += f"""
                            <tr><td>&nbsp;&nbsp;Rigid Dice</td><td>{qc_data['rigid_dice']:.4f}</td></tr>
                            <tr><td>&nbsp;&nbsp;Rigid Status</td><td>{qc_data.get('rigid_status', 'Unknown')}</td></tr>"""
                    
                elif qc_key == 'registration_mni_qc':
                    # Show CSV data from existing files
                    csv_files = [
                        ('mni_registraction_qc.csv', 'MNI Registration CSV'),
                        ('file_existance.csv', 'File Existence CSV')
                    ]
                    csv_dir = os.path.join(self.output_base_dir, 'QC', self.subject_id)
                    csv_session = self.get_session_name(csv_dir)
                    csv_dir = self.find_session_directory(csv_dir)
                    for csv_file, csv_desc in csv_files:
                        csv_path = os.path.join(csv_dir, csv_file)
                        if os.path.exists(csv_path):
                            if csv_session:
                                rel_csv_path = f"../../../QC/{self.subject_id}/{csv_session}/{csv_file}"
                            else:
                                rel_csv_path = f"../../../QC/{self.subject_id}/{csv_file}"
                            html_content += f"""
                            <tr><td>&nbsp;&nbsp;{csv_desc}</td><td><a href="{rel_csv_path}" target="_blank">{csv_file}</a></td></tr>"""
                    
                    # Add MNI registration metrics from existing CSV (Dice coefficient)
                    if 'rigid_dice' in qc_data:
                        html_content += f"""
                        <tr><td>&nbsp;&nbsp;Rigid Dice</td><td>{qc_data['rigid_dice']:.4f}</td></tr>
                        <tr><td>&nbsp;&nbsp;Rigid Status</td><td>{qc_data.get('rigid_status', 'Unknown')}</td></tr>"""
                    if 'affine_dice' in qc_data:
                        html_content += f"""
                        <tr><td>&nbsp;&nbsp;Affine Dice</td><td>{qc_data['affine_dice']:.4f}</td></tr>
                        <tr><td>&nbsp;&nbsp;Affine Status</td><td>{qc_data.get('affine_status', 'Unknown')}</td></tr>"""
                    
                elif qc_key == 'dtifit_qc':
                    # Add DTI fit QC image links with flexible pattern matching
                    dtifit_dir = os.path.join(self.output_base_dir, 'Dtifit', self.subject_id)
                    dtifit_session = self.get_session_name(dtifit_dir)
                    dtifit_dir = self.find_session_directory(dtifit_dir)
                    if os.path.exists(dtifit_dir):
                        # Look for QC images with flexible pattern matching
                        image_files = glob.glob(os.path.join(dtifit_dir, "QC-*.png"))
                        image_links = []
                        for img_path in image_files:
                            img_name = os.path.basename(img_path)
                            if dtifit_session:
                                rel_path = f"../../../Dtifit/{self.subject_id}/{dtifit_session}/{img_name}"
                            else:
                                rel_path = f"../../../Dtifit/{self.subject_id}/{img_name}"
                            image_links.append(f'<a href="{rel_path}" target="_blank">{img_name}</a>')
                        if image_links:
                            html_content += f"""
                            <tr><td>&nbsp;&nbsp;QC Images</td><td>{' | '.join(image_links)}</td></tr>"""
                

                

        
        # Close the main table
        html_content += """
            </table>
        
            <p><em>Generated: """ + datetime.now().strftime('%Y-%m-%d %H:%M:%S') + """</em></p>
        </body>
        </html>
        """
        
        # Save HTML report in subject folder
        try:
            html_report = os.path.join(self.qc_dir, f"{self.subject_id}_report.html")
            with open(html_report, 'w') as f:
                f.write(html_content)
            logging.info(f"HTML report saved to: {html_report}")
            
            # Generate combined HTML report for all subjects
            self.generate_combined_html()
            
        except PermissionError:
            html_report = f"/tmp/{self.subject_id}_report.html"
            with open(html_report, 'w') as f:
                f.write(html_content)
            logging.warning(f"Using fallback location: {html_report}")

    def generate_combined_html(self):
        """Generate simple combined HTML report for all subjects"""
        try:
            # Read all subjects summary CSV
            all_subjects_csv = os.path.join(self.qc_base_dir, "all_subjects_summary.csv")
            if not os.path.exists(all_subjects_csv):
                return
                
            df = pd.read_csv(all_subjects_csv)
            
            # Simple HTML template
            html_content = f"""
            <!DOCTYPE html>
            <html>
            <head>
                <title>DTI QC - All Subjects Summary</title>
                <style>
                    body {{ font-family: Arial, sans-serif; margin: 20px; }}
                    h1 {{ color: #2c3e50; }}
                    table {{ border-collapse: collapse; width: 100%; }}
                    th, td {{ border: 1px solid #ddd; padding: 8px; text-align: left; }}
                    th {{ background-color: #f2f2f2; }}
                    .pass {{ color: green; font-weight: bold; }}
                    .warning {{ color: orange; font-weight: bold; }}
                    .fail {{ color: red; font-weight: bold; }}
                    .stats {{ background: #f9f9f9; padding: 15px; margin: 20px 0; }}
                </style>
            </head>
            <body>
                <h1>DTI Quality Control - Summary Report</h1>
                <p>Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</p>
                
                <div class="stats">
                    <h2>Overall Statistics</h2>
                    <p><strong>Total Subjects:</strong> {len(df)}</p>
                    <p><strong>Passed:</strong> {len(df[df['overall_status'] == 'PASS'])}</p>
                    <p><strong>Warnings:</strong> {len(df[df['overall_status'] == 'WARNING'])}</p>
                    <p><strong>Failed:</strong> {len(df[df['overall_status'] == 'FAIL'])}</p>
                </div>
                
                <h2>Subjects Summary</h2>
                <table>
                    <tr>
                        <th>Subject ID</th>
                        <th>Session</th>
                        <th>Overall Status</th>
                        <th>Brain Volume (mL)</th>
                        <th>FA Mean</th>
                        <th>MD Mean</th>
                        <th>Individual Report</th>
                    </tr>
            """
            
            for _, row in df.iterrows():
                status_class = row['overall_status'].lower()
                brain_vol = row.get('brain_volume_scan0_ml', 'N/A')
                fa_mean = row.get('fa_mean', 'N/A')
                md_mean = row.get('md_mean', 'N/A')
                
                # Handle NaN and empty values safely
                try:
                    if fa_mean != 'N/A' and pd.notna(fa_mean) and fa_mean != '':
                        fa_mean = f"{float(fa_mean):.3f}"
                    else:
                        fa_mean = 'N/A'
                except:
                    fa_mean = 'N/A'
                    
                try:
                    if md_mean != 'N/A' and pd.notna(md_mean) and md_mean != '':
                        md_mean = f"{float(md_mean):.6f}"
                    else:
                        md_mean = 'N/A'
                except:
                    md_mean = 'N/A'
                    
                try:
                    if brain_vol != 'N/A' and pd.notna(brain_vol) and brain_vol != '':
                        brain_vol = f"{float(brain_vol):.1f}"
                    else:
                        brain_vol = 'N/A'
                except:
                    brain_vol = 'N/A'
                
                session_info = row.get('session', 'N/A')
                html_content += f"""
                    <tr>
                        <td>{row['subject_id']}</td>
                        <td>{session_info}</td>
                        <td><span class="{status_class}">{row['overall_status']}</span></td>
                        <td>{brain_vol}</td>
                        <td>{fa_mean}</td>
                        <td>{md_mean}</td>
                        <td><a href="{row['subject_id']}/{row['subject_id']}_report.html">View Report</a></td>
                    </tr>
                """
            
            html_content += """
                </table>
            </body>
            </html>
            """
            
            # Save combined report
            combined_report = os.path.join(self.qc_base_dir, "DTI_QC_Summary.html")
            with open(combined_report, 'w') as f:
                f.write(html_content)
            logging.info(f"Combined HTML report saved to: {combined_report}")
            
        except Exception as e:
            logging.warning(f"Could not generate combined HTML: {e}")

    def create_documentation(self):
        """Create documentation explaining the organized structure"""
        try:
            readme_content = f"""# DTI Quality Control Results
## Subject: {self.subject_id}

### 📁 Directory Structure:
```
DTI_QC/
├── {self.subject_id}/
│   ├── reports/           # HTML reports and summaries
│       ├── qc_images/        # QC images organized by processing step
│       │   ├── topup/        # Top-up correction images
│       │   ├── skull_stripping/  # Skull stripping QC images
│       │   ├── eddy/         # Eddy correction images
│       │   ├── registration/ # Registration QC images
│   │   └── dtifit/       # DTI fit results images
│   └── data/             # Raw QC data (JSON, CSV)
├── DTI_QC_Summary.html      # Combined HTML report
├── all_subjects_summary.csv # Combined CSV
└── logs/                    # Processing logs
└── documentation/           # README files and guides
```

### Files in this subject:
- **reports/{self.subject_id}_qc_report.html** - Main QC report
- **data/{self.subject_id}_qc_results.json** - Detailed results
- **data/{self.subject_id}_qc_summary.csv** - Summary statistics

### Overall QC Status: {self.qc_results.get('overall_status', 'UNKNOWN')}

Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
"""
            
            readme_file = os.path.join(self.subject_dir, "README.md")
            with open(readme_file, 'w') as f:
                f.write(readme_content)
            
            # Also create a main documentation file
            main_readme = os.path.join(self.docs_dir, "DTI_QC_Guide.md")
            if not os.path.exists(main_readme):
                main_content = """# DTI Quality Control System Documentation

## Overview
This directory contains organized DTI Quality Control results with the following structure:

### Directory Organization:
- **{subject_id}/**: Individual subject results  
- **DTI_QC_Summary.html**: Combined results for all subjects
- **logs/**: Processing logs
- **documentation/**: This guide and other docs

### QC Steps Evaluated:
1. **Top-up Correction** - B0 field distortion correction
2. **Skull Stripping** - Brain extraction quality
3. **Eddy Correction** - Motion and eddy current correction
4. **Registration** - Spatial normalization quality
5. **DTI Fit** - Tensor model fitting results

### Status Levels:
- **PASS**: All checks successful
- **WARNING**: Minor issues detected
- **FAIL**: Significant problems requiring attention

### Usage:
1. View individual HTML reports for detailed subject QC
2. Check combined CSV for population statistics
3. Review logs for processing details
"""
                with open(main_readme, 'w') as f:
                    f.write(main_content)
                    
        except Exception as e:
            logging.warning(f"Could not create documentation: {e}")

# ==========================================
# Main Function
# ==========================================

def main():
    """Main function for DTI QC"""
    parser = argparse.ArgumentParser(description='DTI Preprocessing Quality Control')
    parser.add_argument('subject_id', help='Subject ID to process')
    parser.add_argument('--output-dir', '-o', 
                       default=OUTPUT_DIR,
                       help='Output directory containing processed data')
    parser.add_argument('--verbose', '-v', action='store_true',
                       help='Enable verbose logging')
    
    args = parser.parse_args()
    
    # Setup logging
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)
    
    log_file = setup_logging(args.output_dir, args.subject_id)
    
    try:
        # Initialize DTI QC
        dti_qc = DTIQualityControl(args.subject_id, args.output_dir)
        
        # Generate comprehensive QC report
        results = dti_qc.generate_summary_report()
        
        logging.info("DTI QC completed successfully!")
        logging.info(f"Overall status: {results['overall_status']}")
        logging.info(f"QC results saved to: {dti_qc.qc_dir}")
        
        # Return appropriate exit code
        if results['overall_status'] in ['FAIL', 'ERROR']:
            sys.exit(1)
        elif results['overall_status'] == 'WARNING':
            sys.exit(2)
        else:
            sys.exit(0)
            
    except Exception as e:
        logging.error(f"DTI QC failed with error: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()
