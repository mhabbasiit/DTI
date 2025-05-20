import os
import glob

def match_file_pattern(subject_folder,pattern):
    pattern_path = os.path.join(subject_folder, pattern)
    matching_files = glob.glob(pattern_path)
    return matching_files[0]