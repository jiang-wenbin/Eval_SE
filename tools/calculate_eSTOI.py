import argparse
import glob
import os
import numpy as np
import soundfile as sf
import soxr
from pystoi import stoi
from tqdm import tqdm

'''
python tools/calculate_eSTOI.py -c wav/clean/ -e wav/gtcrn_enh/
'''

def calculate_estoi(clean_file, enhanced_file):
    ref, fs = sf.read(clean_file, dtype="float32", always_2d=False)
    inf, fs2 = sf.read(enhanced_file, dtype="float32", always_2d=False)

    if ref.ndim == 2:
        ref = ref[:, 0]
    if inf.ndim == 2:
        inf = inf[:, 0]

    if fs2 != fs:
        inf = soxr.resample(inf, fs2, fs)

    if len(inf) > len(ref):
        inf = inf[:len(ref)]
    elif len(inf) < len(ref):
        inf = np.pad(inf, (0, len(ref) - len(inf)), mode='constant')

    estoi_value = stoi(ref, inf, fs_sig=fs, extended=True)
    return estoi_value

def calculate_average_estoi(clean_folder, enhanced_folder):
    ref_files = glob.glob(os.path.join(clean_folder, "*.wav"))
    ref_files.extend(glob.glob(os.path.join(clean_folder, "*.flac")))
    enhanced_files = glob.glob(os.path.join(enhanced_folder, "*.wav"))
    enhanced_files.extend(glob.glob(os.path.join(enhanced_folder, "*.flac")))

    enhanced_dict = {}
    for f in enhanced_files:
        stem = os.path.splitext(os.path.basename(f))[0]
        enhanced_dict[stem] = f

    estoi_list = []
    for ref_file in tqdm(ref_files, desc='Average eSTOI calculating'):
        ref_stem = os.path.splitext(os.path.basename(ref_file))[0]
        matched = None
        for estem, epath in enhanced_dict.items():
            if estem == ref_stem or estem.startswith(ref_stem + '_'):
                matched = epath
                break
        if matched:
            estoi_value = calculate_estoi(ref_file, matched)
            estoi_list.append(estoi_value)
        else:
            print(f"Enhanced file not found for: {ref_file}")

    if estoi_list:
        average_estoi = sum(estoi_list) / len(estoi_list)
        print("Average eSTOI:", average_estoi)
    else:
        print("No eSTOI values calculated. Check if the folders contain matching audio files.")

def main():
    parser = argparse.ArgumentParser(description="Calculate average eSTOI for audio files in specified folders.")
    parser.add_argument('--clean_folder', '-c', type=str, required=True, help="Path to the folder containing clean audio files")
    parser.add_argument('--enhanced_folder', '-e', type=str, required=True, help="Path to the folder containing enhanced audio files")
    args = parser.parse_args()

    calculate_average_estoi(args.clean_folder, args.enhanced_folder)

if __name__ == "__main__":
    main()
