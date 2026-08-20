import argparse
import glob
import os
import numpy as np
import soundfile as sf
import soxr
from pesq import pesq
from tqdm import tqdm

'''
python tools/calculate_PESQ.py --clean_folder wav/clean --enhanced_folder wav/noisy
'''

def calculate_pesq(clean_file, enhanced_file):
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

    if fs >= 16000:
        if fs > 16000:
            ref = soxr.resample(ref, fs, 16000)
            inf = soxr.resample(inf, fs, 16000)
            fs = 16000
        mode = 'wb'
    elif fs == 8000:
        mode = 'nb'
    else:
        raise ValueError(f"Sample rate must be 8000 or 16000+, but got {fs}")

    pesq_score = pesq(fs, ref, inf, mode)
    return pesq_score

def calculate_average_pesq(clean_folder, enhanced_folder):
    ref_files = glob.glob(os.path.join(clean_folder, "*.wav"))
    ref_files.extend(glob.glob(os.path.join(clean_folder, "*.flac")))
    enhanced_files = glob.glob(os.path.join(enhanced_folder, "*.wav"))
    enhanced_files.extend(glob.glob(os.path.join(enhanced_folder, "*.flac")))

    enhanced_dict = {}
    for f in enhanced_files:
        stem = os.path.splitext(os.path.basename(f))[0]
        enhanced_dict[stem] = f

    pesq_list = []
    for ref_file in tqdm(ref_files, desc='Average PESQ calculating'):
        ref_stem = os.path.splitext(os.path.basename(ref_file))[0]
        matched = None
        for estem, epath in enhanced_dict.items():
            if estem == ref_stem or estem.startswith(ref_stem + '_'):
                matched = epath
                break
        if matched:
            pesq_score = calculate_pesq(ref_file, matched)
            pesq_list.append(pesq_score)
        else:
            print(f"Enhanced file not found for: {ref_file}")

    if pesq_list:
        average_pesq = sum(pesq_list) / len(pesq_list)
        print("Average PESQ:", average_pesq)
    else:
        print("No PESQ values calculated. Check if the folders contain matching audio files.")

def main():
    parser = argparse.ArgumentParser(description="Calculate average PESQ scores for audio files in specified folders.")
    parser.add_argument('--clean_folder', '-c', type=str, required=True, help="Path to the folder containing clean audio files")
    parser.add_argument('--enhanced_folder', '-e', type=str, required=True, help="Path to the folder containing enhanced audio files")
    args = parser.parse_args()

    calculate_average_pesq(args.clean_folder, args.enhanced_folder)

if __name__ == "__main__":
    main()
