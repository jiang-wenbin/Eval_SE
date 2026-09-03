import os
import glob
import argparse
import torch
from torchmetrics.functional.audio import scale_invariant_signal_noise_ratio
import soundfile as sf
from tqdm import tqdm

'''
python tools/calculate_SI-SNR.py --clean_folder testset/clean --enhanced_folder testset/noisy
'''

def calculate_si_snr(enhanced_file, clean_file):
    enhanced_signal, _ = sf.read(enhanced_file)
    clean_signal, _ = sf.read(clean_file)

    min_length = min(len(clean_signal), len(enhanced_signal))
    clean_signal = clean_signal[:min_length]
    enhanced_signal = enhanced_signal[:min_length]

    enhanced_signal_tensor = torch.tensor(enhanced_signal)
    clean_signal_tensor = torch.tensor(clean_signal)

    si_snr_value = scale_invariant_signal_noise_ratio(enhanced_signal_tensor, clean_signal_tensor)
    return si_snr_value.item()

def calculate_average_si_snr(clean_folder, enhanced_folder):
    ref_files = glob.glob(os.path.join(clean_folder, "*.wav"))
    ref_files.extend(glob.glob(os.path.join(clean_folder, "*.flac")))
    enhanced_files = glob.glob(os.path.join(enhanced_folder, "*.wav"))
    enhanced_files.extend(glob.glob(os.path.join(enhanced_folder, "*.flac")))

    enhanced_dict = {}
    for f in enhanced_files:
        stem = os.path.splitext(os.path.basename(f))[0]
        enhanced_dict[stem] = f

    si_snr_list = []
    for ref_file in tqdm(ref_files, desc='Average SI-SNR calculating'):
        ref_stem = os.path.splitext(os.path.basename(ref_file))[0]
        matched = None
        for estem, epath in enhanced_dict.items():
            if estem == ref_stem or estem.startswith(ref_stem + '_'):
                matched = epath
                break
        if matched:
            si_snr_value = calculate_si_snr(matched, ref_file)
            si_snr_list.append(si_snr_value)
        else:
            print(f"Enhanced file not found for: {ref_file}")

    if si_snr_list:
        average_si_snr = sum(si_snr_list) / len(si_snr_list)
        print("Average SI-SNR:", average_si_snr)
    else:
        print("No SI-SNR values calculated. Check if the folders contain matching WAV files.")

def main():
    parser = argparse.ArgumentParser(description="Calculate average SI-SNR for WAV files in specified folders.")
    parser.add_argument('--enhanced_folder','-e', type=str, required=True, help="Path to the folder containing enhanced WAV files")
    parser.add_argument('--clean_folder','-c', type=str, required=True, help="Path to the folder containing clean WAV files")
    args = parser.parse_args()
    
    calculate_average_si_snr(args.clean_folder, args.enhanced_folder)

if __name__ == "__main__":
    main()
