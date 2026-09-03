import argparse
import glob
import os

import numpy as np
import torch
from tqdm import tqdm
from wvmos import get_wvmos

'''
python tools/eval_wvmos.py -t testset/noisy
'''


def main(args):
    clips = glob.glob(os.path.join(args.testset_dir, "*.wav"))
    clips.extend(glob.glob(os.path.join(args.testset_dir, "*.flac")))
    print(f"Processing {len(clips)} audio files...")

    # Evaluate serially because the wav2vec2 model is not thread-safe.
    # wvmos calls torch.load without specifying weights_only. Its checkpoint
    # contains legacy metadata and needs the trusted full-checkpoint loader.
    torch_load = torch.load
    def load_checkpoint(*load_args, **load_kwargs):
        load_kwargs.setdefault("weights_only", False)
        return torch_load(*load_args, **load_kwargs)
    torch.load = load_checkpoint
    wvmos_model = get_wvmos(cuda=True)
    wvmos_scores = []
    for clip in tqdm(clips, desc="WVMOS calculating"):
        try:
            wvmos_scores.append(wvmos_model.calculate_one(clip))
        except Exception as exc:
            print(f"{clip} generated an exception for WVMOS: {exc}")
            wvmos_scores.append(float("nan"))

    if not wvmos_scores:
        print("No audio files processed successfully.")
        return

    mean_wvmos = np.nanmean(wvmos_scores)
    print({"WVMOS": round(mean_wvmos, 3)})


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-t", "--testset_dir", default=".",
        help="Path to the dir containing audio clips in .wav to be evaluated",
    )

    main(parser.parse_args())
