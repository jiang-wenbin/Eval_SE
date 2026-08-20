import argparse
from pathlib import Path

import librosa
import numpy as np
import torch
from tqdm import tqdm

METRIC = "UTMOS"

#python tools/run_utmos.py --enh_dir /home/wz1023/Eval_SE/testset/OMLSA --output_dir /home/wz1023/Eval_SE/results/OMLSA/utmos

def utmos_metric(model, audio_path):
    wave, sr = librosa.load(audio_path, sr=None, mono=True)
    wave = torch.from_numpy(wave).unsqueeze(0).to(device=model.device)
    utmos_score = model(wave, sr)
    return float(utmos_score.cpu().item())


def main(args):
    device = args.device
    if device == "cuda" and not torch.cuda.is_available():
        print("CUDA not available, fallback to CPU", flush=True)
        device = "cpu"

    model = torch.hub.load(
        "tarepan/SpeechMOS:v1.2.0", args.utmos_tag, trust_repo=True
    ).to(device=device)
    model.device = device

    inf_dir = Path(args.enh_dir)
    files = sorted(list(inf_dir.glob("*.wav")) + list(inf_dir.glob("*.flac")))

    outdir = Path(args.output_dir)
    outdir.mkdir(parents=True, exist_ok=True)

    scores = {}
    with (outdir / f"{METRIC}.scp").open("w") as w:
        for f in tqdm(files, desc=METRIC):
            uid = f.stem
            try:
                score = utmos_metric(model, str(f))
            except Exception as e:
                print(f"{f} failed: {e}", flush=True)
                score = float("nan")
            scores[uid] = score
            w.write(f"{uid} {score}\n")

    with (outdir / "RESULTS.txt").open("w") as w:
        w.write(f"{METRIC}: {np.nanmean(list(scores.values())):.4f}\n")
    print(f"Results -> {outdir / 'RESULTS.txt'}", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--enh_dir", type=str, required=True, help="Enhanced audio dir")
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--utmos_tag", type=str, default="utmos22_strong")
    args = parser.parse_args()
    main(args)
