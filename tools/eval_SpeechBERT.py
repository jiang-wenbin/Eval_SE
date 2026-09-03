import argparse
from pathlib import Path

import numpy as np
import soundfile as sf
import soxr
import torch
from discrete_speech_metrics import SpeechBERTScore as SBS
from tqdm import tqdm

METRIC = "SpeechBERTScore"
TARGET_FS = 16000

#python tools/run_speechbert.py --ref_dir testset/clean --enh_dir testset/OMLSA --output_dir results/OMLSA/speechbert

class SpeechBERTScoreModel:
    def __init__(self, device="cpu"):
        self.speech_bert_score = SBS(
            sr=TARGET_FS, model_type="mhubert-147", layer=8, use_gpu="cuda" in device
        )

    def __call__(self, reference, sample):
        precision, recall, f1_score = self.speech_bert_score.score(reference, sample)
        return precision, recall, f1_score


def load_files(dir_path):
    files = {}
    for f in Path(dir_path).glob("*.wav"):
        files[f.stem] = f
    for f in Path(dir_path).glob("*.flac"):
        files[f.stem] = f
    return files


def main(args):
    device = args.device
    if device == "cuda" and not torch.cuda.is_available():
        print("CUDA not available, fallback to CPU", flush=True)
        device = "cpu"

    model = SpeechBERTScoreModel(device=device)
    model.speech_bert_score.model.eval()

    ref_files = load_files(args.ref_dir)
    enh_files = load_files(args.enh_dir)

    outdir = Path(args.output_dir)
    outdir.mkdir(parents=True, exist_ok=True)

    scores = {}
    with (outdir / f"{METRIC}.scp").open("w") as w:
        for uid, enh_path in tqdm(sorted(enh_files.items()), desc=METRIC):
            if uid not in ref_files:
                print(f"ref not found for {uid}", flush=True)
                continue
            ref, fs = sf.read(str(ref_files[uid]), dtype="float32")
            inf, fs2 = sf.read(str(enh_path), dtype="float32")
            if fs != fs2:
                print(f"fs mismatch {uid}: {fs} vs {fs2}", flush=True)
                continue
            if ref.ndim == 2:
                ref = ref[:, 0]
            if inf.ndim == 2:
                inf = inf[:, 0]
            if fs != TARGET_FS:
                ref = soxr.resample(ref, fs, TARGET_FS)
                inf = soxr.resample(inf, fs, TARGET_FS)
            min_len = min(len(ref), len(inf))
            ref = ref[:min_len]
            inf = inf[:min_len]
            try:
                with torch.no_grad():
                    score = model(ref, inf)[0]
            except Exception as e:
                print(f"{uid} failed: {e}", flush=True)
                score = float("nan")
            scores[uid] = score
            w.write(f"{uid} {score}\n")

    with (outdir / "RESULTS.txt").open("w") as w:
        w.write(f"{METRIC}: {np.nanmean(list(scores.values())):.4f}\n")
    print(f"Results -> {outdir / 'RESULTS.txt'}", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--ref_dir", type=str, required=True, help="Clean audio dir")
    parser.add_argument("--enh_dir", type=str, required=True, help="Enhanced audio dir")
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--device", type=str, default="cuda")
    args = parser.parse_args()
    main(args)
