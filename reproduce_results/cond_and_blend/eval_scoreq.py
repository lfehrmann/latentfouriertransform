"""
Offline SCOREQ evaluation for a fixed subset of examples.

This script is separated from train.py/callbacks so final evaluation
can be run without resuming training loops.
"""

from __future__ import annotations

import argparse
import json
import os
import tempfile

import numpy as np
import torch
import torchaudio

import config
from latentft.lightning.lit_fmdiffae import FMDiffAEModule


def load_indices(path: str) -> np.ndarray:
    values: list[int] = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            s = line.strip()
            if not s or s.startswith("#"):
                continue
            values.append(int(s))
    arr = np.asarray(values, dtype=np.int64)
    if arr.ndim != 1:
        raise ValueError("indices file must contain one integer index per line")
    return arr


@torch.no_grad()
def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt_path", type=str, required=True)
    parser.add_argument("--spec_npy", type=str, required=True)
    parser.add_argument("--audio_npy", type=str, required=True)
    parser.add_argument(
        "--indices_file",
        type=str,
        required=True,
        help="Text file with one 0-based index per line.",
    )
    parser.add_argument("--out_json", type=str, required=True)
    parser.add_argument("--num_steps", type=int, default=35)
    parser.add_argument("--scoreq_domain", type=str, default="natural")
    parser.add_argument("--sample_rate", type=int, default=22050)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument(
        "--save_wavs_dir",
        type=str,
        default=None,
        help=(
            "Optional directory to persist generated/reference WAV pairs "
            "(useful for VISQOL). If omitted, temporary WAVs are deleted."
        ),
    )
    args = parser.parse_args()

    try:
        import scoreq
    except ImportError as exc:
        raise SystemExit("Install SCOREQ first: pip install scoreq") from exc

    indices = load_indices(args.indices_file)
    if len(indices) == 0:
        raise SystemExit("No indices found in indices file.")

    specs_all = np.load(args.spec_npy)
    audios_all = np.load(args.audio_npy)
    if specs_all.shape[0] != audios_all.shape[0]:
        raise SystemExit("spec_npy and audio_npy must have equal first dimension")
    if (indices < 0).any() or (indices >= specs_all.shape[0]).any():
        raise SystemExit("indices_file contains out-of-range index")

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")

    lit = FMDiffAEModule.load_from_checkpoint(
        args.ckpt_path, map_location="cpu", strict=True
    )
    model = (
        lit.ema_model.module
        if getattr(lit, "ema_model", None) is not None
        else lit.model
    )
    model.eval().to(device)
    lit.transform.model.to(device)

    batch_specs = torch.from_numpy(specs_all[indices].astype(np.float32)).to(device)
    n = batch_specs.shape[0]
    lows = torch.zeros(n, device=device)
    highs = torch.ones(n, device=device)

    generated_specs = model.generate(
        inputs=batch_specs,
        lows=lows,
        highs=highs,
        num_steps=args.num_steps,
        pbar=True,
    )
    gen_audios = lit.transform.batched_inverse_transform(generated_specs, pbar=True).cpu()
    ref_audios = torch.from_numpy(audios_all[indices].astype(np.float32)).cpu()

    predictor = scoreq.Scoreq(data_domain=args.scoreq_domain, mode="ref")
    distances: list[float] = []
    if args.save_wavs_dir is None:
        tmpdir_cm = tempfile.TemporaryDirectory()
        tmpdir = tmpdir_cm.__enter__()
    else:
        tmpdir_cm = None
        tmpdir = os.path.abspath(args.save_wavs_dir)
        os.makedirs(tmpdir, exist_ok=True)
        print(f"Saving WAV pairs to {tmpdir}")
    try:
        for i, (gen_audio, ref_audio) in enumerate(zip(gen_audios, ref_audios)):
            gen_path = os.path.join(tmpdir, f"gen_{i:04d}.wav")
            ref_path = os.path.join(tmpdir, f"ref_{i:04d}.wav")
            torchaudio.save(gen_path, gen_audio.unsqueeze(0), args.sample_rate)
            torchaudio.save(ref_path, ref_audio.unsqueeze(0), args.sample_rate)
            distances.append(float(predictor.predict(test_path=gen_path, ref_path=ref_path)))
    finally:
        if tmpdir_cm is not None:
            tmpdir_cm.__exit__(None, None, None)

    out_path = os.path.abspath(args.out_json)
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    out = {
        "num_samples": int(n),
        "indices": indices.tolist(),
        "scoreq_domain": args.scoreq_domain,
        "scoreq_mode": "ref",
        "mean": float(np.mean(distances)),
        "std": float(np.std(distances)),
        "distances": distances,
        "out_json": out_path,
    }
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2)

    print(json.dumps({"num_samples": out["num_samples"], "mean": out["mean"], "std": out["std"]}, indent=2))
    print(f"Wrote {out_path}")


if __name__ == "__main__":
    main()
