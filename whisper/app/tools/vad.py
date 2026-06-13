from pathlib import Path
import numpy as np
import soundfile as sf
import torch
import torchaudio
from silero_vad import get_speech_timestamps
import os

MODEL_DIR = "./models"
TORCH_CACHE_DIR = os.path.join(MODEL_DIR, "torch_cache")

os.makedirs(MODEL_DIR, exist_ok=True)
os.makedirs(TORCH_CACHE_DIR, exist_ok=True)

torch.hub.set_dir(TORCH_CACHE_DIR)

print("Loading the VAD model")

vad_model, vad_utils = torch.hub.load(
    "snakers4/silero-vad",
    "silero_vad",
    trust_repo=True
)

(get_speech_timestamps, _, _, _, _) = vad_utils

use_cuda = os.getenv("USE_CUDA", "false").lower() == "true"
if use_cuda:
    vad_model = vad_model.to("cuda")
vad_model.eval()

def _resample_1d_np(x: np.ndarray, sr: int, target_sr: int) -> tuple[np.ndarray, int]:
    x = x.astype(np.float32, copy=False)
    if sr == target_sr:
        return x, sr
    xt = torch.from_numpy(x).unsqueeze(0)  # (1,n)
    xt = torchaudio.functional.resample(xt, sr, target_sr)
    return xt.squeeze(0).numpy(), target_sr

def _ranges_to_mask(n: int, ranges: list[tuple[int, int]]) -> np.ndarray:
    mask = np.zeros(n, dtype=bool)
    for s, e in ranges:
        s = max(0, min(n, int(s)))
        e = max(0, min(n, int(e)))
        if e > s:
            mask[s:e] = True
    return mask

def _dilate_mask(mask: np.ndarray, sr: int, pad_ms: int) -> np.ndarray:
    if pad_ms <= 0:
        return mask
    k = int(sr * pad_ms / 1000)
    if k <= 0:
        return mask
    out = mask.copy()
    out |= np.roll(mask, k)
    out |= np.roll(mask, -k)
    out[:k] |= mask[:k]
    out[-k:] |= mask[-k:]
    return out

def _apply_fade(audio: np.ndarray, sr: int, mask: np.ndarray, fade_ms: int) -> np.ndarray:
    if fade_ms <= 0:
        return audio
    fade = int(sr * fade_ms / 1000)
    if fade <= 1:
        return audio

    y = audio.copy()
    m = mask.astype(np.int8)
    d = np.diff(m, prepend=m[0])
    starts = np.where(d == 1)[0]
    ends = np.where(d == -1)[0]

    for s in starts:
        a = max(0, s - fade)
        b = min(len(y), s + fade)
        w = np.linspace(0.0, 1.0, b - a, dtype=np.float32)
        y[a:b] *= w

    for e in ends:
        a = max(0, e - fade)
        b = min(len(y), e + fade)
        w = np.linspace(1.0, 0.0, b - a, dtype=np.float32)
        y[a:b] *= w

    return y

def _vad_gate_1d(
    audio_1d: np.ndarray,
    sr: int,
    *,
    target_sr: int,
    threshold: float,
    min_speech_ms: int,
    min_silence_ms: int,
    keep_silence_ms: int,
    fade_ms: int,
) -> tuple[np.ndarray, int]:
    """Gating na 1 kanale -> zwraca (audio_po_gate, sr_użyty)."""
    audio_rs, sr2 = _resample_1d_np(audio_1d, sr, target_sr)

    if use_cuda:
        wav = torch.from_numpy(audio_rs).float().to("cuda")
    else:
        wav = torch.from_numpy(audio_rs).float()

    speech = get_speech_timestamps(
        wav,
        vad_model,
        threshold=threshold,
        sampling_rate=sr2,
        min_speech_duration_ms=min_speech_ms,
        min_silence_duration_ms=min_silence_ms,
        return_seconds=False,
    )

    ranges = [(s["start"], s["end"]) for s in speech]
    # ranges w sample index -> zamiana na ms
    ranges_ms = [(int(s * 1000 / sr2), int(e * 1000 / sr2)) for s, e in ranges]
    mask = _ranges_to_mask(len(audio_rs), ranges)
    mask = _dilate_mask(mask, sr2, keep_silence_ms)

    y = audio_rs.copy()
    y[~mask] = np.random.normal(0, 1e-5, size=np.sum(~mask))
    y = _apply_fade(y, sr2, mask, fade_ms)
    return y.astype(np.float32, copy=False), sr2, ranges_ms


@torch.no_grad()
def silero_gate_each_channel_then_merge_mono(
    in_wav: str,
    out_wav: str | None = None,
    *,
    target_sr: int = 16000,
    threshold: float = 0.4,
    min_speech_ms: int = 200,
    min_silence_ms: int = 200,
    keep_silence_ms: int = 500,
    fade_ms: int = 10,
) -> tuple[str, list[list[tuple[int, int]]], list[str]]:
    
    in_path = Path(in_wav)
    out_path = Path(out_wav) if out_wav else in_path.with_name(f"{in_path.stem}_speech_mono{in_path.suffix}")

    x, sr = sf.read(str(in_path), always_2d=True)  # (n, ch)
    n, ch = x.shape

    channels_ranges: list[list[tuple[int, int]]] = []
    audios_per_channels = []

    # --- MONO ---
    if ch == 1:
        ch_audio = x[:, 0].astype(np.float32, copy=False)

        y, sr_used, channel_ranges = _vad_gate_1d(
            ch_audio, sr,
            target_sr=target_sr,
            threshold=threshold,
            min_speech_ms=min_speech_ms,
            min_silence_ms=min_silence_ms,
            keep_silence_ms=keep_silence_ms,
            fade_ms=fade_ms,
        )
        channels_ranges.append(channel_ranges)
        
        print(channels_ranges)

        if not np.isfinite(y).all():
            raise ValueError("Audio contains NaN or Inf")

        max_abs = np.max(np.abs(y))
        if max_abs > 1.0:
            print(f"Warning: max abs = {max_abs}, clipping before save")
            y = np.clip(y, -1.0, 1.0)
        
        sf.write(str(out_path), y, sr_used)
        audios_per_channels.append(str(out_path))
        return str(out_path), channels_ranges, audios_per_channels

    # --- MULTI-CHANNEL ---
    processed: list[np.ndarray] = []
    sr_used: int | None = None

    for c in range(ch):
        ch_audio = x[:, c].astype(np.float32, copy=False)

        y, sr2, channel_ranges = _vad_gate_1d(
            ch_audio, sr,
            target_sr=target_sr,
            threshold=threshold,
            min_speech_ms=min_speech_ms,
            min_silence_ms=min_silence_ms,
            keep_silence_ms=keep_silence_ms,
            fade_ms=fade_ms,
        )

        if sr_used is None:
            sr_used = sr2
        channels_ranges.append(channel_ranges)
        processed.append(y)

        max_abs = np.max(np.abs(y))
        if max_abs > 1.0:
            print(f"Warning: max abs = {max_abs}, clipping before save")
            y = np.clip(y, -1.0, 1.0)
        
        out_path_one_channel = in_path.with_name(f"{in_path.stem}_channel_{c}{in_path.suffix}")
        sf.write(str(out_path_one_channel), y, sr2)
        audios_per_channels.append(str(out_path_one_channel))

    print(channels_ranges)
    
    # wyrównanie długości (czasem resampling daje różnice o 1 próbkę)
    min_len = min(len(p) for p in processed)
    processed = [p[:min_len] for p in processed]

    # downmix do mono
    mono = np.mean(np.stack(processed, axis=0), axis=0).astype(np.float32)
       
    max_abs = np.max(np.abs(mono))
    if max_abs > 1.0:
        print(f"Warning: max abs = {max_abs}, clipping before save")
        mono = np.clip(mono, -1.0, 1.0)
    
    sf.write(str(out_path), mono, int(sr_used))
    return str(out_path), channels_ranges, audios_per_channels

