import numpy as np
import librosa
import math
import torchaudio

# --- Constants ---
SR_TARGET = 32000 # Standard sample rate for music generation

# --- Helper Function: Key/Mode Detection ---
def _to_scalar(x, fallback=None):
    """Return a Python float from a numpy array / list / scalar, else fallback."""
    try:
        arr = np.asarray(x)
        if arr.size == 0:
            return fallback
        return float(arr.ravel()[0])
    except Exception:
        return fallback

def safe_round(x, ndigits=0, fallback=None):
    try:
        return round(float(x), ndigits)
    except Exception:
        return fallback

# ---------- I/O & duration ----------

def load_mono_resample(path: str, sr: int | None = 32000, target_sr: int | None = None):
    """
    Load an audio file as MONO at target sample rate using librosa.
    Accepts either 'sr' (default) or 'target_sr' (alias used by the UI).
    Returns: (y: np.ndarray [T], sr: int)
    """
    use_sr = target_sr if target_sr is not None else sr
    y, _sr = librosa.load(path, sr=use_sr, mono=True)
    y = np.asarray(y, dtype=np.float32)
    return y, use_sr


def get_audio_duration(path: str, sr: int = 32000) -> float:
    """Duration in seconds at the given resample rate."""
    y, _ = load_mono_resample(path, sr=sr)
    return float(len(y)) / float(sr) if len(y) else 0.0

def estimate_bpm_key(vocal_path, sr=32000):
    """
    Returns (bpm:int|None, key_label:str|None, vocal_range:str|None)
    - BPM is rounded to nearest int when valid
    - Key is a nice label (e.g., 'C major' / 'A minor') when detectable
    """
    y, _sr = librosa.load(vocal_path, sr=sr, mono=True)
    # BPM (librosa often returns an array; use first element)
    tempo_arr = librosa.beat.tempo(y=y, sr=_sr, aggregate=None)
    tempo = _to_scalar(tempo_arr, fallback=None)
    bpm = int(round(tempo)) if tempo and np.isfinite(tempo) and tempo > 0 else None

    # crude key guess (fallback if chroma is too weak)
    try:
        chroma = librosa.feature.chroma_cqt(y=y, sr=_sr)
        chroma_mean = chroma.mean(axis=1)
        pitch_class = int(np.argmax(chroma_mean))
        KEY_NAMES = ['C','C#','D','D#','E','F','F#','G','G#','A','A#','B']
        key_guess = KEY_NAMES[pitch_class]
        # very rough major/minor hint from spectral centroid vs. mode energy
        mode = 'major' if chroma_mean[[0,4,7]].sum() >= chroma_mean[[2,5,9]].sum() else 'minor'
        key_label = f"{key_guess} {mode}"
    except Exception:
        key_label = None

    # vocal range (rough)
    try:
        f0 = librosa.yin(y, fmin=librosa.note_to_hz("C2"), fmax=librosa.note_to_hz("C6"), sr=_sr)
        f0 = f0[np.isfinite(f0)]
        if f0.size:
            lo = librosa.hz_to_note(np.percentile(f0, 10))
            hi = librosa.hz_to_note(np.percentile(f0, 90))
            vocal_range = f"{lo}–{hi}"
        else:
            vocal_range = None
    except Exception:
        vocal_range = None

    return bpm, key_label, vocal_range

def detect_key_mode(y: np.ndarray, sr: int) -> tuple[str, str]:
    """
    Automatically detects the musical key and mode (Major/Minor) of an audio signal
    using Chroma features and Krumhansl-Schmuckler templates.
    """
    if librosa is None:
        return "", ""

    # 1. Feature Extraction: Calculate Chroma (Pitch Class Profile)
    # Use CQT (Constant-Q Transform) for better pitch resolution than STFT
    chroma = librosa.feature.chroma_cqt(y=y, sr=sr, bins_per_octave=36)
    chroma_mean = chroma.mean(axis=1)
    
    # Normalize the vector for comparison
    chroma_norm = chroma_mean / chroma_mean.sum() if chroma_mean.sum() else np.zeros(12)

    # 2. Key Profiles (Krumhansl-Schmuckler style templates)
    # 0 = C, 1 = C#, ..., 11 = B
    MAJOR_PROFILE = [6.35, 2.23, 3.48, 2.33, 4.38, 4.09, 2.52, 5.19, 2.39, 3.66, 2.29, 2.88]
    MINOR_PROFILE = [6.33, 2.68, 3.52, 5.38, 2.60, 3.53, 2.54, 4.75, 3.98, 2.69, 3.34, 3.17]

    keys = ['C', 'C#', 'D', 'D#', 'E', 'F', 'F#', 'G', 'G#', 'A', 'A#', 'B']
    best_corr = -1
    best_key = ""
    best_mode = ""

    # 3. Correlation: Shift and correlate profiles against the vocal's chroma
    for i in range(12):
        # Rotate the Major/Minor profile to match the current root note (i)
        major_profile_shifted = np.roll(MAJOR_PROFILE, i)
        minor_profile_shifted = np.roll(MINOR_PROFILE, i)

        # Calculate correlation coefficient
        corr_maj = np.corrcoef(chroma_norm, major_profile_shifted)[0, 1]
        corr_min = np.corrcoef(chroma_norm, minor_profile_shifted)[0, 1]

        # Update best match
        if corr_maj > best_corr:
            best_corr = corr_maj
            best_key = keys[i]
            best_mode = "major"
        if corr_min > best_corr:
            best_corr = corr_min
            best_key = keys[i]
            best_mode = "minor"
    
    # Return Major/Minor for prompt, or blank if confidence is too low (heuristic)
    if best_corr < 0.2: # Low confidence threshold
        return "", "" 
    return best_key, best_mode

# --- Helper Function: Tempo Detection ---

def safe_tempo_detect(vocal_path: str) -> int:
    """Safely estimates the tempo (BPM) from an audio file."""
    if librosa is None:
        return 100
    try:
        y, sr = librosa.load(vocal_path, sr=None, mono=True)
        tempo, _ = librosa.beat.beat_track(y=y, sr=sr)
        tempo = float(tempo[0]) if hasattr(tempo, "__len__") else float(tempo)
        tempo = int(round(tempo)) if math.isfinite(tempo) and tempo > 0 else 100
        return max(40, min(200, tempo))
    except Exception:
        return 100

# --- Helper Function: Vocal Range Detection ---

def detect_vocal_range(y: np.ndarray, sr: int) -> str:
    """Estimates the typical singing range (e.g., C3-C5) from an audio signal."""
    if librosa is None:
        return "C3-C5"
    try:
        # Pitch tracking using YIN
        f0 = librosa.yin(y, fmin=librosa.note_to_hz('C2'), fmax=librosa.note_to_hz('C6'), sr=sr)
        f0 = f0[np.isfinite(f0)]
        if len(f0):
            # Use percentiles to ignore brief outliers
            lo = librosa.hz_to_note(np.percentile(f0, 10))
            hi = librosa.hz_to_note(np.percentile(f0, 90))
            return f"{lo}-{hi}"
        return "C3-C5"
    except Exception:
        return "C3-C5"

# --- Helper Function: Duration Detection ---

def safe_duration_seconds(path: str, clamp_min=8, clamp_max=60) -> int:
    """Safely gets the duration of an audio file in seconds."""
    try:
        info = torchaudio.info(path)
        secs = int(round(info.num_frames / max(1, info.sample_rate)))
        return max(clamp_min, min(clamp_max, secs))
    except Exception:
        # Fallback via librosa if available
        if librosa is not None:
            try:
                secs = int(round(librosa.get_duration(filename=path)))
                return max(clamp_min, min(clamp_max, secs))
            except Exception:
                pass
        return 30

# ==== Loudness + ducking helpers (append to file) =============================

import numpy as _np

def rms_dbfs(x: _np.ndarray) -> float:
    """RMS loudness in dBFS. x must be float32 mono in [-1, 1]."""
    x = _np.asarray(x, dtype=_np.float32)
    return 20.0 * _np.log10(_np.sqrt(_np.mean(_np.square(x)) + 1e-12))

def peak_dbfs(x: _np.ndarray) -> float:
    """Peak level in dBFS."""
    return 20.0 * _np.log10(_np.max(_np.abs(x)) + 1e-12)

def apply_gain_db(x: _np.ndarray, gain_db: float) -> _np.ndarray:
    """Apply linear gain from dB."""
    return x * (10.0 ** (gain_db / 20.0))

def match_target_rms(x: _np.ndarray, target_dbfs: float) -> tuple[_np.ndarray, float]:
    """Return (normalized, applied_gain_db) such that RMS ~= target_dbfs."""
    current = rms_dbfs(x)
    gain_db = float(target_dbfs - current)
    return apply_gain_db(x, gain_db), gain_db

def _env_ma(x: _np.ndarray, sr: int, win_ms: float = 10.0) -> _np.ndarray:
    """Simple envelope via moving average of absolute signal."""
    n = max(1, int(sr * win_ms / 1000.0))
    k = _np.ones(n, dtype=_np.float32) / float(n)
    return _np.convolve(_np.abs(x), k, mode="same")

def duck_instrumental(
    instrumental: _np.ndarray,
    vocal: _np.ndarray,
    sr: int,
    threshold_db: float = -40.0,
    duck_db: float = 4.0,
    attack_ms: float = 12.0,
    release_ms: float = 150.0,
) -> _np.ndarray:
    """
    Light “voice-over” style ducking: when the vocal envelope exceeds a threshold,
    reduce instrumental by duck_db, with attack/release smoothing.
    """
    env = _env_ma(vocal, sr, win_ms=10.0)
    thr_lin = 10.0 ** (threshold_db / 20.0)

    atk_n = max(1, int(sr * attack_ms / 1000.0))
    rel_n = max(1, int(sr * release_ms / 1000.0))

    target_duck = 10.0 ** (-abs(duck_db) / 20.0)  # 0..1
    gain = 1.0
    gains = _np.empty_like(env, dtype=_np.float32)

    for i in range(env.shape[0]):
        want = target_duck if env[i] > thr_lin else 1.0
        # one-pole toward target (different speeds up vs down)
        n = atk_n if want < gain else rel_n
        alpha = 1.0 / float(n)
        gain = (1.0 - alpha) * gain + alpha * want
        gains[i] = gain

    return instrumental * gains


__all__ = [
    "load_mono_resample",
    "get_audio_duration",
    "estimate_bpm_key",
    "safe_round",
]