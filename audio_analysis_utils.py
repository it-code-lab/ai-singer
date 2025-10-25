import numpy as np
import librosa
import math
import torchaudio

# --- Constants ---
SR_TARGET = 32000 # Standard sample rate for music generation

# --- Helper Function: Key/Mode Detection ---

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