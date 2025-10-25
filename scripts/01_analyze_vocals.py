import json, sys, librosa, numpy as np
from pathlib import Path

IN = Path("data/vocals_clean.wav") if Path("data/vocals_clean.wav").exists() else Path("data/vocals_raw.wav")
OUT = Path("data/analysis.json")

# --- load ---
y, sr = librosa.load(IN, sr=48000, mono=True)

# --- tempo (BPM) ---
tempo, beat_frames = librosa.beat.beat_track(y=y, sr=sr, trim=True)
beats_sec = librosa.frames_to_time(beat_frames, sr=sr).tolist()

# --- rough key detection (very heuristic without Essentia) ---
# Pitch class energy
chroma = librosa.feature.chroma_cqt(y=y, sr=sr)
chroma_mean = chroma.mean(axis=1)
pitch_classes = ['C','C#','D','D#','E','F','F#','G','G#','A','A#','B']
key_index = int(np.argmax(chroma_mean))
rough_key = pitch_classes[key_index]

# --- sections via onset detection ---
onset_env = librosa.onset.onset_strength(y=y, sr=sr)
onset_frames = librosa.onset.onset_detect(onset_envelope=onset_env, sr=sr, units='time')
sections = onset_frames.tolist()

analysis = {
    "sample_rate": sr,
    "estimated_bpm": float(tempo),
    "rough_key": rough_key,
    "beats_sec": beats_sec[:256],       # truncate for readability
    "section_markers_sec": sections
}

OUT.write_text(json.dumps(analysis, indent=2))
print(f"Saved analysis → {OUT}\n", json.dumps(analysis, indent=2))
