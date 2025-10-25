import numpy as np, librosa, soundfile as sf

VOCAL = "vocal.wav"                      # your acapella (mono)
INSTR = "instrumental_for_vocal_32k.wav" # or instrumental_32k.wav
SR = 32000

# load & resample
v, sr_v = librosa.load(VOCAL, sr=SR, mono=True)
i, sr_i = librosa.load(INSTR, sr=SR, mono=True)

# crude onset-based alignment (shift instrumental to first vocal onset)
on_v = librosa.onset.onset_detect(y=v, sr=SR, units="time")
shift = on_v[0] if len(on_v) else 0.0
pad = int(shift * SR)
i_shift = np.pad(i, (pad, 0))[:max(len(i)+pad, len(v))]

# match lengths
L = max(len(v), len(i_shift))
v = np.pad(v, (0, L-len(v)))
i_shift = np.pad(i_shift, (0, L-len(i_shift)))

# simple ducking under vocal (RMS-based)
win = 1024
rms = librosa.feature.rms(y=v, frame_length=win, hop_length=win//2)[0]
rms = np.interp(np.linspace(0, len(rms)-1, num=L), np.arange(len(rms)), rms)
duck = 1.0 - 0.35 * np.clip(rms/ (rms.max()+1e-6), 0, 1)  # reduce up to ~35%
mix = (i_shift * duck) + (v * 1.0)

# normalise
mx = np.max(np.abs(mix)) + 1e-9
mix = 0.98 * mix / mx

sf.write("song_mix_32k.wav", mix, SR)
print("Saved: song_mix_32k.wav")
