import numpy as np, torch, torchaudio, librosa
from audiocraft.models import MusicGen

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
SR_VOC = 32000

VOCAL_PATH = "vocal.wav"  # your dry vocal (mono is best)

y, sr = librosa.load(VOCAL_PATH, sr=None, mono=True)
# rough tempo estimate
tempo, _ = librosa.beat.beat_track(y=y, sr=sr)
# tempo can be a numpy array (e.g., [120.0]), so extract scalar
tempo = float(tempo[0]) if isinstance(tempo, (np.ndarray, list)) else float(tempo)
tempo = int(round(tempo)) if np.isfinite(tempo) and tempo > 0 else 100


# rough vocal range estimate
f0 = librosa.yin(y, fmin=librosa.note_to_hz('C2'), fmax=librosa.note_to_hz('C6'), sr=sr)
f0 = f0[np.isfinite(f0)]
if len(f0):
    lo = librosa.hz_to_note(np.percentile(f0, 10))
    hi = librosa.hz_to_note(np.percentile(f0, 90))
    vrange = f"{lo}-{hi}"
else:
    vrange = "C3-C5"

duration = max(10, int(np.ceil(len(y)/sr)))  # seconds

prompt = (
    f"modern pop backing track around {tempo} bpm, support a singer with range {vrange}, "
    "steady drums, electric bass, warm keys, no vocals"
)

print(f"Prompt: {prompt} (duration ~{duration}s)")
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

try:
    # Many builds accept a device kwarg
    model = MusicGen.get_pretrained("facebook/musicgen-medium", device=DEVICE)
except TypeError:
    # Fallback: set after load
    model = MusicGen.get_pretrained("facebook/musicgen-medium")
    try:
        model.set_device(DEVICE)  # Audiocraft API in 1.3.x
    except AttributeError:
        pass  # model will still run on CPU; CUDA still speeds up heavy ops underneath

model.set_generation_params(duration=duration, top_k=250)

with torch.no_grad():
    wav = model.generate(descriptions=[prompt], progress=True)[0].cpu()

# torchaudio.save("instrumental_for_vocal_32k.wav", wav.unsqueeze(0), 32000)
# print("Saved: instrumental_for_vocal_32k.wav")

# wav is a torch Tensor on GPU/CPU returned by model.generate
wav = wav[0].detach().cpu()  # remove batch -> now [1, T] or [T]

# Ensure 2D [channels, time] for torchaudio.save
if wav.dim() == 1:          # [T]
    wav = wav.unsqueeze(0)  # -> [1, T]
elif wav.dim() == 3:        # [1, 1, T]
    wav = wav.squeeze(0)    # -> [1, T]
# if wav.dim() == 2, it's already [C, T]; do nothing

wav = wav.to(torch.float32)
torchaudio.save("instrumental_32k.wav", wav, 32000)
print("Saved: instrumental_32k.wav")