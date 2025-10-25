import torch, torchaudio, numpy as np
from audiocraft.models import MusicGen

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
SR = 32000  # MusicGen uses 32 kHz
DURATION = 20  # seconds

prompt = (
    "uplifting pop backing track, bright acoustic guitar strums, warm bass, "
    "tight drums with snare on 2 and 4, 120 bpm, no vocals"
)

print("Loading model (first run will download weights)...")
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

model.set_generation_params(duration=DURATION, top_k=250, top_p=0.0, temperature=1.0)

print("Generating...")
with torch.no_grad():
    wav = model.generate(descriptions=[prompt], progress=True)  # [B, T]

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
