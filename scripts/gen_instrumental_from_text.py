# --- replace the file body in gen_instrumental_from_text.py ---

import os
import torch
import torchaudio
from audiocraft.models import MusicGen

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
SR = 32000

def _save_wav(wav_t, sr, out_path):
    # (T,) → (1,T) float32 on CPU
    if wav_t.dim() == 2 and wav_t.size(0) == 1:
        wav_t = wav_t[0]
    wav_t = wav_t.detach().to("cpu").float().contiguous()
    wav_t = wav_t.unsqueeze(0)  # mono
    torchaudio.save(out_path, wav_t, sr)

def generate_from_text(
    prompt,
    duration_sec=15,
    model_size="medium",
    cfg_coef=3.0,
    seed=42
):
    repo = {
        "small":  "facebook/musicgen-small",
        "medium": "facebook/musicgen-medium",
        "large":  "facebook/musicgen-large",
    }.get(model_size, "facebook/musicgen-medium")

    model = MusicGen.get_pretrained(repo, device=DEVICE)
    model.set_generation_params(duration=duration_sec, cfg_coef=cfg_coef)
    if seed is not None:
        torch.manual_seed(int(seed))

    with torch.no_grad():
        wav = model.generate(descriptions=[prompt])[0]  # (T,)
    os.makedirs("outputs", exist_ok=True)
    out_path = os.path.join("outputs", "instrumental_32k.wav")
    _save_wav(wav, SR, out_path)
    return out_path

if __name__ == "__main__":
    p = "Devotional bhajan with harmonium and gentle tabla, peaceful, uplifting, no vocals"
    path = generate_from_text(p, duration_sec=14, model_size="medium", cfg_coef=2.8, seed=123)
    print("Saved:", path)
