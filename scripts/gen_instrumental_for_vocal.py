# --- replace the core with this in gen_instrumental_for_vocal.py ---

import os
import torch
import torchaudio
from audiocraft.models import MusicGen
from audio_analysis_utils import estimate_bpm_key

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
SR = 32000

def _save_wav(wav_t, sr, out_path):
    # (B,T) or (T,) → (T,) float32 on CPU for torchaudio
    if wav_t.dim() == 2 and wav_t.size(0) == 1:
        wav_t = wav_t[0]
    wav_t = wav_t.detach().to("cpu").float().contiguous()
    # mono
    wav_t = wav_t.unsqueeze(0)  # (1, T)
    torchaudio.save(out_path, wav_t, sr)

def build_prompt(base_prompt, key_label=None, raag_phrase=None, bpm=None):
    parts = [base_prompt.strip()] if base_prompt else []
    if key_label:
        parts.append(f"key: {key_label}")
    if raag_phrase:
        parts.append(raag_phrase)
    if bpm and bpm > 0:
        parts.append(f"around {int(round(bpm))} BPM")
    parts.append("cinematic, cohesive arrangement, no vocals")
    return ", ".join(parts)

def generate_for_vocal(
    vocal_path,
    extra_prompt="",
    duration_sec=15,
    model_size="medium",
    cfg_coef=3.0,
    seed=42,
    raag_phrase=None
):
    # 1) analyze vocal
    bpm, key_label, _vocal_range = estimate_bpm_key(vocal_path)
    prompt = build_prompt(extra_prompt, key_label, raag_phrase, bpm)

    # 2) load model
    repo = {
        "small":  "facebook/musicgen-small",
        "medium": "facebook/musicgen-medium",
        "large":  "facebook/musicgen-large",
    }.get(model_size, "facebook/musicgen-medium")

    model = MusicGen.get_pretrained(repo, device=DEVICE)
    model.set_generation_params(duration=duration_sec, cfg_coef=cfg_coef)

    # 3) seed (MusicGen uses torch seed)
    if seed is not None:
        torch.manual_seed(int(seed))

    # 4) generate
    with torch.no_grad():
        wav = model.generate(descriptions=[prompt])[0]  # (T,) on device
    os.makedirs("outputs", exist_ok=True)
    out_path = os.path.join("outputs", "instrumental_for_vocal.wav")
    _save_wav(wav, SR, out_path)

    return out_path, bpm, key_label

if __name__ == "__main__":
    # minimal CLI example
    path, bpm, key = generate_for_vocal(
        vocal_path="demo_vocal.wav",
        extra_prompt="Bollywood ballad with warm strings and soft tabla",
        duration_sec=20,
        model_size="medium",
        cfg_coef=3.0,
        seed=123,
        raag_phrase="hints of Raag Yaman (Lydian feel)"
    )
    print("Saved:", path, "| BPM:", bpm, "| Key:", key)
