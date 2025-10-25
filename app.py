# app.py
import os
import math
import time
import tempfile
from pathlib import Path
from typing import Optional, Tuple

import torch
import torchaudio
import gradio as gr

# Optional deps used if present; guarded imports
try:
    import librosa
except Exception:
    librosa = None

try:
    import soundfile as sf
except Exception:
    sf = None

from audiocraft.models import MusicGen

SR = 32000
OUT_DIR = Path("outputs")
OUT_DIR.mkdir(exist_ok=True)

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# ---- Prompt Builder Presets & Helper ---------------------------------------
PROMPT_PRESETS = {
    "Bollywood Romantic Ballad": {
        "default_instruments": ["acoustic guitar", "piano", "warm bass", "soft pads", "light drums"],
        "tempos": ["slow (70–85)", "medium (86–100)"],
        "moods": ["romantic", "emotional", "tender", "nostalgic"],
        "percussion": ["soft kick/snare", "tambourine", "brush kit"],
        "references": ["Arijit Singh", "Amit Trivedi", "Pritam"],
    },
    "Bollywood Upbeat / Dance": {
        "default_instruments": ["synths", "punchy drums", "808 bass", "claps", "bright leads"],
        "tempos": ["fast (105–130)"],
        "moods": ["energetic", "upbeat", "anthemic"],
        "percussion": ["EDM kit", "electronic percussion", "dhol (fusion)"],
        "references": ["Badshah", "Honey Singh", "Pritam"],
    },
    "Devotional (Bhajan)": {
        "default_instruments": ["harmonium", "tabla", "dholak", "acoustic guitar (light)", "tanpura drone", "flute"],
        "tempos": ["slow (65–85)", "medium (86–100)"],
        "moods": ["peaceful", "devotional", "reverent"],
        "percussion": ["tabla", "dholak", "manjeera"],
        "references": ["Anup Jalota", "Jubin Nautiyal (devotional)"],
    },
    "Sufi / Qawwali": {
        "default_instruments": ["harmonium", "tabla", "dholak", "claps", "strings pad", "bass (subtle)"],
        "tempos": ["medium (80–100)"],
        "moods": ["spiritual", "uplifting", "intense (build)"],
        "percussion": ["tabla", "dholak", "handclaps"],
        "references": ["Nusrat Fateh Ali Khan", "Rahat Fateh Ali Khan"],
    },
    "Ghazal": {
        "default_instruments": ["acoustic guitar", "piano", "harmonium (soft)", "violin/strings", "upright bass (soft)"],
        "tempos": ["slow (65–80)"],
        "moods": ["poetic", "melancholic", "intimate"],
        "percussion": ["very light tabla or brush kit"],
        "references": ["Jagjit Singh"],
    },
    "Sanskrit Mantra / Chant": {
        "default_instruments": ["tanpura drone", "gentle pads", "soft bells", "light percussion (anklet/ghungroo)"],
        "tempos": ["slow (60–80)"],
        "moods": ["meditative", "calming", "spiritual"],
        "percussion": ["very light (if any)"],
        "references": ["Deva Premal (style ref)"],
    },
    "Kirtan / Call-and-Response": {
        "default_instruments": ["harmonium", "tabla", "dholak", "acoustic guitar", "claps", "tanpura drone"],
        "tempos": ["medium (80–100)"],
        "moods": ["devotional", "participatory", "uplifting"],
        "percussion": ["tabla", "dholak", "manjeera"],
        "references": ["Krishna Das (style ref)"],
    },
    "Filmi Orchestral / Cinematic": {
        "default_instruments": ["strings (legato + staccato)", "piano", "timpani/taikos", "brass (soft)", "choir pads"],
        "tempos": ["slow build (70–95)"],
        "moods": ["cinematic", "emotional", "grand"],
        "percussion": ["orchestral percussion"],
        "references": ["A.R. Rahman (cinematic)"],
    },
    "Lo-fi Bollywood": {
        "default_instruments": ["lofi drums", "warm keys", "soft guitar", "vinyl crackle", "subtle bass"],
        "tempos": ["slow/medium (70–92)"],
        "moods": ["dreamy", "nostalgic", "chill"],
        "percussion": ["lofi kit"],
        "references": ["lofi remix vibe"],
    },
}

ARRANGEMENT_CHOICES = [
    "steady all through",
    "intro calm → verse moderate → chorus full",
    "intro pad/drone → groove enters at verse → chorus big, then soft outro",
    "soft intro → build through verse → big chorus → breakdown → final chorus",
]

def _clean_join(items):
    return ", ".join([s for s in items if s and s.strip()])

def build_prompt_from_selections(song_type, instruments, extra_instr, mood, tempo_note,
                                 arrangement, era, key_note, scale_mode, percussion, ref_artists,
                                 fx, mix_notes, language_hint, lyric_theme):
    line1 = [song_type]
    if language_hint:
        line1.append(f"({language_hint})")
    if lyric_theme:
        line1.append(f"— theme: {lyric_theme}")
    header = " ".join([s for s in line1 if s])

    # Instruments
    inst_parts = []
    if instruments:
        inst_parts.append(_clean_join(instruments))
    if extra_instr:
        inst_parts.append(extra_instr)
    inst_block = _clean_join(inst_parts)

    # Percussion
    perc_block = _clean_join(percussion)

    # Mood / Tempo / Key
    props = []
    if mood:
        props.append(mood)
    if tempo_note:
        props.append(f"tempo {tempo_note} BPM range")
    if key_note:
        if scale_mode:
            props.append(f"key {key_note} {scale_mode}")
        else:
            props.append(f"key {key_note}")
    elif scale_mode:
        props.append(f"mode {scale_mode}")
    if era:
        props.append(f"{era} aesthetics")

    # Arrangement & mix
    arrange = f"Arrangement: {arrangement}." if arrangement else ""
    fx_block = f"FX: {fx}." if fx else ""
    mix = f"Mix/Production: {mix_notes}." if mix_notes else ""

    # References
    ref = f"Reference vibe: {_clean_join(ref_artists)}." if ref_artists else ""

    # Final prompt
    prompt_lines = [
        header.strip(),
        f"Instruments: {inst_block}." if inst_block else "",
        f"Percussion: {perc_block}." if perc_block else "",
        f"Character: {_clean_join(props)}." if props else "",
        arrange,
        fx_block,
        mix,
        ref
    ]
    prompt = " ".join([p for p in prompt_lines if p]).strip()
    return prompt

# ------------ Utilities ------------

def _ensure_stereo(y: torch.Tensor) -> torch.Tensor:
    """Make [channels, time] with 2 channels."""
    if y.dim() == 1:
        y = y.unsqueeze(0)  # [1, T]
    if y.shape[0] == 1:
        y = y.repeat(2, 1)  # stereo
    return y

def _ensure_2d_for_save(y: torch.Tensor) -> torch.Tensor:
    """torchaudio.save expects 2D [channels, time]."""
    if y.dim() == 3 and y.shape[0] == 1:
        y = y.squeeze(0)
    return y

def _resample_if_needed(wav: torch.Tensor, sr: int, target_sr: int = SR) -> torch.Tensor:
    if sr == target_sr:
        return wav
    # wav: [channels, time]
    return torchaudio.functional.resample(wav, sr, target_sr)

def _normalize_peak(wav: torch.Tensor, target_peak: float = 0.98) -> torch.Tensor:
    peak = wav.abs().max().item()
    if peak > 0:
        wav = wav * (target_peak / peak)
    return wav

def _db_to_lin(db):
    return 10.0 ** (db / 20.0)

def _soft_limiter(wav: torch.Tensor, ceiling_db: float = -1.0) -> torch.Tensor:
    # Gentle arctan/tanh-style soft clip then scale to ceiling
    ceiling = _db_to_lin(ceiling_db)
    wav = torch.tanh(2.5 * wav)
    return _normalize_peak(wav, ceiling)

def _save_wav(wav_t: torch.Tensor, sr: int, out_path: Path):
    # shape to [channels, samples]
    if wav_t.ndim == 1:
        wav_t = wav_t.unsqueeze(0)  # [1, T]
    elif wav_t.ndim == 3:
        # collapse batch if present: [B, C, T] -> [C, T]
        wav_t = wav_t[0]

    # ensure float32 on CPU and sane range
    wav_t = wav_t.detach().to(torch.float32).clamp_(-1.0, 1.0).cpu()

    out_path.parent.mkdir(parents=True, exist_ok=True)
    torchaudio.save(str(out_path), wav_t, sr)  # channels_first=True by default


def _safe_tempo_detect(vocal_path: str) -> int:
    """Best-effort tempo detection; returns fallback 100 if librosa unavailable."""
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

# ------------ Model cache ------------

_MODEL_CACHE = {}

def _get_model(model_size: str = "medium") -> MusicGen:
    """
    Loads and caches a MusicGen model on DEVICE.
    model_size in {"small", "medium", "large"}; 'medium' is a good balance.
    """
    key = (model_size, DEVICE)
    if key in _MODEL_CACHE:
        return _MODEL_CACHE[key]

    repo = {
        "small": "facebook/musicgen-small",
        "medium": "facebook/musicgen-medium",
        "large": "facebook/musicgen-large",
    }.get(model_size, "facebook/musicgen-medium")

    model = MusicGen.get_pretrained(repo)
    # new API: set device on subsequent tensor ops
    # Generate returns torch tensor on CPU by default; we can keep it that way.
    _MODEL_CACHE[key] = model
    return model

# ------------ Generation functions ------------

def gen_from_text_ui(
    prompt: str,
    duration: int = 30,
    model_size: str = "medium",
    top_k: int = 250,
    top_p: float = 0.0,
    temperature: float = 1.0,
    cfg_coef: float = 3.0,
    seed: Optional[int] = 42,
    progress=gr.Progress()
) -> Tuple[str, Tuple[int, list]]:
    """
    Returns: (file_path, (sr, [float_samples])) for Gradio Audio
    """
    progress(0, desc="Loading model…")
    model = _get_model(model_size)

    # Parameters
    if seed is not None and isinstance(seed, int):
        torch.manual_seed(seed)

    # Set params
    model.set_generation_params(
        duration=int(duration),
        top_k=int(top_k),
        top_p=float(top_p),
        temperature=float(temperature),
        cfg_coef=float(cfg_coef),
    )

    progress(0.3, desc="Generating audio…")
    wav = model.generate([prompt])[0]  # [time] (numpy) or torch? audiocraft returns torch on CPU
    if isinstance(wav, torch.Tensor):
        wav_t = wav
    else:
        wav_t = torch.from_numpy(wav)

    # Format: [channels, time] at SR
    wav_t = _ensure_stereo(wav_t)
    wav_t = _normalize_peak(wav_t, 0.98)

    ts = int(time.time())
    out_path = OUT_DIR / f"instrumental_from_text_{ts}.wav"
    _save_wav(wav_t, SR, out_path)

    progress(1.0, desc=f"Saved → {out_path.name}")
    # Return a playable tuple for the Audio component (sr, samples)
    # return str(out_path), (SR, wav_t.mean(0).detach().cpu().numpy().tolist())
    return str(out_path), (SR, wav_t.mean(0).detach().cpu().numpy())

def gen_for_vocal_ui(
    vocal_file: str,
    extra_prompt: str = "",
    duration: int = 30,
    model_size: str = "medium",
    cfg_coef: float = 3.0,
    seed: Optional[int] = 42,
    progress=gr.Progress()
) -> Tuple[str, Tuple[int, list], str]:
    """
    Returns: (instrumental_path, (sr, samples), detected_bpm_text)
    """
    if not vocal_file:
        raise gr.Error("Please upload a vocal file (WAV/MP3/FLAC).")

    progress(0, desc="Estimating tempo…")
    bpm = _safe_tempo_detect(vocal_file)

    base_prompt = f"supportive modern pop backing, clean drums, warm bass, light pads, avoid vocals, {bpm} bpm"
    if extra_prompt and extra_prompt.strip():
        prompt = base_prompt + ", " + extra_prompt.strip()
    else:
        prompt = base_prompt

    progress(0.25, desc="Loading model…")
    model = _get_model(model_size)
    if seed is not None and isinstance(seed, int):
        torch.manual_seed(seed)

    model.set_generation_params(
        duration=int(duration),
        cfg_coef=float(cfg_coef),
    )

    progress(0.6, desc="Generating accompaniment…")
    wav = model.generate([prompt])[0]
    wav_t = wav if isinstance(wav, torch.Tensor) else torch.from_numpy(wav)
    wav_t = _ensure_stereo(wav_t)
    wav_t = _normalize_peak(wav_t, 0.98)

    ts = int(time.time())
    out_path = OUT_DIR / f"instrumental_for_vocal_{ts}.wav"
    _save_wav(wav_t, SR, out_path)

    progress(1.0, desc=f"Saved → {out_path.name}")
    # return str(out_path), (SR, wav_t.mean(0).detach().cpu().numpy().tolist()), f"Detected BPM ≈ {bpm}"
    return str(out_path), (SR, wav_t.mean(0).detach().cpu().numpy()), f"Detected BPM ≈ {bpm}"

def mix_ui(
    vocal_file: str,
    instrumental_file: str,
    vocal_gain_db: float = -2.0,
    music_gain_db: float = -1.0,
    limiter_ceiling_db: float = -1.0,
) -> Tuple[str, Tuple[int, list]]:
    if not vocal_file or not instrumental_file:
        raise gr.Error("Please provide both vocal and instrumental files.")

    # Load with torchaudio to avoid unnecessary deps
    v_wav, v_sr = torchaudio.load(vocal_file)   # [channels, time]
    m_wav, m_sr = torchaudio.load(instrumental_file)

    v_wav = _resample_if_needed(v_wav, v_sr, SR)
    m_wav = _resample_if_needed(m_wav, m_sr, SR)

    v_wav = _ensure_stereo(v_wav)
    m_wav = _ensure_stereo(m_wav)

    # Align lengths
    T = max(v_wav.shape[-1], m_wav.shape[-1])
    if v_wav.shape[-1] < T:
        pad = T - v_wav.shape[-1]
        v_wav = torch.nn.functional.pad(v_wav, (0, pad))
    if m_wav.shape[-1] < T:
        pad = T - m_wav.shape[-1]
        m_wav = torch.nn.functional.pad(m_wav, (0, pad))

    # Apply gain
    v_wav = v_wav * _db_to_lin(vocal_gain_db)
    m_wav = m_wav * _db_to_lin(music_gain_db)

    # Simple blend + limiter
    mix = v_wav + m_wav
    mix = _soft_limiter(mix, limiter_ceiling_db)

    ts = int(time.time())
    out_path = OUT_DIR / f"song_mix_{ts}.wav"
    _save_wav(mix, SR, out_path)
    return str(out_path), (SR, mix.mean(0).detach().cpu().numpy().tolist())

# ------------ UI ------------

def render_prompt_builder(target_extra_prompt_box: gr.Textbox):
    with gr.Accordion("🎼 Prompt Builder (Indian Styles)", open=False):
        with gr.Row():
            song_type = gr.Dropdown(
                label="Song Type",
                choices=list(PROMPT_PRESETS.keys()),
                value="Bollywood Romantic Ballad"
            )
            language = gr.Dropdown(
                label="Language / Lyric Type",
                choices=["Hindi", "Urdu", "Sanskrit", "Punjabi", "Instrumental (no lyrics)"],
                value="Hindi"
            )
            era = gr.Dropdown(
                label="Era / Aesthetic",
                choices=["modern", "2000s", "1990s", "classic filmi", "fusion"],
                value="modern"
            )

        with gr.Row():
            mood = gr.Dropdown(
                label="Mood",
                choices=sum([v["moods"] for v in PROMPT_PRESETS.values()], []),
                value="romantic"
            )
            tempo = gr.Dropdown(
                label="Tempo",
                choices=sum([v["tempos"] for v in PROMPT_PRESETS.values()], []),
                value="medium (86–100)"
            )
            key_note = gr.Textbox(label="Key (optional, e.g., A minor)", value="")
            scale_mode = gr.Dropdown(
                label="Mode (optional)",
                choices=["", "major", "minor", "dorian", "mixolydian", "harmonic minor"],
                value=""
            )

        with gr.Row():
            arrangement = gr.Dropdown(
                label="Arrangement / Energy Curve",
                choices=ARRANGEMENT_CHOICES,
                value="intro calm → verse moderate → chorus full"
            )
            fx = gr.Textbox(label="FX / Space (optional)", placeholder="gentle reverb, light delay, subtle chorus on pads")

        with gr.Row():
            ref_artists = gr.CheckboxGroup(
                label="Reference Artists (optional)",
                choices=list({a for v in PROMPT_PRESETS.values() for a in v.get("references", [])}),
                value=[]
            )

        with gr.Row():
            base_instruments = gr.CheckboxGroup(
                label="Core Instruments",
                choices=list({i for v in PROMPT_PRESETS.values() for i in v["default_instruments"]}),
                value=PROMPT_PRESETS["Bollywood Romantic Ballad"]["default_instruments"]
            )
            extra_instruments = gr.Textbox(label="Add Instruments (comma-separated)", placeholder="sitar, sarod, shehnai")

        with gr.Row():
            percussion = gr.CheckboxGroup(
                label="Percussion",
                choices=list({p for v in PROMPT_PRESETS.values() for p in v.get("percussion", [])}),
                value=["soft kick/snare", "tambourine"]
            )

        with gr.Row():
            lyric_theme = gr.Textbox(label="Lyric / Theme (optional)", placeholder="longing, devotion, festival, sunrise meditation")
            mix_notes = gr.Textbox(label="Mix / Production Notes (optional)", placeholder="warm, intimate vocal pocket, gentle sidechain")

        def _sync_presets(st):
            preset = PROMPT_PRESETS.get(st, PROMPT_PRESETS["Bollywood Romantic Ballad"])
            return (
                gr.update(value=preset["moods"][0] if preset["moods"] else None),
                gr.update(value=preset["tempos"][0] if preset["tempos"] else None),
                gr.update(value=preset["default_instruments"]),
                gr.update(value=preset.get("percussion", [])),
                gr.update(value=[]),  # clear refs by default
            )

        song_type.change(
            _sync_presets,
            inputs=[song_type],
            outputs=[mood, tempo, base_instruments, percussion, ref_artists]
        )

        make_prompt_btn = gr.Button("✨ Generate Prompt")
        out_prompt = gr.Textbox(label="Generated Prompt", lines=4)

        def _make_prompt(st, bi, ei, m, t, arr, e, keyn, mode, perc, refs, fxv, mixv, lang, theme):
            # extract BPM number range for readability ("86–100" → keep as is)
            tempo_note = t
            prompt = build_prompt_from_selections(
                song_type=st,
                instruments=bi,
                extra_instr=ei,
                mood=m,
                tempo_note=tempo_note,
                arrangement=arr,
                era=e,
                key_note=keyn,
                scale_mode=mode,
                percussion=perc,
                ref_artists=refs,
                fx=fxv,
                mix_notes=mixv,
                language_hint=lang,
                lyric_theme=theme,
            )
            return prompt, gr.update(value=prompt)

        make_prompt_btn.click(
            _make_prompt,
            inputs=[song_type, base_instruments, extra_instruments, mood, tempo, arrangement, era,
                    key_note, scale_mode, percussion, ref_artists, fx, mix_notes, language, lyric_theme],
            outputs=[out_prompt, target_extra_prompt_box]
        )

    return out_prompt

with gr.Blocks(title="AI Singer Studio") as demo:
    gr.Markdown(
        f"""# AI Singer Studio
**Device:** `{DEVICE}` • **Sample Rate:** {SR} Hz  
Generate instrumentals with Meta MusicGen, match an uploaded vocal, and mix.
"""
    )

    with gr.Tab("1) Instrumental from Text"):
        with gr.Row():
            prompt = gr.Textbox(label="Text prompt", placeholder="e.g., modern pop instrumental, bright synths, punchy drums, 120 bpm, no vocals", lines=3)
        with gr.Row():
            duration = gr.Slider(8, 60, value=30, step=1, label="Duration (seconds)")
            model_size = gr.Dropdown(["small", "medium", "large"], value="medium", label="Model size")
            seed = gr.Number(value=42, label="Seed (int or blank for random)")
        with gr.Accordion("Advanced", open=False):
            top_k = gr.Slider(0, 1000, value=250, step=1, label="top_k")
            top_p = gr.Slider(0.0, 1.0, value=0.0, step=0.01, label="top_p")
            temperature = gr.Slider(0.1, 2.0, value=1.0, step=0.05, label="temperature")
            cfg_coef = gr.Slider(0.0, 6.0, value=3.0, step=0.1, label="cfg_coef")

        gen_btn = gr.Button("Generate Instrumental")
        out_path1 = gr.File(label="Saved WAV")
        audio1 = gr.Audio(label="Preview", type="numpy")

        gen_btn.click(
            gen_from_text_ui,
            inputs=[prompt, duration, model_size, top_k, top_p, temperature, cfg_coef, seed],
            outputs=[out_path1, audio1],
        )

    with gr.Tab("2) Instrumental for Vocal"):
        vocal_in = gr.Audio(label="Upload Vocal (WAV/MP3/FLAC)", type="filepath")
        extra_prompt = gr.Textbox(label="Extra prompt (optional)", placeholder="e.g., airy pads, subtle guitar")
        _ = render_prompt_builder(target_extra_prompt_box=extra_prompt)
        with gr.Row():
            duration2 = gr.Slider(8, 60, value=30, step=1, label="Duration (seconds)")
            model_size2 = gr.Dropdown(["small", "medium", "large"], value="medium", label="Model size")
            seed2 = gr.Number(value=42, label="Seed (int or blank for random)")
            cfg2 = gr.Slider(0.0, 6.0, value=3.0, step=0.1, label="cfg_coef")
        gen2_btn = gr.Button("Generate Matching Instrumental")
        bpm_text = gr.Markdown()
        out_path2 = gr.File(label="Saved WAV")
        audio2 = gr.Audio(label="Preview", type="numpy")

        def _wrap_gen_for_vocal(vocal_file, extra_prompt, duration, model_size, seed, cfg):
            path, audio, bpm_label = gen_for_vocal_ui(vocal_file, extra_prompt, duration, model_size, cfg, seed)
            bpm_text = f"**{bpm_label}**"
            return bpm_text, path, audio

        gen2_btn.click(
            _wrap_gen_for_vocal,
            inputs=[vocal_in, extra_prompt, duration2, model_size2, seed2, cfg2],
            outputs=[bpm_text, out_path2, audio2]
        )

    with gr.Tab("3) Mix Vocal + Instrumental"):
        v_in = gr.Audio(label="Vocal file", type="filepath")
        m_in = gr.Audio(label="Instrumental file", type="filepath")
        with gr.Row():
            vocal_gain = gr.Slider(-12, 6, value=-2.0, step=0.5, label="Vocal gain (dB)")
            music_gain = gr.Slider(-12, 6, value=-1.0, step=0.5, label="Music gain (dB)")
            tp = gr.Slider(-6, -0.1, value=-1.0, step=0.1, label="Limiter Ceiling (dBTP)")
        mix_btn = gr.Button("Mix")
        out_mix = gr.File(label="Saved Mix")
        audio_mix = gr.Audio(label="Preview", type="numpy")

        mix_btn.click(
            mix_ui,
            inputs=[v_in, m_in, vocal_gain, music_gain, tp],
            outputs=[out_mix, audio_mix]
        )

if __name__ == "__main__":
    # share=False keeps it local; set to True if you want a public link
    demo.launch(share=False)
