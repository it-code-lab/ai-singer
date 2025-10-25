import os
import math
import time
from pathlib import Path
from typing import Tuple, Optional, List

import torch
import torchaudio
import numpy as np
import gradio as gr

# Audiocraft / MusicGen
from audiocraft.models import MusicGen

# Local helpers (you already have this file)
from audio_analysis_utils import (
    load_mono_resample,
    safe_tempo_detect,
    detect_key_mode,
    detect_vocal_range,
    safe_duration_seconds,
)
from threading import Timer
# -----------------------------------------------------------------------------
# Global config
# -----------------------------------------------------------------------------
SR = 32000  # MusicGen sample rate
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
DEFAULT_MODEL = "facebook/musicgen-medium"

# Cache models by size so switching is instant
_MODEL_CACHE = {}




_mix_timer = None
def debounce_mix(vocal_file, instr_file, vg, ig, og):
    global _mix_timer
    if _mix_timer:
        _mix_timer.cancel()
    _mix_timer = Timer(0.8, lambda: mix_tracks(vocal_file, instr_file, vg, ig, og))
    _mix_timer.start()
    return None, None  # no updates until the Timer runs


def get_model(model_size: str) -> MusicGen:
    """Return a MusicGen model on correct device with sane defaults."""
    if model_size in _MODEL_CACHE:
        return _MODEL_CACHE[model_size]

    name = {
        "Small (Melody)": "facebook/musicgen-small",
        "Medium": "facebook/musicgen-medium",
        "Large": "facebook/musicgen-large",
    }.get(model_size, DEFAULT_MODEL)

    model = MusicGen.get_pretrained(name, device=DEVICE)
    # Slightly conservative defaults; we always override via set_generation_params
    model.set_generation_params(
        use_sampling=True,
        top_k=250,
        top_p=0.0,
        temperature=1.0,
        cfg_coef=3.0,
        duration=10,
    )
    _MODEL_CACHE[model_size] = model
    return model

# -----------------------------------------------------------------------------
# Utility helpers
# -----------------------------------------------------------------------------

def _ensure_2d_cpu_f32(wav_t: torch.Tensor) -> torch.Tensor:
    """Torchaudio.save wants (channels, time) on CPU float32."""
    if wav_t is None:
        raise ValueError("Empty audio tensor")

    if wav_t.dim() == 3:  # (B, C, T) or (B, T, 1)
        # squeeze batch
        wav_t = wav_t[0]
    if wav_t.dim() == 1:  # (T,) -> (1, T)
        wav_t = wav_t.unsqueeze(0)
    elif wav_t.dim() == 2:
        pass
    else:
        # Some models return (T, C) – transpose
        if wav_t.shape[0] < wav_t.shape[1]:
            wav_t = wav_t.transpose(0, 1)
    return wav_t.detach().to("cpu", dtype=torch.float32)


def _save_wav(wav_t: torch.Tensor, sr: int, out_path: Path) -> Path:
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    wav = _ensure_2d_cpu_f32(wav_t)
    torchaudio.save(str(out_path), wav, sr)
    return out_path


def _estimate_duration_from_file(path: str) -> float:
    if not path:
        return 10.0
    info = torchaudio.info(path)
    return max(1.0, info.num_frames / float(info.sample_rate))


# -----------------------------------------------------------------------------
# Prompt building
# -----------------------------------------------------------------------------

BOLLY_INSTRUMENTS = [
    "tabla", "dholak", "bansuri", "sitar", "tanpura",
    "acoustic guitar", "electric bass", "strings section",
    "pads", "piano", "shehnai", "santoor", "harmonium"
]

DEVOTIONAL_INSTRUMENTS = [
    "tabla", "mridangam", "tanpura", "harmonium", "bansuri",
    "soft strings", "temple bells", "santoor", "light pads"
]

MANTRA_INSTRUMENTS = [
    "tanpura", "bells", "drones", "oceanic pads", "subtle percussion"
]

RAGA_HINTS = {
    "None": "",
    "Yaman (Kalyan)": "hints of Raag Yaman (Lydian feel)",
    "Bhairav": "hints of Raag Bhairav (Phrygian dominant flavor)",
    "Bhairavi": "hints of Raag Bhairavi (minor / Komal notes)",
    "Kafi": "hints of Raag Kafi (Dorian mood)",
    "Khamaj": "hints of Raag Khamaj (Mixolydian color)",
    "Bilawal": "hints of Raag Bilawal (Ionian/major)",
}

SONG_TYPES = [
    "Bollywood Ballad",
    "Upbeat Bollywood",
    "Devotional (Bhajan)",
    "Sanskrit Mantra / Chant",
    "Lo-fi Bollywood",
    "Classical Fusion",
]

STRUCTURE_PRESETS = [
    "intro-verse-chorus-verse-chorus-outro",
    "pad intro-verse-chorus-bridge-chorus",
    "short intro-verse-chorus",
]


ALL_INSTRUMENTS = sorted(set(
    BOLLY_INSTRUMENTS + DEVOTIONAL_INSTRUMENTS + MANTRA_INSTRUMENTS
))

def sanitize_instruments(selected):
    return [x for x in (selected or []) if x in ALL_INSTRUMENTS]

# --- replace your current suggested_instruments with this ---
def suggested_instruments(song_type: str) -> List[str]:
    if song_type in ("Bollywood Ballad", "Upbeat Bollywood", "Lo-fi Bollywood"):
        recs = BOLLY_INSTRUMENTS
    elif song_type == "Devotional (Bhajan)":
        recs = DEVOTIONAL_INSTRUMENTS
    elif song_type == "Sanskrit Mantra / Chant":
        recs = MANTRA_INSTRUMENTS
    else:
        # Fallback MUST use only items present in ALL_INSTRUMENTS
        recs = ["piano", "strings section", "acoustic guitar", "electric bass", "pads"]
    # ensure we never return anything outside the fixed choices
    return [x for x in recs if x in ALL_INSTRUMENTS]



def make_prompt(
    song_type: str,
    mood: str,
    energy: str,
    extra: str,
    instruments: List[str],
    raga_name: str,
    bpm: Optional[float],
    key_text: Optional[str],
    vocal_range: Optional[str],
    structure: Optional[str],
) -> str:
    lines = []
    if song_type:
        lines.append(song_type)
    if mood:
        lines.append(mood)
    if energy:
        lines.append(f"energy: {energy}")

    if raga_name and RAGA_HINTS.get(raga_name):
        lines.append(RAGA_HINTS[raga_name])

    if instruments:
        lines.append("instruments: " + ", ".join(instruments))

    if bpm and bpm > 0:
        lines.append(f"around {int(round(bpm))} BPM")

    if key_text:
        lines.append(f"key: {key_text}")

    if vocal_range:
        lines.append(f"supporting vocal range {vocal_range}")

    if structure:
        lines.append(f"structure: {structure}")

    if extra:
        lines.append(extra)

    # Gentle production directions that help MusicGen
    lines.append(
        "clean studio mix, warm low-end, wide stereo, subtle reverb, avoid vocals"
    )

    return ", ".join([s for s in lines if s])


# -----------------------------------------------------------------------------
# Core generation
# -----------------------------------------------------------------------------

def generate_music(
    prompt: str,
    duration: float,
    model_size: str,
    cfg: float,
    seed: int,
) -> Tuple[Path, torch.Tensor]:
    model = get_model(model_size)
    # Re-seed
    if seed is None or seed < 0:
        seed = int(time.time()) & 0x7FFFFFFF
    torch.manual_seed(seed)
    if DEVICE == "cuda":
        torch.cuda.manual_seed_all(seed)

    # Set params per request
    model.set_generation_params(
        use_sampling=True,
        top_k=250,
        temperature=1.0,
        duration=max(1, int(round(duration))),
        cfg_coef=float(cfg),
    )

    with torch.inference_mode():
        wav = model.generate(descriptions=[prompt], progress=True)
        # MusicGen returns (B, T) at SR=32000
        wav = wav[0]  # (T,)

    out_dir = Path("outputs")
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"musicgen_{int(time.time())}.wav"
    saved = _save_wav(wav, SR, out_path)
    return saved, wav


# -----------------------------------------------------------------------------
# Analysis pipeline for vocals
# -----------------------------------------------------------------------------

def analyze_vocal(vocal_path: str):
    if not vocal_path:
        return None, None, None

    # Load mono @ SR for downstream analyzers
    y, sr = load_mono_resample(vocal_path, target_sr=SR)

    # tempo: helper expects the *path*, not raw audio
    bpm = safe_tempo_detect(vocal_path)

    # key/mode: helper returns (key_root, mode) -> make "C major" style label
    key_root, key_mode = detect_key_mode(y, sr)
    key_text = f"{key_root} {key_mode}".strip() if key_root and key_mode else None

    # vocal range: helper already returns a single string label like "C3-C5"
    vlabel = detect_vocal_range(y, sr)

    return bpm, key_text, vlabel



# -----------------------------------------------------------------------------
# Gradio UI
# -----------------------------------------------------------------------------

with gr.Blocks(title="AI Singer Studio – Instrumental Generator") as demo:
    gr.Markdown("""
    # 🎵 AI Singer Studio
    Create instrumentals **from text** or **to match your vocal**.
    - Uses **Meta MusicGen** under the hood (AudioCraft).
    - Auto-detects **tempo**, **key**, and **vocal range** from your voice.
    - Smart **prompt generator** with Bollywood/Devotional/Mantra presets and **Raag hints**.
    """)

    with gr.Tabs():
        # ----------------------------- TAB 1: Text → Instrumental -----------------------------
        with gr.Tab("From Text"):
            with gr.Row():
                with gr.Column(scale=1):
                    song_type = gr.Dropdown(SONG_TYPES, label="Song Type", value="Bollywood Ballad")
                    mood = gr.Textbox(label="Mood (e.g., romantic, cinematic)")
                    energy = gr.Textbox(label="Energy (e.g., mellow, driving)")
                    raga = gr.Dropdown(list(RAGA_HINTS.keys()), label="Raag/Thaat (optional)", value="None")
                    structure = gr.Dropdown(STRUCTURE_PRESETS, label="Structure", value=STRUCTURE_PRESETS[0])
                    # instr = gr.CheckboxGroup(choices=suggested_instruments("Bollywood Ballad"), label="Suggested Instruments")

                    instr = gr.CheckboxGroup(
                        choices=ALL_INSTRUMENTS,
                        value=suggested_instruments("Bollywood Ballad"),
                        label="Instruments"
                    )

                    # def _update_instr(song_type_val):
                    #     return gr.update(value=suggested_instruments(song_type_val))

                    def _update_instr(song_type_val):
                        return gr.update(value=sanitize_instruments(suggested_instruments(song_type_val)))

                    song_type.change(_update_instr, inputs=[song_type], outputs=instr)

                    extra = gr.Textbox(label="Extra prompt (optional)")

                    with gr.Accordion("Pro Tips", open=False):
                        gr.Markdown(
                            "- **CFG**: 2.5–3.5 is usually musical.\n"
                            "- **Seed**: Fix a seed for repeatable results; change it to explore variations.\n"
                            "- **Duration**: Generate shorter ideas first (8–16s) then iterate."
                        )
                        auto_btn = gr.Button("Use Recommended CFG & Seed")

                with gr.Column(scale=1):
                    duration1 = gr.Slider(4, 60, value=16, step=1, label="Duration (sec)")
                    cfg1 = gr.Slider(1.5, 4.5, value=3.0, step=0.1, label="CFG (guidance strength)")
                    seed1 = gr.Number(value=42, precision=0, label="Seed (int)")
                    model1 = gr.Dropdown(["Small (Melody)", "Medium", "Large"], value="Medium", label="Model size")
                    prompt_out = gr.Textbox(label="Final Prompt Preview",lines=4, interactive=False)
                    gen1 = gr.Button("Generate Instrumental")

            audio1 = gr.Audio(label="Preview", interactive=False)
            path1 = gr.File(label="Saved WAV")

            # def _update_instr(song_type_val, current_sel):
            #     choices = suggested_instruments(song_type_val)
            #     # Keep only selections that are still valid for the new choice set
            #     keep = [x for x in (current_sel or []) if x in choices]
            #     return gr.update(choices=choices, value=keep)

            # pass BOTH the song type and the current selection into the callback
            # song_type.change(_update_instr, inputs=[song_type, instr], outputs=instr)


            def _build_prompt_ui(song_type, mood, energy, extra, instruments, raga, duration):
                # No key/tempo/vocal range on text tab
                p = make_prompt(song_type, mood, energy, extra, instruments or [], raga, None, None, None, structure=None)
                return p

            for w in [song_type, mood, energy, extra, instr, raga]:
                w.change(_build_prompt_ui, [song_type, mood, energy, extra, instr, raga, duration1], prompt_out)

            def _pro_tips():
                # Very light heuristics
                # return gr.Slider.update(value=3.2), gr.Number.update(value=42)
                return gr.update(value=3.2), gr.update(value=42)

            auto_btn.click(_pro_tips, [], [cfg1, seed1])

            def run_text(song_type, mood, energy, extra, instruments, raga, duration, model, cfg, seed):
                # prompt = make_prompt(song_type, mood, energy, extra, instruments or [], raga, None, None, None, structure=None)

                prompt = make_prompt(
                    song_type, mood, energy, extra,
                    sanitize_instruments(instruments),
                    raga, None, None, None, structure=None
                )

                out_path, wav = generate_music(prompt, duration, model, cfg, int(seed))
                # Return audio numpy for immediate preview
                wav_np = _ensure_2d_cpu_f32(wav).numpy().T  # (T, C)
                return str(out_path), (SR, wav_np), prompt

            gen1.click(
                run_text,
                [song_type, mood, energy, extra, instr, raga, duration1, model1, cfg1, seed1],
                [path1, audio1, prompt_out],
            )

        # ----------------------------- TAB 2: Vocal → Instrumental ----------------------------
        with gr.Tab("From Vocal"):
            with gr.Row():
                with gr.Column(scale=1):
                    vocal = gr.Audio(type="filepath", label="Upload Dry Vocal (mono or stereo)")
                    auto_duration = gr.Number(value=16, precision=0, label="Auto Duration (sec)")
                    bpm_lbl = gr.Markdown("**Tempo:** –")
                    key_lbl = gr.Markdown("**Key/Mode:** –")
                    range_lbl = gr.Markdown("**Vocal Range:** –")

                with gr.Column(scale=1):
                    song_type2 = gr.Dropdown(SONG_TYPES, label="Song Type", value="Devotional (Bhajan)")



                    raga2 = gr.Dropdown(list(RAGA_HINTS.keys()), label="Raag/Thaat (optional)", value="Yaman (Kalyan)")
                    extra2 = gr.Textbox(label="Extra prompt (optional)")
                    structure2 = gr.Dropdown(STRUCTURE_PRESETS, label="Structure", value=STRUCTURE_PRESETS[0])

                    # Editable creative controls that feed into the prompt
                    mood2 = gr.Textbox(label="Mood (e.g., emotive, devotional, cinematic)", value="emotive, cinematic")
                    energy2 = gr.Textbox(label="Energy (e.g., mellow, uplifting, driving)", value="mellow")
                    # Instruments that start from a sensible preset for the chosen song type
                    
                    
                    # instruments2 = gr.CheckboxGroup(
                    #     choices=suggested_instruments("Devotional (Bhajan)"),
                    #     value=suggested_instruments("Devotional (Bhajan)"),
                    #     label="Instruments (you can add/remove)"
                    # )

                    # def _update_instr2(song_type_val, current_sel):
                    #     choices = suggested_instruments(song_type_val)
                    #     keep = [x for x in (current_sel or []) if x in choices] or choices
                    #     return gr.update(choices=choices, value=keep)

                    # song_type2.change(_update_instr2, inputs=[song_type2, instruments2], outputs=instruments2)

                    instruments2 = gr.CheckboxGroup(
                        choices=ALL_INSTRUMENTS,
                        value=suggested_instruments("Devotional (Bhajan)"),
                        label="Instruments (you can add/remove)"
                    )

                    # def _update_instr2(song_type_val):
                    #     return gr.update(value=suggested_instruments(song_type_val))

                    def _update_instr2(song_type_val):
                        return gr.update(value=sanitize_instruments(suggested_instruments(song_type_val)))

                    song_type2.change(_update_instr2, inputs=[song_type2], outputs=instruments2)

                    with gr.Accordion("Use / override detected musical info (optional)", open=False):
                        override_bpm = gr.Textbox(label="BPM override (leave blank to use auto-detected)")
                        override_key = gr.Textbox(label="Key override, e.g., C major (leave blank to use auto-detected)")
                        override_range = gr.Textbox(label="Vocal range override, e.g., C3–C5 (leave blank to use auto-detected)")

                    cfg2 = gr.Slider(1.5, 4.5, value=3.0, step=0.1, label="CFG (guidance strength)")
                    seed2 = gr.Number(value=99, precision=0, label="Seed (int)")
                    model2 = gr.Dropdown(["Small (Melody)", "Medium", "Large"], value="Medium", label="Model size")

            prompt2 = gr.Textbox(label="Final Prompt Preview", lines=4, interactive=False)
            gen2 = gr.Button("Analyze & Generate Backing Track")
            audio2 = gr.Audio(label="Instrumental Preview", interactive=False)
            path2 = gr.File(label="Saved WAV (instrumental)")

            # When vocal uploaded → auto-duration + analysis
            # def on_vocal_uploaded(path):
            #     if not path:
            #         return gr.Number.update(value=16), gr.Markdown.update(value="**Tempo:** –"), gr.Markdown.update(value="**Key/Mode:** –"), gr.Markdown.update(value="**Vocal Range:** –")
            #     dur = int(round(_estimate_duration_from_file(path)))
            #     bpm, key_text, vlabel = analyze_vocal(path)
            #     bpm_md = f"**Tempo:** ~{int(round(bpm))} BPM" if bpm else "**Tempo:** –"
            #     key_md = f"**Key/Mode:** {key_text}" if key_text else "**Key/Mode:** –"
            #     rng_md = f"**Vocal Range:** {vlabel}" if vlabel else "**Vocal Range:** –"
            #     return gr.Number.update(value=dur), gr.Markdown.update(value=bpm_md), gr.Markdown.update(value=key_md), gr.Markdown.update(value=rng_md)

            def on_vocal_uploaded(path):
                if not path:
                    return (
                        gr.update(value=16),
                        gr.update(value="**Tempo:** –"),
                        gr.update(value="**Key/Mode:** –"),
                        gr.update(value="**Vocal Range:** –"),
                        gr.update(value=""),  # override_bpm
                        gr.update(value=""),  # override_key
                        gr.update(value=""),  # override_range
                    )

                try:
                    dur = safe_duration_seconds(path, clamp_min=8, clamp_max=60)
                except Exception:
                    dur = 30

                try:
                    bpm, key_text, vlabel = analyze_vocal(path)
                except Exception:
                    bpm, key_text, vlabel = None, None, None

                bpm_md = f"**Tempo:** ~{int(round(bpm))} BPM" if bpm else "**Tempo:** –"
                key_md = f"**Key/Mode:** {key_text}" if key_text else "**Key/Mode:** –"
                rng_md = f"**Vocal Range:** {vlabel}" if vlabel else "**Vocal Range:** –"

                # Prefill overrides with detected values (as plain text the user can edit)
                return (
                    gr.update(value=dur),
                    gr.update(value=bpm_md),
                    gr.update(value=key_md),
                    gr.update(value=rng_md),
                    gr.update(value=str(int(round(bpm))) if bpm else ""),  # override_bpm
                    gr.update(value=key_text or ""),                       # override_key
                    gr.update(value=vlabel or ""),                         # override_range
                )



            vocal.change(
                on_vocal_uploaded,
                [vocal],
                [auto_duration, bpm_lbl, key_lbl, range_lbl, override_bpm, override_key, override_range],
            )

            def build_prompt_from_vocal(
                vocal_path,
                song_type,
                raga,
                extra,
                structure,
                mood,
                energy,
                instruments,
                bpm_override,
                key_override,
                range_override,
            ):
                # Auto-analysis if we have a vocal
                det_bpm, det_key_text, det_vlabel = analyze_vocal(vocal_path) if vocal_path else (None, None, None)

                # Apply overrides if provided
                bpm_val = None
                if bpm_override:
                    try:
                        bpm_val = float(bpm_override)
                    except Exception:
                        bpm_val = det_bpm
                else:
                    bpm_val = det_bpm

                key_val = key_override.strip() if (key_override and key_override.strip()) else det_key_text
                range_val = range_override.strip() if (range_override and range_override.strip()) else det_vlabel

                # p = make_prompt(
                #     song_type=song_type,
                #     mood=mood,
                #     energy=energy,
                #     extra=extra,
                #     instruments=instruments or suggested_instruments(song_type),
                #     raga_name=raga,
                #     bpm=bpm_val,
                #     key_text=key_val,
                #     vocal_range=range_val,
                #     structure=structure,
                # )

                p = make_prompt(
                    song_type=song_type,
                    mood=mood,
                    energy=energy,
                    extra=extra,
                    instruments=sanitize_instruments(instruments) or suggested_instruments(song_type),
                    raga_name=raga,
                    bpm=bpm_val,
                    key_text=key_val,
                    vocal_range=range_val,
                    structure=structure,
                )

                return p


            # Live prompt preview
            # for w in [vocal, song_type2, raga2, extra2, structure2]:
            #     w.change(build_prompt_from_vocal, [vocal, song_type2, raga2, extra2, structure2], [prompt2])

            for w in [vocal, song_type2, raga2, extra2, structure2, mood2, energy2, instruments2, override_bpm, override_key, override_range]:
                w.change(
                    build_prompt_from_vocal,
                    [vocal, song_type2, raga2, extra2, structure2, mood2, energy2, instruments2, override_bpm, override_key, override_range],
                    [prompt2],
                )

            # def run_vocal(path, song_type, raga, extra, structure, duration, model, cfg, seed):
            #     prompt = build_prompt_from_vocal(path, song_type, raga, extra, structure)
            #     out_path, wav = generate_music(prompt, duration, model, cfg, int(seed))
            #     wav_np = _ensure_2d_cpu_f32(wav).numpy().T
            #     return str(out_path), (SR, wav_np), prompt

            # gen2.click(
            #     run_vocal,
            #     [vocal, song_type2, raga2, extra2, structure2, auto_duration, model2, cfg2, seed2],
            #     [path2, audio2, prompt2],
            # )

            def run_vocal(path, song_type, raga, extra, structure, duration, model, cfg, seed,
                        mood, energy, instruments, bpm_override, key_override, range_override):
                prompt = build_prompt_from_vocal(
                    path, song_type, raga, extra, structure, mood, energy, instruments,
                    bpm_override, key_override, range_override
                )
                out_path, wav = generate_music(prompt, duration, model, cfg, int(seed))
                wav_np = _ensure_2d_cpu_f32(wav).numpy().T
                return str(out_path), (SR, wav_np), prompt

            gen2.click(
                run_vocal,
                [
                    vocal, song_type2, raga2, extra2, structure2, auto_duration, model2, cfg2, seed2,
                    mood2, energy2, instruments2, override_bpm, override_key, override_range
                ],
                [path2, audio2, prompt2],
            )

        # ----------------------------- TAB 3: Mix Vocal + Instrumental ------------------------
        with gr.Tab("Mix"):
            with gr.Row():
                with gr.Column():
                    # Use File widgets to avoid audio streaming quirks on Windows
                    mix_vocal = gr.File(file_types=["audio"], label="Vocal track (WAV/MP3/M4A/…)")
                    mix_instr = gr.File(file_types=["audio"], label="Instrumental track (WAV/MP3/M4A/…)")
                with gr.Column():
                    vocal_gain = gr.Slider(-12, 12, value=0, step=0.5, label="Vocal gain (dB)")
                    instr_gain = gr.Slider(-12, 12, value=0, step=0.5, label="Instrumental gain (dB)")
                    out_gain = gr.Slider(-12, 12, value=-1.0, step=0.5, label="Output gain / limiter target (dBFS)")
                    mix_btn = gr.Button("Mix & Export")

            mix_audio = gr.Audio(label="Mix Preview", interactive=False, streaming=False)
            mix_path = gr.File(label="Saved Mix WAV")

            def _db_to_lin(db):
                return 10 ** (db / 20.0)

            def _soft_limiter(x: torch.Tensor, thresh_db=-1.0):
                # Simple hard-knee limiter for safety
                peak = x.abs().max().item() if x.numel() else 1.0
                if peak > 0:
                    x = x / max(1.0, peak)
                target = _db_to_lin(thresh_db)
                return x.clamp(-target, target)

            # def mix_tracks(vocal_path, instr_path, vocal_db, instr_db, out_db):
            #     if not vocal_path or not instr_path:
            #         raise gr.Error("Please provide both vocal and instrumental.")
            #     v, vsr = load_mono_resample(vocal_path, target_sr=SR)
            #     i, isr = load_mono_resample(instr_path, target_sr=SR)
            #     T = max(len(v), len(i))
            #     v = np.pad(v, (0, T - len(v)))
            #     i = np.pad(i, (0, T - len(i)))
            #     v_t = torch.tensor(v).unsqueeze(0)  # (1, T)
            #     i_t = torch.tensor(i).unsqueeze(0)
            #     v_t = v_t * _db_to_lin(vocal_db)
            #     i_t = i_t * _db_to_lin(instr_db)
            #     mix = v_t + i_t
            #     mix = _soft_limiter(mix, thresh_db=out_db)
            #     out_path = Path("outputs") / f"mix_{int(time.time())}.wav"
            #     saved = _save_wav(mix.squeeze(0), SR, out_path)
            #     mix_np = _ensure_2d_cpu_f32(mix.squeeze(0)).numpy().T
            #     return str(saved), (SR, mix_np)

            def _file_to_path(file_val):
                # Gradio 5 returns {'name': 'C:\\...\\file.wav', 'size': ..., ...}
                # Some setups pass a plain path string. Normalize to string path.
                if isinstance(file_val, dict) and "name" in file_val:
                    return file_val["name"]
                return str(file_val) if file_val else None

            def mix_tracks(vocal_file, instr_file, vocal_db, instr_db, out_db):
                vpath = _file_to_path(vocal_file)
                ipath = _file_to_path(instr_file)
                if not vpath or not ipath:
                    raise gr.Error("Please provide both vocal and instrumental.")

                v, _ = load_mono_resample(vpath, target_sr=SR)
                i, _ = load_mono_resample(ipath, target_sr=SR)

                T = max(len(v), len(i))
                if T == 0:
                    raise gr.Error("One of the files appears to be silent/empty.")

                v = np.pad(v, (0, T - len(v)))
                i = np.pad(i, (0, T - len(i)))

                v_t = torch.tensor(v, dtype=torch.float32).unsqueeze(0)  # (1, T)
                i_t = torch.tensor(i, dtype=torch.float32).unsqueeze(0)  # (1, T)

                def _db_to_lin(db):
                    return 10 ** (db / 20.0)

                def _soft_limiter(x: torch.Tensor, thresh_db=-1.0):
                    peak = x.abs().max().item() if x.numel() else 1.0
                    if peak > 0:
                        x = x / max(1.0, peak)
                    target = _db_to_lin(thresh_db)
                    return x.clamp(-target, target)

                v_t = v_t * _db_to_lin(vocal_db)
                i_t = i_t * _db_to_lin(instr_db)
                mix = v_t + i_t
                mix = _soft_limiter(mix, thresh_db=out_db)

                out_path = Path("outputs") / f"mix_{int(time.time())}.wav"
                saved = _save_wav(mix.squeeze(0), SR, out_path)
                mix_np = _ensure_2d_cpu_f32(mix.squeeze(0)).numpy().T
                return str(saved), (SR, mix_np)


            mix_btn.click(mix_tracks, [mix_vocal, mix_instr, vocal_gain, instr_gain, out_gain], [mix_path, mix_audio])

            # Real-time preview when sliders move
            vocal_gain.release(mix_tracks, [mix_vocal, mix_instr, vocal_gain, instr_gain, out_gain], [mix_path, mix_audio])
            instr_gain.release(mix_tracks, [mix_vocal, mix_instr, vocal_gain, instr_gain, out_gain], [mix_path, mix_audio])
            out_gain.release(mix_tracks, [mix_vocal, mix_instr, vocal_gain, instr_gain, out_gain], [mix_path, mix_audio])


    gr.Markdown("""
    ---
    **Notes**
    - If you want **repeatable** results, keep the seed fixed.
    - If your output feels too random, lower CFG slightly (e.g., 2.8–3.2).
    - For longer songs, generate in **sections** (verse/chorus) and stitch in a DAW.
    """)


if __name__ == "__main__":
    demo.queue().launch(server_name="127.0.0.1", server_port=7860, inbrowser=False)
