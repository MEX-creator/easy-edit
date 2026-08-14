"""Beat and energy analysis via librosa (local, free, no API).

detect_beats_from_waveform() is pure and unit-testable on synthetic audio;
analyze_audio() wraps librosa.load() so it works on video files too (requires
ffmpeg for non-WAV containers).
"""

from __future__ import annotations

from typing import Optional

from core.analysis import AudioAnalysis, Beat
from core.errors import AnalysisError, MissingDependency


def detect_beats_from_waveform(y, sr: int, hop_length: int = 512) -> AudioAnalysis:
    """Beat markers + normalized energy curve from a raw mono waveform."""
    try:
        import librosa
        import numpy as np
    except ImportError:
        raise MissingDependency("Audio/beat analysis", "audio")

    y = np.asarray(y, dtype=np.float32)
    if y.size == 0 or float(np.max(np.abs(y))) < 1e-6:
        raise AnalysisError("audio track is silent; skipping beat analysis")

    tempo, beat_frames = librosa.beat.beat_track(y=y, sr=sr)
    tempo = float(np.atleast_1d(tempo)[0]) if tempo is not None else 0.0
    beat_times = librosa.frames_to_time(beat_frames, sr=sr) if beat_frames.size else np.array([])

    onset_env = librosa.onset.onset_strength(y=y, sr=sr)
    if onset_env.size:
        onset_env = librosa.util.normalize(onset_env)

    if beat_times.size == 0:
        # beat_track can fail on very clean/synthetic tracks. Fall back to
        # onset energy peaks — the punchy points cuts want to land on anyway —
        # and estimate the tempo from their median interval.
        onset_frames = (
            librosa.onset.onset_detect(onset_envelope=onset_env, sr=sr)
            if onset_env.size
            else np.array([])
        )
        beat_times = librosa.frames_to_time(onset_frames, sr=sr)
    if tempo <= 0.0 and beat_times.size >= 2:
        interval = float(np.median(np.diff(beat_times)))
        if interval > 0:
            tempo = 60.0 / interval

    beats = []
    for i, t in enumerate(beat_times):
        strength = float(onset_env[i]) if i < len(onset_env) else 1.0
        beats.append(Beat(time_sec=float(t), strength=round(strength, 3)))

    hop = int(sr * 0.1)
    rms = librosa.feature.rms(y=y, hop_length=hop)[0]
    times = librosa.frames_to_time(np.arange(rms.size), sr=sr, hop_length=hop)
    curve = [(float(t), float(e)) for t, e in zip(times, rms)]
    if curve:
        peak = max(e for _, e in curve)
        if peak > 0:
            curve = [(t, e / peak) for t, e in curve]

    return AudioAnalysis(
        beats=beats, energy_curve=curve, tempo_bpm=tempo if beats else None, has_audio=True
    )


def analyze_audio(path: str) -> Optional[AudioAnalysis]:
    """Load audio (from a video or audio file) and analyze it.

    Returns None when the file has no audio track; raises AnalysisError on
    decode problems that even the ffmpeg fallback cannot handle.
    """
    try:
        import librosa
    except ImportError:
        raise MissingDependency("Audio/beat analysis", "audio")
    loaded = _load_audio(librosa, path)
    if loaded is None:  # file has no audio track
        return None
    y, sr = loaded
    if y is None or len(y) == 0:
        return None
    return detect_beats_from_waveform(y, sr)


def _load_audio(librosa, path: str, target_sr: int = 22050):
    """librosa.load() with an ffmpeg fallback.

    soundfile (librosa's only backend since 1.0) cannot decode mp4/m4a/aac, so
    for video containers we shell out to ffmpeg to extract a WAV first. WAV /
    FLAC / OGG files never need ffmpeg.
    """
    try:
        return librosa.load(path, sr=target_sr, mono=True)
    except Exception as exc:
        return _load_via_ffmpeg(path, target_sr, original_error=exc)


def _load_via_ffmpeg(path: str, target_sr: int, original_error: Exception):
    import os
    import shutil
    import subprocess
    import tempfile

    import librosa

    ffmpeg = shutil.which("ffmpeg")
    if ffmpeg is None:
        raise AnalysisError(
            f"could not read audio from {path} ({original_error}); install ffmpeg "
            "and put it on PATH, or provide a WAV/FLAC/OGG file"
        )
    proc = subprocess.run(
        [
            ffmpeg, "-v", "error", "-i", str(path),
            "-ac", "1", "-ar", str(target_sr), "-f", "wav", "pipe:1",
        ],
        capture_output=True,
    )
    if proc.returncode != 0:
        stderr = proc.stderr.decode("utf-8", errors="replace").lower()
        if "does not contain any stream" in stderr or "no audio" in stderr:
            return None  # file simply has no audio track
        raise AnalysisError(
            f"ffmpeg could not decode audio from {path}: "
            f"{proc.stderr.decode('utf-8', errors='replace')[:200]}"
        )
    if not proc.stdout:
        return None
    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
        tmp.write(proc.stdout)
        tmp_path = tmp.name
    try:
        return librosa.load(tmp_path, sr=target_sr, mono=True)
    finally:
        try:
            os.unlink(tmp_path)
        except OSError:
            pass
