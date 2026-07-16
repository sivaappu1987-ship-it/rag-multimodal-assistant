"""
audio.py — Audio processing service.
Provides hybrid speech-to-text (STT) via faster-whisper (local) and Sarvam AI (API),
and text-to-speech (TTS) via edge-tts (local Microsoft Neural Voices).
"""
import os
import re
import tempfile
import subprocess  # nosec B404
import httpx
import edge_tts
from langdetect import detect
from app.config import SARVAM_API_KEY

VOICE_TABLE = {
    "en": "en-IN-NeerjaNeural",
    "hi": "hi-IN-SwaraNeural",
    "ta": "ta-IN-PallaviNeural",
    "te": "te-IN-ShrutiNeural",
    "kn": "kn-IN-SapnaNeural",
    "ml": "ml-IN-SobhanaNeural",
    "bn": "bn-IN-TanishaaNeural",
    "mr": "mr-IN-AarohiNeural",
}

SARVAM_LANG_MAP = {
    "en": "en-IN",
    "hi": "hi-IN",
    "ta": "ta-IN",
    "te": "te-IN",
    "kn": "kn-IN",
    "ml": "ml-IN",
    "bn": "bn-IN",
    "mr": "mr-IN",
    "auto": "unknown",
}

_whisper_model = None


def get_whisper_model():
    global _whisper_model
    if _whisper_model is None:
        from faster_whisper import WhisperModel
        print("[Audio] Loading local faster-whisper ('small', int8, CPU)...")
        _whisper_model = WhisperModel("small", device="cpu", compute_type="int8")
        print("[Audio] local faster-whisper model ready.")
    return _whisper_model


def convert_to_wav(input_path: str) -> str:
    """
    Transcode uploaded audio file to standard 16kHz mono WAV using FFmpeg.
    """
    temp_dir = tempfile.gettempdir()
    output_path = os.path.join(
        temp_dir,
        f"transcoded_{os.path.basename(input_path)}.wav"
    )

    import imageio_ffmpeg
    ffmpeg_exe = imageio_ffmpeg.get_ffmpeg_exe()

    cmd = [
        ffmpeg_exe,
        "-y",
        "-i",
        input_path,
        "-ar",
        "16000",
        "-ac",
        "1",
        output_path
    ]

    try:
        subprocess.run(
            cmd,
            shell=False,
            check=True,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )  # nosec B603
        return output_path

    except subprocess.CalledProcessError as e:
        raise RuntimeError("Audio transcoding failed") from e


async def transcribe_audio(file_path: str, hint_lang: str) -> dict:
    transcoded_path = convert_to_wav(file_path)

    try:
        if hint_lang == "en":
            model = get_whisper_model()
            segments, info = model.transcribe(transcoded_path, beam_size=5)
            text = " ".join([s.text for s in segments]).strip()
            return {"text": text, "detected_language": info.language}

        if not SARVAM_API_KEY:
            raise RuntimeError("SARVAM_API_KEY not configured")

        sarvam_lang = SARVAM_LANG_MAP.get(hint_lang, "unknown")

        async with httpx.AsyncClient() as client:
            with open(transcoded_path, "rb") as audio_file:
                response = await client.post(
                    "https://api.sarvam.ai/speech-to-text",
                    files={
                        "file": (
                            os.path.basename(transcoded_path),
                            audio_file,
                            "audio/wav"
                        )
                    },
                    data={
                        "model": "saaras:v3",
                        "mode": "transcribe",
                        "language_code": sarvam_lang,
                    },
                    headers={"api-subscription-key": SARVAM_API_KEY},
                    timeout=30.0,
                )

        if response.status_code != 200:
            raise RuntimeError(f"Sarvam API error: {response.text}")

        res_data = response.json()
        transcript = res_data.get("transcript", "")

        detected_lang = hint_lang
        if hint_lang == "auto" and transcript:
            try:
                detected_lang = detect(transcript)
            except Exception:
                detected_lang = "unknown"

        return {
            "text": transcript.strip(),
            "detected_language": detected_lang,
        }

    finally:
        if os.path.exists(transcoded_path):
            os.remove(transcoded_path)


def truncate_text(text: str, max_sentences: int = 3) -> str:
    sentences = re.split(r"(?<=[.!?।])\s+", text.strip())
    return " ".join(sentences[:max_sentences])


def detect_language(text: str) -> str:
    try:
        return detect(text)
    except Exception:
        return "en"


async def speak_text(text: str, language: str = None) -> bytes:
    truncated = truncate_text(text, 3) or "No response text to read."

    if not language or language in ("auto", "unknown"):
        language = detect_language(truncated)

    lang_key = language.split("-")[0].lower()
    voice = VOICE_TABLE.get(lang_key, "en-IN-NeerjaNeural")

    communicate = edge_tts.Communicate(truncated, voice)

    with tempfile.NamedTemporaryFile(suffix=".mp3", delete=False) as tmp:
        tmp_path = tmp.name

    try:
        await communicate.save(tmp_path)
        with open(tmp_path, "rb") as f:
            return f.read()
    finally:
        if os.path.exists(tmp_path):
            os.remove(tmp_path)