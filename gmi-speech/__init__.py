"""GMI Cloud speech synthesis without streaming."""
from pathlib import Path

from agent.tts_provider import TTSProvider
from agent.secret_scope import get_secret_str

from .gmi_media import (
    BackendSupport, CANDIDATES, GmiError, audio_extension, choose_param, download_audio,
    map_payload, media_urls, options, parameters, pick_default_model,
)

SPEECH_MODELS = ("minimax-tts-speech-2.6-turbo", "inworld-tts-1.5-mini")
FALLBACK_VOICES = ("English_expressive_narrator", "English_radiant_girl")


class GmiSpeechProvider(BackendSupport, TTSProvider):
    name = "gmi-speech"
    kind = "audio"
    preferences = SPEECH_MODELS
    voice_compatible = False

    def __init__(self, client=None, catalog=None):
        self._init_backend(lambda: get_secret_str("GMI_API_KEY"), client, catalog)

    def list_voices(self):
        record = pick_default_model(self.records(), "audio", prefer=SPEECH_MODELS)
        specs = parameters(record or {})
        name = choose_param(specs, CANDIDATES["voice"])
        voices = options(specs[name]) if name else []
        return [{"id": voice, "display": voice} for voice in (voices or FALLBACK_VOICES)]

    def synthesize(self, text, output_path, *, voice=None, model=None, speed=None, format="mp3", **extra):
        try:
            self.require_ready(text)
            record = self.select_model(model)
            payload, _ = map_payload(record, prompt=text, voice=voice, speed=speed, format=format)
            specs = parameters(record)
            name = choose_param(specs, CANDIDATES["format"])
            actual_format = payload.get(name, specs.get(name, {}).get("default_value"))
            extension = audio_extension(actual_format)
            if extension is None:
                raise GmiError("The model does not declare a supported audio file format. Choose another speech model.", "invalid_request")
            result = self.client.run(record["model"], payload, 300)
            return download_audio(media_urls(result, "audio")[0], Path(output_path).with_suffix(extension))
        except GmiError as exc:
            error = ValueError if exc.kind == "invalid_request" else RuntimeError
            raise error(str(exc)) from None


def register(ctx):
    ctx.register_tts_provider(GmiSpeechProvider())
