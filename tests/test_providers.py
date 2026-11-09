from copy import deepcopy
from pathlib import Path
import sys

import httpx
import pytest

from agent.image_gen_provider import ImageGenProvider
from agent.video_gen_provider import VideoGenProvider
from agent.tts_provider import TTSProvider
from fake_gmi import MEDIA


@pytest.mark.parametrize("kind,cls,base,register", [
    ("image", "GmiImageProvider", ImageGenProvider, "register_image_gen_provider"),
    ("video", "GmiVideoProvider", VideoGenProvider, "register_video_gen_provider"),
    ("speech", "GmiSpeechProvider", TTSProvider, "register_tts_provider"),
])
def test_register(plugins, kind, cls, base, register):
    seen = []
    ctx = type("Context", (), {register: lambda self, provider: seen.append(provider)})()
    plugins[kind].register(ctx)
    assert len(seen) == 1 and isinstance(seen[0], base)
    assert seen[0].name == "gmi-" + kind
    assert seen[0].get_setup_schema()["badge"] == "paid"
    if kind == "speech":
        assert seen[0].voice_compatible is False and seen[0].streams_pcm is False


def test_image_generation_and_edit(plugins, client, fake, monkeypatch, tmp_path):
    module = plugins["image"]
    path = tmp_path / "hh/cache/generated/images/mock.png"
    saved = []
    def save(url, **kwargs):
        saved.append((url, kwargs))
        return path
    monkeypatch.setattr(module, "save_url_image", save)
    provider = module.GmiImageProvider(client=client)
    assert isinstance(provider, ImageGenProvider)
    result = provider.generate("A tree", "square", n=2)
    assert result["success"] and result["image"] == str(path)
    assert result["model"] == "seedream-5.0-lite"
    assert result["provider"] == "gmi-image" and result["modality"] == "text"
    assert all(k in result for k in ("price", "request_id", "dropped"))
    assert saved[0][1] == {"prefix": "gmi"}
    result = provider.generate("Edit", image_url="https://example.com/a")
    assert result["success"] and result["modality"] == "image"
    result = provider.generate("Edit", image_url="/tmp/a.png")
    assert not result["success"] and result["error_type"] == "invalid_request"
    assert "upload the file first" in result["error"]
    assert not path.exists()


def test_image_config_precedence_and_passthrough(plugins, client, fake, monkeypatch):
    module = plugins["image"]
    monkeypatch.setattr(module, "save_url_image", lambda *a, **k: (_ for _ in ()).throw(RuntimeError()))
    config = {"image_gen": {"model": "another-provider-model", "gmi-image": {"model": "gemini-2.5-flash-image"}}}
    original = deepcopy(config)
    monkeypatch.setattr(module, "load_config_readonly", lambda: config)
    provider = module.GmiImageProvider(client=client)
    assert provider.generate("Hello")["model"] == "gemini-2.5-flash-image"
    assert provider.generate("Hello", model="gpt-image-2-generate")["model"] == "gpt-image-2-generate"
    assert config == original
    config["image_gen"].pop("gmi-image")
    assert provider.generate("Hello")["model"] == "seedream-5.0-lite"
    assert not any("another-provider-model" in r["path"] for r in fake.requests)
    fresh = deepcopy(fake.models["gemini-2.5-flash-image"])
    fresh["model"] = "new-public-model"
    fake.models[fresh["model"]] = fresh
    result = provider.generate("Hello", model=fresh["model"])
    assert result["success"] and result["image"] == fake.url + "/media/image"


def test_video_defaults_duration_and_references(plugins, client, fake, monkeypatch, tmp_path):
    module = plugins["video"]
    path = tmp_path / "hh/cache/generated/videos/mock.mp4"
    def save(url, **kwargs):
        assert kwargs == {"prefix": "gmi", "headers": None}
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(client.http.get(url).content)
        return path
    monkeypatch.setattr(module, "save_url_video", save)
    provider = module.GmiVideoProvider(client=client)
    result = provider.generate("A tree", duration=7, reference_image_urls=["https://example.com/a"])
    assert result["success"] and result["model"] == "Minimax-Hailuo-02"
    assert result["duration"] == 6 and "reference_image_urls" in result["dropped"]
    assert Path(result["video"]).read_bytes() == MEDIA["video"]
    assert result["price"]
    result = provider.generate("A tree", image_url="https://example.com/a")
    assert result["success"] and result["model"] == "Minimax-Hailuo-2.3-Fast"
    result = provider.generate("A tree", model="happyhorse-1.0-t2v", image_url="https://example.com/a")
    assert result["error_type"] == "invalid_request" and "Minimax-Hailuo-2.3-Fast" in result["error"]
    monkeypatch.setattr(module, "save_url_video", lambda *a, **k: (_ for _ in ()).throw(RuntimeError()))
    assert provider.generate("Hello")["video"] == fake.url + "/media/video"
    assert provider.capabilities()["max_reference_images"] == 0


@pytest.mark.parametrize("model,fmt,extension", [
    ("minimax-tts-speech-2.6-turbo", "flac", ".flac"),
    ("minimax-tts-speech-2.6-turbo", "unknown", ".mp3"),
    ("inworld-tts-1.5-mini", "ogg", ".ogg"),
    ("inworld-tts-1.5-mini", "wav", ".wav"),
])
def test_speech_download(plugins, client, fake, monkeypatch, tmp_path, model, fmt, extension):
    module = plugins["speech"]
    provider = module.GmiSpeechProvider(client=client)
    assert isinstance(provider, TTSProvider)
    assert all("music" not in entry["id"] for entry in provider.list_models())
    assert provider.list_voices()[0]["id"]
    fake.poll_replies.append((200, {"status": "success", "outcome": {"audio_url": "https://media.example/audio"}}))
    # Keep URL policy intact. Route only this HTTPS transport to the local fake.
    real_client = httpx.Client
    def route(request):
        assert request.url.host == "media.example"
        local = client.http.get(fake.url + "/media/audio")
        return httpx.Response(200, content=local.content)
    copy = sys.modules[module.__name__ + ".gmi_media"]
    monkeypatch.setattr(copy.httpx, "Client", lambda **kwargs: real_client(transport=httpx.MockTransport(route)))
    result = provider.synthesize("Hello", str(tmp_path / "speech.bin"), model=model, format=fmt, speed=1)
    assert Path(result).suffix == extension
    assert Path(result).read_bytes() == MEDIA["audio"]
    with pytest.raises(ValueError, match="Valid values"):
        provider.synthesize("Hello", str(tmp_path / "bad"), model=model, voice="not-a-voice")


def test_missing_credentials_empty_prompts(plugins, client, fake):
    for kind, cls in [("image", "GmiImageProvider"), ("video", "GmiVideoProvider")]:
        provider = getattr(plugins[kind], cls)(client=client)
        assert provider.generate("")["error_type"] == "invalid_request"
        client.api_key_fn = lambda: None
        assert provider.generate("Hello")["error_type"] == "missing_credentials"
    with pytest.raises(RuntimeError, match="Set GMI_API_KEY"):
        plugins["speech"].GmiSpeechProvider(client=client).synthesize("Hello", "/unused")
    assert not fake.requests
