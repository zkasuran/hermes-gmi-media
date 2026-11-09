from copy import deepcopy

import pytest

from shared.gmi_media import (
    GmiError, accepts_media, map_payload, nearest_option, options, parameters,
    pick_default_model, required_media, validate_payload,
)


@pytest.mark.parametrize("aspect,ratio,free,size", [
    ("landscape", "16:9", "1536x1024", "1280x720"),
    ("square", "1:1", "1024x1024", "1024x1024"),
    ("portrait", "9:16", "1024x1536", "720x1280"),
])
def test_image_aspects(models, aspect, ratio, free, size):
    for model in ("gemini-2.5-flash-image", "luma-uni-1.1"):
        payload, _ = map_payload(models[model], prompt="A tree", aspect_ratio=aspect)
        assert payload["aspect_ratio"] == ratio
    payload, _ = map_payload(models["gpt-image-2-generate"], prompt="A tree", aspect_ratio=aspect)
    assert payload["size"] == free
    payload, _ = map_payload(models["seedream-3-0-t2i-250415"], prompt="A tree", aspect_ratio=aspect)
    assert payload["size"] == size
    payload, dropped = map_payload(models["seedream-5.0-lite"], prompt="A tree", aspect_ratio=aspect)
    # Every explicit dimension in this captured schema exceeds the area cap.
    assert "size" not in payload
    assert dropped == ["aspect_ratio"]


@pytest.mark.parametrize("model,expected", [
    ("Veo3-Fast", {"durationSeconds": "6", "resolution": "720p", "aspectRatio": "16:9", "generateAudio": True}),
    ("Minimax-Hailuo-02", {"duration": 6, "resolution": "768P"}),
    ("kling-v2-6", {"duration": "5", "sound": "on"}),
    ("happyhorse-1.0-t2v", {"duration": 7, "resolution": "720P"}),
])
def test_video_schemas(models, model, expected):
    payload, dropped = map_payload(models[model], prompt="A tree", duration=7,
        resolution="720p", aspect_ratio="16:9", audio=True)
    for key, value in expected.items():
        assert payload[key] == value and type(payload[key]) is type(value)
    assert set(payload) <= parameters(models[model]).keys()
    if model == "Minimax-Hailuo-02":
        assert "aspect_ratio" in dropped


@pytest.mark.parametrize("model,speed_name,fmt", [
    ("minimax-tts-speech-2.6-turbo", "speed", "mp3"),
    ("inworld-tts-1.5-mini", "speaking_rate", "MP3"),
])
def test_speech_schemas(models, model, speed_name, fmt):
    specs = parameters(models[model])
    voice = options(specs["voice_id"])[0]
    payload, _ = map_payload(models[model], prompt="Hello", voice=voice.lower(), speed=100, format="mp3")
    assert payload["voice_id"] == voice
    assert payload[speed_name] == specs[speed_name]["max_value"]
    assert payload["format" if "format" in specs else "audio_encoding"] == fmt
    with pytest.raises(GmiError, match="Valid values"):
        map_payload(models[model], prompt="Hello", voice="nonexistent-voice")


def test_required_dropped_seed_and_validation(models):
    record = deepcopy(models["Veo3-Fast"])
    payload, dropped = map_payload(record, prompt="Hello", seed=10**20, unsupported="x")
    assert payload["seed"] == 4294967295
    assert payload["durationSeconds"] == "8"
    assert dropped == ["unsupported"]
    record["parameters"].append({"name": "mandatory", "required": True, "type": "string"})
    with pytest.raises(GmiError, match="mandatory"):
        map_payload(record, prompt="Hello")
    with pytest.raises(GmiError, match="Unknown parameter"):
        validate_payload(models["Veo3-Fast"], {"unknown": 1})
    with pytest.raises(GmiError, match="Valid values"):
        validate_payload(models["Veo3-Fast"], {"durationSeconds": 8})


@pytest.mark.parametrize("value", ["/tmp/photo.png", "file:///tmp/photo.png", "ftp://example.com/a", []])
def test_image_paths_rejected(models, value):
    with pytest.raises(GmiError, match=r"public http\(s\) URLs"):
        map_payload(models["gemini-2.5-flash-image"], prompt="Hello", image_url=value)


def test_image_counts(models):
    urls = [f"https://example.com/{n}" for n in range(6)]
    payload, _ = map_payload(models["gemini-2.5-flash-image"], image_url=urls)
    assert payload["image"] == urls[:3]
    payload, _ = map_payload(models["Minimax-Hailuo-02"], prompt="Hello", image_url=urls)
    assert payload["first_frame_image"] == urls[0]


def test_every_fixture_declared_keys_only(records):
    for record in records:
        payload, _ = map_payload(record, prompt="Hello", image_url="https://example.com/a",
            duration=6, resolution="720p", seed=99999999999, unknown="ignored")
        assert set(payload) <= parameters(record).keys()
        validate_payload(record, payload)


def test_defaults_never_need_media(records, models):
    for kind in ("image", "video", "audio"):
        assert not required_media(pick_default_model(records, kind))
    preferred = ("Minimax-Hailuo-2.3-Fast",)
    assert pick_default_model(records, "video", prefer=preferred)["model"] != preferred[0]
    assert pick_default_model(records, "video", need_image_input=True, prefer=preferred)["model"] == preferred[0]
    assert accepts_media(models[preferred[0]], "image")


def test_nearest_and_size_limits():
    assert nearest_option(7, [8, 6]) == 6
    assert nearest_option("720p", [{"value": "768P"}, {"value": "1080P"}]) == "768P"
    record = {"parameters": [{"name": "size", "type": "enum", "options": ["1024x1024", "2048x2048", "4096x4096"]}]}
    assert map_payload(record, aspect_ratio="square")[0]["size"] == "2048x2048"
    record["parameters"][0]["options"] = ["2K"]
    assert map_payload(record, aspect_ratio="square") == ({}, ["aspect_ratio"])
