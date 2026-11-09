from copy import deepcopy
import time

from shared.gmi_media import Catalog, price_text, speech_model


def test_cache_ttl_and_stale(client, fake):
    now = [0]
    catalog = Catalog(client, clock=lambda: now[0], ttl=10)
    first = catalog.models("image")
    assert first
    count = len(fake.requests)
    assert Catalog(client, clock=lambda: now[0], ttl=10).models("image") == first
    assert len(fake.requests) == count
    now[0] = 11
    assert catalog.models("image") == first
    assert len(fake.requests) > count
    now[0] = 22
    client.list_model_ids = lambda: (_ for _ in ()).throw(RuntimeError("Offline"))
    assert catalog.models("image") == first


def test_filters(client, fake, models):
    for mid, changes in [("hidden-model", {"model_metadata": {"visibility": "hidden"}}),
                         ("inactive-model", {"status": "inactive"}),
                         ("zero-workers", {"qworkers": 0}),
                         ("probe-model", {}),
                         ("00000000-0000-0000-0000-000000000000", {})]:
        record = deepcopy(models["gemini-2.5-flash-image"])
        record.update(model=mid, **changes)
        fake.models[mid] = record
    records = Catalog(client).models("image")
    assert {m["model"] for m in records} <= models.keys()
    assert not any("probe-model" in r["path"] for r in fake.requests)
    assert not speech_model(models["minimax-music-2.5"])


def test_fallback_fast_and_availability(plugins, client, fake, models):
    for kind, cls in [("image", "GmiImageProvider"), ("video", "GmiVideoProvider"), ("speech", "GmiSpeechProvider")]:
        provider = getattr(plugins[kind], cls)(client=client)
        assert provider.is_available()
    assert not fake.requests
    client.list_model_ids = lambda: (_ for _ in ()).throw(RuntimeError("Offline"))
    provider = plugins["image"].GmiImageProvider(client=client)
    start = time.monotonic()
    assert {m["id"] for m in provider.list_models()} == set(plugins["image"].FALLBACK_IMAGE_MODELS)
    assert time.monotonic() - start < 5
    assert set(plugins["image"].FALLBACK_IMAGE_MODELS) <= models.keys()


def test_slow_catalog_is_bounded(client, monkeypatch):
    import threading
    release = threading.Event()
    monkeypatch.setattr(client, "list_model_ids", lambda: release.wait(2) and [])
    catalog = Catalog(client, picker_timeout=0.05)
    start = time.monotonic()
    try:
        assert catalog.models("image") == []
        assert time.monotonic() - start < 0.5
    finally:
        release.set()
        Catalog._pending[client.base_url].result(timeout=2)


def test_price_is_verbatim():
    assert price_text({"pricing_details": "0.2 USD per odd unit"}) == "0.2 USD per odd unit"
    assert price_text({"price_info": {"price": 3, "unit": "per character"}}) == "3 micro-dollars / per character"
