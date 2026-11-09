from pathlib import Path

import httpx
import pytest

from shared import gmi_media as media


@pytest.mark.parametrize("url", ["http://example.com/a", "https://127.0.0.1/a", "https://[::1]/a", "https://localhost/a", "https://sub.localhost/a"])
def test_reject_url(url, tmp_path):
    with pytest.raises(media.GmiError):
        media.download_audio(url, tmp_path / "audio.mp3")
    assert not list(tmp_path.iterdir())


@pytest.mark.parametrize("body,limit,kind", [(b"", 10, "empty_response"), (b"12345", 4, "invalid_request"), (b"ID3", 10, None)])
@pytest.mark.parametrize("length_header", [True, False])
def test_atomic_download(monkeypatch, tmp_path, body, limit, kind, length_header):
    real_client = httpx.Client
    def reply(request):
        response = httpx.Response(200, content=body)
        if not length_header:
            response.headers.pop("content-length", None)
        return response
    monkeypatch.setattr(media.httpx, "Client", lambda **kwargs: real_client(transport=httpx.MockTransport(reply)))
    dest = tmp_path / "audio.mp3"
    if kind:
        dest.write_bytes(b"previous")
        with pytest.raises(media.GmiError) as error:
            media.download_audio("https://media.example/a", dest, max_bytes=limit)
        assert error.value.kind == kind
        assert dest.read_bytes() == b"previous"
    else:
        assert media.download_audio("https://media.example/a", dest, max_bytes=limit) == str(dest)
        assert dest.read_bytes() == body
    assert list(tmp_path.iterdir()) == [dest]


def test_redirect_downgrade(monkeypatch, tmp_path):
    real_client = httpx.Client
    transport = httpx.MockTransport(lambda request: httpx.Response(302, headers={"location": "http://example.com/a"}))
    monkeypatch.setattr(media.httpx, "Client", lambda **kwargs: real_client(transport=transport))
    with pytest.raises(media.GmiError, match="HTTPS"):
        media.download_audio("https://media.example/a", tmp_path / "audio.mp3")
    assert not list(tmp_path.iterdir())
