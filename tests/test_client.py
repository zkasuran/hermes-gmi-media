import socket

import httpx
import pytest

from shared.gmi_media import GmiError, GmiMediaClient, media_urls, scrub

MODEL = "gemini-2.5-flash-image"


def test_socket_boundary():
    with socket.socket() as sock:
        with pytest.raises(AssertionError, match="non-loopback"):
            sock.connect(("203.0.113.1", 443))
        with pytest.raises(AssertionError, match="non-loopback"):
            sock.connect_ex(("203.0.113.1", 443))
    with pytest.raises(AssertionError, match="non-loopback"):
        socket.getaddrinfo("example.com", 443)


@pytest.mark.parametrize("immediate", [False, True])
def test_success(client, fake, immediate):
    fake.immediate = immediate
    result = client.run(MODEL, {"prompt": "Hello"}, 300)
    assert result["request_id"] == "1"
    assert media_urls(result) == [fake.url + "/media/image"]
    assert len(fake.requests) == (1 if immediate else 2)
    assert all(r["Authorization"] == "Bearer test-key-not-real" for r in fake.requests)


def test_polling_backoff(client, fake):
    delays = []
    client.sleep = delays.append
    fake.poll_replies.extend([(200, {"status": "running"})] * 7)
    assert client.run(MODEL, {}, 300)["status"] == "success"
    assert delays == [1, 1.5, 2.25, 3.375, 5, 5, 5]


@pytest.mark.parametrize("status", ["failed", "error", "cancelled"])
def test_failed(client, fake, status):
    fake.poll_replies.append((200, {"status": status, "message": "Please shorten the prompt"}))
    with pytest.raises(GmiError, match="shorten") as error:
        client.run(MODEL, {}, 300)
    assert error.value.kind == "failed"


def test_timeout(client, fake):
    now = [0]
    client.monotonic = lambda: now[0]
    client.sleep = lambda seconds: now.__setitem__(0, now[0] + seconds)
    fake.poll_replies.extend([(200, {"status": "running"})] * 10)
    with pytest.raises(GmiError) as error:
        client.run(MODEL, {}, 2)
    assert error.value.kind == "timeout"


def test_daily_rate_limit(client, fake):
    fake.post_replies.append((429, {"rate_limit": {"limit_type": "RPD", "limit": 10}, "retry_after_ms": 4000}))
    with pytest.raises(GmiError) as error:
        client.run(MODEL, {}, 300)
    assert error.value.kind == "rate_limit"
    assert all(word in str(error.value) for word in (MODEL, "10", "4000", "another model"))


@pytest.mark.parametrize("status,kind", [(401, "auth"), (403, "auth"), (400, "invalid_request"), (500, "api_error")])
def test_http_errors_and_redaction(client, fake, caplog, status, kind):
    fake.post_replies.append((status, {"error": "Authorization: Bearer test-key-not-real"}))
    with pytest.raises(GmiError) as error:
        client.run(MODEL, {}, 300)
    assert error.value.kind == kind
    assert "test-key-not-real" not in str(error.value) + caplog.text
    assert len([r for r in fake.requests if r["method"] == "POST"]) == 1


def test_poll_retry(client, fake):
    fake.poll_replies.extend([(503, {"error": "Busy"}), (500, {"error": "Busy"})])
    assert client.run(MODEL, {}, 300)["status"] == "success"
    assert sum(r["method"] == "GET" for r in fake.requests) == 3


def test_network_retry_and_post_no_retry():
    calls = []
    def fail(request):
        calls.append(request.method)
        raise httpx.ConnectError("network unavailable")
    with httpx.Client(transport=httpx.MockTransport(fail)) as http:
        client = GmiMediaClient(lambda: "test-key-not-real", http=http, sleep=lambda _: None)
        with pytest.raises(GmiError):
            client.submit(MODEL, {})
        assert calls == ["POST"]
        with pytest.raises(GmiError):
            client.poll("1", 100)
        assert calls == ["POST", "GET", "GET", "GET"]


def test_key_per_request(client, fake):
    calls = []
    client.api_key_fn = lambda: calls.append(1) or "test-key-not-real"
    client.run(MODEL, {}, 300)
    assert len(calls) == len(fake.requests) == 2
    fake.requests.clear()
    client.api_key_fn = lambda: None
    with pytest.raises(GmiError) as error:
        client.submit(MODEL, {})
    assert error.value.kind == "missing_credentials"
    assert fake.requests == []


def test_empty_outcome(client, fake):
    fake.poll_replies.append((200, {"status": "completed", "outcome": {}}))
    with pytest.raises(GmiError) as error:
        client.run(MODEL, {}, 300)
    assert error.value.kind == "empty_response"


def test_url_precedence_and_scrub():
    assert media_urls({"outcome": {"thumbnail_image_url": "thumb", "media_urls": [{"url": "real"}]}}) == ["real"]
    assert media_urls({"outcome": {"thumbnail_image_url": "thumb"}}) == ["thumb"]
    assert "test-key-not-real" not in scrub("Bearer test-key-not-real")
