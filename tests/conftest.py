"""Offline test boundary. Real credentials are never requested."""
import importlib.util
import ipaddress
import json
from pathlib import Path
import socket
import sys

import pytest

sys.dont_write_bytecode = True
ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture(autouse=True)
def offline(monkeypatch, tmp_path):
    original_connect = socket.socket.connect
    original_connect_ex = socket.socket.connect_ex
    original_lookup = socket.getaddrinfo

    def check(host):
        if host == "localhost":
            return
        try:
            if ipaddress.ip_address(host).is_loopback:
                return
        except ValueError:
            pass
        raise AssertionError("Offline test blocked a non-loopback connection")

    def connect(sock, address):
        check(address[0])
        return original_connect(sock, address)

    def connect_ex(sock, address):
        check(address[0])
        return original_connect_ex(sock, address)

    def lookup(host, *args, **kwargs):
        check(host)
        return original_lookup(host, *args, **kwargs)

    monkeypatch.setattr(socket.socket, "connect", connect)
    monkeypatch.setattr(socket.socket, "connect_ex", connect_ex)
    monkeypatch.setattr(socket, "getaddrinfo", lookup)
    monkeypatch.setenv("HERMES_HOME", str(tmp_path / "hh"))
    import agent.secret_scope
    monkeypatch.setattr(agent.secret_scope, "get_secret_str", lambda name: "test-key-not-real")


@pytest.fixture
def records():
    return json.loads((ROOT / "tests/fixtures/gmi_models.json").read_text())["models"]


@pytest.fixture
def models(records):
    return {record["model"]: record for record in records}


@pytest.fixture
def fake(records):
    from fake_gmi import FakeGmi
    with FakeGmi(records) as server:
        yield server


@pytest.fixture
def client(fake):
    from shared.gmi_media import GmiMediaClient
    value = GmiMediaClient(lambda: "test-key-not-real", base_url=fake.url, sleep=lambda _: None)
    yield value
    value.http.close()


@pytest.fixture
def plugins(monkeypatch):
    result = {}
    for kind in ("image", "video", "speech"):
        name = "fixture_plugin_" + kind
        spec = importlib.util.spec_from_file_location(name, ROOT / f"gmi-{kind}/__init__.py",
            submodule_search_locations=[str(ROOT / f"gmi-{kind}")])
        module = importlib.util.module_from_spec(spec)
        monkeypatch.setitem(sys.modules, name, module)
        spec.loader.exec_module(module)
        result[kind] = module
    monkeypatch.setattr(result["image"], "load_config_readonly", lambda: {})
    return result
