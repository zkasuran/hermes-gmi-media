"""GMI media transport and schema mapping. Python 3.11 or later."""
from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from contextvars import copy_context
from copy import deepcopy
import ipaddress
import json
import math
from pathlib import Path
import re
import tempfile
import threading
import time
from urllib.parse import quote, urljoin, urlsplit
import uuid

import httpx

DEFAULT_BASE_URL = "https://console.gmicloud.ai/api/v1/ie/requestqueue/apikey"
INTERNAL = re.compile(r"capture|probe|ssrf|test|hunt|bypass", re.I)
IMAGE_NAMES = ("image", "first_frame_image", "first_frame", "image_url", "input_image")
CANDIDATES = {
    "prompt": ("prompt", "text"), "aspect_ratio": ("aspect_ratio", "aspectRatio"),
    "duration": ("duration", "durationSeconds", "duration_seconds"),
    "resolution": ("resolution",), "negative_prompt": ("negative_prompt", "negativePrompt"),
    "seed": ("seed",), "audio": ("generateAudio", "generate_audio", "sound"),
    "image_url": IMAGE_NAMES, "voice": ("voice_id", "voice"),
    "speed": ("speed", "speaking_rate"), "format": ("format", "audio_encoding", "output_format"),
    "num_images": ("n", "num_images", "num_outputs", "max_images"),
}
ASPECTS = {"landscape": "16:9", "square": "1:1", "portrait": "9:16"}
IMAGE_INPUT_ERROR = "GMI takes image inputs as public http(s) URLs; upload the file first or pass a URL"


def scrub(text):
    text = str(text)
    text = re.sub(r"(?i)authorization\s*[:=]\s*[^\r\n]+", "Authorization: [redacted]", text)
    text = re.sub(r"(?i)bearer\s+[^\s\"'<>]+", "Bearer [redacted]", text)
    text = re.sub(r"(?i)\b(?:gmi_[a-z0-9_-]{8,}|sk-[a-z0-9_-]{16,})", "[redacted]", text)
    return re.sub(r"(?i)((?:api[_ -]?key|token|voucher)\s*[=:]\s*)[^\s,;]+", r"\1[redacted]", text)


def _reset_text(ms):
    try:
        seconds = int(float(ms) / 1000)
    except (TypeError, ValueError):
        return "Reset time unspecified."
    hours, minutes = divmod(max(0, seconds) // 60, 60)
    return f"Resets in about {hours} h {minutes} min ({int(ms)} ms)."


class GmiError(RuntimeError):
    def __init__(self, message, kind="api_error"):
        super().__init__(scrub(message))
        self.kind = kind


def public_id(model_id):
    if not isinstance(model_id, str) or not model_id or INTERNAL.search(model_id):
        return False
    try:
        uuid.UUID(model_id)
        return False
    except ValueError:
        return True


def media_urls(result, kind=None):
    outcome = result.get("outcome") or {}
    if not isinstance(outcome, dict):
        return []
    urls = []
    keys = ([f"{kind}_url"] if kind else []) + ["media_urls", "image_url", "video_url", "audio_url"]
    for key in dict.fromkeys(keys):
        items = outcome.get(key) or []
        for item in items if isinstance(items, list) else [items]:
            url = item.get("url") if isinstance(item, dict) else item
            if isinstance(url, str) and url and url not in urls:
                urls.append(url)
    if not urls and isinstance(outcome.get("thumbnail_image_url"), str):
        urls.append(outcome["thumbnail_image_url"])
    return urls


class GmiMediaClient:
    def __init__(self, api_key_fn, base_url=DEFAULT_BASE_URL, http=None,
                 sleep=time.sleep, monotonic=time.monotonic):
        self.api_key_fn = api_key_fn
        self.base_url = base_url.rstrip("/")
        self.http = http or httpx.Client(timeout=httpx.Timeout(30, connect=10), trust_env=False)
        self.sleep = sleep
        self.monotonic = monotonic

    def _request(self, method, path, *, body=None, model=None, deadline=None, retry=False):
        for attempt in range(3 if retry else 1):
            key = self.api_key_fn()
            if not key:
                raise GmiError("Set GMI_API_KEY in Hermes setup before using GMI media.", "missing_credentials")
            remaining = deadline - self.monotonic() if deadline is not None else 30
            if remaining <= 0:
                raise GmiError("GMI job deadline expired. Check its status before submitting again.", "timeout")
            try:
                reply = self.http.request(method, f"{self.base_url}/{path}", json=body,
                    headers={"Authorization": "Bearer " + key},
                    timeout=httpx.Timeout(min(30, remaining), connect=min(10, remaining)))
            except httpx.RequestError:
                if retry and attempt < 2:
                    self.sleep(min(1, max(0, remaining)))
                    continue
                raise GmiError("GMI network request failed. Check connectivity and job status before submitting again.", "api_error") from None
            if retry and reply.status_code >= 500 and attempt < 2:
                self.sleep(min(1, max(0, remaining)))
                continue
            try:
                data = reply.json()
            except ValueError:
                data = {}
            if not isinstance(data, dict):
                data = {}
            if not reply.is_success:
                detail = scrub(str(data.get("error") or data.get("message") or "No error detail").replace(key, "[redacted]"))
                if reply.status_code in (401, 403):
                    raise GmiError("GMI rejected the credential. Check the active profile key and its permissions.", "auth")
                if reply.status_code == 429:
                    limit = data.get("rate_limit") or {}
                    if limit.get("limit_type") == "RPD":
                        raise GmiError(f"GMI daily model limit for {model or path}: {limit.get('limit', limit.get('limit_value', 'unspecified'))}. "
                            f"{_reset_text(data.get('retry_after_ms', limit.get('retry_after_ms')))} Choose another model.", "rate_limit")
                    raise GmiError(f"GMI rate limit for {model or path}. Wait before trying again. {detail}", "rate_limit")
                kind = "invalid_request" if 400 <= reply.status_code < 500 else "api_error"
                raise GmiError(f"GMI returned HTTP {reply.status_code}: {detail}. Check the model inputs and job status.", kind)
            # An error can be returned inside a successful polling response too.
            for field in ("error", "message"):
                if field in data:
                    data[field] = scrub(str(data[field]).replace(key, "[redacted]"))
            return data

    def list_model_ids(self):
        return [mid for mid in self._request("GET", "models").get("model_ids", []) if public_id(mid)]

    def get_model(self, model_id):
        data = self._request("GET", "models/" + quote(model_id, safe=""))
        record = data.get("data", data)
        if not isinstance(record, dict) or not record.get("model"):
            raise GmiError("GMI returned no model schema. Choose another model.", "empty_response")
        return record

    def submit(self, model, payload):
        data = self._request("POST", "requests", body={"model": model, "payload": payload}, model=model)
        request_id = data.get("request_id") or data.get("id")
        if not request_id:
            raise GmiError("GMI returned no request id. Check the console before submitting again.", "empty_response")
        data["request_id"] = request_id
        return data

    def _terminal(self, data):
        status = str(data.get("status", "")).lower()
        if status in ("failed", "error", "cancelled"):
            raise GmiError(f"GMI job {status}: {data.get('error') or data.get('message') or 'no detail'}. Review inputs before trying again.", "failed")
        if status in ("success", "completed"):
            if not media_urls(data):
                raise GmiError("GMI job returned no media URL. Check the request in the GMI console.", "empty_response")
            return True
        return False

    def poll(self, request_id, timeout_s, on_status=None):
        deadline = self.monotonic() + timeout_s
        delay = 1.0
        while self.monotonic() < deadline:
            data = self._request("GET", "requests/" + quote(str(request_id), safe=""), deadline=deadline, retry=True)
            if on_status:
                on_status(data.get("status"))
            if self._terminal(data):
                data["request_id"] = request_id
                return data
            self.sleep(min(delay, max(0, deadline - self.monotonic())))
            delay = min(5, delay * 1.5)
        raise GmiError(f"GMI job {request_id} timed out. Check its status before submitting again.", "timeout")

    def run(self, model, payload, timeout_s):
        start = self.monotonic()
        data = self.submit(model, payload)
        if self._terminal(data):
            return data
        return self.poll(data["request_id"], max(0, timeout_s - (self.monotonic() - start)))


def parameters(record):
    return {p["name"]: p for p in record.get("parameters", []) if isinstance(p, dict) and "name" in p}


def options(spec):
    return [v.get("value") if isinstance(v, dict) else v for v in (spec.get("options") or spec.get("enum") or [])]


def choose_param(spec_names, names):
    return next((name for name in names if name in spec_names), None)


def required_media(record):
    required = {m["type"] for m in record.get("modalities", {}).get("input", [])
                if m.get("required") and m.get("type") != "text"}
    for p in parameters(record).values():
        if p.get("required"):
            typ = p.get("type", "")
            if "image" in typ or p["name"] in IMAGE_NAMES:
                required.add("image")
            elif typ in ("audio", "video"):
                required.add(typ)
    return required


def accepts_media(record, media_type):
    specs = parameters(record)
    if media_type == "image":
        return bool(choose_param(specs, IMAGE_NAMES))
    return any(p.get("type") == media_type for p in specs.values())


def usable(record, output_type):
    try:
        return (public_id(record.get("model")) and record.get("status") == "active"
            and int(record.get("qworkers", 0)) > 0
            and record.get("model_metadata", {}).get("visibility") != "hidden"
            and any(m.get("type") == output_type for m in record.get("modalities", {}).get("output", [])))
    except (ValueError, TypeError, AttributeError):
        return False


def speech_model(record):
    return (usable(record, "audio") and record.get("model_type") == "audio"
            and "music" not in record["model"].lower() and "text" in parameters(record)
            and not parameters(record).get("lyrics", {}).get("required"))


def price_text(record):
    details = record.get("pricing_details")
    if details:
        return details if isinstance(details, str) else json.dumps(details, ensure_ascii=True)
    info = record.get("price_info") or {}
    return f"{info['price']} micro-dollars / {info.get('unit', 'unspecified unit')}" if "price" in info else "Price unavailable"


class Catalog:
    """Bounded picker reads with one shared refresh per API origin."""
    _lock = threading.Lock()
    _cache = {}
    _pending = {}
    _refreshers = ThreadPoolExecutor(max_workers=2, thread_name_prefix="gmi-catalog")

    def __init__(self, client, ttl=3600, clock=time.monotonic, picker_timeout=4.5):
        self.client, self.ttl, self.clock = client, ttl, clock
        self.picker_timeout = min(4.5, max(0, picker_timeout))

    def _fetch(self):
        ids = self.client.list_model_ids()
        with ThreadPoolExecutor(max_workers=8, thread_name_prefix="gmi-schema") as pool:
            futures = [pool.submit(copy_context().run, self.client.get_model, mid) for mid in ids]
            records = []
            for future in futures:
                try:
                    records.append(future.result())
                except GmiError as exc:
                    if exc.kind in ("auth", "missing_credentials"):
                        raise
        if ids and not records:
            raise GmiError("GMI returned no model schemas.", "empty_response")
        return records

    def cached(self, kind):
        """Fresh cached records for kind, or None when the cache is cold or expired. No network."""
        with self._lock:
            cached = self._cache.get(self.client.base_url)
            if cached and self.clock() - cached[0] < self.ttl:
                return deepcopy([m for m in cached[1] if usable(m, kind)])
        return None

    def models(self, kind, wait=None):
        key = self.client.base_url
        try:
            with self._lock:
                cached = self._cache.get(key)
                if cached and self.clock() - cached[0] < self.ttl:
                    records = cached[1]
                    return deepcopy([m for m in records if usable(m, kind)])
                future = self._pending.get(key)
                if future is None:
                    future = self._refreshers.submit(copy_context().run, self._fetch)
                    self._pending[key] = future
            records = future.result(timeout=self.picker_timeout if wait is None else wait)
            with self._lock:
                self._cache[key] = (self.clock(), records)
                self._pending.pop(key, None)
        except Exception:
            with self._lock:
                future = self._pending.get(key)
                if future is not None and future.done():
                    self._pending.pop(key, None)
                records = self._cache.get(key, (0, []))[1]
        return deepcopy([m for m in records if usable(m, kind)])


def pick_default_model(records, kind, need_image_input=False, prefer=None):
    candidates = [m for m in records if usable(m, kind)
                  and (kind != "audio" or speech_model(m))
                  and (not required_media(m) if not need_image_input else
                       accepts_media(m, "image") and required_media(m) <= {"image"})]
    for model_id in prefer or ():
        match = next((m for m in candidates if m["model"] == model_id), None)
        if match:
            return match
    def price(m):
        try:
            return float(m.get("price_info", {}).get("price", math.inf))
        except (ValueError, TypeError):
            return math.inf
    return min(candidates, key=price) if candidates else None


def _number(value):
    value = str(value)
    if re.fullmatch(r"\d+(?:\.\d+)?[:x]\d+(?:\.\d+)?", value, re.I):
        a, b = re.split(r"[:x]", value, flags=re.I)
        return float(a) / float(b)
    match = re.fullmatch(r"(-?\d+(?:\.\d+)?)(?:p)?", value, re.I)
    return float(match[1]) if match else None


def nearest_option(requested, choices):
    choices = options({"options": choices})
    for choice in choices:
        if str(choice).lower() == str(requested).lower():
            return choice
    number = _number(requested)
    numeric = [(choice, _number(choice)) for choice in choices]
    numeric = [(choice, val) for choice, val in numeric if val is not None]
    if number is not None and numeric:
        return min(numeric, key=lambda item: (abs(item[1] - number), item[1]))[0]
    raise GmiError(f"Invalid value {requested!r}. Valid values: {choices[:10]}", "invalid_request")


def _coerce(value, spec, *, nearest=False):
    values = options(spec)
    if values:
        for option in values:
            if str(value).lower() == str(option).lower():
                return option
        if nearest:
            return nearest_option(value, values)
        raise GmiError(f"Invalid {spec['name']} value {value!r}. Valid values: {values[:10]}", "invalid_request")
    typ = spec.get("type")
    if typ in ("integer", "float"):
        try:
            value = float(value)
            if not math.isfinite(value):
                raise ValueError
            value = max(spec.get("min_value") if spec.get("min_value") is not None else -math.inf, value)
            value = min(spec.get("max_value") if spec.get("max_value") is not None else math.inf, value)
            return int(value) if typ == "integer" else value
        except (ValueError, TypeError):
            raise GmiError(f"{spec['name']} needs a number.", "invalid_request") from None
    if typ == "boolean" and not isinstance(value, bool):
        raise GmiError(f"{spec['name']} needs true or false.", "invalid_request")
    return value


def validate_payload(record, payload):
    specs = parameters(record)
    for name, value in payload.items():
        if name not in specs:
            raise GmiError(f"Unknown parameter {name}. Use the model's declared parameters.", "invalid_request")
        spec = specs[name]
        choices = options(spec)
        if choices and not any(type(value) is type(v) and value == v for v in choices):
            raise GmiError(f"Invalid {name}. Valid values: {choices[:10]}", "invalid_request")
    for name, spec in specs.items():
        if spec.get("required") and (name not in payload or payload[name] is None):
            raise GmiError(f"Required parameter {name} has no input or default. Supply it or choose another model.", "invalid_request")


def _image_value(value, spec):
    images = value if isinstance(value, (list, tuple)) else [value]
    if not images:
        raise GmiError(IMAGE_INPUT_ERROR, "invalid_request")
    for url in images:
        try:
            parsed = urlsplit(url)
            valid = isinstance(url, str) and parsed.scheme in ("http", "https") and parsed.hostname
        except (ValueError, TypeError):
            valid = False
        if not valid:
            raise GmiError(IMAGE_INPUT_ERROR, "invalid_request")
    limit = max(1, int(spec.get("max_value") or 1))
    return list(images[:limit]) if limit > 1 else images[0]


def _size(aspect, spec):
    choices = options(spec)
    if choices:
        sizes = [v for v in choices if isinstance(v, str) and re.fullmatch(r"\d+x\d+", v)]
        safe = [v for v in sizes if math.prod(map(int, v.split("x"))) <= 4194304]
        sizes = safe or (sizes if len(set(sizes)) == 1 else [])
        if not sizes:
            return None
        target = _number(ASPECTS.get(aspect, aspect))
        if target is None:
            raise GmiError("Use a named aspect or a width:height ratio.", "invalid_request")
        return min(sizes, key=lambda v: (abs(_number(v) - target), -math.prod(map(int, v.split("x")))))
    if spec.get("type") == "string":
        ratio = _number(ASPECTS.get(aspect, aspect)) or 1
        return "1536x1024" if ratio > 1 else "1024x1536" if ratio < 1 else "1024x1024"
    return None


def map_payload(record, **inputs):
    specs, payload, dropped = parameters(record), {}, []
    if "n" in inputs:
        inputs.setdefault("num_images", inputs.pop("n"))
    for concept, value in inputs.items():
        if value is None:
            continue
        name = choose_param(specs, CANDIDATES.get(concept, ()))
        if concept == "aspect_ratio" and not name and "size" in specs:
            size = _size(value, specs["size"])
            if size is not None:
                payload["size"] = size
            else:
                dropped.append(concept)
            continue
        if not name:
            dropped.append(concept)
            continue
        spec = specs[name]
        if concept == "image_url":
            value = _image_value(value, spec)
        elif concept == "aspect_ratio":
            value = ASPECTS.get(value, value)
        elif concept == "audio" and {str(v).lower() for v in options(spec)} == {"on", "off"}:
            value = "on" if value else "off"
        elif concept == "format" and options(spec):
            aliases = {"wav": "LINEAR16", "ogg": "OGG_OPUS", "opus": "OGG_OPUS"}
            choices = options(spec)
            if not any(str(v).lower() == str(value).lower() for v in choices):
                value = aliases.get(str(value).lower(), value)
            if not any(str(v).lower() == str(value).lower() for v in choices):
                value = next((v for v in choices if str(v).lower() == "mp3"), spec.get("default_value") or choices[0])
        payload[name] = _coerce(value, spec, nearest=concept in ("duration", "resolution", "aspect_ratio"))
    for name, spec in specs.items():
        if spec.get("required") and name not in payload:
            value = spec.get("default_value")
            if value is None or (name in IMAGE_NAMES and not value):
                raise GmiError(f"Required parameter {name} has no input or default. Supply it or choose another model.", "invalid_request")
            payload[name] = _coerce(value, spec)
    validate_payload(record, payload)
    return payload, dropped


def audio_extension(value):
    return {"mp3": ".mp3", "flac": ".flac", "ogg": ".ogg", "opus": ".ogg",
            "ogg_opus": ".ogg", "linear16": ".wav", "wav": ".wav"}.get(str(value).lower())


def download_audio(url, dest_path, max_bytes=50_000_000, timeout=60):
    def check_url(value):
        parsed = urlsplit(value)
        host = (parsed.hostname or "").rstrip(".").lower()
        if parsed.scheme != "https" or not host or host == "localhost" or host.endswith(".localhost") or parsed.username or parsed.password:
            raise GmiError("Audio URL must use HTTPS with a public hostname.", "invalid_request")
        try:
            ipaddress.ip_address(host)
        except ValueError:
            return
        raise GmiError("Audio URL must not use an IP address.", "invalid_request")
    check_url(url)
    dest = Path(dest_path)
    temp = None
    try:
        with httpx.Client(timeout=timeout, follow_redirects=False, trust_env=False) as http:
            for redirect in range(6):
                check_url(url)
                with http.stream("GET", url) as response:
                    if response.is_redirect:
                        url = urljoin(url, response.headers.get("location", ""))
                        continue
                    response.raise_for_status()
                    length = response.headers.get("content-length")
                    if length and int(length) > max_bytes:
                        raise GmiError("Audio exceeds the download byte limit.", "invalid_request")
                    with tempfile.NamedTemporaryFile(dir=dest.parent, prefix=".gmi-audio-", delete=False) as output:
                        temp = Path(output.name)
                        count = 0
                        for chunk in response.iter_bytes():
                            count += len(chunk)
                            if count > max_bytes:
                                raise GmiError("Audio exceeds the download byte limit.", "invalid_request")
                            output.write(chunk)
                    if not count:
                        raise GmiError("GMI audio download was empty. Check the result URL.", "empty_response")
                    temp.replace(dest)
                    return str(dest)
            raise GmiError("Too many audio redirects. Check the result URL.", "api_error")
    except GmiError:
        raise
    except Exception:
        raise GmiError("Audio download failed. Check the result URL and output directory.", "api_error") from None
    finally:
        if temp is not None:
            temp.unlink(missing_ok=True)


class BackendSupport:
    """Picker metadata and model selection shared by the standalone backends."""
    display_name = "GMI Cloud"
    kind = ""
    preferences = ()

    def _init_backend(self, api_key_fn, client=None, catalog=None):
        self.client = client or GmiMediaClient(api_key_fn)
        self.catalog = catalog or Catalog(self.client)

    def is_available(self):
        try:
            return bool(self.client.api_key_fn())
        except Exception:
            return False

    def get_setup_schema(self):
        return {"name": self.display_name, "badge": "paid", "tag": "Media via GMI Cloud. Spends GMI credits.",
                "env_vars": [{"key": "GMI_API_KEY", "prompt": "GMI Cloud API key", "url": "https://console.gmicloud.ai"}]}

    def records(self, wait=None):
        records = self.catalog.models(self.kind, wait)
        return [m for m in records if speech_model(m)] if self.kind == "audio" else records

    def list_models(self):
        return [{"id": m["model"], "display": m.get("display_name") or m["model"],
                 "price": price_text(m), "strengths": str(m.get("brief_description", ""))[:80],
                 "modalities": ["text", "image"] if accepts_media(m, "image") else ["text"]}
                for m in self.records()]

    def default_model(self):
        record = pick_default_model(self.records(), self.kind, prefer=self.preferences)
        return record["model"] if record else None

    def _fits(self, record, need_image):
        return (usable(record, self.kind) and (self.kind != "audio" or speech_model(record))
                and (accepts_media(record, "image") and required_media(record) <= {"image"}
                     if need_image else not required_media(record)))

    def _probe(self, ids, need_image):
        """Read single schemas while the full catalog is cold (the full read takes tens of seconds)."""
        for model_id in dict.fromkeys(i for i in ids if i):
            try:
                record = self.client.get_model(model_id)
            except GmiError as exc:
                if exc.kind in ("auth", "missing_credentials", "rate_limit"):
                    raise
                continue
            if self._fits(record, need_image):
                return record
        return None

    def select_model(self, explicit=None, *, need_image=False, scoped=None, shared=None, prefer=None):
        prefer = self.preferences if prefer is None else prefer
        chosen = explicit or scoped
        records = self.catalog.cached(self.kind)
        record = None
        if chosen:
            record = next((m for m in records or () if m["model"] == chosen), None) or self.client.get_model(chosen)
        elif records is None:
            record = self._probe(prefer, need_image)
            if record is None:
                records = self.records(wait=90)
        if record is None:
            record = next((m for m in records or () if m["model"] == shared), None) if shared else None
            record = record or pick_default_model(records or [], self.kind, need_image_input=need_image, prefer=prefer)
        if record is None:
            raise GmiError("No usable GMI model found. Refresh the catalog or set an explicit model.", "invalid_request")
        if not self._fits(record, need_image):
            suggestion = pick_default_model(self.records(wait=90), self.kind, need_image_input=need_image, prefer=prefer)
            alternative = suggestion["model"] if suggestion else "another model from the catalog"
            raise GmiError(f"Model {record['model']} does not accept these inputs. Use {alternative}.", "invalid_request")
        return record

    def require_ready(self, prompt):
        if not isinstance(prompt, str) or not prompt.strip():
            raise GmiError("A nonempty text prompt is required.", "invalid_request")
        if not self.is_available():
            raise GmiError("Set GMI_API_KEY in Hermes setup before using GMI media.", "missing_credentials")
