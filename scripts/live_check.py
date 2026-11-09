"""Manual paid checks. Never invoked by the offline test suite."""
from __future__ import annotations

import argparse
import importlib.util
import json
from pathlib import Path
import shutil
import subprocess
import sys
from decimal import Decimal, InvalidOperation

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from shared.gmi_media import map_payload, parameters, price_text, scrub

TEXT = "A small blue boat on a calm lake."


def load_provider(kind):
    name = "live_plugin_" + kind
    folder = ROOT / ("gmi-" + kind)
    spec = importlib.util.spec_from_file_location(name, folder / "__init__.py",
                                               submodule_search_locations=[str(folder)])
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return getattr(module, "Gmi" + kind.title() + "Provider")()


def estimate(record, payload):
    """Only exact billing units with known quantities can produce an estimate."""
    info = record.get("price_info") or {}
    try:
        price = Decimal(str(info["price"])) / Decimal(1_000_000)
        if not price.is_finite() or price < 0:
            return None
    except (KeyError, InvalidOperation):
        return None
    unit = str(info.get("unit", "")).lower().strip()
    if unit in ("per request", "per image", "request", "image"):
        return price
    if unit in ("per second", "second"):
        specs = parameters(record)
        name = next((n for n in ("duration", "durationSeconds", "duration_seconds") if n in specs), None)
        duration = payload.get(name, specs.get(name, {}).get("default_value"))
        if duration is not None:
            return price * Decimal(str(duration))
    # Character billing labels have known inconsistencies. Never estimate them.
    return None


def describe(path, kind):
    print(f"Output: {path} ({path.stat().st_size} bytes)")
    if kind == "image":
        # Read metadata only. Never decode pixels or display an image.
        from PIL import Image
        with Image.open(path) as image:
            width, height = image.size
            print(f"Image size: {width}x{height}")
            if max(width, height) > 1990:
                print("Image exceeds the preview limit. Do not open or display it.")
    elif kind == "video" and shutil.which("ffprobe"):
        result = subprocess.run(["ffprobe", "-v", "error", "-show_entries", "format=duration",
                                 "-of", "json", str(path)], capture_output=True, text=True)
        if result.returncode == 0:
            duration = json.loads(result.stdout).get("format", {}).get("duration")
            if duration:
                print(f"Video duration: {duration} seconds")


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=("image", "image-edit", "speech", "video", "video-image", "all"))
    parser.add_argument("--model")
    parser.add_argument("--max-usd", type=Decimal, default=Decimal("3"))
    parser.add_argument("--i-accept-unknown-cost", action="store_true")
    parser.add_argument("--image-url", help="Public input URL for edit and image-to-video checks")
    args = parser.parse_args(argv)
    if not args.max_usd.is_finite() or args.max_usd <= 0:
        parser.error("--max-usd must be a positive finite amount")
    modes = ["image", "image-edit", "speech", "video", "video-image"] if args.mode == "all" else [args.mode]
    if args.mode == "all" and args.model:
        parser.error("--model applies to one mode at a time")
    if any(mode in ("image-edit", "video-image") for mode in modes) and not args.image_url:
        parser.error("--image-url is required for image inputs")
    plans = []
    known_total = Decimal(0)
    unknown = False
    for mode in modes:
        kind = "speech" if mode == "speech" else mode.split("-")[0]
        provider = load_provider(kind)
        if not provider.is_available():
            parser.error("Set GMI_API_KEY in the active Hermes profile before running")
        image_url = args.image_url if mode in ("image-edit", "video-image") else None
        prefer = ("Minimax-Hailuo-2.3-Fast", "kling-v2-6") if mode == "video-image" else None
        record = provider.select_model(args.model, need_image=bool(image_url), prefer=prefer)
        inputs = {"prompt": TEXT}
        if kind == "speech":
            inputs["format"] = "mp3"
        else:
            inputs.update(aspect_ratio="square" if kind == "image" else "16:9", image_url=image_url)
        if kind == "video":
            inputs.update(duration=3, resolution="720p")
        payload, _ = map_payload(record, **inputs)
        cost = estimate(record, payload)
        print(f"{mode}: {record['model']} | GMI price: {price_text(record)}", flush=True)
        if cost is None:
            unknown = True
            print("Cost cannot be estimated from this unit.", flush=True)
        else:
            known_total += cost
            print(f"Estimate for this check: USD {cost}", flush=True)
        plans.append((mode, kind, provider, record, inputs))
    if known_total > args.max_usd:
        parser.error(f"Known estimate USD {known_total} exceeds --max-usd")
    if unknown and not args.i_accept_unknown_cost:
        parser.error("Unknown cost requires --i-accept-unknown-cost; a hard spend cap cannot be guaranteed")
    output = ROOT / "live-output"
    output.mkdir(exist_ok=True)
    for mode, kind, provider, record, inputs in plans:
        if kind == "speech":
            path = Path(provider.synthesize(TEXT, str(output / "speech.mp3"), model=record["model"]))
        else:
            inputs.pop("prompt")
            result = provider.generate(TEXT, model=record["model"], **inputs)
            if not result["success"]:
                raise RuntimeError(result["error"])
            value = result[kind]
            if value.startswith(("https://", "http://")):
                raise RuntimeError("Job succeeded but download failed. Check the console before rerunning.")
            source = Path(value)
            path = output / (mode + source.suffix)
            shutil.copyfile(source, path)
        describe(path, kind)
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as exc:
        print(scrub(exc), file=sys.stderr)
        raise SystemExit(1)
