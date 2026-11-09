"""GMI Cloud text and image to video backend."""
from agent.video_gen_provider import (
    VideoGenProvider, COMMON_ASPECT_RATIOS, COMMON_RESOLUTIONS,
    error_response, save_url_video, success_response,
)
from agent.secret_scope import get_secret_str

from .gmi_media import (
    BackendSupport, CANDIDATES, GmiError, accepts_media, choose_param, map_payload,
    media_urls, options, parameters, pick_default_model, price_text,
)

TEXT_MODELS = ("Minimax-Hailuo-02",)
IMAGE_MODELS = ("Minimax-Hailuo-2.3-Fast", "kling-v2-6")


class GmiVideoProvider(BackendSupport, VideoGenProvider):
    name = "gmi-video"
    kind = "video"
    preferences = TEXT_MODELS

    def __init__(self, client=None, catalog=None):
        self._init_backend(lambda: get_secret_str("GMI_API_KEY"), client, catalog)

    def capabilities(self):
        caps = {"modalities": ["text", "image"], "aspect_ratios": list(COMMON_ASPECT_RATIOS),
                "resolutions": list(COMMON_RESOLUTIONS), "min_duration": 3, "max_duration": 15,
                "supports_audio": True, "supports_negative_prompt": True,
                "supports_seed": True, "max_reference_images": 0}
        record = pick_default_model(self.records(), "video", prefer=TEXT_MODELS)
        if not record:
            return caps
        specs = parameters(record)
        caps["modalities"] = ["text", "image"] if accepts_media(record, "image") else ["text"]
        for concept in ("audio", "negative_prompt", "seed"):
            caps["supports_" + concept] = bool(choose_param(specs, CANDIDATES[concept]))
        for concept, key in (("aspect_ratio", "aspect_ratios"), ("resolution", "resolutions")):
            name = choose_param(specs, CANDIDATES[concept])
            caps[key] = options(specs[name]) if name else []
        name = choose_param(specs, CANDIDATES["duration"])
        if name:
            spec = specs[name]
            values = [float(v) for v in options(spec)]
            caps["min_duration"] = min(values) if values else spec.get("min_value", 3)
            caps["max_duration"] = max(values) if values else spec.get("max_value", 15)
        return caps

    def generate(self, prompt, *, model=None, image_url=None, reference_image_urls=None,
                 duration=None, aspect_ratio="16:9", resolution="720p", negative_prompt=None,
                 audio=None, seed=None, **kwargs):
        model_id = model or ""
        try:
            self.require_ready(prompt)
            record = self.select_model(model, need_image=bool(image_url),
                                       prefer=IMAGE_MODELS if image_url else TEXT_MODELS)
            model_id = record["model"]
            payload, dropped = map_payload(record, prompt=prompt, image_url=image_url,
                duration=duration, aspect_ratio=aspect_ratio, resolution=resolution,
                negative_prompt=negative_prompt, audio=audio, seed=seed)
            if reference_image_urls:
                dropped.append("reference_image_urls")
            specs = parameters(record)
            duration_name = choose_param(specs, CANDIDATES["duration"])
            actual_duration = payload.get(duration_name, specs.get(duration_name, {}).get("default_value", 0))
            result = self.client.run(model_id, payload, 900)
            url = media_urls(result, "video")[0]
            try:
                video = str(save_url_video(url, prefix="gmi", headers=None))
            except Exception:
                video = url
            return success_response(video=video, model=model_id, prompt=prompt, provider=self.name,
                modality="image" if image_url else "text", aspect_ratio=aspect_ratio,
                duration=int(float(actual_duration or 0)),
                extra={"request_id": result["request_id"], "price": price_text(record), "dropped": dropped})
        except GmiError as exc:
            return error_response(error=str(exc), error_type=exc.kind, provider=self.name,
                                  model=model_id, prompt=prompt, aspect_ratio=aspect_ratio)


def register(ctx):
    ctx.register_video_gen_provider(GmiVideoProvider())
