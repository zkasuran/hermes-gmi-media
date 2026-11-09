"""GMI Cloud image generation and URL based image editing."""
from agent.image_gen_provider import ImageGenProvider, error_response, save_url_image, success_response
from agent.secret_scope import get_secret_str
from hermes_cli.config import load_config_readonly

from .gmi_media import BackendSupport, GmiError, map_payload, media_urls, price_text

FALLBACK_IMAGE_MODELS = ("seedream-5.0-lite", "gemini-2.5-flash-image", "gpt-image-2-generate")


class GmiImageProvider(BackendSupport, ImageGenProvider):
    name = "gmi-image"
    kind = "image"
    preferences = FALLBACK_IMAGE_MODELS

    def __init__(self, client=None, catalog=None):
        self._init_backend(lambda: get_secret_str("GMI_API_KEY"), client, catalog)

    def capabilities(self):
        return {"modalities": ["text", "image"], "max_reference_images": 3}

    def list_models(self):
        return super().list_models() or [{"id": mid, "display": mid, "price": "Price unavailable"}
                                         for mid in FALLBACK_IMAGE_MODELS]

    def generate(self, prompt, aspect_ratio="landscape", *, image_url=None, reference_image_urls=None, **kwargs):
        model_id = kwargs.get("model") or ""
        try:
            self.require_ready(prompt)
            config = load_config_readonly().get("image_gen", {})
            refs = [reference_image_urls] if isinstance(reference_image_urls, str) else list(reference_image_urls or [])
            images = ([image_url] if image_url else []) + refs
            record = self.select_model(model_id, need_image=bool(images),
                scoped=config.get(self.name, {}).get("model"), shared=config.get("model"))
            model_id = record["model"]
            payload, dropped = map_payload(record, prompt=prompt, aspect_ratio=aspect_ratio,
                image_url=images[:3] if images else None, num_images=kwargs.get("n", kwargs.get("num_images")))
            result = self.client.run(model_id, payload, 300)
            url = media_urls(result, "image")[0]
            try:
                image = str(save_url_image(url, prefix="gmi"))
            except Exception:
                image = url
            return success_response(image=image, model=model_id, prompt=prompt, aspect_ratio=aspect_ratio,
                provider=self.name, modality="image" if images else "text",
                extra={"request_id": result["request_id"], "price": price_text(record), "dropped": dropped})
        except GmiError as exc:
            return error_response(error=str(exc), error_type=exc.kind, provider=self.name,
                                  model=model_id, prompt=prompt, aspect_ratio=aspect_ratio)


def register(ctx):
    ctx.register_image_gen_provider(GmiImageProvider())
