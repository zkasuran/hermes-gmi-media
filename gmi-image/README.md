# gmi-image

Creates images and edits public image URLs with Seedream, Gemini and other live GMI models.
Requires Hermes 0.21.6 or later and Python 3.11 or later.
Install with `hermes plugins install gmi-image` once listed in the catalog.

## Setup

Set `GMI_API_KEY` through Hermes setup for the active profile.
Get a key at <https://console.gmicloud.ai>.

```yaml
image_gen:
  provider: gmi-image
  gmi-image:
    model: seedream-5.0-lite
```

Example: Generate a square image of a blue boat. Use gmi-image.
An explicit model argument selects another GMI model.

## Cost and data disclosure

Every generation call spends the user's GMI credits with no Hermes approval prompt.
Prices come live from GMI and are shown as supplied. No total cost is computed.
Prompts and image URLs go to console.gmicloud.ai. Results are fetched from returned URLs.
No telemetry or background processes. No user media files are read outside the Hermes cache.

## Limits

Images in must be public HTTP or HTTPS URLs. Local files are rejected.
Up to three reference URLs are accepted when the model supports them.
Model-specific sizes may differ from the requested aspect.
An explicit model wins over the scoped model then the shared image model.
Unsupported options are dropped and reported where the response supports metadata.
See the repository README for offline tests and manual live checks.
