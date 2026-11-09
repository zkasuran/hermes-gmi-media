# gmi-video

Creates video from text or a public image URL with Hailuo, Kling and other live GMI models.
Requires Hermes 0.21.6 or later and Python 3.11 or later.
Install with `hermes plugins install gmi-video` once listed in the catalog.

## Setup

Set `GMI_API_KEY` through Hermes setup for the active profile.
Get a key at <https://console.gmicloud.ai>.

```yaml
video_gen:
  provider: gmi-video
  model: Minimax-Hailuo-02
```

Example: Generate a short video of a blue boat. Use gmi-video.
An explicit model argument selects another GMI model.

## Cost and data disclosure

Every generation call spends the user's GMI credits with no Hermes approval prompt.
Prices come live from GMI and are shown as supplied. No total cost is computed.
Prompts and image URLs go to console.gmicloud.ai. Results are fetched from returned URLs.
No telemetry or background processes. No user media files are read outside the Hermes cache.

## Limits

Use a public image URL for image-to-video. Local files are rejected.
Reference image lists are ignored and reported.
Text and image jobs use different default models.
Duration is clamped to the selected model schema. Video may cost money per second.
Unsupported options are dropped and reported where the response supports metadata.
See the repository README for offline tests and manual live checks.
