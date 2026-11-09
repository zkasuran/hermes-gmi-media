# gmi-speech

Synthesizes speech with MiniMax and Inworld models. Music models are excluded.
Requires Hermes 0.21.6 or later and Python 3.11 or later.
Install with `hermes plugins install gmi-speech` once listed in the catalog.

## Setup

Set `GMI_API_KEY` through Hermes setup for the active profile.
Get a key at <https://console.gmicloud.ai>.

```yaml
tts:
  provider: gmi-speech
  model: minimax-tts-speech-2.6-turbo
```

Example: Read this sentence aloud with gmi-speech: Hello from the lake.
An explicit model argument selects another GMI model.

## Cost and data disclosure

Every generation call spends the user's GMI credits with no Hermes approval prompt.
Prices come live from GMI and are shown as supplied. No total cost is computed.
Prompts and image URLs go to console.gmicloud.ai. Results are fetched from returned URLs.
No telemetry or background processes. No user media files are read outside the Hermes cache.

## Limits

No streaming TTS or voice-bubble compatibility is provided.
Choose voices from the live model list. Invalid voices return an error.
Unsupported formats fall back to an offered format with the correct file extension.
Speed is clamped to the model range.
Unsupported options are dropped and reported where the response supports metadata.
See the repository README for offline tests and manual live checks.
