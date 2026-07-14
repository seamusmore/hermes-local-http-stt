# http-stt

Hermes STT plugin — transcribes audio via a self-hosted HTTP speech-to-text service.

## Install

```bash
cp -r http-stt ~/.hermes/plugins/
```

Then restart the Hermes gateway.

## Configure

Add to `~/.hermes/config.yaml`:

```yaml
stt:
  enabled: true
  provider: http_stt
  http_stt:
    service_url: http://your-stt-service:port
    language: auto
```

- **`service_url`** — required. Your STT service URL (no trailing slash).
- **`language`** — optional, defaults to `auto`.

## How it works

The plugin registers a `TranscriptionProvider` named `http_stt`. When the gateway receives a voice message, it POSTs the audio file to `{service_url}/transcribe` and expects a JSON response:

```json
{"transcript": "recognized text", "text": "alternative field"}
```

## STT service example (SenseVoice)

```python
# service.py — run with: uvicorn service:app --port 8001
from fastapi import FastAPI, UploadFile, File, Query
from funasr import AutoModel

app = FastAPI()
model = AutoModel(model="iic/SenseVoiceSmall")

@app.post("/transcribe")
async def transcribe(file: UploadFile = File(...), language: str = "auto"):
    audio = await file.read()
    result = model.generate(input=audio, language=language)
    return {"transcript": result[0]["text"]}
```

## License

MIT
