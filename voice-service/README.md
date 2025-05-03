# Voice Service API

A production-grade speech-to-text (STT) and text-to-speech (TTS) service with multiple provider support and hybrid optimization.

## Features

- Speech-to-Text (STT) transcription using:
  - Local Whisper models (tiny, base, small, medium, large)
  - Google Speech-to-Text API (optional)
  - Azure Speech-to-Text API (optional)
  - AWS Transcribe (optional)
- Text-to-Speech (TTS) synthesis using:
  - Local Coqui TTS models
  - Google Text-to-Speech API (optional)
  - Azure Text-to-Speech API (optional)
  - AWS Polly (optional)
- Production-grade features:
  - Caching via Redis
  - Rate limiting
  - Cost tracking and limits
  - Fallback strategies
  - Provider selection by tier (economy, standard, premium)
  - Robust error handling

## Setup

### Prerequisites

- Python 3.10+
- FFmpeg installed on your system
- Redis server (optional but recommended)
- Docker and Docker Compose (for containerized setup)

### Installation

#### Using Docker Compose (recommended)

1. Clone this repository
2. Update environment variables in `docker-compose.yml` if needed
3. Run the service:

```bash
docker-compose up -d
```

#### Local Installation

1. Clone this repository
2. Install dependencies:

```bash
pip install -r requirements.txt
```

3. Start the service:

```bash
uvicorn main:app --host 0.0.0.0 --port 8000
```

## API Endpoints

### Transcription (STT)

```
POST /api/transcribe
```

Parameters:
- `file`: Audio file (multipart/form-data)
- `language`: Optional language code (e.g., "en", "es")
- `provider`: Optional provider name
- `use_cache`: Whether to use caching (default: true)
- `fallback`: Whether to try other providers if the specified one fails (default: true)

### Speech Synthesis (TTS)

```
POST /api/synthesize
```

Parameters:
- `text`: Text to synthesize
- `voice`: Optional voice ID
- `language`: Optional language code
- `provider`: Optional provider name
- `use_cache`: Whether to use caching (default: true)
- `fallback`: Whether to try other providers if the specified one fails (default: true)

### Get Synthesized Audio

```
GET /api/audio/{audio_id}
```

Returns the previously synthesized audio file.

### Health Check

```
GET /health
```

Returns the service health status and available providers.

## Configuration

The service can be configured using environment variables, which can be set in the `docker-compose.yml` file or directly in your environment.

### Environment Variables

| Variable | Description | Default |
|----------|-------------|---------|
| LOG_LEVEL | Logging level | INFO |
| LOG_FILE | Log file path | speech_service.log |
| REDIS_HOST | Redis host | localhost |
| REDIS_PORT | Redis port | 6379 |
| REDIS_PASSWORD | Redis password | None |
| REDIS_DB | Redis database number | 0 |
| CACHE_TTL | Cache time-to-live (seconds) | 86400 (24h) |
| RATE_LIMIT_TIER1 | Rate limit for tier 1 (req/min) | 100 |
| RATE_LIMIT_TIER2 | Rate limit for tier 2 (req/min) | 300 |
| MAX_DAILY_COST | Maximum daily cost in USD | 10.0 |
| COST_TIER_RATIO | Ratio of paid API usage | 0.2 |

### Third-Party API Keys

To use third-party services, set the following environment variables:

| Variable | Description |
|----------|-------------|
| GOOGLE_API_KEY | Google Cloud API key |
| AZURE_SPEECH_KEY | Azure Speech Services key |
| AZURE_SPEECH_REGION | Azure Speech Services region |
| AWS_ACCESS_KEY | AWS access key ID |
| AWS_SECRET_KEY | AWS secret access key |
| AWS_REGION | AWS region |

## License

MIT 