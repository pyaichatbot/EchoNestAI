# Voice Service

A production-grade Speech-to-Text and Text-to-Speech system with WebSocket support for real-time audio streaming.

## Features

- **Multi-Provider STT**: Whisper, Google, Azure, AWS, ElevenLabs, AssemblyAI
- **Multi-Provider TTS**: Coqui TTS, AI4Bharat TTS, Google, Azure, AWS
- **Real-time WebSocket Audio Streaming**: Full-duplex audio communication
- **Intent Detection**: AI-powered intent recognition
- **Guard Rails**: Content safety and moderation
- **Rate Limiting**: Redis-based rate limiting
- **Cost Management**: Usage tracking and cost limits
- **Multi-language Support**: English, Hindi, German, French, Spanish, and more

## Quick Start

### Prerequisites

- Docker and Docker Compose
- At least 4GB RAM
- 10GB free disk space

### Environment Variables

Create a `.env` file in the project root:

```bash
# Redis Configuration
REDIS_PASSWORD=your_redis_password

# API Keys (optional - leave empty if not using)
GOOGLE_API_KEY=
AZURE_SPEECH_KEY=
AZURE_SPEECH_REGION=eastus
AWS_ACCESS_KEY_ID=
AWS_SECRET_ACCESS_KEY=
AWS_REGION=us-east-1
OPENAI_API_KEY=
ANTHROPIC_API_KEY=
ELEVENLABS_API_KEY=
ASSEMBLYAI_API_KEY=

# Service Configuration
LOG_LEVEL=INFO
MAX_DAILY_COST=100.0
COST_TIER_RATIO=0.1

# TTS Configuration
TTS_MODEL_ID=ai4bharat/indic-parler-tts
TTS_USE_CUDA=false
TTS_SAMPLING_RATE=16000
```

### Build and Run

1. **Build the containers:**
   ```bash
   docker-compose build
   ```

2. **Start the services:**
   ```bash
   docker-compose up -d
   ```

3. **Check the logs:**
   ```bash
   docker-compose logs -f voice-service
   ```

4. **Access the service:**
   - API: http://localhost:8003
   - WebSocket Client: http://localhost:8003/static/websocket_client.html
   - Health Check: http://localhost:8003/health

## API Endpoints

### REST API

- `POST /api/transcribe` - Transcribe audio file
- `POST /api/synthesize` - Synthesize text to speech
- `GET /api/audio/{audio_id}` - Get synthesized audio
- `GET /api/ai4b_speakers` - Get available AI4Bharat speakers
- `GET /health` - Health check

### WebSocket API

- `WS /ws/audio/{session_id}?user_id={user_id}&language={language}` - Real-time audio streaming

## Usage Examples

### REST API

**Transcribe Audio:**
```bash
curl -X POST "http://localhost:8003/api/transcribe" \
  -H "Content-Type: multipart/form-data" \
  -F "file=@audio.wav" \
  -F "language=en"
```

**Synthesize Speech:**
```bash
curl -X POST "http://localhost:8003/api/synthesize" \
  -H "Content-Type: application/json" \
  -d '{
    "text": "Hello, world!",
    "language": "en",
    "voice": "default_female"
  }'
```

### WebSocket Client

1. Open http://localhost:8003/static/websocket_client.html
2. Enter Session ID and User ID
3. Select language
4. Click "Connect"
5. Use "Start Recording" to begin voice chat

## Architecture

### Components

- **STT Engine**: Multi-provider speech-to-text with fallback
- **TTS Engine**: Multi-provider text-to-speech with language-specific routing
- **WebSocket Service**: Real-time audio streaming with VAD and turn-taking
- **Intent Detection**: AI-powered intent recognition
- **Guard Rails**: Content safety and moderation
- **Rate Limiting**: Redis-based rate limiting
- **Cost Management**: Usage tracking and cost limits

### Providers

**STT Providers:**
- Whisper (local, multiple models)
- Google Speech-to-Text
- Azure Speech Services
- AWS Transcribe
- ElevenLabs
- AssemblyAI

**TTS Providers:**
- Coqui TTS (local, multiple models)
- AI4Bharat TTS (local, Indian languages)
- Google Text-to-Speech
- Azure Speech Services
- AWS Polly

## Configuration

### Model Cache Directories

The service uses the following cache directories:
- `/app/models/whisper` - Whisper models
- `/app/models/coqui` - Coqui TTS models
- `/app/models/huggingface` - Hugging Face models
- `/app/parler_models` - AI4Bharat TTS models

### Performance Tuning

- **CUDA**: Set `TTS_USE_CUDA=true` for GPU acceleration
- **Memory**: Increase Docker memory limit for large models
- **Concurrency**: Adjust rate limits based on your needs

## Troubleshooting

### Common Issues

1. **Build fails with memory error:**
   - Increase Docker memory limit to 8GB+
   - Use `--no-cache` flag: `docker-compose build --no-cache`

2. **Model download fails:**
   - Check internet connection
   - Verify disk space (10GB+ required)
   - Check firewall settings

3. **WebSocket connection fails:**
   - Verify Redis is running: `docker-compose ps`
   - Check logs: `docker-compose logs voice-service`
   - Ensure port 8003 is accessible

4. **Audio quality issues:**
   - Use 16kHz mono WAV format
   - Ensure proper microphone permissions
   - Check browser audio settings

### Logs

View logs with:
```bash
# All services
docker-compose logs

# Voice service only
docker-compose logs voice-service

# Follow logs
docker-compose logs -f voice-service
```

### Health Check

Check service health:
```bash
curl http://localhost:8003/health
```

Expected response:
```json
{
  "status": "ok",
  "stt_providers": ["whisper_small", "whisper_base"],
  "tts_providers": ["coqui_tacotron2-DDC", "ai4bharat_indic-parler-tts"]
}
```

## Development

### Local Development

1. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

2. **Set up Redis:**
   ```bash
   docker run -d -p 6379:6379 redis:7-alpine
   ```

3. **Run the service:**
   ```bash
   python main.py
   ```

### Testing

Run tests:
```bash
# Unit tests
python -m pytest tests/

# Integration tests
python -m pytest tests/integration/
```

## License

This project is licensed under the MIT License.

## Support

For issues and questions:
1. Check the troubleshooting section
2. Review the logs
3. Open an issue with detailed error information 