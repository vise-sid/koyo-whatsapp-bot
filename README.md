# Koyo Voice & WhatsApp Integration

A FastAPI application that provides voice calling and WhatsApp messaging capabilities using Twilio, Pipecat, OpenAI, Deepgram, ElevenLabs, and Firebase.

## Project Structure

The codebase has been refactored into a clean, modular architecture:

```
koyo_v9/
├── app.py                          # Main FastAPI application
├── app_original.py                 # Original monolithic app.py (backup)
├── requirements.txt                # Python dependencies
├── render.yaml                     # Deployment configuration
├── prompts/                        # AI prompt templates
│   ├── meher_text_prompt.py       # Text conversation prompts
│   └── meher_voice_prompt.py      # Voice conversation prompts
├── services/                       # Business logic services
│   ├── __init__.py
│   ├── whatsapp_service.py        # WhatsApp messaging via Twilio
│   └── voice_session.py           # Voice call session management
├── webhooks/                       # Webhook handlers
│   ├── __init__.py
│   ├── whatsapp_handler.py        # WhatsApp webhook processing
│   └── voice_handler.py           # Twilio voice webhook processing
├── database/                       # Database operations
│   ├── __init__.py
│   └── firebase_service.py        # Firebase Firestore operations
└── utils/                          # Utility functions
    ├── __init__.py
    └── helpers.py                  # Common helper functions
```

## Key Features

### Voice Calls
- Real-time voice conversations using Pipecat
- Speech-to-text with Deepgram
- Text-to-speech with ElevenLabs
- Multilingual support (Hindi/English)
- Intelligent idle handling and call termination
- Function calling for WhatsApp messaging and call termination

### WhatsApp Integration
- Template-based messaging for outbound messages
- Freeform messaging for inbound conversations
- Media processing (audio transcription, image captioning, document handling)
- Context-aware off-call conversations

### Data Management
- Firebase Firestore integration for conversation storage
- Batch operations for efficient message saving
- Session management with automatic cleanup
- Conversation metadata tracking

## Architecture Improvements

### Modular Design
- **Separation of Concerns**: Each module has a single responsibility
- **Service Layer**: Business logic encapsulated in service classes
- **Handler Layer**: Webhook processing separated from business logic
- **Utility Layer**: Common functions centralized for reusability

### Code Quality
- **Type Hints**: Comprehensive type annotations throughout
- **Documentation**: Detailed docstrings for all functions and classes
- **Error Handling**: Robust exception handling with proper logging
- **Async/Await**: Optimized asynchronous operations

### Maintainability
- **Single Responsibility**: Each class and function has one clear purpose
- **Dependency Injection**: Services are initialized with required dependencies
- **Configuration**: Environment-based configuration management
- **Logging**: Structured logging for debugging and monitoring

## API Endpoints

- `GET /health` - Health check endpoint
- `POST /whatsapp-webhook` - Handle incoming WhatsApp messages
- `POST /voice` - Handle Twilio voice webhooks
- `POST /status` - Twilio status callbacks
- `WebSocket /ws` - Voice call WebSocket connection

## Environment Variables

```bash
TWILIO_ACCOUNT_SID=your_twilio_account_sid
TWILIO_AUTH_TOKEN=your_twilio_auth_token
OPENAI_API_KEY=your_openai_api_key
DEEPGRAM_API_KEY=your_deepgram_api_key
ELEVENLABS_API_KEY=your_elevenlabs_api_key
ELEVENLABS_VOICE_ID=your_elevenlabs_voice_id
FIREBASE_CREDENTIALS=your_firebase_credentials_json
TWILIO_WHATSAPP_FROM=whatsapp:+your_twilio_whatsapp_number
VALIDATE_TWILIO_SIGNATURE=true
```

## Development

1. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

2. Set environment variables

3. Run the application:
   ```bash
   python app.py
   ```

## Deployment

The application is configured for deployment on Render with the provided `render.yaml` configuration.

## Benefits of the Refactored Architecture

1. **Maintainability**: Code is organized logically and easy to navigate
2. **Testability**: Each module can be tested independently
3. **Scalability**: Services can be scaled or replaced individually
4. **Reusability**: Common functionality is centralized in utils
5. **Debugging**: Clear separation makes issues easier to isolate
6. **Documentation**: Self-documenting code with comprehensive docstrings
7. **Type Safety**: Type hints prevent runtime errors and improve IDE support
