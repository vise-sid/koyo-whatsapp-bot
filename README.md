# Koyo WhatsApp Voice Bot

A FastAPI application that provides AI-powered voice calling capabilities through Twilio WhatsApp integration using Pipecat AI framework.

## Features

- **WhatsApp Voice Calls**: Handle incoming voice calls through Twilio WhatsApp
- **AI-Powered Conversations**: Uses OpenAI GPT-4o-mini for intelligent responses
- **Speech-to-Text**: Deepgram STT for accurate voice recognition
- **Text-to-Speech**: ElevenLabs TTS for natural voice synthesis
- **Real-time Processing**: WebSocket-based audio streaming
- **Voice Activity Detection**: Silero VAD for optimal conversation flow

## Tech Stack

- **FastAPI**: Web framework
- **Pipecat AI**: Real-time AI pipeline framework
- **Twilio**: WhatsApp voice calling and media streams
- **OpenAI**: GPT-4o-mini for language processing
- **Deepgram**: Speech-to-text service
- **ElevenLabs**: Text-to-speech service
- **Uvicorn**: ASGI server

## Environment Variables

The following environment variables need to be configured:

| Variable | Description | Required |
|----------|-------------|----------|
| `TWILIO_ACCOUNT_SID` | Your Twilio Account SID | Yes |
| `TWILIO_AUTH_TOKEN` | Your Twilio Auth Token | Yes |
| `OPENAI_API_KEY` | OpenAI API key for GPT-4o-mini | Yes |
| `DEEPGRAM_API_KEY` | Deepgram API key for STT | Yes |
| `ELEVENLABS_API_KEY` | ElevenLabs API key for TTS | Yes |
| `ELEVENLABS_VOICE_ID` | ElevenLabs voice ID (default: "Rachel") | No |
| `VALIDATE_TWILIO_SIGNATURE` | Validate Twilio webhook signatures (default: "false") | No |

## Local Development

1. **Clone the repository**
   ```bash
   git clone <your-repo-url>
   cd koyo_v9
   ```

2. **Create a virtual environment**
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

4. **Set up environment variables**
   Create a `.env` file with your API keys:
   ```env
   TWILIO_ACCOUNT_SID=your_twilio_account_sid
   TWILIO_AUTH_TOKEN=your_twilio_auth_token
   OPENAI_API_KEY=your_openai_api_key
   DEEPGRAM_API_KEY=your_deepgram_api_key
   ELEVENLABS_API_KEY=your_elevenlabs_api_key
   ELEVENLABS_VOICE_ID=Rachel
   VALIDATE_TWILIO_SIGNATURE=false
   ```

5. **Run the application**
   ```bash
   uvicorn app:app --reload
   ```

The application will be available at `http://localhost:8000`

## Deployment on Render.com

1. **Push to GitHub**
   - This repository is already configured for GitHub deployment

2. **Deploy on Render**
   - Connect your GitHub repository to Render
   - Render will automatically detect the `render.yaml` configuration
   - Set the required environment variables in Render dashboard
   - Deploy!

3. **Configure Twilio Webhooks**
   - Set your Twilio Voice Request URL to: `https://your-render-app.onrender.com/voice`
   - Set your Twilio Status Callback URL to: `https://your-render-app.onrender.com/status`

## API Endpoints

- `GET /health` - Health check endpoint
- `POST /voice` - Twilio webhook for incoming calls
- `POST /status` - Twilio status callback endpoint
- `WebSocket /ws` - WebSocket endpoint for media streams

## Twilio Configuration

1. **WhatsApp Sandbox Setup**
   - Enable WhatsApp in your Twilio Console
   - Configure your WhatsApp number
   - Set up webhook URLs pointing to your deployed app

2. **Voice Configuration**
   - Create a TwiML App in Twilio Console
   - Set the Voice Request URL to your `/voice` endpoint
   - Configure your WhatsApp number to use this TwiML App

## Architecture

The application uses a pipeline architecture:

1. **Incoming Call**: Twilio receives WhatsApp voice call
2. **Webhook**: Twilio sends webhook to `/voice` endpoint
3. **TwiML Response**: Returns TwiML to connect to WebSocket
4. **Media Stream**: Establishes WebSocket connection for audio streaming
5. **AI Pipeline**: Processes audio through STT → LLM → TTS pipeline
6. **Response**: Sends audio back to caller through Twilio

## Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Test thoroughly
5. Submit a pull request

## License

This project is licensed under the MIT License.
