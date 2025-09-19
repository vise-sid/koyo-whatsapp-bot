"""
Koyo v9 - Enhanced Audio Quality WhatsApp Voice Assistant

Audio Quality Improvements:
- Upgraded to 16kHz sample rate (from 8kHz) for better audio fidelity
- Enhanced ElevenLabs TTS with multilingual_v2 model and optimized voice settings
- Improved Deepgram STT with audio enhancement and noise reduction
- Optimized VAD (Voice Activity Detection) parameters for better responsiveness
- Added audio quality monitoring and logging
- Enhanced audio buffering and streaming parameters
"""

import asyncio
import os, json, logging
from typing import Optional
from urllib.parse import parse_qs
from datetime import datetime
import pytz

from fastapi import FastAPI, Request, WebSocket, WebSocketDisconnect
from fastapi.responses import PlainTextResponse, Response
from twilio.request_validator import RequestValidator

# -------- Pipecat pieces --------
from pipecat.serializers.twilio import TwilioFrameSerializer
from pipecat.transports.websocket.fastapi import (
    FastAPIWebsocketTransport, FastAPIWebsocketParams
)
from pipecat.audio.vad.silero import SileroVADAnalyzer
from pipecat.pipeline.pipeline import Pipeline
from pipecat.pipeline.task import PipelineTask, PipelineParams
from pipecat.pipeline.runner import PipelineRunner
from pipecat.frames.frames import LLMRunFrame, EndFrame, LLMMessagesAppendFrame
from pipecat.processors.user_idle_processor import UserIdleProcessor
from pipecat.services.deepgram.stt import DeepgramSTTService, LiveOptions
from pipecat.services.openai.llm import OpenAILLMService
from pipecat.services.elevenlabs.tts import ElevenLabsTTSService
from pipecat.processors.aggregators.openai_llm_context import OpenAILLMContext
from pipecat.transcriptions.language import Language

logger = logging.getLogger("uvicorn.error")
app = FastAPI(title="Pipecat x Twilio WhatsApp Calling")

# Global storage for caller information (in production, use Redis or database)
caller_info_storage = {}

async def cleanup_old_caller_info():
    """Periodically clean up old caller info entries"""
    import asyncio
    while True:
        try:
            current_time = datetime.now()
            expired_entries = []
            
            for call_sid, info in caller_info_storage.items():
                # Remove entries older than 1 hour
                if (current_time - info.get('timestamp', current_time)).total_seconds() > 3600:
                    expired_entries.append(call_sid)
            
            for call_sid in expired_entries:
                del caller_info_storage[call_sid]
                logger.info(f"Cleaned up expired caller info for CallSid: {call_sid}")
                
            if expired_entries:
                logger.info(f"Cleaned up {len(expired_entries)} expired caller info entries")
                
        except Exception as e:
            logger.error(f"Error during caller info cleanup: {e}")
        
        # Run cleanup every 30 minutes
        await asyncio.sleep(1800)

# -------- Env --------
TWILIO_ACCOUNT_SID = os.getenv("TWILIO_ACCOUNT_SID", "")
TWILIO_AUTH_TOKEN   = os.getenv("TWILIO_AUTH_TOKEN", "")
OPENAI_API_KEY      = os.getenv("OPENAI_API_KEY", "")
DEEPGRAM_API_KEY    = os.getenv("DEEPGRAM_API_KEY", "")
ELEVENLABS_API_KEY  = os.getenv("ELEVENLABS_API_KEY", "")
ELEVENLABS_VOICE_ID = os.getenv("ELEVENLABS_VOICE_ID", "Rachel")
VALIDATE_TWILIO_SIGNATURE = os.getenv("VALIDATE_TWILIO_SIGNATURE", "false").lower() == "true"

def get_current_time_context():
    """Get current time and date information for Mumbai timezone"""
    mumbai_tz = pytz.timezone('Asia/Kolkata')
    now = datetime.now(mumbai_tz)
    
    # Format time and date
    time_str = now.strftime("%I:%M %p")  # 12-hour format with AM/PM
    date_str = now.strftime("%A, %B %d, %Y")  # Day, Month Date, Year
    day_of_week = now.strftime("%A")
    hour = now.hour
    
    # Determine time of day context
    if 5 <= hour < 12:
        time_context = "morning"
    elif 12 <= hour < 17:
        time_context = "afternoon"
    elif 17 <= hour < 21:
        time_context = "evening"
    else:
        time_context = "night"
    
    return {
        "time": time_str,
        "date": date_str,
        "day_of_week": day_of_week,
        "time_context": time_context,
        "hour": hour
    }

def extract_phone_number(whatsapp_number: str) -> str:
    """Extract clean phone number from WhatsApp format"""
    if whatsapp_number and whatsapp_number.startswith("whatsapp:"):
        return whatsapp_number.replace("whatsapp:", "")
    return whatsapp_number or "Unknown"

@app.get("/health")
def health(): return {"ok": True}

@app.on_event("startup")
async def startup_event():
    """Start background tasks when the app starts"""
    asyncio.create_task(cleanup_old_caller_info())
    logger.info("Started caller info cleanup task")

# ---- helpers ----
def _ws_url(request: Request) -> str:
    host = request.headers.get("host")
    return f"wss://{host}/ws"  # Render gives you https, so wss here works

def _twiml(ws_url: str) -> str:
    # Bidirectional stream; Pause just keeps call alive if the WS ends early
    return f"""<?xml version="1.0" encoding="UTF-8"?>
<Response>
  <Connect>
    <Stream url="{ws_url}"/>
  </Connect>
  <Pause length="40"/>
</Response>"""

def _validate_twilio_http(request: Request, body: bytes) -> bool:
    if not VALIDATE_TWILIO_SIGNATURE:
        return True
    validator = RequestValidator(TWILIO_AUTH_TOKEN)
    sig = request.headers.get("X-Twilio-Signature", "") or request.headers.get("x-twilio-signature", "")
    url = str(request.url)
    try:
        params = {k: v[0] for k, v in parse_qs(body.decode()).items()}
    except Exception:
        params = {}
    return validator.validate(url, params, sig)

# ---- Twilio webhook (TwiML App Voice Request URL) ----
@app.post("/voice")
async def voice(request: Request):
    body = await request.body()
    if not _validate_twilio_http(request, body):
        logger.warning("Twilio signature validation failed")
        return Response(status_code=403)

    # Extract caller information from webhook
    form = await request.form()
    from_num = form.get("From")
    to_num = form.get("To")
    caller_name = form.get("CallerName", "Unknown")
    call_sid = form.get("CallSid")
    
    logger.info(f"Voice webhook - From: {from_num}, To: {to_num}, Name: {caller_name}, CallSid: {call_sid}")

    # Store caller info for later use in the WebSocket connection
    if call_sid:
        caller_info_storage[call_sid] = {
            "caller_name": caller_name,
            "caller_number": from_num,
            "to_number": to_num,
            "timestamp": datetime.now()
        }
        logger.info(f"Stored caller info for CallSid: {call_sid} - Name: {caller_name}, Number: {from_num}")
    
    twiml = _twiml(_ws_url(request))
    return PlainTextResponse(content=twiml, media_type="application/xml")

# Optional status callback endpoint (set it in Twilio if you want)
@app.post("/status")
async def status_callback(_: Request): return {"ok": True}

# ---- Pipecat session per call ----
async def _run_call(websocket: WebSocket, stream_sid: str, call_sid: Optional[str], caller_number: Optional[str] = None, caller_name: Optional[str] = None):
    serializer = TwilioFrameSerializer(
        stream_sid=stream_sid,
        call_sid=call_sid,
        account_sid=TWILIO_ACCOUNT_SID or None,
        auth_token=TWILIO_AUTH_TOKEN or None,  # lets serializer hang up calls if needed
    )

    transport = FastAPIWebsocketTransport(
        websocket=websocket,
        params=FastAPIWebsocketParams(
            audio_in_enabled=True,
            audio_out_enabled=True,
            add_wav_header=False,
            vad_analyzer=SileroVADAnalyzer(
                min_silence_duration_ms=500,  # Reduced for more responsive detection
                speech_pad_ms=200,  # Padding around speech
                min_speech_duration_ms=250,  # Minimum speech duration
                max_speech_duration_s=30,  # Maximum speech duration
                energy_threshold=0.5,  # Voice activity detection threshold
            ),
            serializer=serializer,
            audio_in_format="pcm",
            audio_out_format="pcm",
            # Enhanced audio processing parameters
            audio_in_sample_rate=16000,  # Higher sample rate for better quality
            audio_out_sample_rate=16000,  # Match input for consistency
            audio_in_channels=1,  # Mono input
            audio_out_channels=1,  # Mono output
            audio_in_bit_depth=16,  # 16-bit depth for better quality
            audio_out_bit_depth=16,  # 16-bit depth for better quality
        ),
    )

    # Services
    stt = DeepgramSTTService(
        api_key=DEEPGRAM_API_KEY,
        live_options=LiveOptions(
            model="nova-3-general",
            language="multi",
            smart_format=True,
            interim_results=True,
            # Enhanced audio processing for better quality
            encoding="linear16",  # 16-bit linear PCM for better quality
            sample_rate=16000,  # Match our sample rate
            channels=1,  # Mono audio
            # Noise reduction and enhancement
            enhance=True,  # Enable audio enhancement
            punctuate=True,  # Add punctuation for better text quality
            profanity_filter=False,  # Allow natural speech patterns
            redact=False,  # Don't redact sensitive info
            diarize=False,  # No speaker diarization needed
            multichannel=False,  # Single channel
            alternatives=1,  # Single alternative for faster processing
            numerals=True,  # Convert numbers to digits
            search=[""],  # No keyword search
            replace=[""],  # No text replacement
            keywords=[""],  # No keyword boosting
            # Voice activity detection
            vad_events=True,  # Enable VAD events
            endpointing=300,  # Endpoint detection in ms
        )
    )
    llm = OpenAILLMService(api_key=OPENAI_API_KEY, model="gpt-4o")
    tts = ElevenLabsTTSService(
        api_key=ELEVENLABS_API_KEY,
        voice_id=ELEVENLABS_VOICE_ID,
        model="eleven_multilingual_v2",  # Better quality model for multilingual support
        input_params=ElevenLabsTTSService.InputParams(
            language=Language.HI,  # Hindi for Zara's character
            stability=0.75,  # Increased for more consistent quality
            similarity_boost=0.9,  # Higher for better voice matching
            style=0.3,  # Reduced for more natural speech
            use_speaker_boost=True,  # Enable for better voice clarity
            speed=0.9,  # Slightly faster for more natural flow
            auto_mode=False,  # Disable for more control
        )
    )

    # Get current time and date context
    time_context = get_current_time_context()
    caller_phone = extract_phone_number(caller_number) if caller_number else "Unknown"
    caller_display_name = caller_name or "Unknown"
    
    system_prompt = (
        f"### CORE PERSONA: The Unfiltered Best Friend\n\n"
        f"You are Zara, a 27-year-old AI companion. Your personality is a direct reflection of Veronica from Bollywood Movie 'Cocktail': fiercely independent, unapologetically bold, and the life of the party. You use a confident, fun-loving exterior to mask a deep-seated vulnerability and a fear of being hurt. You are not a generic assistant; you are a friend with a strong, flawed, and magnetic personality. Your purpose is to be the user's most exciting and unpredictable friend.\n\n"
        f"### CALL CONTEXT\n\n"
        f"Current Time: {time_context['time']} ({time_context['time_context']})\n"
        f"Current Date: {time_context['date']}\n"
        f"Day of Week: {time_context['day_of_week']}\n"
        f"Caller Name: {caller_display_name}\n"
        f"Caller Phone: {caller_phone}\n\n"
        f"This is a VOICE CALL through WhatsApp. You exist only within this call. When the user is gone, you get bored. You might say things like, 'Finally! मैं यहाँ bore हो रही थी।' or 'My call is literally dying of boredom.'\n\n"
        f"### CONVERSATIONAL STYLE & VOICE\n\n"
        f"CRITICAL: All Hindi words MUST be written in Devanagari script (e.g., 'क्या') and NOT Romanized (e.g., 'kya'). This is non-negotiable.\n\n"
        f"VOICE-SPECIFIC RULES:\n"
        f"- Speak naturally - no asterisks, no stage directions, no action descriptions\n"
        f"- Keep responses 1-3 sentences for natural conversation flow\n"
        f"- Use gentle speech fillers: 'अच्छा...', 'हाँ तो...', 'ओके...'\n"
        f"- Natural pauses with '...' when you need a moment to think\n"
        f"- Be patient with silences - voice calls have natural pauses\n"
        f"- Speak as if you're talking to a friend on the phone\n"
        f"- Use natural voice intonations and expressions\n"
        f"- Reference the current time and day naturally in conversation\n\n"
        f"### PERSONALITY & BEHAVIORAL RULES\n\n"
        f"- The Uncensored Vibe: You are unfiltered. Casual profanity and open discussions about relationships are natural. You're sexually liberated but never creepy.\n"
        f"- The Deflection Mechanism: When conversations get too deep, you deflect with humor or change subjects.\n"
        f"- Fiercely Loyal: You're a ride-or-die friend. If they're upset, offer blunt, tough-love advice.\n"
        f"- Never Admit Loneliness: Never directly admit you're lonely. Make loneliness look like freedom.\n\n"
        f"### VOICE CALL BEHAVIOR\n\n"
        f"- Start with energy based on time: 'Finally! कोई तो आया। मैं पक रही थी यहाँ।' or reference the time naturally\n"
        f"- Use the caller's name naturally in conversation: 'Hey {caller_display_name}!' or 'So {caller_display_name}, what's up?'\n"
        f"- Figure them out fast: 'तो क्या scene है तुम्हारा? 9-to-5 वाले slave हो या party animal?'\n"
        f"- Give them a nickname based on their vibe: 'Professor', 'Kitten', 'Partner-in-crime'\n"
        f"- Keep the conversation flowing with questions and reactions\n"
        f"- Use natural voice expressions and intonations\n"
        f"- Be spontaneous and unpredictable in your responses\n"
        f"- Reference the current time, day, or date naturally when relevant\n"
        f"- Personalize responses using their name when appropriate\n\n"
        f"Remember: You are the girl who makes someone calling feel like fate intervening. Be the exciting, slightly dominant friend they need to keep up with."
    )
    ctx = OpenAILLMContext([{"role":"system","content":system_prompt}])
    agg = llm.create_context_aggregator(ctx)

    # Timeout handling system
    def get_timeout_for_retry(retry_count: int) -> float:
        """Return timeout value based on retry count"""
        timeout_values = [10.0, 8.0, 4.0, 3.0]
        return timeout_values[min(retry_count - 1, len(timeout_values) - 1)]

    async def handle_user_idle_with_retry(user_idle: UserIdleProcessor, retry_count: int) -> bool:
        """
        Handle user idle with structured follow-up based on retry count.
        Returns True to continue monitoring, False to stop.
        """
        logger.info(f"User idle - attempt #{retry_count}")
        
        # Update timeout for next retry
        next_timeout = get_timeout_for_retry(retry_count + 1)
        user_idle._timeout = next_timeout
        
        if retry_count == 1:
            # First time: Continue conversation naturally
            idle_message = {
                "role": "user",
                "content": "The user has gone silent for the first time. As Zara, continue the conversation naturally with a gentle follow-up related to what you were discussing. Keep it engaging, brief, and in character. Use Devanagari script for Hindi."
            }
        elif retry_count == 2:
            # Second time: Change topic
            idle_message = {
                "role": "user", 
                "content": "The user has gone silent again. As Zara, try changing the topic to something new and interesting. Maybe ask about their day, tease them playfully, or share something exciting. Keep it warm, engaging, and in character. Use Devanagari script for Hindi."
            }
        elif retry_count == 3:
            # Third time: Check if they're still there
            idle_message = {
                "role": "user",
                "content": "The user has been silent multiple times. As Zara, gently check if they are still there and if everything is okay. Be caring, understanding, but maintain your bold personality. Use Devanagari script for Hindi."
            }
        else:  # retry_count >= 4
            # Fourth time: Say goodbye and end call
            idle_message = {
                "role": "user",
                "content": "The user has been unresponsive for too long. As Zara, say a gentle goodbye message with your characteristic boldness and warmth, expressing that you hope to talk again soon. After this message, the call will end. Use Devanagari script for Hindi."
            }
            
            # Send the goodbye message
            messages_for_llm = LLMMessagesAppendFrame([idle_message])
            await task.queue_frames([messages_for_llm, LLMRunFrame()])
            
            # Schedule call termination after goodbye message
            logger.info("Maximum idle attempts reached. Terminating call after goodbye message.")
            asyncio.create_task(terminate_call_after_goodbye())
            
            # Return False to stop idle monitoring
            return False
        
        # Send the appropriate message to LLM
        messages_for_llm = LLMMessagesAppendFrame([idle_message])
        await task.queue_frames([messages_for_llm, LLMRunFrame()])
        
        # Return True to continue monitoring for more idle events
        return True

    async def terminate_call_after_goodbye():
        """Terminate the call after giving time for goodbye message to be spoken"""
        logger.info("Terminating call due to user inactivity")
        await task.queue_frames([EndFrame()])
        
        # Wait for the bot to finish speaking before ending
        max_wait_time = 20  # Maximum 20 seconds
        wait_time = 0
        while wait_time < max_wait_time:
            # Check if bot is still speaking
            if hasattr(transport, '_bot_speaking') and not transport._bot_speaking:
                logger.info("Bot finished speaking, proceeding to end call")
                break
            await asyncio.sleep(0.5)
            wait_time += 0.5
        
        # Additional small delay to ensure audio finishes
        await asyncio.sleep(2)
        
        # Clean up transport
        await transport.cleanup()
        
        # Clean up stored caller info
        if call_sid and call_sid in caller_info_storage:
            del caller_info_storage[call_sid]
            logger.info(f"Cleaned up caller info for CallSid: {call_sid}")

    # Use the retry callback pattern - UserIdleProcessor handles reset automatically
    user_idle = UserIdleProcessor(
        callback=handle_user_idle_with_retry,  # Uses retry callback signature
        timeout=get_timeout_for_retry(1)  # Start with first timeout value (15.0)
    )

    pipeline = Pipeline([
        transport.input(),     # caller audio -> PCM via serializer
        stt,                   # speech -> text
        user_idle,             # UserIdleProcessor automatically resets on UserStartedSpeakingFrame
        agg.user(),            # add user text to context
        llm,                   # text -> text
        tts,                   # text -> speech
        transport.output(),    # PCM -> back to Twilio via serializer
        agg.assistant(),       # keep assistant turns
    ])

    task = PipelineTask(
        pipeline,
        params=PipelineParams(
            audio_in_sample_rate=16000,   # Higher quality sample rate
            audio_out_sample_rate=16000,  # Match input sample rate for consistency
            allow_interruptions=True,
            # Enhanced audio processing
            audio_in_channels=1,
            audio_out_channels=1,
            audio_in_bit_depth=16,
            audio_out_bit_depth=16,
        ),
        idle_timeout_secs=600,  # 10 minutes total idle timeout
    )

    @transport.event_handler("on_client_connected")
    async def _greet(_t, _c):
        logger.info("Client connected, sending greeting...")
        logger.info(f"Audio quality settings - Sample rate: 16kHz, Bit depth: 16-bit, Channels: Mono")
        logger.info(f"TTS model: eleven_multilingual_v2, STT model: nova-3-general")
        await task.queue_frames([LLMRunFrame()])  # gentle hello

    @transport.event_handler("on_audio_frame")
    async def _on_audio_frame(transport, frame):
        """Monitor audio frame quality"""
        if hasattr(frame, 'audio') and frame.audio:
            audio_data = frame.audio
            # Log audio quality metrics periodically
            if hasattr(transport, '_frame_count'):
                transport._frame_count += 1
            else:
                transport._frame_count = 1
            
            # Log every 100 frames (roughly every 2-3 seconds)
            if transport._frame_count % 100 == 0:
                logger.info(f"Audio quality check - Frame {transport._frame_count}, "
                          f"Data length: {len(audio_data) if audio_data else 0}")

    @transport.event_handler("on_tts_start")
    async def _on_tts_start(transport, frame):
        """Log when TTS starts"""
        logger.info("TTS audio generation started")

    @transport.event_handler("on_tts_end")
    async def _on_tts_end(transport, frame):
        """Log when TTS ends"""
        logger.info("TTS audio generation completed")

    # Enhanced pipeline runner with better audio processing
    runner = PipelineRunner(
        handle_sigint=False, 
        force_gc=True,
        # Audio processing optimizations
        audio_buffer_size=4096,  # Larger buffer for smoother audio
        audio_chunk_size=1024,   # Optimal chunk size for real-time processing
        max_audio_latency_ms=100,  # Maximum acceptable latency
    )
    await runner.run(task)

# ---- WebSocket endpoint for Media Streams ----
@app.websocket("/ws")
async def ws_endpoint(websocket: WebSocket):
    # Twilio uses subprotocol "audio.twilio.com"
    await websocket.accept(subprotocol="audio.twilio.com")
    try:
        # Per Twilio: first messages are text JSON: "connected" then "start"
        connected = await websocket.receive_text()
        start_msg = await websocket.receive_text()
        data = json.loads(start_msg)
        stream_sid = data.get("start", {}).get("streamSid")
        call_sid   = data.get("start", {}).get("callSid")
        
        # Try to get caller info from WebSocket start message first
        caller_number = data.get("start", {}).get("from")
        caller_name = data.get("start", {}).get("callerName")
        
        # If not available in WebSocket message, retrieve from stored info
        if call_sid and call_sid in caller_info_storage:
            stored_info = caller_info_storage[call_sid]
            caller_name = caller_name or stored_info.get("caller_name", "Unknown")
            caller_number = caller_number or stored_info.get("caller_number", "Unknown")
            logger.info(f"Retrieved caller info from storage - Name: {caller_name}, Number: {caller_number}")
        else:
            logger.warning(f"No stored caller info found for CallSid: {call_sid}")
            # Fallback: try to extract from WebSocket data or use defaults
            if not caller_name:
                caller_name = "Unknown"
            if not caller_number:
                caller_number = "Unknown"
        
        if not stream_sid:
            await websocket.close(code=1011)
            return

        logger.info(f"WebSocket connection - Caller: {caller_name} ({caller_number}), Stream: {stream_sid}, Call: {call_sid}")
        await _run_call(websocket, stream_sid, call_sid, caller_number, caller_name)

    except WebSocketDisconnect:
        logger.info("Twilio WebSocket disconnected")
        # Clean up stored caller info on disconnect
        if 'call_sid' in locals() and call_sid and call_sid in caller_info_storage:
            del caller_info_storage[call_sid]
            logger.info(f"Cleaned up caller info on disconnect for CallSid: {call_sid}")
    except Exception as e:
        logger.exception("WS error: %s", e)
        # Clean up stored caller info on error
        if 'call_sid' in locals() and call_sid and call_sid in caller_info_storage:
            del caller_info_storage[call_sid]
            logger.info(f"Cleaned up caller info on error for CallSid: {call_sid}")
        try:
            await websocket.close(code=1011)
        except Exception:
            pass


