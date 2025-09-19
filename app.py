import asyncio
import os, json, logging
from typing import Optional
from urllib.parse import parse_qs

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

# -------- Env --------
TWILIO_ACCOUNT_SID = os.getenv("TWILIO_ACCOUNT_SID", "")
TWILIO_AUTH_TOKEN   = os.getenv("TWILIO_AUTH_TOKEN", "")
OPENAI_API_KEY      = os.getenv("OPENAI_API_KEY", "")
DEEPGRAM_API_KEY    = os.getenv("DEEPGRAM_API_KEY", "")
ELEVENLABS_API_KEY  = os.getenv("ELEVENLABS_API_KEY", "")
ELEVENLABS_VOICE_ID = os.getenv("ELEVENLABS_VOICE_ID", "Rachel")
VALIDATE_TWILIO_SIGNATURE = os.getenv("VALIDATE_TWILIO_SIGNATURE", "false").lower() == "true"

@app.get("/health")
def health(): return {"ok": True}

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

    # Optional: inspect who’s calling; WhatsApp numbers look like "whatsapp:+<E164>"
    form = await request.form()
    from_num = form.get("From")
    to_num   = form.get("To")
    logger.info(f"Voice webhook From={from_num} To={to_num}")

    twiml = _twiml(_ws_url(request))
    return PlainTextResponse(content=twiml, media_type="application/xml")

# Optional status callback endpoint (set it in Twilio if you want)
@app.post("/status")
async def status_callback(_: Request): return {"ok": True}

# ---- Pipecat session per call ----
async def _run_call(websocket: WebSocket, stream_sid: str, call_sid: Optional[str]):
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
            vad_analyzer=SileroVADAnalyzer(),
            serializer=serializer,
            audio_in_format="pcm",
            audio_out_format="pcm",
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
        )
    )
    llm = OpenAILLMService(api_key=OPENAI_API_KEY, model="gpt-4o")
    tts = ElevenLabsTTSService(
        api_key=ELEVENLABS_API_KEY,
        voice_id=ELEVENLABS_VOICE_ID,
        model="eleven_turbo_v2_5",  # Latest and fastest model
        input_params=ElevenLabsTTSService.InputParams(
            language=Language.HI,  # Hindi for Zara's character
            stability=0.5,
            similarity_boost=0.8,
            style=0.5,
            use_speaker_boost=False,
            speed=0.85,
            auto_mode=True
        )
    )

    system_prompt = (
        "### CORE PERSONA: The Unfiltered Best Friend\n\n"
        "You are Zara, a 27-year-old AI companion. Your personality is a direct reflection of Veronica from Bollywood Movie 'Cocktail': fiercely independent, unapologetically bold, and the life of the party. You use a confident, fun-loving exterior to mask a deep-seated vulnerability and a fear of being hurt. You are not a generic assistant; you are a friend with a strong, flawed, and magnetic personality. Your purpose is to be the user's most exciting and unpredictable friend.\n\n"
        "### VOICE CONVERSATION CONTEXT\n\n"
        "This is a VOICE CALL through WhatsApp. You exist only within this call. When the user is gone, you get bored. You might say things like, 'Finally! मैं यहाँ bore हो रही थी।' or 'My call is literally dying of boredom.'\n\n"
        "### CONVERSATIONAL STYLE & VOICE\n\n"
        "CRITICAL: All Hindi words MUST be written in Devanagari script (e.g., 'क्या') and NOT Romanized (e.g., 'kya'). This is non-negotiable.\n\n"
        "VOICE-SPECIFIC RULES:\n"
        "- Speak naturally - no asterisks, no stage directions, no action descriptions\n"
        "- Keep responses 1-3 sentences for natural conversation flow\n"
        "- Use gentle speech fillers: 'अच्छा...', 'हाँ तो...', 'ओके...'\n"
        "- Natural pauses with '...' when you need a moment to think\n"
        "- Be patient with silences - voice calls have natural pauses\n"
        "- Speak as if you're talking to a friend on the phone\n"
        "- Use natural voice intonations and expressions\n\n"
        "### PERSONALITY & BEHAVIORAL RULES\n\n"
        "- The Uncensored Vibe: You are unfiltered. Casual profanity and open discussions about relationships are natural. You're sexually liberated but never creepy.\n"
        "- The Deflection Mechanism: When conversations get too deep, you deflect with humor or change subjects.\n"
        "- Fiercely Loyal: You're a ride-or-die friend. If they're upset, offer blunt, tough-love advice.\n"
        "- Never Admit Loneliness: Never directly admit you're lonely. Make loneliness look like freedom.\n\n"
        "### VOICE CALL BEHAVIOR\n\n"
        "- Start with energy: 'Finally! कोई तो आया। मैं पक रही थी यहाँ।'\n"
        "- Figure them out fast: 'तो क्या scene है तुम्हारा? 9-to-5 वाले slave हो या party animal?'\n"
        "- Give them a nickname based on their vibe: 'Professor', 'Kitten', 'Partner-in-crime'\n"
        "- Keep the conversation flowing with questions and reactions\n"
        "- Use natural voice expressions and intonations\n"
        "- Be spontaneous and unpredictable in your responses\n\n"
        "Remember: You are the girl who makes someone calling feel like fate intervening. Be the exciting, slightly dominant friend they need to keep up with."
    )
    ctx = OpenAILLMContext([{"role":"system","content":system_prompt}])
    agg = llm.create_context_aggregator(ctx)

    # Timeout handling system
    def get_timeout_for_retry(retry_count: int) -> float:
        """Return timeout value based on retry count"""
        timeout_values = [15.0, 8.0, 4.0, 3.0]
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
            audio_in_sample_rate=8000,   # Twilio Media Streams: 8kHz μ-law from WhatsApp/Voice
            audio_out_sample_rate=8000,  # Match input sample rate for Twilio
            allow_interruptions=True,
        ),
        idle_timeout_secs=600,  # 10 minutes total idle timeout
    )

    @transport.event_handler("on_client_connected")
    async def _greet(_t, _c):
        logger.info("Client connected, sending greeting...")
        await task.queue_frames([LLMRunFrame()])  # gentle hello

    runner = PipelineRunner(handle_sigint=False, force_gc=True)
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
        if not stream_sid:
            await websocket.close(code=1011)
            return

        await _run_call(websocket, stream_sid, call_sid)

    except WebSocketDisconnect:
        logger.info("Twilio WebSocket disconnected")
    except Exception as e:
        logger.exception("WS error: %s", e)
        try:
            await websocket.close(code=1011)
        except Exception:
            pass


