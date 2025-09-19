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
from pipecat.pipeline.task import PipelineTask
from pipecat.pipeline.params import PipelineParams
from pipecat.pipeline.runner import PipelineRunner
from pipecat.frames.frames import LLMRunFrame

from pipecat.services.deepgram.stt import DeepgramSTTService
from pipecat.services.openai.llm import OpenAILLMService
from pipecat.services.elevenlabs.tts import ElevenLabsTTSService
from pipecat.processors.aggregators.openai_llm_context import OpenAILLMContext

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
        ),
    )

    # Services
    stt = DeepgramSTTService(api_key=DEEPGRAM_API_KEY, interim_results=True, model="nova-2-general")
    llm = OpenAILLMService(api_key=OPENAI_API_KEY, model="gpt-4o-mini")
    tts = ElevenLabsTTSService(api_key=ELEVENLABS_API_KEY, voice_id=ELEVENLABS_VOICE_ID)

    system_prompt = (
        "You are a friendly, concise WhatsApp call assistant. "
        "Keep answers brief and speak-friendly. Offer help proactively."
    )
    ctx = OpenAILLMContext([{"role":"system","content":system_prompt}])
    agg = llm.create_context_aggregator(ctx)

    pipeline = Pipeline([
        transport.input(),     # caller audio -> PCM via serializer
        stt,                   # speech -> text
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
            audio_out_sample_rate=8000,
            allow_interruptions=True,
        ),
    )

    @transport.event_handler("on_client_connected")
    async def _greet(_t, _c):
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

