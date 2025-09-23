import asyncio
import os, json, logging
from typing import Optional, Dict, Any
from urllib.parse import parse_qs
from datetime import datetime
import pytz
import httpx

from fastapi import FastAPI, Request, WebSocket, WebSocketDisconnect
from fastapi.responses import PlainTextResponse, Response, HTMLResponse, JSONResponse
from twilio.request_validator import RequestValidator
from twilio.rest import Client

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
from pipecat.adapters.schemas.function_schema import FunctionSchema
from pipecat.adapters.schemas.tools_schema import ToolsSchema
from pipecat.services.llm_service import FunctionCallParams
from openai import OpenAI
from pipecat.services.mem0.memory import Mem0MemoryService
from prompts.meher_voice_prompt import get_voice_system_prompt
from prompts.meher_text_prompt import get_text_system_prompt
try:
    # Mem0 platform SDK
    from mem0 importMemoryClient
    _MEM0_AVAILABLE = True
except Exception:
    _MEM0_AVAILABLE = False

logger = logging.getLogger("uvicorn.error")
app = FastAPI(title="Pipecat x Twilio WhatsApp Calling")

# Global storage for caller information (in production, use Redis or database)
caller_info_storage = {}
active_sessions = {}
offcall_context: Dict[str, list] = {}
mem0_client = None

# ---- Mem0 Platform compatibility helpers (handle API name differences) ----
def _mem_platform_search(user_id: str, query: str, top_k: int = 5, score_threshold: float = 0.35) -> list:
    if not _MEM0_AVAILABLE or mem0_client is None:
        return []
    try:
        if hasattr(mem0_client, "search_memories"):
            return mem0_client.search_memories(user_id=user_id, query=query, top_k=top_k, score_threshold=score_threshold) or []
        if hasattr(mem0_client, "get_memories"):
            return mem0_client.get_memories(user_id=user_id, query=query, top_k=top_k, score_threshold=score_threshold) or []
    except Exception as e:
        logger.error(f"Mem0 search error: {e}")
    return []

def _mem_platform_add(user_id: str, text: str, metadata: dict | None = None) -> None:
    if not _MEM0_AVAILABLE or mem0_client is None:
        return
    try:
        if hasattr(mem0_client, "add_memory"):
            mem0_client.add_memory(user_id=user_id, memory=text, metadata=metadata or {})
            return
        if hasattr(mem0_client, "add"):
            mem0_client.add(user_id=user_id, memory=text, metadata=metadata or {})
            return
    except Exception as e:
        logger.error(f"Mem0 add error: {e}")

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
TWILIO_WHATSAPP_FROM = os.getenv("TWILIO_WHATSAPP_FROM", "whatsapp:+15557344000")  # Your Twilio WhatsApp number

class WhatsAppMessagingService:
    """Service for sending WhatsApp messages via Twilio"""
    
    def __init__(self, account_sid: str, auth_token: str, from_number: str):
        self.client = Client(account_sid, auth_token)
        self.from_number = from_number
        self.logger = logging.getLogger(__name__)
    
    async def send_message(self, to_number: str, message: str, recipient_name: str = "User") -> Dict[str, Any]:
        """
        Send a WhatsApp message using the approved koyo_simple template
        
        Args:
            to_number: The recipient's phone number (with country code, e.g., +1234567890)
            message: The message content to send
            recipient_name: The recipient's name for personalization
            
        Returns:
            Dict containing the result of the message sending operation
        """
        try:
            # Ensure the to_number is in WhatsApp format
            if not to_number.startswith("whatsapp:"):
                to_number = f"whatsapp:{to_number}"
            
            # Ensure the from_number is in WhatsApp format
            from_number = self.from_number
            if not from_number.startswith("whatsapp:"):
                from_number = f"whatsapp:{from_number}"
            
            # Use the approved koyo_simple template with content_sid
            message_obj = self.client.messages.create(
                from_=from_number,
                to=to_number,
                content_sid="HXe35b0e8a3ebf215e7407f5131ea03510",  # koyo_simple template SID
                content_variables=json.dumps({
                    "1": recipient_name,
                    "2": message
                })
            )
            
            self.logger.info(f"WhatsApp template message sent - SID: {message_obj.sid}, To: {to_number}")
            
            return {
                "success": True,
                "message_sid": message_obj.sid,
                "status": message_obj.status,
                "to": to_number,
                "message": message,
                "template_used": True,
                "template_name": "koyo_simple"
            }
            
        except Exception as e:
            self.logger.error(f"Failed to send WhatsApp message to {to_number}: {str(e)}")
            return {
                "success": False,
                "error": str(e),
                "to": to_number,
                "message": message,
                "template_used": False
            }

    async def send_freeform_message(self, to_number: str, message: str) -> Dict[str, Any]:
        """Send a freeform WhatsApp message (no template/content sid).

        Note: For inbound user-initiated sessions within the 24-hour window, freeform messages are allowed.
        """
        try:
            if not to_number.startswith("whatsapp:"):
                to_number = f"whatsapp:{to_number}"
            from_number = self.from_number
            if not from_number.startswith("whatsapp:"):
                from_number = f"whatsapp:{from_number}"

            message_obj = self.client.messages.create(
                from_=from_number,
                to=to_number,
                body=message,
            )
            self.logger.info(f"WhatsApp freeform message sent - SID: {message_obj.sid}, To: {to_number}")
            return {"success": True, "message_sid": message_obj.sid, "status": message_obj.status}
        except Exception as e:
            self.logger.error(f"Failed to send freeform WhatsApp message to {to_number}: {str(e)}")
            return {"success": False, "error": str(e)}

async def _fetch_with_twilio_auth(url: str) -> bytes:
    async with httpx.AsyncClient(timeout=20) as client:
        r = await client.get(url, auth=(TWILIO_ACCOUNT_SID, TWILIO_AUTH_TOKEN))
        r.raise_for_status()
        return r.content

async def _transcribe_twilio_audio(url: str) -> str:
    try:
        audio_bytes = await _fetch_with_twilio_auth(url)
        client = OpenAI(api_key=OPENAI_API_KEY)
        # OpenAI whisper transcription
        from io import BytesIO
        file_like = BytesIO(audio_bytes)
        file_like.name = "audio.ogg"
        tr = client.audio.transcriptions.create(model="whisper-1", file=file_like)
        return (tr.text or "").strip()
    except Exception as e:
        logger.error(f"Audio transcription failed: {e}")
        return ""

async def _caption_image_url(image_url: str) -> str:
    try:
        client = OpenAI(api_key=OPENAI_API_KEY)
        prompt = "Caption the image in one short casual line and infer the likely intent if obvious."
        resp = client.chat.completions.create(
            model="gpt-4o",
            messages=[
                {"role": "user", "content": [
                    {"type": "text", "text": prompt},
                    {"type": "image_url", "image_url": {"url": image_url}},
                ]}
            ],
            max_tokens=120,
            temperature=0.7,
        )
        return (resp.choices[0].message.content or "").strip()
    except Exception as e:
        logger.error(f"Image captioning failed: {e}")
        return ""

async def _fetch_text_excerpt(url: str, max_chars: int = 800) -> str:
    try:
        data = await _fetch_with_twilio_auth(url)
        text = data.decode("utf-8", errors="ignore")
        text = " ".join(text.split())
        return text[:max_chars]
    except Exception as e:
        logger.error(f"Text fetch failed: {e}")
        return ""

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

# ---- Admin/Dashboard: Inspect memories and conversation snippets ----
@app.get("/memories/{user_id}")
async def get_user_memories(user_id: str, limit: int = 20, threshold: float = 0.0):
    if not _MEM0_AVAILABLE or mem0_client is None:
        return JSONResponse({"success": False, "error": "Mem0 not initialized"}, status_code=500)
    try:
        # broad search to list latest/top vectors – if OSS returns empty on empty query, use a wildcard phrase
        query = "recent user memories"
        res = _mem_platform_search(user_id=user_id, query=query, top_k=limit, score_threshold=threshold)
        items = []
        for r in res:
            text = r.get("text") or r.get("memory") or str(r)
            score = r.get("score")
            meta = r.get("metadata") or {}
            items.append({"text": text, "score": score, "metadata": meta})
        return {"success": True, "user_id": user_id, "count": len(items), "items": items}
    except Exception as e:
        logger.error(f"Mem0 list error: {e}")
        return JSONResponse({"success": False, "error": str(e)}, status_code=500)

@app.get("/memories/search")
async def search_user_memories(user_id: str, query: str, limit: int = 10, threshold: float = 0.35):
    if not _MEM0_AVAILABLE or mem0_client is None:
        return JSONResponse({"success": False, "error": "Mem0 not initialized"}, status_code=500)
    try:
        res = _mem_platform_search(user_id=user_id, query=query, top_k=limit, score_threshold=threshold)
        items = []
        for r in res:
            text = r.get("text") or r.get("memory") or str(r)
            score = r.get("score")
            meta = r.get("metadata") or {}
            items.append({"text": text, "score": score, "metadata": meta})
        return {"success": True, "user_id": user_id, "query": query, "count": len(items), "items": items}
    except Exception as e:
        logger.error(f"Mem0 search error: {e}")
        return JSONResponse({"success": False, "error": str(e)}, status_code=500)

@app.get("/memories/dashboard")
async def memories_dashboard(user_id: str = "", query: str = "", limit: int = 10, threshold: float = 0.35):
    if not _MEM0_AVAILABLE or mem0_client is None:
        return HTMLResponse("<h3>Mem0 not initialized</h3>")
    html_head = """
    <style>
      body{font-family:Inter,system-ui,Arial;padding:24px;max-width:900px;margin:0 auto}
      h1{font-size:22px;margin:0 0 12px}
      form{margin:16px 0;padding:12px;border:1px solid #eee;border-radius:8px}
      input,button{padding:8px 10px;margin:4px}
      .item{padding:10px;border-bottom:1px solid #eee}
      .meta{color:#666;font-size:12px}
      code{background:#f6f8fa;padding:2px 4px;border-radius:4px}
    </style>
    """
    items_html = ""
    count = 0
    if user_id:
        try:
            q = query or "recent user memories"
            res = _mem_platform_search(user_id=user_id, query=q, top_k=limit, score_threshold=threshold)
            for r in res:
                text = (r.get("text") or r.get("memory") or str(r)).replace("<", "&lt;")
                score = r.get("score")
                meta = r.get("metadata") or {}
                items_html += f"<div class='item'><div>{text}</div><div class='meta'>score={score} | metadata={meta}</div></div>"
            count = len(res)
        except Exception as e:
            items_html = f"<div class='item'>Error: {str(e)}</div>"
    form_html = f"""
      <h1>Memories Dashboard</h1>
      <form method='get'>
        <label>User ID (phone):</label>
        <input name='user_id' value='{user_id}' placeholder='e.g. +15551234567' />
        <label>Query:</label>
        <input name='query' value='{query}' placeholder='topic or keyword' />
        <label>Limit:</label>
        <input name='limit' type='number' min='1' max='100' value='{limit}' />
        <label>Threshold:</label>
        <input name='threshold' type='number' step='0.01' value='{threshold}' />
        <button type='submit'>Search</button>
      </form>
      <div><b>Results:</b> {count}</div>
      <div>{items_html}</div>
    """
    return HTMLResponse(html_head + form_html)

@app.on_event("startup")
async def startup_event():
    """Start background tasks when the app starts"""
    asyncio.create_task(cleanup_old_caller_info())
    logger.info("Started caller info cleanup task")
    # Initialize Mem0 Platform if available
    global mem0_client
    if _MEM0_AVAILABLE and mem0_client is None:
        try:
            mem0_client =MemoryClient(api_key=os.getenv("MEM0_API_KEY", ""))
            logger.info("Mem0 initialized (Platform)")
        except Exception as e:
            logger.error(f"Mem0 init failed: {e}")

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

# ---- WhatsApp webhook (Messaging Inbound) ----
@app.post("/whatsapp-webhook")
async def whatsapp_webhook(request: Request):
    """Handle inbound WhatsApp messages.

    Behavior:
    - If there's an active voice session for the sender, inject the text/media summary into the live LLM context
    - Otherwise, run a lightweight off-call multimodal handler and reply via WhatsApp
    """
    body_bytes = await request.body()
    if not _validate_twilio_http(request, body_bytes):
        logger.warning("Twilio signature validation failed for WhatsApp webhook")
        return Response(status_code=403)

    try:
        form = await request.form()
        from_num = (form.get("From") or "").replace("whatsapp:", "")
        text_body = form.get("Body", "") or ""
        num_media = int(form.get("NumMedia", "0") or "0")

        media_items = []
        for i in range(num_media):
            url = form.get(f"MediaUrl{i}")
            ctype = form.get(f"MediaContentType{i}")
            if url:
                media_items.append({"url": url, "content_type": ctype or ""})

        logger.info(f"Incoming WhatsApp from {from_num} | text='{text_body}' | media_count={num_media}")
        session = active_sessions.get(from_num)

        if session:
            # Inject into ongoing call context with media enrichment
            parts = []
            if text_body.strip():
                parts.append(text_body.strip())
            # Enrich media: transcribe audio, caption images, mark documents
            for m in media_items or []:
                ctype = (m.get("content_type") or "").lower()
                url = m.get("url")
                if not url:
                    continue
                try:
                    if "audio/" in ctype:
                        transcript = await _transcribe_twilio_audio(url)
                        parts.append(f"[audio transcript] {transcript}" if transcript else "[audio note received]")
                    elif "image/" in ctype:
                        caption = await _caption_image_url(url)
                        parts.append(f"[image] {caption}" if caption else "[image received]")
                    elif "text/" in ctype:
                        excerpt = await _fetch_text_excerpt(url, max_chars=500)
                        parts.append(f"[text doc excerpt] {excerpt}" if excerpt else "[text document received]")
                    elif "application/pdf" in ctype or ctype.startswith("application/"):
                        parts.append("[document received]")
                    else:
                        parts.append("[media received]")
                except Exception as me:
                    logger.error(f"Media handling error (in-call) for {url}: {me}")
                    parts.append("[media received]")

            content = (" ".join(parts)).strip() or "[empty message]"
            frames = [
                LLMMessagesAppendFrame([{ "role": "user", "content": f"(WhatsApp) {content}" }]),
                LLMRunFrame(),
            ]
            await session["task"].queue_frames(frames)
            # Store inbound message as memory
            try:
                if _MEM0_AVAILABLE and mem0_client is not None and from_num:
                    _mem_platform_add(user_id=from_num, text=f"user_said: {content[:200]}", metadata={"tags":["whatsapp","voice","user"]})
            except Exception as me:
                logger.error(f"Mem0 add (in-call WA) failed: {me}")
            return Response(status_code=204)

        # No live session -> off-call LLM chat with context and reply over WhatsApp (freeform)
        reply_text = await handle_multimodal_offcall(text_body, media_items, from_num)

        whatsapp_service = WhatsAppMessagingService(
            account_sid=TWILIO_ACCOUNT_SID,
            auth_token=TWILIO_AUTH_TOKEN,
            from_number=TWILIO_WHATSAPP_FROM,
        )
        await whatsapp_service.send_freeform_message(to_number=from_num, message=reply_text)
        # Return 204 to avoid Twilio echoing body as a user-visible message
        return Response(status_code=204)

    except Exception as e:
        logger.exception("WhatsApp webhook error: %s", e)
        return Response(status_code=500)

async def handle_multimodal_offcall(text: str, media: list, user_phone: str) -> str:
    """Off-call handler: build/maintain chat context and generate a Meher-style reply via LLM.

    - Maintains per-user message history in offcall_context[user_phone]
    - Supports text + media summaries
    - Generates a concise Hinglish reply (≤ ~75 tokens)
    """
    # 1) Build user turn with media summary
    parts = []
    if text and text.strip():
        parts.append(text.strip())
    # Enrich media for off-call: transcribe/caption/excerpt
    for m in media or []:
        ctype = (m.get("content_type") or "").lower()
        url = m.get("url")
        if not url:
            continue
        try:
            if "audio/" in ctype:
                transcript = await _transcribe_twilio_audio(url)
                parts.append(f"[audio transcript] {transcript}" if transcript else "[audio note received]")
            elif "image/" in ctype:
                caption = await _caption_image_url(url)
                parts.append(f"[image] {caption}" if caption else "[image received]")
            elif "text/" in ctype:
                excerpt = await _fetch_text_excerpt(url, max_chars=800)
                parts.append(f"[text doc excerpt] {excerpt}" if excerpt else "[text document received]")
            elif "application/pdf" in ctype or ctype.startswith("application/"):
                parts.append("[document received]")
            else:
                parts.append("[media received]")
        except Exception as me:
            logger.error(f"Media handling error (off-call) for {url}: {me}")
            parts.append("[media received]")
    user_text = (" ".join(parts)).strip() or "[empty message]"

    # 2) Prepare context store for user
    history = offcall_context.get(user_phone) or []

    # 3) System prompt for off-call chat: use dedicated texting prompt module
    time_context = get_current_time_context()
    system_text = get_text_system_prompt(
        caller_display_name=user_phone,
        caller_phone=user_phone,
        time_context=time_context,
    )

    # 4) Build messages array (bounded history). Allow tool call for memory fetch.
    messages = [{"role": "system", "content": system_text}]
    # include last ~8 messages from history for context
    for m in history[-8:]:
        messages.append(m)
    messages.append({"role": "user", "content": user_text})

    # 5) Call OpenAI for a short response
    try:
        client = OpenAI(api_key=OPENAI_API_KEY)
        tools = []
        if _MEM0_AVAILABLE and mem0_client is not None:
            tools = [{
                "type": "function",
                "function": {
                    "name": "fetch_memories",
                    "description": "Retrieve relevant memories for the current topic. Call only if needed.",
                    "parameters": {
                        "type": "object",
                        "properties": {"query": {"type": "string"}},
                        "required": ["query"]
                    }
                }
            }]

        resp = client.chat.completions.create(
            model="gpt-4o",
            messages=messages,
            tools=tools if tools else None,
            tool_choice="auto" if tools else None,
            max_tokens=120,
            temperature=0.7,
        )
        msg = resp.choices[0].message
        if tools and getattr(msg, "tool_calls", None):
            import json
            for call in msg.tool_calls:
                if call.function.name == "fetch_memories":
                    args = json.loads(call.function.arguments or "{}")
                    query = args.get("query", user_text)
                    try:
                        res = _mem_platform_search(user_id=user_phone, query=query, top_k=5, score_threshold=0.35)
                        hits = [r.get("text") or r.get("memory") or str(r) for r in res]
                        messages.append({
                            "role": "tool",
                            "tool_call_id": call.id,
                            "name": "fetch_memories",
                            "content": "\n".join(hits) if hits else "NO_MEMORIES",
                        })
                    except Exception as me:
                        logger.error(f"Mem0 search failed: {me}")
                        messages.append({
                            "role": "tool",
                            "tool_call_id": call.id,
                            "name": "fetch_memories",
                            "content": "NO_MEMORIES",
                        })
            resp = client.chat.completions.create(
                model="gpt-4o",
                messages=messages,
                max_tokens=120,
                temperature=0.7,
            )
        reply = (resp.choices[0].message.content or "")[:800]
    except Exception as e:
        logger.error(f"Off-call LLM error: {e}")
        reply = "थोड़ी technical दिक्कत हो गयी मेरी तरफ—एक छोटा सा message फिर भेजो, मैं तुरंत जवाब दूँगी।"

    # 6) Update history
    history.append({"role": "user", "content": user_text})
    history.append({"role": "assistant", "content": reply})
    offcall_context[user_phone] = history[-20:]  # cap to last 20 turns

    # 7) Store salient memory (non-blocking best-effort)
    try:
        if _MEM0_AVAILABLE and mem0_client is not None:
            _mem_platform_add(user_id=user_phone, text=f"user_said: {user_text[:200]}", metadata={"tags":["whatsapp","user"]})
            _mem_platform_add(user_id=user_phone, text=f"assistant_hint: {reply[:200]}", metadata={"tags":["whatsapp","assistant"]})
    except Exception as me:
        logger.error(f"Mem0 add failed: {me}")

    return reply

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
            # Enhanced audio processing for better quality
            encoding="linear16",  # 16-bit linear PCM for better quality
            channels=1,  # Mono audio
        )
    )
    # Initialize WhatsApp messaging service
    whatsapp_service = WhatsAppMessagingService(
        account_sid=TWILIO_ACCOUNT_SID,
        auth_token=TWILIO_AUTH_TOKEN,
        from_number=TWILIO_WHATSAPP_FROM
    )
    
    # Define WhatsApp function schema using Pipecat's FunctionSchema
    whatsapp_function = FunctionSchema(
        name="send_whatsapp_message",
        description="Send a WhatsApp text message to a user during the conversation. Use this when the user asks you to send them a message, reminder, or any text via WhatsApp.",
        properties={
            "to_number": {
                "type": "string",
                "description": "The recipient's phone number with country code (e.g., +1234567890). If not provided, use the caller's number."
            },
            "message": {
                "type": "string",
                "description": "The message content to send via WhatsApp"
            }
        },
        required=["message"]
    )
    
    # Optional: memory fetch function (LLM calls only when needed)
    if _MEM0_AVAILABLE and mem0_client is not None:
        fetch_memories_function = FunctionSchema(
            name="fetch_memories",
            description="Retrieve relevant user memories for current topic. Call only if helpful.",
            properties={
                "query": {"type":"string","description":"What to search in the user's memories."}
            },
            required=["query"]
        )
        tools = ToolsSchema(standard_tools=[whatsapp_function, fetch_memories_function])
    else:
        tools = ToolsSchema(standard_tools=[whatsapp_function])
    
    # Initialize LLM service
    llm = OpenAILLMService(
        api_key=OPENAI_API_KEY, 
        model="gpt-4o",
        params=OpenAILLMService.InputParams(
            temperature=0.7,
        )
    )
    
    # Register WhatsApp function handler
    async def send_whatsapp_message_handler(params: FunctionCallParams):
        """Handle WhatsApp message sending function calls"""
        try:
            # Check if connection is still active
            if hasattr(transport, '_websocket') and transport._websocket.client_state.name != 'CONNECTED':
                logger.info("Connection closed, skipping WhatsApp message")
                await params.result_callback("❌ Connection closed, cannot send message")
                return
            
            # Get the phone number, use caller's number if not provided
            to_number = params.arguments.get("to_number", caller_phone)
            message = params.arguments.get("message", "")
            
            if not message:
                result = "❌ No message content provided"
                await params.result_callback(result)
                return
            
            # Send the WhatsApp message using the approved template
            result = await whatsapp_service.send_message(
                to_number=to_number, 
                message=message, 
                recipient_name=caller_display_name
            )
            
            if result["success"]:
                logger.info(f"WhatsApp template message sent successfully: {result}")
                response = f"✅ WhatsApp message sent successfully to {to_number} using template '{result.get('template_name', 'koyo_simple')}'"
            else:
                logger.error(f"Failed to send WhatsApp message: {result}")
                response = f"❌ Failed to send WhatsApp message: {result.get('error', 'Unknown error')}"
            
            await params.result_callback(response)
            
        except Exception as e:
            logger.error(f"Error in WhatsApp function handler: {str(e)}")
            await params.result_callback(f"❌ Error sending WhatsApp message: {str(e)}")
    
    # Register the function handler with the LLM service
    llm.register_function("send_whatsapp_message", send_whatsapp_message_handler)

    # Register memory fetch handler if available
    if _MEM0_AVAILABLE and mem0_client is not None:
        async def fetch_memories_handler(params: FunctionCallParams):
            try:
                query = params.arguments.get("query", "") or ""
                res = mem0_client.search(query=query, user_id=caller_phone, limit=5, threshold=0.35) or []
                hits = [r.get("memory") or r.get("text") or str(r) for r in res]
                await params.result_callback("\n".join(hits) if hits else "NO_MEMORIES")
            except Exception as me:
                logger.error(f"fetch_memories error: {me}")
                await params.result_callback("NO_MEMORIES")

        llm.register_function("fetch_memories", fetch_memories_handler)
    tts = ElevenLabsTTSService(
        api_key=ELEVENLABS_API_KEY,
        voice_id=ELEVENLABS_VOICE_ID,
        model="eleven_flash_v2_5",  # Better quality model for multilingual support
        input_params=ElevenLabsTTSService.InputParams(
            language=Language.HI,  # Hindi for Meher's character
            stability=0.5,  # Increased for more consistent quality
            similarity_boost=0.9,  # Higher for better voice matching
            style=0.5,  # Reduced for more natural speech
            use_speaker_boost=False,  # Enable for better voice clarity
            speed=0.9,  # Slightly faster for more natural flow
            auto_mode=True,
        )
    )

    # Get current time and date context
    time_context = get_current_time_context()
    caller_phone = extract_phone_number(caller_number) if caller_number else "Unknown"
    caller_display_name = caller_name or "Unknown"
    
    system_prompt = get_voice_system_prompt(caller_display_name, caller_phone, time_context)

    ctx = OpenAILLMContext(
        messages=[{"role":"system","content":system_prompt}],
        tools=tools
    )
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
        
        # Check if the connection is still active
        if hasattr(transport, '_websocket') and transport._websocket.client_state.name != 'CONNECTED':
            logger.info("Connection closed, stopping idle monitoring")
            return False
        
        # Update timeout for next retry
        next_timeout = get_timeout_for_retry(retry_count + 1)
        user_idle._timeout = next_timeout
        
        if retry_count == 1:
            # First time: Continue conversation naturally
            idle_message = {
                "role": "user",
                "content": "The user has gone silent for the first time. As Meher, continue the conversation naturally with a gentle follow-up related to what you were discussing. Keep it engaging, brief, and in character. Use Devanagari script for Hindi."
            }
        elif retry_count == 2:
            # Second time: Change topic
            idle_message = {
                "role": "user", 
                "content": "The user has gone silent again. As Meher, try changing the topic to something new and interesting. Maybe ask about their day, tease them playfully, or share something exciting. Keep it warm, engaging, and in character. Use Devanagari script for Hindi."
            }
        elif retry_count == 3:
            # Third time: Check if they're still there
            idle_message = {
                "role": "user",
                "content": "The user has been silent multiple times. As Meher, gently check if they are still there and if everything is okay. Be caring, understanding, but maintain your bold personality. Use Devanagari script for Hindi."
            }
        else:  # retry_count >= 4
            # Fourth time: Say goodbye and end call
            idle_message = {
                "role": "user",
                "content": "The user has been unresponsive for too long. As Meher, say a gentle goodbye message with your characteristic boldness and warmth, expressing that you hope to talk again soon. After this message, the call will end. Use Devanagari script for Hindi."
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
        
        # Check if connection is still active before proceeding
        if hasattr(transport, '_websocket') and transport._websocket.client_state.name != 'CONNECTED':
            logger.info("Connection already closed, skipping termination")
            return
        
        await task.queue_frames([EndFrame()])
        
        # Wait for the bot to finish speaking before ending
        max_wait_time = 20  # Maximum 20 seconds
        wait_time = 0
        while wait_time < max_wait_time:
            # Check if connection is still active
            if hasattr(transport, '_websocket') and transport._websocket.client_state.name != 'CONNECTED':
                logger.info("Connection closed during termination, stopping")
                break
                
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

    # Memory service (per-turn storage + retrieval) using Mem0 Platform
    memory_service = None
    if _MEM0_AVAILABLE and mem0_client is not None:
        try:
            memory_service = Mem0MemoryService(
                api_key=os.getenv("MEM0_API_KEY"),
                user_id=caller_phone or "unknown",
                agent_id="meher",
                run_id=call_sid or "voice",
                params={
                    "search_limit": 5,
                    "search_threshold": 0.35,
                    "system_prompt": "[MEMORY CONTEXT] Use only if relevant:\n",
                    "add_as_system_message": True,
                    "position": 0,
                },
            )
        except Exception as e:
            logger.error(f"Mem0MemoryService init failed: {e}")
            memory_service = None

    pipeline_steps = [
        transport.input(),     # caller audio -> PCM via serializer
        stt,                   # speech -> text
        user_idle,             # UserIdleProcessor automatically resets on UserStartedSpeakingFrame
        agg.user(),            # add user text to context
    ]
    if memory_service is not None:
        pipeline_steps.append(memory_service)
    pipeline_steps += [
        llm,                   # text -> text
        tts,                   # text -> speech
        transport.output(),    # PCM -> back to Twilio via serializer
        agg.assistant(),       # keep assistant turns
    ]

    pipeline = Pipeline(pipeline_steps)

    task = PipelineTask(
        pipeline,
        params=PipelineParams(
            audio_in_sample_rate=16000,   # Silero VAD compatible sample rate
            audio_out_sample_rate=16000,  # Match input sample rate for consistency
            allow_interruptions=True,
        ),
        idle_timeout_secs=600,  # 10 minutes total idle timeout
    )

    # Register active session for this caller (enables WA->voice bridging)
    try:
        if caller_phone and caller_phone != "Unknown":
            active_sessions[caller_phone] = {
                "call_sid": call_sid,
                "task": task,
                "transport": transport,
                "display_name": caller_display_name,
            }
            logger.info(f"Registered active session for {caller_phone}")
    except Exception:
        pass

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
        # Clean up active session
        try:
            if 'caller_number' in locals() and caller_number:
                phone = extract_phone_number(caller_number)
                if phone in active_sessions:
                    del active_sessions[phone]
                    logger.info(f"Deregistered active session for {phone}")
        except Exception:
            pass
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


