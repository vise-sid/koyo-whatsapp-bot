"""
Main FastAPI application for Koyo voice and WhatsApp integration.

This application provides endpoints for:
- WhatsApp webhook handling
- Twilio voice webhook handling  
- WebSocket connections for voice calls
- Health check endpoint

The application uses Pipecat for voice processing and integrates with
Twilio, OpenAI, Deepgram, ElevenLabs, and Firebase services.
"""

import asyncio
import json
import logging
import os
from datetime import datetime
from typing import Dict, Any

from fastapi import FastAPI, Request, WebSocket, WebSocketDisconnect
from fastapi.responses import PlainTextResponse, Response, JSONResponse
import hmac
import hashlib
import time
import json

# Import our modular services
from services.whatsapp_service import WhatsAppMessagingService
from services.voice_session import VoiceSessionManager
from webhooks.whatsapp_handler import WhatsAppHandler
from webhooks.voice_handler import VoiceHandler
from database.firebase_service import firebase_service
from utils.helpers import extract_phone_number
from services.memory_trigger_service import memory_router
from services.firebase_trigger import firebase_router

# Configure logging
logger = logging.getLogger("uvicorn.error")

# Initialize FastAPI app
app = FastAPI(title="Koyo Voice & WhatsApp Integration")

# Include memory and Firebase trigger routers
app.include_router(memory_router)
app.include_router(firebase_router)

# ElevenLabs SIP webhook: returns XML instructing Twilio to dial ElevenLabs SIP
@app.post("/elevenlabs-voice-webhook")
async def elevenlabs_voice_webhook() -> Response:
    xml_body = (
        "<?xml version=\"1.0\" encoding=\"UTF-8\"?>\n"
        "<Response>\n"
        "<Dial>\n"
        "<Sip>sip:+19346497573@sip.rtc.elevenlabs.io:5060;transport=tcp</Sip>\n"
        "</Dial>\n"
        "</Response>\n"
    )
    return Response(content=xml_body, media_type="application/xml")

# ElevenLabs conversation initiation webhook: provides caller data for personalization
@app.post("/elevenlabs-init-webhook")
async def elevenlabs_init_webhook(request: Request) -> JSONResponse:
    """
    Webhook endpoint for ElevenLabs to fetch conversation initiation data.
    Receives caller information and returns personalized conversation data.
    """
    try:
        # Optional shared-secret validation (recommended)
        expected_secret = os.getenv("ELEVENLABS_INIT_WEBHOOK_SECRET", "")
        if expected_secret:
            provided_secret = (
                request.headers.get("X-EL-Secret")
                or request.headers.get("x-el-secret")
                or request.headers.get("Authorization", "").replace("Bearer ", "")
            )
            if not provided_secret or provided_secret != expected_secret:
                logger.warning("Init webhook secret validation failed")
                return JSONResponse({"error": "Unauthorized"}, status_code=401)
        # Parse request body
        body = await request.json()
        caller_id = body.get("caller_id")
        agent_id = body.get("agent_id")
        called_number = body.get("called_number")
        call_sid = body.get("call_sid")
        
        logger.info(f"ElevenLabs init webhook called with caller_id: {caller_id}, call_sid: {call_sid}")
        
        if not caller_id:
            logger.warning("Missing caller_id in ElevenLabs init webhook")
            return JSONResponse({"error": "Missing caller_id"}, status_code=400)
        
        # Store call information for later use in WebSocket and post-call webhook
        # We'll use call_sid as the key to map back to the caller
        if call_sid:
            # Align keys with voice_handler storage keys for consistent retrieval
            caller_info_storage[call_sid] = {
                "caller_id": caller_id,               # raw field from ElevenLabs
                "agent_id": agent_id,
                "called_number": called_number,
                "caller_number": caller_id,           # normalized key used elsewhere
                "caller_name": body.get("caller_name", "Unknown"),
                "to_number": called_number,
                "timestamp": datetime.utcnow().isoformat()
            }
        
        # Get user's recent conversation context for personalization (messages, counts)
        user_context = await _get_user_context_for_call(caller_id)

        # Prefer caller details captured by Twilio voice webhook (voice_handler)
        stored_info = caller_info_storage.get(call_sid or "", {})
        resolved_user_name = stored_info.get("caller_name") or "User"
        resolved_user_phone = stored_info.get("caller_number") or caller_id

        # Return conversation initiation data
        response_data = {
            "type": "conversation_initiation_client_data",
            "dynamic_variables": {
                "user_phone": resolved_user_phone,
                "user_name": resolved_user_name,
                "last_interaction": user_context.get("last_interaction", ""),
                "conversation_count": user_context.get("conversation_count", 0)
            }
            # "conversation_config_override": {
            #     "agent": {
            #         "first_message": f"Hello! I'm Meher, your AI companion. How can I help you today?",
            #         "language": "en"
            #     }
            # }
        }
        
        logger.info(f"Returning conversation initiation data for caller {caller_id}")
        return JSONResponse(response_data)
        
    except Exception as e:
        logger.error(f"Error processing ElevenLabs init webhook: {e}")
        return JSONResponse({"error": "Internal server error"}, status_code=500)

async def _get_user_context_for_call(caller_id: str) -> dict:
    """Get user context for conversation personalization"""
    try:
        from database.firebase_service import firebase_service
        
        if not firebase_service.db:
            return {}
        
        # Get user's conversation metadata
        user_ref = firebase_service.db.collection("users").document(caller_id)
        user_doc = user_ref.get()
        
        if user_doc.exists:
            user_data = user_doc.to_dict()
            return {
                "name": user_data.get("name", "User"),
                "last_interaction": user_data.get("last_interaction", ""),
                "conversation_count": user_data.get("conversation_count", 0)
            }
        
        return {}
        
    except Exception as e:
        logger.error(f"Error getting user context: {e}")
        return {}

# ElevenLabs post-call webhook: handles transcript data from completed calls
@app.post("/post-call-eleven")
async def post_call_eleven_webhook(request: Request) -> JSONResponse:
    try:
        # Get raw body for HMAC validation
        body = await request.body()
        
        # Get signature header
        signature_header = request.headers.get("elevenlabs-signature")
        if not signature_header:
            logger.warning("Missing ElevenLabs signature header")
            return JSONResponse({"error": "Missing signature header"}, status_code=401)
        
        # Parse signature header
        headers = signature_header.split(",")
        timestamp = None
        hmac_signature = None
        
        for header in headers:
            if header.startswith("t="):
                timestamp = header[2:]
            elif header.startswith("v0="):
                hmac_signature = header
        
        if not timestamp or not hmac_signature:
            logger.warning("Invalid signature format")
            return JSONResponse({"error": "Invalid signature format"}, status_code=401)
        
        # Validate timestamp (within 30 minutes)
        tolerance = int(time.time()) - 30 * 60
        if int(timestamp) < tolerance:
            logger.warning("Request timestamp expired")
            return JSONResponse({"error": "Request expired"}, status_code=403)
        
        # Validate HMAC signature
        webhook_secret = os.getenv("ELEVENLABS_WEBHOOK_SECRET")
        if not webhook_secret:
            logger.error("ELEVENLABS_WEBHOOK_SECRET not configured")
            return JSONResponse({"error": "Webhook secret not configured"}, status_code=500)
        
        full_payload_to_sign = f"{timestamp}.{body.decode('utf-8')}"
        mac = hmac.new(
            key=webhook_secret.encode("utf-8"),
            msg=full_payload_to_sign.encode("utf-8"),
            digestmod=hashlib.sha256,
        )
        digest = 'v0=' + mac.hexdigest()
        
        if hmac_signature != digest:
            logger.warning("Invalid HMAC signature")
            return JSONResponse({"error": "Invalid signature"}, status_code=401)
        
        # Parse webhook payload
        try:
            payload = json.loads(body.decode('utf-8'))
        except json.JSONDecodeError as e:
            logger.error(f"Invalid JSON payload: {e}")
            return JSONResponse({"error": "Invalid JSON"}, status_code=400)
        
        # Process webhook based on type
        webhook_type = payload.get("type")
        if webhook_type == "post_call_transcription":
            await _handle_transcription_webhook(payload)
        elif webhook_type == "post_call_audio":
            await _handle_audio_webhook(payload)
        else:
            logger.warning(f"Unknown webhook type: {webhook_type}")
            return JSONResponse({"error": "Unknown webhook type"}, status_code=400)
        
        logger.info(f"Successfully processed {webhook_type} webhook")
        return JSONResponse({"status": "received"})
        
    except Exception as e:
        logger.error(f"Error processing ElevenLabs webhook: {e}")
        return JSONResponse({"error": "Internal server error"}, status_code=500)

async def _handle_transcription_webhook(payload: dict):
    """Handle post-call transcription webhook from ElevenLabs"""
    try:
        data = payload.get("data", {})
        conversation_id = data.get("conversation_id")
        agent_id = data.get("agent_id")
        user_id = data.get("user_id")
        transcript = data.get("transcript", [])
        metadata = data.get("metadata", {})
        analysis = data.get("analysis", {})
        
        if not conversation_id:
            logger.error("Missing conversation_id in webhook payload")
            return
        
        # Try to get caller information from our stored data
        # Look for call_sid in metadata or try to match by conversation_id
        caller_phone = None
        call_sid = metadata.get("call_sid")
        
        if call_sid and call_sid in caller_info_storage:
            caller_phone = caller_info_storage[call_sid]["caller_id"]
            logger.info(f"Found caller phone {caller_phone} for call_sid {call_sid}")
        else:
            # Fallback: try to extract from user_id or other fields
            caller_phone = user_id
            logger.warning(f"Using fallback caller phone {caller_phone}")
        
        # Store transcript in Firebase
        from database.firebase_service import firebase_service
        
        # Create transcript document structure
        transcript_data = {
            "conversation_id": conversation_id,
            "agent_id": agent_id,
            "user_id": user_id,
            "caller_phone": caller_phone,
            "transcript": transcript,
            "metadata": metadata,
            "analysis": analysis,
            "webhook_timestamp": payload.get("event_timestamp"),
            "created_at": datetime.utcnow().isoformat(),
            "source": "elevenlabs_webhook"
        }
        
        # Store in Firebase
        await firebase_service.save_transcript_to_firebase(transcript_data)
        
        # If we have caller phone, also store individual messages in user's collection
        if caller_phone:
            await _store_user_messages_from_transcript(caller_phone, transcript, conversation_id, metadata)
        
        logger.info(f"Stored transcript for conversation {conversation_id}")
        
    except Exception as e:
        logger.error(f"Error handling transcription webhook: {e}")

async def _store_user_messages_from_transcript(caller_phone: str, transcript: list, conversation_id: str, metadata: dict):
    """Store individual messages from transcript in user's collection"""
    try:
        from database.firebase_service import firebase_service
        
        if not firebase_service.db:
            return
        
        character_name = "meher"
        call_sid = metadata.get("call_sid")
        
        # Create or update conversation metadata
        await firebase_service.create_or_update_conversation_metadata(caller_phone, character_name, call_sid)
        
        # Store each message from transcript
        messages_to_store = []
        for turn in transcript:
            role = turn.get("role")
            message = turn.get("message", "")
            time_in_call = turn.get("time_in_call_secs", 0)
            
            if role in ["user", "agent"] and message.strip():
                sender = "user" if role == "user" else "character"
                messages_to_store.append({
                    "sender": sender,
                    "content": message[:1000],  # Limit content length
                    "timestamp": datetime.utcnow(),
                    "sync": False,
                    "conversation_type": "voice",
                    "call_sid": call_sid,
                    "conversation_id": conversation_id,
                    "time_in_call_secs": time_in_call
                })
        
        if messages_to_store:
            # Use batch save for efficiency
            await firebase_service.save_voice_messages_to_firebase_batch(
                caller_phone, messages_to_store, call_sid
            )
            logger.info(f"Stored {len(messages_to_store)} messages for user {caller_phone}")
        
    except Exception as e:
        logger.error(f"Error storing user messages from transcript: {e}")

async def _handle_audio_webhook(payload: dict):
    """Handle post-call audio webhook from ElevenLabs"""
    try:
        data = payload.get("data", {})
        conversation_id = data.get("conversation_id")
        agent_id = data.get("agent_id")
        full_audio = data.get("full_audio")
        
        if not conversation_id or not full_audio:
            logger.error("Missing required fields in audio webhook")
            return
        
        # Store audio metadata in Firebase (not the actual audio data due to size)
        from database.firebase_service import firebase_service
        
        audio_data = {
            "conversation_id": conversation_id,
            "agent_id": agent_id,
            "has_audio": True,
            "audio_size_bytes": len(full_audio.encode('utf-8')),
            "webhook_timestamp": payload.get("event_timestamp"),
            "created_at": datetime.utcnow().isoformat(),
            "source": "elevenlabs_webhook"
        }
        
        # Store audio metadata in Firebase
        await firebase_service.save_audio_metadata_to_firebase(audio_data)
        
        logger.info(f"Stored audio metadata for conversation {conversation_id}")
        
    except Exception as e:
        logger.error(f"Error handling audio webhook: {e}")

# Global storage for session management (in production, use Redis or database)
caller_info_storage: Dict[str, Any] = {}
active_sessions: Dict[str, Any] = {}
offcall_context: Dict[str, list] = {}

# Environment variables
TWILIO_ACCOUNT_SID = os.getenv("TWILIO_ACCOUNT_SID", "")
TWILIO_AUTH_TOKEN = os.getenv("TWILIO_AUTH_TOKEN", "")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")
DEEPGRAM_API_KEY = os.getenv("DEEPGRAM_API_KEY", "")
ELEVENLABS_API_KEY = os.getenv("ELEVENLABS_API_KEY", "")
ELEVENLABS_VOICE_ID = os.getenv("ELEVENLABS_VOICE_ID", "Rachel")
VALIDATE_TWILIO_SIGNATURE = os.getenv("VALIDATE_TWILIO_SIGNATURE", "false").lower() == "true"
TWILIO_WHATSAPP_FROM = os.getenv("TWILIO_WHATSAPP_FROM", "whatsapp:+15557344000")

# Initialize service handlers
whatsapp_handler = WhatsAppHandler(
    twilio_account_sid=TWILIO_ACCOUNT_SID,
    twilio_auth_token=TWILIO_AUTH_TOKEN,
    openai_api_key=OPENAI_API_KEY,
    validate_signature=VALIDATE_TWILIO_SIGNATURE
)

voice_handler = VoiceHandler(
    twilio_auth_token=TWILIO_AUTH_TOKEN,
    validate_signature=VALIDATE_TWILIO_SIGNATURE
)

voice_session_manager = VoiceSessionManager(
    twilio_account_sid=TWILIO_ACCOUNT_SID,
    twilio_auth_token=TWILIO_AUTH_TOKEN,
    openai_api_key=OPENAI_API_KEY,
    deepgram_api_key=DEEPGRAM_API_KEY,
    elevenlabs_api_key=ELEVENLABS_API_KEY,
    elevenlabs_voice_id=ELEVENLABS_VOICE_ID,
    whatsapp_from_number=TWILIO_WHATSAPP_FROM
)

whatsapp_service = WhatsAppMessagingService(
    account_sid=TWILIO_ACCOUNT_SID,
    auth_token=TWILIO_AUTH_TOKEN,
    from_number=TWILIO_WHATSAPP_FROM
)


async def cleanup_old_caller_info():
    """Periodically clean up old caller info entries"""
    while True:
        try:
            current_time = datetime.now()
            expired_entries = []
            
            # Clean up old caller info entries (older than 1 hour)
            for call_sid, info in caller_info_storage.items():
                if (current_time - info.get('timestamp', current_time)).total_seconds() > 3600:
                    expired_entries.append(call_sid)
            
            for call_sid in expired_entries:
                del caller_info_storage[call_sid]
                logger.info(f"Cleaned up expired caller info for CallSid: {call_sid}")
            
            # Clean up old disconnected sessions (older than 1 hour)
            expired_sessions = []
            for caller_phone, session in active_sessions.items():
                if session.get("disconnected", False):
                    disconnected_at = session.get("disconnected_at", current_time)
                    if (current_time - disconnected_at).total_seconds() > 3600:
                        expired_sessions.append(caller_phone)
            
            for caller_phone in expired_sessions:
                del active_sessions[caller_phone]
                logger.info(f"Cleaned up expired disconnected session for caller: {caller_phone}")
                
            if expired_entries or expired_sessions:
                logger.info(f"Cleaned up {len(expired_entries)} caller info entries and {len(expired_sessions)} expired sessions")
                
        except Exception as e:
            logger.error(f"Error during caller info cleanup: {e}")
        
        # Run cleanup every 30 minutes
        await asyncio.sleep(1800)


@app.get("/health")
async def health_check() -> Dict[str, str]:
    """Health check endpoint"""
    return {"status": "ok", "timestamp": datetime.now().isoformat()}


@app.on_event("startup")
async def startup_event():
    """Start background tasks when the app starts"""
    # Start cleanup task
    cleanup_task = asyncio.create_task(cleanup_old_caller_info())
    cleanup_task.add_done_callback(
        lambda t: logger.error(f"Cleanup task failed: {t.exception()}") 
        if t.exception() else None
    )
    logger.info("Started caller info cleanup task")
    
    # Firebase is initialized in the firebase_service module
    logger.info("Application startup completed")


@app.post("/whatsapp-webhook")
async def whatsapp_webhook(request: Request) -> Response:
    """
    Handle inbound WhatsApp messages.
    
    This endpoint processes incoming WhatsApp messages and either:
    - Injects them into an active voice session if one exists
    - Processes them as off-call text conversation
    """
    try:
        # Handle the webhook using our modular handler
        result = await whatsapp_handler.handle_webhook(request, active_sessions, offcall_context)
        
        # If it's a dict with reply_text, send the reply
        if isinstance(result, dict) and "reply_text" in result:
            # Extract phone number from form data
            form = await request.form()
            from_num = (form.get("From") or "").replace("whatsapp:", "")
            
            # Send reply via WhatsApp
            await whatsapp_service.send_freeform_message(
                to_number=from_num, 
                message=result["reply_text"]
            )
            return Response(status_code=204)
        
        # Otherwise return the result directly
        return result
        
    except Exception as e:
        logger.exception("WhatsApp webhook error: %s", e)
        return Response(status_code=500)


@app.post("/voice")
async def voice_webhook(request: Request) -> PlainTextResponse:
    """
    Handle Twilio voice webhook.
    
    This endpoint receives voice call initiation requests from Twilio
    and returns TwiML to connect the call to our WebSocket stream.
    """
    return await voice_handler.handle_voice_webhook(request, caller_info_storage)


@app.post("/status")
async def status_callback(request: Request) -> Dict[str, str]:
    """Optional status callback endpoint for Twilio"""
    return {"status": "ok"}


@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    """
    WebSocket endpoint for Twilio Media Streams.
    
    This endpoint handles the WebSocket connection for voice calls,
    managing the Pipecat pipeline for speech-to-text, LLM processing,
    and text-to-speech.
    """
    # Twilio uses subprotocol "audio.twilio.com"
    await websocket.accept(subprotocol="audio.twilio.com")
    
    try:
        # Receive initial connection messages from Twilio
        connected = await websocket.receive_text()
        start_msg = await websocket.receive_text()
        data = json.loads(start_msg)
        
        # Extract connection details
        stream_sid = data.get("start", {}).get("streamSid")
        call_sid = data.get("start", {}).get("callSid")
        caller_number = data.get("start", {}).get("from")
        caller_name = data.get("start", {}).get("callerName")
        
        # Retrieve caller info from stored data if not in WebSocket message
        if call_sid and call_sid in caller_info_storage:
            stored_info = caller_info_storage[call_sid]
            caller_name = caller_name or stored_info.get("caller_name", "Unknown")
            caller_number = caller_number or stored_info.get("caller_number", "Unknown")
            logger.info(f"Retrieved caller info from storage - Name: {caller_name}, Number: {caller_number}")
        
        if not stream_sid:
            await websocket.close(code=1011)
            return

        logger.info(f"WebSocket connection - Caller: {caller_name} ({caller_number}), Stream: {stream_sid}, Call: {call_sid}")
        
        # Run the voice call session
        await voice_session_manager.run_call(
            websocket=websocket,
            stream_sid=stream_sid,
            call_sid=call_sid,
            caller_number=caller_number,
            caller_name=caller_name,
            active_sessions=active_sessions,
            caller_info_storage=caller_info_storage
        )

    except WebSocketDisconnect:
        logger.info("Twilio WebSocket disconnected")
        
        # Clean up on WebSocket disconnect
        try:
            if 'caller_number' in locals() and caller_number and caller_number != "Unknown":
                caller_phone = extract_phone_number(caller_number)
                if caller_phone and caller_phone != "Unknown":
                    session = active_sessions.get(caller_phone)
                    if session:
                        # Mark as disconnected and save conversation
                        session["disconnected"] = True
                        session["disconnected_at"] = datetime.now()
                        
                        # Save conversation asynchronously
                        llm_ctx = session.get("llm_context")
                        if llm_ctx:
                            conversation_messages = firebase_service.extract_conversation_messages(llm_ctx)
                            if conversation_messages:
                                task = asyncio.create_task(
                                    firebase_service.save_voice_messages_to_firebase_batch(
                                        caller_phone, conversation_messages, 
                                        session.get("call_sid")
                                    )
                                )
                                task.add_done_callback(
                                    lambda t: logger.error(f"Firebase save task failed: {t.exception()}") 
                                    if t.exception() else None
                                )
                        
                        # Clean up session
                        del active_sessions[caller_phone]
                        logger.info(f"WebSocket disconnect cleanup completed for {caller_phone}")
        except Exception as e:
            logger.error(f"Error in WebSocket disconnect cleanup: {e}")
            
    except Exception as e:
        logger.exception("WebSocket error: %s", e)
        
        # Clean up stored caller info on error
        if 'call_sid' in locals() and call_sid and call_sid in caller_info_storage:
            del caller_info_storage[call_sid]
            logger.info(f"Cleaned up caller info on error for CallSid: {call_sid}")
        
        try:
            await websocket.close(code=1011)
        except Exception:
            pass


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
