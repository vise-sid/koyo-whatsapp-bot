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
