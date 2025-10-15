import asyncio
import os
import json
import logging
from typing import Optional, Dict, Any, List
from urllib.parse import parse_qs
from datetime import datetime, timedelta
import pytz
import httpx
import google.generativeai as genai
from apscheduler.schedulers.asyncio import AsyncIOScheduler
from apscheduler.triggers.interval import IntervalTrigger

# Firebase imports
import firebase_admin
from firebase_admin import credentials, firestore

from fastapi import FastAPI, Request, WebSocket, WebSocketDisconnect
from fastapi.responses import PlainTextResponse, Response
from twilio.request_validator import RequestValidator
from twilio.rest import Client

# dskfsdflkjdf

# -------- Pipecat pieces --------
from pipecat.serializers.twilio import TwilioFrameSerializer
from pipecat.transports.websocket.fastapi import (
    FastAPIWebsocketTransport,
    FastAPIWebsocketParams,
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
from prompts.meher_voice_prompt import get_voice_system_prompt
from prompts.meher_text_prompt import get_text_system_prompt

logger = logging.getLogger("uvicorn.error")
app = FastAPI(title="Pipecat x Twilio WhatsApp Calling")

# Global storage for caller information (in production, use Redis or database)
caller_info_storage = {}
active_sessions = {}
offcall_context: Dict[str, list] = {}

# Firebase client
db = None

# APScheduler for background tasks
scheduler = AsyncIOScheduler()


async def create_or_update_conversation_metadata(
    user_id: str, character_name: str, call_sid: str = None
):
    """Create or update conversation metadata document"""
    if db is None:
        return

    try:
        conversation_metadata = {
            "last_updated": datetime.now(),
            "message_count": 0,  # This will be updated when messages are added
            "character_name": character_name,
        }

        if call_sid:
            conversation_metadata["call_sid"] = call_sid

        # Update conversation metadata (create if doesn't exist)
        # Structure: users/{user_id}/conversations/{character_name}
        conversation_ref = (
            db.collection("users")
            .document(user_id)
            .collection("conversations")
            .document(character_name)
        )
        conversation_ref.set(conversation_metadata, merge=True)

    except Exception as e:
        logger.error(f"Failed to update conversation metadata: {e}")


def extract_conversation_messages(llm_context):
    """Extract conversation messages from LLM context"""
    messages = llm_context.get_messages()
    conversation_messages = []
    for msg in messages:
        role = msg.get("role")
        content = msg.get("content", "")
        if role in ["user", "assistant"] and content.strip():
            sender = "user" if role == "user" else "character"
            conversation_messages.append(
                {"sender": sender, "content": content, "timestamp": datetime.now()}
            )
    return conversation_messages


async def save_voice_messages_to_firebase_batch(
    caller_phone: str,
    messages: list,
    call_sid: str = None,
    character_name: str = "meher",
):
    """Global helper function to save voice messages to Firebase using batch operations"""
    try:
        if not messages:
            return

        logger.info(
            f"Batch saving {len(messages)} voice messages to Firebase for {caller_phone}"
        )

        # Create batch for faster operations
        batch = db.batch()

        # Update conversation metadata once
        await create_or_update_conversation_metadata(
            caller_phone, character_name, call_sid
        )

        # Add all messages to batch
        conversation_ref = (
            db.collection("users")
            .document(caller_phone)
            .collection("conversations")
            .document(character_name)
        )
        messages_ref = conversation_ref.collection("messages")

        for msg in messages:
            message_data = {
                "sender": msg["sender"],
                "content": msg["content"][:1000],
                "timestamp": msg["timestamp"],
                "sync": False,
                "conversation_type": "voice",
            }
            if call_sid:
                message_data["call_sid"] = call_sid

            # Add to batch
            new_message_ref = messages_ref.document()
            batch.set(new_message_ref, message_data)

        # Commit batch operation (much faster than individual saves)
        batch.commit()
        logger.info(
            f"Batch saved {len(messages)} voice messages to Firebase for {caller_phone}"
        )

    except Exception as e:
        logger.error(f"Failed to batch save voice messages to Firebase: {e}")


async def save_message_to_firebase(
    user_id: str,
    sender: str,
    content: str,
    timestamp: datetime = None,
    sync: bool = False,
    conversation_type: str = "text",  # "text" or "voice"
    call_sid: str = None,
    character_name: str = "meher",
) -> bool:
    """Save a message to Firebase Firestore in nested collection structure:
    users/{user_id}/conversations/{character_name}/messages/{message_id}

    Args:
        user_id: Phone number or user identifier
        sender: "user" or "character" (Meher)
        content: Message content
        timestamp: Message timestamp (defaults to now)
        sync: Sync status (defaults to False)
        conversation_type: "text" or "voice"
        call_sid: Call SID for voice conversations

    Returns:
        bool: True if saved successfully, False otherwise
    """
    if db is None:
        logger.warning("Firebase not initialized, skipping message save")
        return False

    try:
        if timestamp is None:
            timestamp = get_mumbai_datetime()

        # Use the provided character name (defaults to "meher")

        # Create or update conversation metadata
        await create_or_update_conversation_metadata(user_id, character_name, call_sid)

        # Create message document
        message_data = {
            "sender": sender,
            "content": content[:1000],  # Limit content length
            "timestamp": timestamp,
            "sync": sync,
            "conversation_type": conversation_type,
        }

        # Add call_sid for voice conversations
        if call_sid:
            message_data["call_sid"] = call_sid

        # Save to nested collection: users/{user_id}/conversations/{character_name}/messages
        doc_ref = (
            db.collection("users")
            .document(user_id)
            .collection("conversations")
            .document(character_name)
            .collection("messages")
            .add(message_data)
        )
        logger.info(
            f"Saved {sender} message to Firebase for user {user_id} in conversation with {character_name}"
        )

        # Update message count in conversation metadata
        try:
            conversation_ref = (
                db.collection("users")
                .document(user_id)
                .collection("conversations")
                .document(character_name)
            )
            conversation_ref.update({"message_count": firestore.Increment(1)})
        except Exception as e:
            logger.warning(f"Failed to update message count: {e}")

        # Update notification tracking for re-engagement messages
        # Only update if this is a user message (not character response)
        if sender == "user":
            try:
                # Update the notifications collection with last message info
                notification_data = {
                    "user_id": user_id,
                    "character_name": character_name,
                    "last_message_timestamp": timestamp,
                    "last_user_message": content[:500],  # Store last user message
                    "sent_notification": False,  # Reset notification flag
                    "updated_at": timestamp,
                    "message_count": firestore.Increment(1),  # Track total messages
                }

                doc_ref = db.collection("notifications").document(
                    f"{user_id}_{character_name}"
                )
                doc_ref.set(notification_data, merge=True)
                logger.info(
                    f"Updated notification tracking for user {user_id} with character {character_name}"
                )

            except Exception as e:
                logger.warning(f"Failed to update notification tracking: {e}")

        return True

    except Exception as e:
        logger.error(f"Failed to save message to Firebase: {e}")
        return False


async def cleanup_old_caller_info():
    """Periodically clean up old caller info entries"""
    import asyncio

    while True:
        try:
            current_time = datetime.now()
            expired_entries = []

            for call_sid, info in caller_info_storage.items():
                # Remove entries older than 1 hour
                if (
                    current_time - info.get("timestamp", current_time)
                ).total_seconds() > 3600:
                    expired_entries.append(call_sid)

            for call_sid in expired_entries:
                del caller_info_storage[call_sid]
                logger.info(f"Cleaned up expired caller info for CallSid: {call_sid}")

            # Also cleanup old disconnected active sessions
            expired_sessions = []
            for caller_phone, session in active_sessions.items():
                if session.get("disconnected", False):
                    # Remove disconnected sessions older than 1 hour
                    disconnected_at = session.get("disconnected_at", current_time)
                    if (current_time - disconnected_at).total_seconds() > 3600:
                        expired_sessions.append(caller_phone)

            for caller_phone in expired_sessions:
                del active_sessions[caller_phone]
                logger.info(
                    f"Cleaned up expired disconnected session for caller: {caller_phone}"
                )

            if expired_entries or expired_sessions:
                logger.info(
                    f"Cleaned up {len(expired_entries)} caller info entries and {len(expired_sessions)} expired sessions"
                )

        except Exception as e:
            logger.error(f"Error during caller info cleanup: {e}")

        # Run cleanup every 30 minutes
        await asyncio.sleep(1800)


# -------- Env --------
TWILIO_ACCOUNT_SID = os.getenv("TWILIO_ACCOUNT_SID", "")
TWILIO_AUTH_TOKEN = os.getenv("TWILIO_AUTH_TOKEN", "")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")
DEEPGRAM_API_KEY = os.getenv("DEEPGRAM_API_KEY", "")
ELEVENLABS_API_KEY = os.getenv("ELEVENLABS_API_KEY", "")
ELEVENLABS_VOICE_ID = os.getenv("ELEVENLABS_VOICE_ID", "Rachel")
VALIDATE_TWILIO_SIGNATURE = (
    os.getenv("VALIDATE_TWILIO_SIGNATURE", "false").lower() == "true"
)
TWILIO_WHATSAPP_FROM = os.getenv(
    "TWILIO_WHATSAPP_FROM", "whatsapp:+15557344000"
)  # Your Twilio WhatsApp number
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "")  # For web search functionality


class WhatsAppMessagingService:
    """Service for sending WhatsApp messages via Twilio"""

    def __init__(self, account_sid: str, auth_token: str, from_number: str):
        self.client = Client(account_sid, auth_token)
        self.from_number = from_number
        self.logger = logging.getLogger(__name__)

    async def send_message(
        self, to_number: str, message: str, recipient_name: str = "User"
    ) -> Dict[str, Any]:
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
                content_variables=json.dumps({"1": recipient_name, "2": message}),
            )

            self.logger.info(
                f"WhatsApp template message sent - SID: {message_obj.sid}, To: {to_number}"
            )

            return {
                "success": True,
                "message_sid": message_obj.sid,
                "status": message_obj.status,
                "to": to_number,
                "message": message,
                "template_used": True,
                "template_name": "koyo_simple",
            }

        except Exception as e:
            self.logger.error(
                f"Failed to send WhatsApp message to {to_number}: {str(e)}"
            )
            return {
                "success": False,
                "error": str(e),
                "to": to_number,
                "message": message,
                "template_used": False,
            }

    async def send_freeform_message(
        self, to_number: str, message: str
    ) -> Dict[str, Any]:
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
            self.logger.info(
                f"WhatsApp freeform message sent - SID: {message_obj.sid}, To: {to_number}"
            )
            return {
                "success": True,
                "message_sid": message_obj.sid,
                "status": message_obj.status,
            }
        except Exception as e:
            self.logger.error(
                f"Failed to send freeform WhatsApp message to {to_number}: {str(e)}"
            )
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
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": prompt},
                        {"type": "image_url", "image_url": {"url": image_url}},
                    ],
                }
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
    mumbai_tz = pytz.timezone("Asia/Kolkata")
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
        "hour": hour,
    }


def extract_phone_number(whatsapp_number: str) -> str:
    """Extract clean phone number from WhatsApp format"""
    if whatsapp_number and whatsapp_number.startswith("whatsapp:"):
        return whatsapp_number.replace("whatsapp:", "")
    return whatsapp_number or "Unknown"


def get_mumbai_datetime() -> datetime:
    """Get current datetime in Mumbai timezone"""
    mumbai_tz = pytz.timezone("Asia/Kolkata")
    return datetime.now(mumbai_tz)


def normalize_datetime_to_mumbai(dt: datetime) -> datetime:
    """Normalize a datetime to Mumbai timezone for consistent comparison"""
    mumbai_tz = pytz.timezone("Asia/Kolkata")
    if dt.tzinfo is None:
        # If datetime is naive, make it timezone-aware
        return mumbai_tz.localize(dt)
    elif dt.tzinfo != mumbai_tz:
        # Convert to Mumbai timezone if it's in a different timezone
        return dt.astimezone(mumbai_tz)
    return dt


class WebSearchService:
    """Service for web search using Gemini 2.5 Flash"""

    def __init__(self, api_key: str):
        if api_key:
            genai.configure(api_key=api_key)
            self.model = genai.GenerativeModel("gemini-2.0-flash")
        else:
            self.model = None
            logger.warning("Gemini API key not provided - web search disabled")

    async def search_web(
        self, query: str, max_results: int = 3
    ) -> List[Dict[str, str]]:
        """
        Search the web using Gemini 2.5 Flash and return relevant results

        Args:
            query: Search query
            max_results: Maximum number of results to return

        Returns:
            List of search results with title, snippet, and url
        """
        if not self.model:
            return []

        try:
            search_prompt = f"""
            Search the web for: {query}

            Please provide {max_results} relevant search results in the following format:
            Title: [result title]
            Snippet: [brief description]
            URL: [source url]

            ---

            Focus on recent, relevant information. If you find current news or recent updates, prioritize those.
            """

            response = self.model.generate_content(search_prompt)
            results = []

            # Parse the response to extract structured results
            content = response.text
            sections = content.split("---")
            logger.info(f"Search results content: {content}")
            logger.info(f"Search results sections: {sections}")

            for section in sections:
                if "Title:" in section and "Snippet:" in section:
                    lines = section.strip().split("\n")
                    result = {}
                    for line in lines:
                        if line.startswith("Title:"):
                            result["title"] = line.replace("Title:", "").strip()
                        elif line.startswith("Snippet:"):
                            result["snippet"] = line.replace("Snippet:", "").strip()
                        elif line.startswith("URL:"):
                            result["url"] = line.replace("URL:", "").strip()

                    if len(result) >= 2:  # At least title and snippet
                        results.append(result)

            return results[:max_results]

        except Exception as e:
            logger.error(f"Web search error: {e}")
            return []


async def update_notification_tracking(
    user_id: str, last_message_timestamp: datetime = None, character_name: str = "meher"
):
    """
    Update notification tracking in Firebase for re-engagement messages

    Args:
        user_id: User's phone number
        last_message_timestamp: Timestamp of the last message (defaults to now)
        character_name: Character name for the conversation
    """
    if db is None:
        return

    try:
        if last_message_timestamp is None:
            last_message_timestamp = get_mumbai_datetime()

        # Update the notifications collection with the structure shown in the image
        notification_data = {
            "user_id": user_id,
            "character_name": character_name,
            "last_message_timestamp": last_message_timestamp,
            "last_user_message": "",  # Will be updated when we have the actual message
            "notification_sent_at": None,
            "sent_notification": False,
            "updated_at": last_message_timestamp,
        }

        # Use the document ID format: {user_id}_{character_name}
        doc_ref = db.collection("notifications").document(f"{user_id}_{character_name}")
        doc_ref.set(notification_data, merge=True)

        logger.info(
            f"Updated notification tracking for user {user_id} with character {character_name}"
        )

    except Exception as e:
        logger.error(f"Failed to update notification tracking: {e}")


async def get_last_25_messages(
    user_id: str, character_name: str = "meher"
) -> List[Dict[str, Any]]:
    """
    Fetch the last 25 messages from Firebase for AI context

    Args:
        user_id: User's phone number
        character_name: Character name (defaults to "meher")

    Returns:
        List of message dictionaries with sender, content, and timestamp
    """
    if db is None:
        return []

    try:
        # Query messages from the nested collection structure
        messages_ref = (
            db.collection("users")
            .document(user_id)
            .collection("conversations")
            .document(character_name)
            .collection("messages")
        )

        # Order by timestamp descending and limit to 25
        query = messages_ref.order_by(
            "timestamp", direction=firestore.Query.DESCENDING
        ).limit(25)
        docs = query.stream()

        messages = []
        for doc in docs:
            data = doc.to_dict()
            messages.append(
                {
                    "sender": data.get("sender", "unknown"),
                    "content": data.get("content", ""),
                    "timestamp": data.get("timestamp"),
                    "conversation_type": data.get("conversation_type", "text"),
                }
            )

        # Reverse to get chronological order (oldest first)
        messages.reverse()

        logger.info(f"Retrieved {len(messages)} messages for user {user_id}")
        return messages

    except Exception as e:
        logger.error(f"Failed to fetch messages for user {user_id}: {e}")
        return []


async def generate_reengagement_message(
    user_id: str, character_name: str = "meher"
) -> str:
    """
    Generate a re-engagement message using AI based on the last 25 messages

    Args:
        user_id: User's phone number
        character_name: Character name for the conversation

    Returns:
        Generated re-engagement message
    """
    try:
        # Get conversation history
        messages = await get_last_25_messages(user_id, character_name)

        if not messages:
            return "Hey! कैसी हो? बहुत दिनों से बात नहीं हुई। कुछ interesting हो रहा है?"

        # Build context for AI
        conversation_context = ""
        for msg in messages[-10:]:  # Use last 10 messages for context
            sender = "User" if msg["sender"] == "user" else character_name
            conversation_context += f"{sender}: {msg['content']}\n"

        # Create AI prompt for re-engagement
        reengagement_prompt = f"""
        You are {character_name}, a witty and loyal friend. Based on the recent conversation history below, create a natural re-engagement message to reconnect with the user after 8 hours of inactivity.

        Recent conversation:
        {conversation_context}

        Guidelines:
        - Keep it casual and friendly in Hinglish (mix of Hindi and English)
        - Reference something from the conversation naturally
        - Be playful and engaging
        - Keep it under 50 words
        - Use romanized Hindi (not Devanagari script)
        - Don't mention the 8-hour gap explicitly
        - Make it feel like a natural continuation

        Generate a warm, engaging message that feels like {character_name} reaching out to continue the conversation.
        """

        client = OpenAI(api_key=OPENAI_API_KEY)
        response = client.chat.completions.create(
            model="gpt-4o",
            messages=[{"role": "user", "content": reengagement_prompt}],
            max_tokens=100,
            temperature=0.8,
        )

        message = response.choices[0].message.content.strip()
        logger.info(f"Generated re-engagement message for user {user_id}")
        return message

    except Exception as e:
        logger.error(f"Failed to generate re-engagement message: {e}")
        return "Hey! कैसी हो? बहुत दिनों से बात नहीं हुई। कुछ interesting हो रहा है?"


async def check_and_send_reengagement_messages():
    """
    Background task to check for users who need re-engagement messages after 2 minutes
    """
    if db is None:
        logger.warning("Firebase not available for re-engagement check")
        return

    try:
        # Use timezone-aware datetime for consistent comparison
        current_time = get_mumbai_datetime()
        two_minutes_ago = current_time - timedelta(minutes=2)

        # Query notifications collection for users who haven't received notifications
        notifications_ref = db.collection("notifications")
        query = notifications_ref.where("sent_notification", "==", False)

        docs = query.stream()
        users_to_notify = []

        for doc in docs:
            data = doc.to_dict()
            last_message_time = data.get("last_message_timestamp")

            if last_message_time and isinstance(last_message_time, datetime):
                # Normalize datetime to Mumbai timezone for consistent comparison
                last_message_time = normalize_datetime_to_mumbai(last_message_time)

                # Check if 2 minutes have passed since last message
                if last_message_time <= two_minutes_ago:
                    users_to_notify.append(
                        {
                            "user_id": data.get("user_id"),
                            "doc_id": doc.id,
                            "last_message_time": last_message_time,
                            "character_name": data.get("character_name", "meher"),
                        }
                    )

        logger.info(
            f"Found {len(users_to_notify)} users needing re-engagement messages"
        )

        # Send re-engagement messages
        whatsapp_service = WhatsAppMessagingService(
            account_sid=TWILIO_ACCOUNT_SID,
            auth_token=TWILIO_AUTH_TOKEN,
            from_number=TWILIO_WHATSAPP_FROM,
        )

        for user_info in users_to_notify:
            try:
                user_id = user_info["user_id"]
                doc_id = user_info["doc_id"]
                character_name = user_info.get("character_name", "meher")
                last_message_time = user_info.get("last_message_time")

                logger.info(
                    f"Processing re-engagement for user {user_id} (character: {character_name})"
                )
                logger.info(
                    f"Last message time: {last_message_time}, Current time: {current_time}"
                )

                # Generate personalized re-engagement message
                message = await generate_reengagement_message(user_id, character_name)
                logger.info(f"Generated re-engagement message: {message[:100]}...")

                # Send WhatsApp message
                result = await whatsapp_service.send_message(
                    to_number=user_id, message=message, recipient_name="User"
                )

                if result["success"]:
                    # Update notification tracking with timezone-aware datetime
                    doc_ref = db.collection("notifications").document(doc_id)
                    doc_ref.update(
                        {
                            "sent_notification": True,
                            "notification_sent_at": current_time,
                            "updated_at": current_time,
                            "last_reengagement_message": message[:500],
                        }
                    )

                    logger.info(
                        f"✅ Successfully sent re-engagement message to user {user_id}"
                    )
                else:
                    logger.error(
                        f"❌ Failed to send re-engagement message to user {user_id}: {result}"
                    )

            except Exception as e:
                logger.error(
                    f"❌ Error sending re-engagement message to user {user_info.get('user_id', 'unknown')}: {e}"
                )

    except Exception as e:
        logger.error(f"Error in re-engagement check: {e}")


async def reengagement_scheduler():
    """Background scheduler for re-engagement messages using APScheduler"""
    try:
        # Add job to scheduler - run every 30 seconds to check for users
        scheduler.add_job(
            check_and_send_reengagement_messages,
            trigger=IntervalTrigger(seconds=30),
            id="reengagement_check",
            name="Check for users needing re-engagement messages",
            replace_existing=True,
        )

        # Start the scheduler
        scheduler.start()
        logger.info("APScheduler started for re-engagement messages")

    except Exception as e:
        logger.error(f"Error starting APScheduler: {e}")


@app.get("/health")
def health():
    return {
        "ok": True,
        "timestamp": datetime.now().isoformat(),
        "firebase_connected": db is not None,
        "scheduler_running": scheduler.running if scheduler else False,
        "environment": {
            "twilio_account_sid": bool(TWILIO_ACCOUNT_SID),
            "twilio_auth_token": bool(TWILIO_AUTH_TOKEN),
            "twilio_whatsapp_from": bool(TWILIO_WHATSAPP_FROM),
            "openai_api_key": bool(OPENAI_API_KEY),
            "gemini_api_key": bool(GEMINI_API_KEY),
            "elevenlabs_api_key": bool(ELEVENLABS_API_KEY),
            "deepgram_api_key": bool(DEEPGRAM_API_KEY),
        },
    }


@app.post("/test-whatsapp")
async def test_whatsapp(request: Request):
    """Test endpoint to verify WhatsApp message sending"""
    try:
        data = await request.json()
        to_number = data.get("to_number")
        message = data.get("message", "Test message from Koyo bot")

        if not to_number:
            return {"error": "to_number is required"}

        whatsapp_service = WhatsAppMessagingService(
            account_sid=TWILIO_ACCOUNT_SID,
            auth_token=TWILIO_AUTH_TOKEN,
            from_number=TWILIO_WHATSAPP_FROM,
        )

        result = await whatsapp_service.send_message(
            to_number=to_number, message=message, recipient_name="Test User"
        )

        return result

    except Exception as e:
        logger.error(f"Test WhatsApp error: {e}")
        return {"error": str(e)}


@app.on_event("startup")
async def startup_event():
    """Start background tasks when the app starts"""
    # Create background task with proper error handling
    cleanup_task = asyncio.create_task(cleanup_old_caller_info())
    cleanup_task.add_done_callback(
        lambda t: logger.error(f"Cleanup task failed: {t.exception()}")
        if t.exception()
        else None
    )
    logger.info("Started caller info cleanup task")

    # Initialize Firebase
    global db
    try:
        # Try to initialize with service account key from environment
        firebase_credentials = os.getenv("FIREBASE_CREDENTIALS")
        if firebase_credentials:
            # Parse JSON credentials from environment variable
            cred_dict = json.loads(firebase_credentials)
            cred = credentials.Certificate(cred_dict)
            firebase_admin.initialize_app(cred)
        else:
            # Try to initialize with default credentials (for local development)
            firebase_admin.initialize_app()

        db = firestore.client()
        # Test Firebase connection
        test_doc = db.collection("_health_check").document("test")
        test_doc.set({"timestamp": datetime.now()}, merge=True)
        test_doc.delete()
        logger.info("Firebase initialized and tested successfully")

        # Start re-engagement scheduler
        await reengagement_scheduler()
        logger.info("Started APScheduler for re-engagement messages")

    except Exception as e:
        logger.error(f"Firebase initialization failed: {e}")
        db = None
        # Don't crash the app, but log the critical error
        logger.critical("Firebase is not available - message saving will fail")


@app.on_event("shutdown")
async def shutdown_event():
    """Clean up resources when the app shuts down"""
    try:
        if scheduler.running:
            scheduler.shutdown()
            logger.info("APScheduler stopped")
    except Exception as e:
        logger.error(f"Error stopping scheduler: {e}")


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
    sig = request.headers.get("X-Twilio-Signature", "") or request.headers.get(
        "x-twilio-signature", ""
    )
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

        # Log all form data for debugging
        logger.info(f"WhatsApp webhook form data: {dict(form)}")

        media_items = []
        for i in range(num_media):
            url = form.get(f"MediaUrl{i}")
            ctype = form.get(f"MediaContentType{i}")
            if url:
                media_items.append({"url": url, "content_type": ctype or ""})

        logger.info(
            f"Incoming WhatsApp from {from_num} | text='{text_body}' | media_count={num_media}"
        )
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
                        parts.append(
                            f"[audio transcript] {transcript}"
                            if transcript
                            else "[audio note received]"
                        )
                    elif "image/" in ctype:
                        caption = await _caption_image_url(url)
                        parts.append(
                            f"[image] {caption}" if caption else "[image received]"
                        )
                    elif "text/" in ctype:
                        excerpt = await _fetch_text_excerpt(url, max_chars=500)
                        parts.append(
                            f"[text doc excerpt] {excerpt}"
                            if excerpt
                            else "[text document received]"
                        )
                    elif "application/pdf" in ctype or ctype.startswith("application/"):
                        parts.append("[document received]")
                    else:
                        parts.append("[media received]")
                except Exception as me:
                    logger.error(f"Media handling error (in-call) for {url}: {me}")
                    parts.append("[media received]")

            content = (" ".join(parts)).strip() or "[empty message]"
            frames = [
                LLMMessagesAppendFrame(
                    [{"role": "user", "content": f"(WhatsApp) {content}"}]
                ),
                LLMRunFrame(),
            ]
            await session["task"].queue_frames(frames)

            # Save user message to Firebase
            await save_message_to_firebase(
                user_id=from_num,
                sender="user",
                content=content,
                conversation_type="text",
                character_name="meher",
            )

            return Response(status_code=204)

        # No live session -> off-call LLM chat with context and reply over WhatsApp (freeform)
        try:
            reply_text = await handle_multimodal_offcall(
                text_body, media_items, from_num, "meher"
            )
        except Exception as e:
            logger.error(f"Error in multimodal offcall handler: {e}")
            reply_text = "Sorry, I'm having some technical difficulties. Please try again in a moment."

        whatsapp_service = WhatsAppMessagingService(
            account_sid=TWILIO_ACCOUNT_SID,
            auth_token=TWILIO_AUTH_TOKEN,
            from_number=TWILIO_WHATSAPP_FROM,
        )

        try:
            # Use template-based message instead of freeform for better reliability
            result = await whatsapp_service.send_message(
                to_number=from_num, message=reply_text, recipient_name="User"
            )
            if result["success"]:
                logger.info(f"Successfully sent WhatsApp reply to {from_num}")
            else:
                logger.error(
                    f"Failed to send WhatsApp message to {from_num}: {result.get('error', 'Unknown error')}"
                )
        except Exception as e:
            logger.error(f"Failed to send WhatsApp message to {from_num}: {e}")
            # Don't fail the webhook, just log the error

        # Save both user message and character response to Firebase
        try:
            await save_message_to_firebase(
                user_id=from_num,
                sender="user",
                content=text_body,
                conversation_type="text",
                character_name="meher",
            )
            await save_message_to_firebase(
                user_id=from_num,
                sender="character",
                content=reply_text,
                conversation_type="text",
                character_name="meher",
            )
            logger.info(f"Successfully saved messages to Firebase for {from_num}")
        except Exception as e:
            logger.error(f"Failed to save messages to Firebase for {from_num}: {e}")
            # Don't fail the webhook, just log the error

        # Return 204 to avoid Twilio echoing body as a user-visible message
        return Response(status_code=204)

    except Exception as e:
        logger.exception("WhatsApp webhook error: %s", e)
        return Response(status_code=500)


async def handle_multimodal_offcall(
    text: str, media: list, user_phone: str, character_name: str = "meher"
) -> str:
    """Off-call handler: build/maintain chat context and generate a Meher-style reply via LLM.

    - Maintains per-user message history in offcall_context[user_phone]
    - Supports text + media summaries
    - Generates a concise Hinglish reply (≤ ~75 tokens)
    - Includes conversation history from Firebase
    - Supports web search functionality
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
                parts.append(
                    f"[audio transcript] {transcript}"
                    if transcript
                    else "[audio note received]"
                )
            elif "image/" in ctype:
                caption = await _caption_image_url(url)
                parts.append(f"[image] {caption}" if caption else "[image received]")
            elif "text/" in ctype:
                excerpt = await _fetch_text_excerpt(url, max_chars=800)
                parts.append(
                    f"[text doc excerpt] {excerpt}"
                    if excerpt
                    else "[text document received]"
                )
            elif "application/pdf" in ctype or ctype.startswith("application/"):
                parts.append("[document received]")
            else:
                parts.append("[media received]")
        except Exception as me:
            logger.error(f"Media handling error (off-call) for {url}: {me}")
            parts.append("[media received]")
    user_text = (" ".join(parts)).strip() or "[empty message]"

    # 2) Get conversation history from Firebase
    conversation_history = ""
    try:
        messages = await get_last_25_messages(user_phone, character_name)
        if messages:
            # Build conversation context from last 10 messages
            recent_messages = messages[-10:] if len(messages) > 10 else messages
            conversation_history = "\n".join(
                [
                    f"{'User' if msg['sender'] == 'user' else character_name}: {msg['content']}"
                    for msg in recent_messages
                ]
            )
    except Exception as e:
        logger.warning(f"Failed to fetch conversation history: {e}")

    # 3) Prepare context store for user
    history = offcall_context.get(user_phone) or []

    # 4) System prompt for off-call chat: use dedicated texting prompt module with history
    time_context = get_current_time_context()
    system_text = get_text_system_prompt(
        caller_display_name=user_phone,
        caller_phone=user_phone,
        time_context=time_context,
        conversation_history=conversation_history,
    )

    # 5) Build messages array (bounded history)
    messages = [{"role": "system", "content": system_text}]
    # include last ~8 messages from history for context
    for m in history[-8:]:
        messages.append(m)
    messages.append({"role": "user", "content": user_text})

    # 6) Check if user is asking for web search
    web_search_query = None
    search_keywords = [
        "latest",
        "news",
        "current",
        "recent",
        "today",
        "weather",
        "stock",
        "trending",
        "update",
    ]
    if any(keyword in user_text.lower() for keyword in search_keywords):
        # Extract potential search query
        web_search_query = user_text

    # 7) Call OpenAI for a short response
    try:
        client = OpenAI(api_key=OPENAI_API_KEY)

        # If web search is needed, perform it first
        if web_search_query and GEMINI_API_KEY:
            try:
                web_search_service = WebSearchService(GEMINI_API_KEY)
                search_results = await web_search_service.search_web(web_search_query)

                if search_results:
                    search_context = "\n".join(
                        [
                            f"Title: {result.get('title', '')}\nSnippet: {result.get('snippet', '')}\nURL: {result.get('url', '')}"
                            for result in search_results
                        ]
                    )

                    # Add search results to the conversation
                    messages.append(
                        {
                            "role": "system",
                            "content": f"WEB SEARCH RESULTS:\n{search_context}\n\nUse this information to provide current, accurate information to the user.",
                        }
                    )
            except Exception as e:
                logger.error(f"Web search error: {e}")

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

    # 8) Update history
    history.append({"role": "user", "content": user_text})
    history.append({"role": "assistant", "content": reply})
    offcall_context[user_phone] = history[-20:]  # cap to last 20 turns

    # Save messages to Firebase (this will be called by the webhook handler)
    # Note: We don't save here to avoid duplication since webhook handler already saves

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

    logger.info(
        f"Voice webhook - From: {from_num}, To: {to_num}, Name: {caller_name}, CallSid: {call_sid}"
    )

    # Store caller info for later use in the WebSocket connection
    if call_sid:
        caller_info_storage[call_sid] = {
            "caller_name": caller_name,
            "caller_number": from_num,
            "to_number": to_num,
            "timestamp": datetime.now(),
        }
        logger.info(
            f"Stored caller info for CallSid: {call_sid} - Name: {caller_name}, Number: {from_num}"
        )

    twiml = _twiml(_ws_url(request))
    return PlainTextResponse(content=twiml, media_type="application/xml")


# Optional status callback endpoint (set it in Twilio if you want)
@app.post("/status")
async def status_callback(_: Request):
    return {"ok": True}


# ---- Pipecat session per call ----
async def _run_call(
    websocket: WebSocket,
    stream_sid: str,
    call_sid: Optional[str],
    caller_number: Optional[str] = None,
    caller_name: Optional[str] = None,
    character_name: str = "meher",
):
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
        ),
    )
    # Initialize WhatsApp messaging service
    whatsapp_service = WhatsAppMessagingService(
        account_sid=TWILIO_ACCOUNT_SID,
        auth_token=TWILIO_AUTH_TOKEN,
        from_number=TWILIO_WHATSAPP_FROM,
    )

    # Define WhatsApp function schema using Pipecat's FunctionSchema
    whatsapp_function = FunctionSchema(
        name="send_whatsapp_message",
        description="Send a WhatsApp text message to a user during the conversation. Use this when the user asks you to send them a message, reminder, or any text via WhatsApp.",
        properties={
            "to_number": {
                "type": "string",
                "description": "The recipient's phone number with country code (e.g., +1234567890). If not provided, use the caller's number.",
            },
            "message": {
                "type": "string",
                "description": "The message content to send via WhatsApp",
            },
        },
        required=["message"],
    )

    # Define call termination function schema
    terminate_call_function = FunctionSchema(
        name="terminate_voice_call",
        description="Terminate the current voice call. Use this when the conversation has naturally concluded, when the user says goodbye, or when it's appropriate to end the call. Always say goodbye before terminating.",
        properties={
            "reason": {
                "type": "string",
                "description": "Brief reason for ending the call (e.g., 'conversation concluded', 'user said goodbye', 'call completed successfully')",
            }
        },
        required=["reason"],
    )

    # Define web search function schema
    web_search_function = FunctionSchema(
        name="search_web",
        description="Search the web for real-time information using Gemini 2.5 Flash. Use this when users ask about current news, events, weather, stock prices, or any information that might be outdated.",
        properties={
            "query": {
                "type": "string",
                "description": "The search query to look up on the web",
            }
        },
        required=["query"],
    )

    tools = ToolsSchema(
        standard_tools=[whatsapp_function, terminate_call_function, web_search_function]
    )

    # Initialize LLM service
    llm = OpenAILLMService(
        api_key=OPENAI_API_KEY,
        model="gpt-4o",
        params=OpenAILLMService.InputParams(
            temperature=0.7,
        ),
    )

    # Register WhatsApp function handler
    async def send_whatsapp_message_handler(params: FunctionCallParams):
        """Handle WhatsApp message sending function calls"""
        try:
            # Check if connection is still active
            if (
                hasattr(transport, "_websocket")
                and transport._websocket.client_state.name != "CONNECTED"
            ):
                logger.info("Connection closed, skipping WhatsApp message")
                await params.result_callback(
                    "❌ Connection closed, cannot send message"
                )
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
                to_number=to_number, message=message, recipient_name=caller_display_name
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

    # Register call termination function handler
    async def terminate_call_handler(params: FunctionCallParams):
        """Handle voice call termination function calls using Pipecat best practices"""
        try:
            reason = params.arguments.get("reason", "agent initiated termination")
            logger.info(f"Agent requested to terminate call: {reason}")

            # Use our unified termination function for consistent behavior
            success = await terminate_call(
                reason, save_conversation=True, immediate=True
            )

            if success:
                await params.result_callback(
                    f"✅ Call terminated successfully: {reason}"
                )
            else:
                await params.result_callback(f"❌ Failed to terminate call: {reason}")

        except Exception as e:
            logger.error(f"Error in terminate call handler: {str(e)}")
            await params.result_callback(f"❌ Error terminating call: {str(e)}")

    # Register the terminate call function handler
    llm.register_function("terminate_voice_call", terminate_call_handler)

    # Register web search function handler
    async def web_search_handler(params: FunctionCallParams):
        """Handle web search function calls"""
        try:
            query = params.arguments.get("query", "")

            if not query:
                result = "❌ No search query provided"
                await params.result_callback(result)
                return

            # Perform web search using Gemini
            if GEMINI_API_KEY:
                web_search_service = WebSearchService(GEMINI_API_KEY)
                search_results = await web_search_service.search_web(query)

                if search_results:
                    # Format results for voice response
                    result_text = f"Latest info about {query}:\n"
                    for i, result in enumerate(search_results[:3], 1):
                        title = result.get("title", "No title")
                        snippet = result.get("snippet", "No description")
                        result_text += f"{i}. {title}: {snippet}\n"

                    await params.result_callback(
                        f"✅ Found {len(search_results)} results: {result_text}"
                    )
                else:
                    await params.result_callback(
                        f"❌ No recent information found about {query}"
                    )
            else:
                await params.result_callback(
                    "❌ Web search not available - API key missing"
                )

        except Exception as e:
            logger.error(f"Error in web search handler: {str(e)}")
            await params.result_callback(f"❌ Error searching web: {str(e)}")

    # Register the web search function handler
    llm.register_function("search_web", web_search_handler)
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
        ),
    )

    # Get current time and date context
    time_context = get_current_time_context()
    caller_phone = extract_phone_number(caller_number) if caller_number else "Unknown"
    caller_display_name = caller_name or "Unknown"

    # Get conversation history from Firebase for voice calls
    conversation_history = ""
    try:
        messages = await get_last_25_messages(caller_phone, character_name)
        if messages:
            # Build conversation context from last 10 messages
            recent_messages = messages[-10:] if len(messages) > 10 else messages
            conversation_history = "\n".join(
                [
                    f"{'User' if msg['sender'] == 'user' else character_name}: {msg['content']}"
                    for msg in recent_messages
                ]
            )
    except Exception as e:
        logger.warning(f"Failed to fetch conversation history for voice call: {e}")

    system_prompt = get_voice_system_prompt(
        caller_display_name, caller_phone, time_context, conversation_history
    )

    ctx = OpenAILLMContext(
        messages=[{"role": "system", "content": system_prompt}], tools=tools
    )
    agg = llm.create_context_aggregator(ctx)

    # Timeout handling system
    def get_timeout_for_retry(retry_count: int) -> float:
        """Return timeout value based on retry count"""
        timeout_values = [10.0, 8.0, 4.0, 3.0]
        return timeout_values[min(retry_count - 1, len(timeout_values) - 1)]

    async def handle_user_idle_with_retry(
        user_idle: UserIdleProcessor, retry_count: int
    ) -> bool:
        """
        Handle user idle with structured follow-up based on retry count.
        Returns True to continue monitoring, False to stop.
        """
        logger.info(f"User idle - attempt #{retry_count}")

        # Check if the connection is still active
        if (
            hasattr(transport, "_websocket")
            and transport._websocket.client_state.name != "CONNECTED"
        ):
            logger.info("Connection closed, stopping idle monitoring")
            return False

        # Check if the session has been marked as disconnected
        if caller_phone and caller_phone != "Unknown":
            session = active_sessions.get(caller_phone)
            if session and session.get("disconnected"):
                logger.info("Session marked as disconnected, stopping idle monitoring")
                return False

        # Update timeout for next retry
        next_timeout = get_timeout_for_retry(retry_count + 1)
        user_idle._timeout = next_timeout

        if retry_count == 1:
            # First time: Continue conversation naturally
            idle_message = {
                "role": "user",
                "content": "The user has gone silent for the first time. As Meher, continue the conversation naturally with a gentle follow-up related to what you were discussing. Keep it engaging, brief, and in character. Use Devanagari script for Hindi.",
            }
        elif retry_count == 2:
            # Second time: Change topic
            idle_message = {
                "role": "user",
                "content": "The user has gone silent again. As Meher, try changing the topic to something new and interesting. Maybe ask about their day, tease them playfully, or share something exciting. Keep it warm, engaging, and in character. Use Devanagari script for Hindi.",
            }
        elif retry_count == 3:
            # Third time: Check if they're still there
            idle_message = {
                "role": "user",
                "content": "The user has been silent multiple times. As Meher, gently check if they are still there and if everything is okay. Be caring, understanding, but maintain your bold personality. Use Devanagari script for Hindi.",
            }
        else:  # retry_count >= 4
            # Fourth time: Say goodbye and end call
            idle_message = {
                "role": "user",
                "content": "The user has been unresponsive for too long. As Meher, say a gentle goodbye message with your characteristic boldness and warmth, expressing that you hope to talk again soon. After this message, the call will end. Use Devanagari script for Hindi.",
            }

            # Send the goodbye message
            messages_for_llm = LLMMessagesAppendFrame([idle_message])
            await task.queue_frames([messages_for_llm, LLMRunFrame()])

            # Schedule call termination after goodbye message
            logger.info(
                "Maximum idle attempts reached. Terminating call after goodbye message."
            )
            terminate_task = asyncio.create_task(terminate_call_after_goodbye())
            terminate_task.add_done_callback(
                lambda t: logger.error(f"Terminate call task failed: {t.exception()}")
                if t.exception()
                else None
            )

            # Return False to stop idle monitoring
            return False

        # Send the appropriate message to LLM
        messages_for_llm = LLMMessagesAppendFrame([idle_message])
        await task.queue_frames([messages_for_llm, LLMRunFrame()])

        # Return True to continue monitoring for more idle events
        return True

    async def terminate_call(
        reason: str = "call terminated",
        save_conversation: bool = True,
        immediate: bool = False,
    ):
        """
        Unified call termination function following Pipecat best practices.
        Args:
            reason: Reason for termination
            save_conversation: Whether to save conversation to Firebase
            immediate: If True, cancel immediately; if False, graceful shutdown
        """
        termination_type = "Immediate" if immediate else "Graceful"
        logger.info(f"{termination_type} call termination initiated: {reason}")

        try:
            # Check if we have valid caller info
            if not caller_phone or caller_phone == "Unknown":
                logger.warning("No valid caller phone found for termination")
                return False

            # Get the active session
            session = active_sessions.get(caller_phone)
            if not session:
                logger.warning(f"No active session found for caller: {caller_phone}")
                return False

            # Mark session as disconnected to prevent further processing
            session["disconnected"] = True
            session["disconnected_at"] = datetime.now()
            logger.info(f"Marked session as disconnected for caller: {caller_phone}")

            # Save conversation to Firebase if requested
            if save_conversation:
                try:
                    llm_ctx = session.get("llm_context")
                    if llm_ctx:
                        conversation_messages = extract_conversation_messages(llm_ctx)
                        if conversation_messages:
                            await save_voice_messages_to_firebase_batch(
                                caller_phone,
                                conversation_messages,
                                call_sid,
                                character_name,
                            )
                            logger.info("Conversation saved to Firebase successfully")
                except Exception as e:
                    logger.error(f"Failed to save conversation to Firebase: {e}")

            # Terminate based on mode
            task = session.get("task")
            if task:
                if immediate:
                    # Cancel task immediately - pushes CancelFrame downstream
                    await task.cancel()
                    logger.info("Task cancelled for immediate termination")
                else:
                    # Queue EndFrame for graceful shutdown after pending frames
                    await task.queue_frames([EndFrame()])
                    logger.info("EndFrame queued for graceful termination")
            else:
                logger.warning("No task found in session to terminate")

            return True

        except Exception as e:
            logger.error(f"Error in {termination_type.lower()} call termination: {e}")
            return False

    async def terminate_call_after_goodbye():
        """Terminate the call after giving time for goodbye message to be spoken"""
        await terminate_call(
            "user inactivity timeout", save_conversation=True, immediate=False
        )

    # Use the retry callback pattern - UserIdleProcessor handles reset automatically
    user_idle = UserIdleProcessor(
        callback=handle_user_idle_with_retry,  # Uses retry callback signature
        timeout=get_timeout_for_retry(1),  # Start with first timeout value (15.0)
    )

    pipeline_steps = [
        transport.input(),  # caller audio -> PCM via serializer
        stt,  # speech -> text
        user_idle,  # UserIdleProcessor automatically resets on UserStartedSpeakingFrame
        agg.user(),  # add user text to context
        llm,  # text -> text
        tts,  # text -> speech
        transport.output(),  # PCM -> back to Twilio via serializer
        agg.assistant(),  # keep assistant turns
    ]

    pipeline = Pipeline(pipeline_steps)

    task = PipelineTask(
        pipeline,
        params=PipelineParams(
            audio_in_sample_rate=16000,  # Silero VAD compatible sample rate
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
                "llm_context": ctx,  # Store the LLM context for message access
                "disconnected": False,  # Flag to track disconnection
            }
            logger.info(f"Registered active session for {caller_phone}")
    except Exception:
        pass

    @transport.event_handler("on_client_connected")
    async def _greet(_t, _c):
        logger.info("Client connected, sending greeting...")
        logger.info(
            "Audio quality settings - Sample rate: 16kHz, Bit depth: 16-bit, Channels: Mono"
        )
        logger.info("TTS model: eleven_multilingual_v2, STT model: nova-3-general")
        await task.queue_frames([LLMRunFrame()])  # gentle hello

    @transport.event_handler("on_audio_frame")
    async def _on_audio_frame(transport, frame):
        """Monitor audio frame quality"""
        if hasattr(frame, "audio") and frame.audio:
            audio_data = frame.audio
            # Log audio quality metrics periodically
            if hasattr(transport, "_frame_count"):
                transport._frame_count += 1
            else:
                transport._frame_count = 1

            # Log every 100 frames (roughly every 2-3 seconds)
            if transport._frame_count % 100 == 0:
                logger.info(
                    f"Audio quality check - Frame {transport._frame_count}, "
                    f"Data length: {len(audio_data) if audio_data else 0}"
                )

    @transport.event_handler("on_tts_start")
    async def _on_tts_start(transport, frame):
        """Log when TTS starts"""
        logger.info("TTS audio generation started")

    @transport.event_handler("on_tts_end")
    async def _on_tts_end(transport, frame):
        """Log when TTS ends"""
        logger.info("TTS audio generation completed")

    @transport.event_handler("on_client_disconnected")
    async def _on_client_disconnected(transport, client):
        """Handle client disconnection - save conversation and cleanup"""
        logger.info("Client disconnected, saving conversation and cleaning up")

        # Use immediate termination for client disconnect (connection already closed)
        if caller_phone and caller_phone != "Unknown":
            await terminate_call(
                "client disconnected", save_conversation=True, immediate=True
            )
        else:
            logger.warning("No valid caller phone found for client disconnect cleanup")

    # Enhanced pipeline runner with better audio processing
    runner = PipelineRunner(
        handle_sigint=False,
        force_gc=True,
    )

    # Add proper cleanup handlers following Pipecat best practices
    async def cleanup_on_pipeline_end():
        """Clean up resources when pipeline ends naturally"""
        try:
            if caller_phone and caller_phone != "Unknown":
                # Clean up stored caller info
                if call_sid and call_sid in caller_info_storage:
                    del caller_info_storage[call_sid]
                    logger.info(
                        f"Cleaned up caller info for completed call: {call_sid}"
                    )

                # Clean up active session
                if caller_phone in active_sessions:
                    del active_sessions[caller_phone]
                    logger.info(
                        f"Deregistered active session for completed call: {caller_phone}"
                    )

                logger.info(f"Pipeline cleanup completed for caller: {caller_phone}")
        except Exception as e:
            logger.error(f"Error in pipeline cleanup: {e}")

    try:
        await runner.run(task)
    except Exception as e:
        logger.error(f"Pipeline error: {e}")
        # Ensure cleanup happens even on errors
        await cleanup_on_pipeline_end()
    finally:
        # Always clean up when pipeline ends
        await cleanup_on_pipeline_end()


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
        call_sid = data.get("start", {}).get("callSid")

        # Try to get caller info from WebSocket start message first
        caller_number = data.get("start", {}).get("from")
        caller_name = data.get("start", {}).get("callerName")

        # If not available in WebSocket message, retrieve from stored info
        if call_sid and call_sid in caller_info_storage:
            stored_info = caller_info_storage[call_sid]
            caller_name = caller_name or stored_info.get("caller_name", "Unknown")
            caller_number = caller_number or stored_info.get("caller_number", "Unknown")
            logger.info(
                f"Retrieved caller info from storage - Name: {caller_name}, Number: {caller_number}"
            )
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

        logger.info(
            f"WebSocket connection - Caller: {caller_name} ({caller_number}), Stream: {stream_sid}, Call: {call_sid}"
        )
        await _run_call(
            websocket, stream_sid, call_sid, caller_number, caller_name, "meher"
        )

    except WebSocketDisconnect:
        logger.info("Twilio WebSocket disconnected")

        # Clean up on WebSocket disconnect (connection already closed)
        try:
            if "caller_number" in locals() and caller_number:
                caller_phone = extract_phone_number(caller_number)
                if caller_phone and caller_phone != "Unknown":
                    # Use our unified termination function for immediate cleanup
                    # Note: We can't call terminate_call directly due to scope, so do minimal cleanup
                    session = active_sessions.get(caller_phone)
                    if session:
                        # Mark as disconnected and save conversation asynchronously
                        session["disconnected"] = True
                        session["disconnected_at"] = datetime.now()
                        llm_ctx = session.get("llm_context")
                        if llm_ctx:
                            conversation_messages = extract_conversation_messages(
                                llm_ctx
                            )
                            if conversation_messages:
                                # Create task with proper error handling
                                task = asyncio.create_task(
                                    save_voice_messages_to_firebase_batch(
                                        caller_phone,
                                        conversation_messages,
                                        call_sid,
                                        "meher",
                                    )
                                )
                                task.add_done_callback(
                                    lambda t: logger.error(
                                        f"Firebase save task failed: {t.exception()}"
                                    )
                                    if t.exception()
                                    else None
                                )
                        # Clean up session
                        del active_sessions[caller_phone]
                        logger.info(
                            f"WebSocket disconnect cleanup completed for {caller_phone}"
                        )
        except Exception as e:
            logger.error(f"Error in WebSocket disconnect cleanup: {e}")
    except Exception as e:
        logger.exception("WS error: %s", e)
        # Clean up stored caller info on error
        if "call_sid" in locals() and call_sid and call_sid in caller_info_storage:
            del caller_info_storage[call_sid]
            logger.info(f"Cleaned up caller info on error for CallSid: {call_sid}")
        try:
            await websocket.close(code=1011)
        except Exception:
            pass
