"""
Utility helper functions for the application.

This module contains various utility functions used throughout the application
including time context, phone number extraction, and media processing.
"""

import logging
from datetime import datetime
from typing import Dict, Any, List
from urllib.parse import parse_qs

import httpx
import pytz
from twilio.request_validator import RequestValidator
from openai import OpenAI


def get_current_time_context() -> Dict[str, Any]:
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


def validate_twilio_http(request, body: bytes, auth_token: str, validate_signature: bool = True) -> bool:
    """Validate Twilio HTTP request signature"""
    if not validate_signature:
        return True
    validator = RequestValidator(auth_token)
    sig = request.headers.get("X-Twilio-Signature", "") or request.headers.get("x-twilio-signature", "")
    url = str(request.url)
    try:
        params = {k: v[0] for k, v in parse_qs(body.decode()).items()}
    except Exception:
        params = {}
    return validator.validate(url, params, sig)


async def fetch_with_twilio_auth(url: str, account_sid: str, auth_token: str) -> bytes:
    """Fetch content from URL with Twilio authentication"""
    async with httpx.AsyncClient(timeout=20) as client:
        r = await client.get(url, auth=(account_sid, auth_token))
        r.raise_for_status()
        return r.content


async def transcribe_twilio_audio(url: str, account_sid: str, auth_token: str, openai_api_key: str) -> str:
    """Transcribe audio from Twilio URL using OpenAI Whisper"""
    try:
        audio_bytes = await fetch_with_twilio_auth(url, account_sid, auth_token)
        client = OpenAI(api_key=openai_api_key)
        # OpenAI whisper transcription
        from io import BytesIO
        file_like = BytesIO(audio_bytes)
        file_like.name = "audio.ogg"
        tr = client.audio.transcriptions.create(model="whisper-1", file=file_like)
        return (tr.text or "").strip()
    except Exception as e:
        logging.getLogger(__name__).error(f"Audio transcription failed: {e}")
        return ""


async def caption_image_url(image_url: str, openai_api_key: str) -> str:
    """Generate caption for image URL using OpenAI Vision"""
    try:
        client = OpenAI(api_key=openai_api_key)
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
        logging.getLogger(__name__).error(f"Image captioning failed: {e}")
        return ""


async def fetch_text_excerpt(url: str, account_sid: str, auth_token: str, max_chars: int = 800) -> str:
    """Fetch and excerpt text from URL"""
    try:
        data = await fetch_with_twilio_auth(url, account_sid, auth_token)
        text = data.decode("utf-8", errors="ignore")
        text = " ".join(text.split())
        return text[:max_chars]
    except Exception as e:
        logging.getLogger(__name__).error(f"Text fetch failed: {e}")
        return ""


def get_timeout_for_retry(retry_count: int) -> float:
    """Return timeout value based on retry count"""
    timeout_values = [10.0, 8.0, 4.0, 3.0]
    return timeout_values[min(retry_count - 1, len(timeout_values) - 1)]


def ws_url(request) -> str:
    """Generate WebSocket URL from request"""
    host = request.headers.get("host")
    return f"wss://{host}/ws"  # Render gives you https, so wss here works


def twiml(ws_url: str) -> str:
    """Generate TwiML response for voice calls"""
    # Bidirectional stream; Pause just keeps call alive if the WS ends early
    return f"""<?xml version="1.0" encoding="UTF-8"?>
<Response>
  <Connect>
    <Stream url="{ws_url}"/>
  </Connect>
  <Pause length="40"/>
</Response>"""
