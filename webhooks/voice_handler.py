"""
Voice webhook handler for processing Twilio voice calls.

This module handles Twilio voice webhooks and manages voice call sessions
with proper cleanup and session management.
"""

import logging
from datetime import datetime
from typing import Dict, Any, Optional

from fastapi import Request
from fastapi.responses import PlainTextResponse

from utils.helpers import validate_twilio_http, twiml, ws_url


class VoiceHandler:
    """Handler for Twilio voice webhook requests"""
    
    def __init__(self, twilio_auth_token: str, validate_signature: bool = True):
        """
        Initialize voice handler.
        
        Args:
            twilio_auth_token: Twilio auth token
            validate_signature: Whether to validate Twilio signatures
        """
        self.twilio_auth_token = twilio_auth_token
        self.validate_signature = validate_signature
        self.logger = logging.getLogger(__name__)
    
    async def handle_voice_webhook(
        self, 
        request: Request, 
        caller_info_storage: Dict[str, Any]
    ) -> PlainTextResponse:
        """
        Handle Twilio voice webhook and return TwiML response.
        
        Args:
            request: FastAPI request object
            caller_info_storage: Dictionary to store caller information
            
        Returns:
            PlainTextResponse with TwiML content
        """
        body = await request.body()
        if not validate_twilio_http(request, body, self.twilio_auth_token, self.validate_signature):
            self.logger.warning("Twilio signature validation failed")
            return PlainTextResponse(content="", status_code=403)

        # Extract caller information from webhook
        form = await request.form()
        from_num = form.get("From")
        to_num = form.get("To")
        caller_name = form.get("CallerName", "Unknown")
        call_sid = form.get("CallSid")
        
        self.logger.info(f"Voice webhook - From: {from_num}, To: {to_num}, Name: {caller_name}, CallSid: {call_sid}")

        # Store caller info for later use in the WebSocket connection
        if call_sid:
            caller_info_storage[call_sid] = {
                "caller_name": caller_name,
                "caller_number": from_num,
                "to_number": to_num,
                "timestamp": datetime.now()
            }
            self.logger.info(f"Stored caller info for CallSid: {call_sid} - Name: {caller_name}, Number: {from_num}")
        
        twiml_response = twiml(ws_url(request))
        return PlainTextResponse(content=twiml_response, media_type="application/xml")
