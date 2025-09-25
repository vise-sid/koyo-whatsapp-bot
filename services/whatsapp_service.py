"""
WhatsApp messaging service for sending messages via Twilio.

This module provides functionality to send WhatsApp messages using Twilio's
messaging API with approved templates and freeform messages.
"""

import json
import logging
from typing import Dict, Any
from twilio.rest import Client


class WhatsAppMessagingService:
    """Service for sending WhatsApp messages via Twilio"""
    
    def __init__(self, account_sid: str, auth_token: str, from_number: str):
        """
        Initialize the WhatsApp messaging service.
        
        Args:
            account_sid: Twilio account SID
            auth_token: Twilio auth token
            from_number: WhatsApp number to send from (with country code)
        """
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
        """
        Send a freeform WhatsApp message (no template/content sid).

        Note: For inbound user-initiated sessions within the 24-hour window, 
        freeform messages are allowed.
        
        Args:
            to_number: The recipient's phone number
            message: The message content to send
            
        Returns:
            Dict containing the result of the message sending operation
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
