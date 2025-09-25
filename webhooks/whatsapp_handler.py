"""
WhatsApp webhook handler for processing incoming messages.

This module handles incoming WhatsApp messages, processes media content,
and manages both active voice sessions and off-call text conversations.
"""

import logging
from typing import List, Dict, Any, Optional

from fastapi import Request, Response
from openai import OpenAI

from utils.helpers import (
    validate_twilio_http, 
    transcribe_twilio_audio, 
    caption_image_url, 
    fetch_text_excerpt,
    get_current_time_context
)
from prompts.meher_text_prompt import get_text_system_prompt
from database.firebase_service import firebase_service


class WhatsAppHandler:
    """Handler for WhatsApp webhook requests"""
    
    def __init__(
        self, 
        twilio_account_sid: str, 
        twilio_auth_token: str, 
        openai_api_key: str,
        validate_signature: bool = True
    ):
        """
        Initialize WhatsApp handler.
        
        Args:
            twilio_account_sid: Twilio account SID
            twilio_auth_token: Twilio auth token  
            openai_api_key: OpenAI API key
            validate_signature: Whether to validate Twilio signatures
        """
        self.twilio_account_sid = twilio_account_sid
        self.twilio_auth_token = twilio_auth_token
        self.openai_api_key = openai_api_key
        self.validate_signature = validate_signature
        self.logger = logging.getLogger(__name__)
    
    async def handle_webhook(self, request: Request, active_sessions: Dict[str, Any]) -> Response:
        """
        Handle inbound WhatsApp messages.
        
        Behavior:
        - If there's an active voice session for the sender, inject the text/media summary into the live LLM context
        - Otherwise, run a lightweight off-call multimodal handler and reply via WhatsApp
        
        Args:
            request: FastAPI request object
            active_sessions: Dictionary of active voice sessions
            
        Returns:
            Response object
        """
        body_bytes = await request.body()
        if not validate_twilio_http(request, body_bytes, self.twilio_auth_token, self.validate_signature):
            self.logger.warning("Twilio signature validation failed for WhatsApp webhook")
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

            self.logger.info(f"Incoming WhatsApp from {from_num} | text='{text_body}' | media_count={num_media}")
            session = active_sessions.get(from_num)

            if session:
                # Inject into ongoing call context with media enrichment
                content = await self._process_media_content(text_body, media_items, from_num, is_in_call=True)
                
                # Import here to avoid circular imports
                from pipecat.frames.frames import LLMMessagesAppendFrame, LLMRunFrame
                
                frames = [
                    LLMMessagesAppendFrame([{ "role": "user", "content": f"(WhatsApp) {content}" }]),
                    LLMRunFrame(),
                ]
                await session["task"].queue_frames(frames)
                
                # Save user message to Firebase
                await firebase_service.save_message_to_firebase(
                    user_id=from_num,
                    sender="user",
                    content=content,
                    conversation_type="text"
                )
                
                return Response(status_code=204)

            # No live session -> off-call LLM chat with context and reply over WhatsApp (freeform)
            reply_text = await self._handle_multimodal_offcall(text_body, media_items, from_num)
            
            # Send reply via WhatsApp (this would need to be injected as dependency)
            # For now, we'll return the reply text to be sent by the main handler
            
            # Save both user message and character response to Firebase
            await firebase_service.save_message_to_firebase(
                user_id=from_num,
                sender="user",
                content=text_body,
                conversation_type="text"
            )
            await firebase_service.save_message_to_firebase(
                user_id=from_num,
                sender="character",
                content=reply_text,
                conversation_type="text"
            )
            
            # Return the reply text for the main handler to send
            return {"reply_text": reply_text, "status": 200}

        except Exception as e:
            self.logger.exception("WhatsApp webhook error: %s", e)
            return Response(status_code=500)
    
    async def _process_media_content(
        self, 
        text_body: str, 
        media_items: List[Dict[str, str]], 
        from_num: str,
        is_in_call: bool = False
    ) -> str:
        """Process media content and return enriched text"""
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
                    transcript = await transcribe_twilio_audio(
                        url, self.twilio_account_sid, self.twilio_auth_token, self.openai_api_key
                    )
                    parts.append(f"[audio transcript] {transcript}" if transcript else "[audio note received]")
                elif "image/" in ctype:
                    caption = await caption_image_url(url, self.openai_api_key)
                    parts.append(f"[image] {caption}" if caption else "[image received]")
                elif "text/" in ctype:
                    excerpt = await fetch_text_excerpt(
                        url, self.twilio_account_sid, self.twilio_auth_token, max_chars=500 if is_in_call else 800
                    )
                    parts.append(f"[text doc excerpt] {excerpt}" if excerpt else "[text document received]")
                elif "application/pdf" in ctype or ctype.startswith("application/"):
                    parts.append("[document received]")
                else:
                    parts.append("[media received]")
            except Exception as me:
                context = "in-call" if is_in_call else "off-call"
                self.logger.error(f"Media handling error ({context}) for {url}: {me}")
                parts.append("[media received]")

        return (" ".join(parts)).strip() or "[empty message]"
    
    async def _handle_multimodal_offcall(self, text: str, media: List[Dict[str, str]], user_phone: str) -> str:
        """
        Off-call handler: build/maintain chat context and generate a Meher-style reply via LLM.

        - Maintains per-user message history in offcall_context[user_phone]
        - Supports text + media summaries
        - Generates a concise Hinglish reply (≤ ~75 tokens)
        
        Args:
            text: User's text message
            media: List of media items
            user_phone: User's phone number
            
        Returns:
            Generated reply text
        """
        # Import here to avoid circular imports
        # Note: offcall_context should be passed as parameter in real implementation
        from app_clean import offcall_context  # Temporary fix
        
        # 1) Build user turn with media summary
        user_text = await self._process_media_content(text, media, user_phone, is_in_call=False)

        # 2) Prepare context store for user
        history = offcall_context.get(user_phone) or []

        # 3) System prompt for off-call chat: use dedicated texting prompt module
        time_context = get_current_time_context()
        system_text = get_text_system_prompt(
            caller_display_name=user_phone,
            caller_phone=user_phone,
            time_context=time_context,
        )

        # 4) Build messages array (bounded history)
        messages = [{"role": "system", "content": system_text}]
        # include last ~8 messages from history for context
        for m in history[-8:]:
            messages.append(m)
        messages.append({"role": "user", "content": user_text})

        # 5) Call OpenAI for a short response
        try:
            client = OpenAI(api_key=self.openai_api_key)
            resp = client.chat.completions.create(
                model="gpt-4o",
                messages=messages,
                max_tokens=120,
                temperature=0.7,
            )
            reply = (resp.choices[0].message.content or "")[:800]
        except Exception as e:
            self.logger.error(f"Off-call LLM error: {e}")
            reply = "थोड़ी technical दिक्कत हो गयी मेरी तरफ—एक छोटा सा message फिर भेजो, मैं तुरंत जवाब दूँगी।"

        # 6) Update history
        history.append({"role": "user", "content": user_text})
        history.append({"role": "assistant", "content": reply})
        offcall_context[user_phone] = history[-20:]  # cap to last 20 turns

        return reply
