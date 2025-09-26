"""
Voice session manager for handling Twilio voice calls with Pipecat.

This module manages voice call sessions, including pipeline setup, 
function handlers, and session lifecycle management.
"""

import asyncio
import logging
from datetime import datetime
from typing import Dict, Any, Optional

from pipecat.serializers.twilio import TwilioFrameSerializer
from pipecat.transports.websocket.fastapi import (
    FastAPIWebsocketTransport, FastAPIWebsocketParams
)
from pipecat.audio.vad.silero import SileroVADAnalyzer
from pipecat.pipeline.pipeline import Pipeline
from pipecat.pipeline.task import PipelineTask, PipelineParams
from pipecat.pipeline.runner import PipelineRunner
from pipecat.frames.frames import LLMRunFrame, EndFrame, LLMMessagesAppendFrame
from pipecat.processors.frame_processor import FrameDirection
from pipecat.processors.user_idle_processor import UserIdleProcessor
from pipecat.services.deepgram.stt import DeepgramSTTService, LiveOptions
from pipecat.services.openai.llm import OpenAILLMService
from pipecat.services.elevenlabs.tts import ElevenLabsTTSService
from pipecat.processors.aggregators.openai_llm_context import OpenAILLMContext
from pipecat.transcriptions.language import Language
from pipecat.adapters.schemas.function_schema import FunctionSchema
from pipecat.adapters.schemas.tools_schema import ToolsSchema
from pipecat.services.llm_service import FunctionCallParams

from utils.helpers import get_timeout_for_retry
from database.firebase_service import firebase_service
from services.whatsapp_service import WhatsAppMessagingService


class VoiceSessionManager:
    """Manages voice call sessions with Pipecat pipeline"""
    
    def __init__(
        self,
        twilio_account_sid: str,
        twilio_auth_token: str,
        openai_api_key: str,
        deepgram_api_key: str,
        elevenlabs_api_key: str,
        elevenlabs_voice_id: str,
        whatsapp_from_number: str
    ):
        """
        Initialize voice session manager.
        
        Args:
            twilio_account_sid: Twilio account SID
            twilio_auth_token: Twilio auth token
            openai_api_key: OpenAI API key
            deepgram_api_key: Deepgram API key
            elevenlabs_api_key: ElevenLabs API key
            elevenlabs_voice_id: ElevenLabs voice ID
            whatsapp_from_number: WhatsApp from number
        """
        self.twilio_account_sid = twilio_account_sid
        self.twilio_auth_token = twilio_auth_token
        self.openai_api_key = openai_api_key
        self.deepgram_api_key = deepgram_api_key
        self.elevenlabs_api_key = elevenlabs_api_key
        self.elevenlabs_voice_id = elevenlabs_voice_id
        self.whatsapp_from_number = whatsapp_from_number
        self.logger = logging.getLogger(__name__)
    
    async def run_call(
        self,
        websocket,
        stream_sid: str,
        call_sid: Optional[str],
        caller_number: Optional[str] = None,
        caller_name: Optional[str] = None,
        active_sessions: Dict[str, Any] = None,
        caller_info_storage: Dict[str, Any] = None
    ):
        """
        Run a voice call session with Pipecat pipeline.
        
        Args:
            websocket: WebSocket connection
            stream_sid: Stream SID from Twilio
            call_sid: Call SID from Twilio
            caller_number: Caller's phone number
            caller_name: Caller's name
            active_sessions: Dictionary of active sessions
            caller_info_storage: Dictionary of caller info storage
        """
        if active_sessions is None:
            active_sessions = {}
        if caller_info_storage is None:
            caller_info_storage = {}
        
        # Initialize serializer and transport
        serializer = TwilioFrameSerializer(
            stream_sid=stream_sid,
            call_sid=call_sid,
            account_sid=self.twilio_account_sid or None,
            auth_token=self.twilio_auth_token or None,
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

        # Initialize services
        stt = DeepgramSTTService(
            api_key=self.deepgram_api_key,
            live_options=LiveOptions(
                model="nova-3-general",
                language="multi",
                smart_format=True,
                interim_results=True,
                encoding="linear16",
                channels=1,
            )
        )
        
        whatsapp_service = WhatsAppMessagingService(
            account_sid=self.twilio_account_sid,
            auth_token=self.twilio_auth_token,
            from_number=self.whatsapp_from_number
        )
        
        # Setup function tools
        tools = self._setup_function_tools()
        
        # Initialize LLM service
        llm = OpenAILLMService(
            api_key=self.openai_api_key, 
            model="gpt-4o",
            params=OpenAILLMService.InputParams(
                temperature=0.7,
            )
        )
        
        # Register function handlers
        self._register_function_handlers(
            llm, whatsapp_service, caller_number, caller_name, 
            call_sid, active_sessions, caller_info_storage
        )
        
        tts = ElevenLabsTTSService(
            api_key=self.elevenlabs_api_key,
            voice_id=self.elevenlabs_voice_id,
            model="eleven_flash_v2_5",
            input_params=ElevenLabsTTSService.InputParams(
                language=Language.HI,
                stability=0.5,
                similarity_boost=0.9,
                style=0.5,
                use_speaker_boost=False,
                speed=0.9,
                auto_mode=True,
            )
        )

        # Setup system prompt and context
        from utils.helpers import get_current_time_context
        from prompts.meher_voice_prompt import get_voice_system_prompt
        
        time_context = get_current_time_context()
        caller_phone = caller_number.replace("whatsapp:", "") if caller_number else "Unknown"
        caller_display_name = caller_name or "Unknown"
        
        system_prompt = get_voice_system_prompt(caller_display_name, caller_phone, time_context)

        ctx = OpenAILLMContext(
            messages=[{"role":"system","content":system_prompt}],
            tools=tools
        )
        agg = llm.create_context_aggregator(ctx)

        # Setup user idle handling
        user_idle = UserIdleProcessor(
            callback=lambda retry_count: self._handle_user_idle_with_retry(
                retry_count, caller_phone, active_sessions
            ),
            timeout=get_timeout_for_retry(1)
        )
        
        # Setup pipeline
        pipeline_steps = [
            transport.input(),
            stt,
            user_idle,
            agg.user(),
            llm,
            tts,
            transport.output(),
            agg.assistant(),
        ]

        pipeline = Pipeline(pipeline_steps)

        task = PipelineTask(
            pipeline,
            params=PipelineParams(
                audio_in_sample_rate=16000,
                audio_out_sample_rate=16000,
                allow_interruptions=True,
            ),
            idle_timeout_secs=600,
        )

        # Register active session
        if caller_phone and caller_phone != "Unknown":
            active_sessions[caller_phone] = {
                "call_sid": call_sid,
                "task": task,
                "transport": transport,
                "display_name": caller_display_name,
                "llm_context": ctx,
                "disconnected": False,
            }
            self.logger.info(f"Registered active session for {caller_phone}")

        # Setup event handlers
        self._setup_event_handlers(transport, task, caller_phone, call_sid, active_sessions, caller_info_storage)

        # Run pipeline
        runner = PipelineRunner(
            handle_sigint=False, 
            force_gc=True,
        )
        
        try:
            await runner.run(task)
        except Exception as e:
            self.logger.error(f"Pipeline error: {e}")
            await self._cleanup_on_pipeline_end(caller_phone, call_sid, active_sessions, caller_info_storage)
        finally:
            await self._cleanup_on_pipeline_end(caller_phone, call_sid, active_sessions, caller_info_storage)
    
    def _setup_function_tools(self) -> ToolsSchema:
        """Setup function tools for the LLM"""
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
        
        terminate_call_function = FunctionSchema(
            name="terminate_voice_call",
            description="Terminate the current voice call. Use this when the conversation has naturally concluded, when the user says goodbye, or when it's appropriate to end the call. Always say goodbye before terminating.",
            properties={
                "reason": {
                    "type": "string",
                    "description": "Brief reason for ending the call (e.g., 'conversation concluded', 'user said goodbye', 'call completed successfully')"
                }
            },
            required=["reason"]
        )
        
        return ToolsSchema(standard_tools=[whatsapp_function, terminate_call_function])
    
    def _register_function_handlers(
        self,
        llm: OpenAILLMService,
        whatsapp_service: WhatsAppMessagingService,
        caller_number: str,
        caller_name: str,
        call_sid: str,
        active_sessions: Dict[str, Any],
        caller_info_storage: Dict[str, Any]
    ):
        """Register function handlers with the LLM service"""
        
        async def send_whatsapp_message_handler(params: FunctionCallParams):
            """Handle WhatsApp message sending function calls"""
            try:
                to_number = params.arguments.get("to_number", caller_number)
                message = params.arguments.get("message", "")
                
                if not message:
                    await params.result_callback("❌ No message content provided")
                    return
                
                result = await whatsapp_service.send_message(
                    to_number=to_number, 
                    message=message, 
                    recipient_name=caller_name or "User"
                )
                
                if result["success"]:
                    response = f"✅ WhatsApp message sent successfully to {to_number}"
                else:
                    response = f"❌ Failed to send WhatsApp message: {result.get('error', 'Unknown error')}"
                
                await params.result_callback(response)
                
            except Exception as e:
                self.logger.error(f"Error in WhatsApp function handler: {str(e)}")
                await params.result_callback(f"❌ Error sending WhatsApp message: {str(e)}")
        
        async def terminate_call_handler(params: FunctionCallParams):
            """Handle voice call termination function calls using EndTaskFrame"""
            try:
                reason = params.arguments.get("reason", "agent initiated termination")
                self.logger.info(f"Agent requested to terminate call: {reason}")
                
                # Save conversation before terminating
                try:
                    caller_phone = caller_number.replace("whatsapp:", "") if caller_number else "Unknown"
                    if caller_phone != "Unknown" and caller_phone in active_sessions:
                        session = active_sessions[caller_phone]
                        llm_ctx = session.get("llm_context")
                        if llm_ctx:
                            conversation_messages = firebase_service.extract_conversation_messages(llm_ctx)
                            if conversation_messages:
                                await firebase_service.save_voice_messages_to_firebase_batch(
                                    caller_phone, conversation_messages, call_sid
                                )
                                self.logger.info("Conversation saved to Firebase before termination")
                except Exception as e:
                    self.logger.error(f"Failed to save conversation before termination: {e}")
                
                # Use simple EndTaskFrame approach
                from pipecat.frames.frames import EndTaskFrame, TTSSpeakFrame
                from pipecat.processors.frame_processor import FrameDirection
                
                # Say goodbye first
                await params.llm.push_frame(TTSSpeakFrame("Have a nice day! Talk to you soon!"))
                
                # Signal that the task should end after processing this frame
                await params.llm.push_frame(EndTaskFrame(), FrameDirection.UPSTREAM)
                
                await params.result_callback(f"✅ Call termination initiated: {reason}")
                
            except Exception as e:
                self.logger.error(f"Error in terminate call handler: {str(e)}")
                await params.result_callback(f"❌ Error terminating call: {str(e)}")
        
        llm.register_function("send_whatsapp_message", send_whatsapp_message_handler)
        llm.register_function("terminate_voice_call", terminate_call_handler)
    
    async def _handle_user_idle_with_retry(
        self, 
        retry_count: int, 
        caller_phone: str, 
        active_sessions: Dict[str, Any]
    ) -> bool:
        """Handle user idle with structured follow-up based on retry count"""
        self.logger.info(f"User idle - attempt #{retry_count}")
        
        session = active_sessions.get(caller_phone)
        if not session:
            return False
        
        # Check if session is disconnected
        if session.get("disconnected"):
            return False
        
        # Create appropriate idle message based on retry count
        if retry_count == 1:
            idle_message = {
                "role": "user",
                "content": "The user has gone silent for the first time. As Meher, continue the conversation naturally with a gentle follow-up related to what you were discussing. Keep it engaging, brief, and in character. Use Devanagari script for Hindi."
            }
        elif retry_count == 2:
            idle_message = {
                "role": "user", 
                "content": "The user has gone silent again. As Meher, try changing the topic to something new and interesting. Maybe ask about their day, tease them playfully, or share something exciting. Keep it warm, engaging, and in character. Use Devanagari script for Hindi."
            }
        elif retry_count == 3:
            idle_message = {
                "role": "user",
                "content": "The user has been silent multiple times. As Meher, gently check if they are still there and if everything is okay. Be caring, understanding, but maintain your bold personality. Use Devanagari script for Hindi."
            }
        else:  # retry_count >= 4
            # Use simple EndTaskFrame approach for idle timeout
            task = session.get("task")
            if task:
                try:
                    # Save conversation before terminating
                    caller_phone = caller_number.replace("whatsapp:", "") if caller_number else "Unknown"
                    if caller_phone != "Unknown" and caller_phone in active_sessions:
                        session = active_sessions[caller_phone]
                        llm_ctx = session.get("llm_context")
                        if llm_ctx:
                            conversation_messages = firebase_service.extract_conversation_messages(llm_ctx)
                            if conversation_messages:
                                await firebase_service.save_voice_messages_to_firebase_batch(
                                    caller_phone, conversation_messages, session.get("call_sid")
                                )
                                self.logger.info("Conversation saved to Firebase before idle timeout termination")
                except Exception as e:
                    self.logger.error(f"Failed to save conversation before idle timeout: {e}")
                
                # Use EndTaskFrame for clean termination
                from pipecat.frames.frames import EndTaskFrame, TTSSpeakFrame
                from pipecat.processors.frame_processor import FrameDirection
                
                # Say goodbye first
                await task.queue_frames([TTSSpeakFrame("Thanks for the chat! Talk to you soon!")])
                
                # Signal that the task should end
                await task.queue_frames([EndTaskFrame()])
            
            return False
        
        # Send the appropriate message to LLM
        task = session.get("task")
        if task:
            messages_for_llm = LLMMessagesAppendFrame([idle_message])
            await task.queue_frames([messages_for_llm, LLMRunFrame()])
        
        return True
    
    async def _terminate_call(
        self,
        reason: str,
        caller_phone: str,
        call_sid: str,
        active_sessions: Dict[str, Any],
        caller_info_storage: Dict[str, Any],
        save_conversation: bool = True,
        immediate: bool = False
    ) -> bool:
        """Unified call termination function"""
        self.logger.info(f"Call termination initiated: {reason}")
        self.logger.info(f"Caller phone: {caller_phone}, Call SID: {call_sid}")
        self.logger.info(f"Active sessions keys: {list(active_sessions.keys())}")
        
        try:
            if not caller_phone or caller_phone == "Unknown":
                self.logger.warning(f"Invalid caller phone: {caller_phone}")
                return False
            
            session = active_sessions.get(caller_phone)
            if not session:
                self.logger.warning(f"No active session found for caller: {caller_phone}")
                return False
            
            # Mark session as disconnected
            session["disconnected"] = True
            session["disconnected_at"] = datetime.now()
            
            # Save conversation if requested
            if save_conversation:
                try:
                    llm_ctx = session.get("llm_context")
                    if llm_ctx:
                        conversation_messages = firebase_service.extract_conversation_messages(llm_ctx)
                        if conversation_messages:
                            await firebase_service.save_voice_messages_to_firebase_batch(
                                caller_phone, conversation_messages, call_sid
                            )
                except Exception as e:
                    self.logger.error(f"Failed to save conversation: {e}")
            
            # Terminate task
            task = session.get("task")
            if task:
                if immediate:
                    await task.cancel()
                else:
                    await task.queue_frames([EndFrame()])
            
            return True
            
        except Exception as e:
            self.logger.error(f"Error in call termination: {e}")
            return False
    
    async def _terminate_call_after_goodbye(
        self,
        caller_phone: str,
        call_sid: str,
        active_sessions: Dict[str, Any],
        caller_info_storage: Dict[str, Any]
    ):
        """Terminate call after giving time for goodbye message"""
        await self._terminate_call(
            "user inactivity timeout", caller_phone, call_sid, 
            active_sessions, caller_info_storage, 
            save_conversation=True, immediate=False
        )
    
    def _setup_event_handlers(
        self,
        transport: FastAPIWebsocketTransport,
        task: PipelineTask,
        caller_phone: str,
        call_sid: str,
        active_sessions: Dict[str, Any],
        caller_info_storage: Dict[str, Any]
    ):
        """Setup transport event handlers"""
        
        @transport.event_handler("on_client_connected")
        async def _greet(_t, _c):
            self.logger.info("Client connected, sending greeting...")
            await task.queue_frames([LLMRunFrame()])
        
        @transport.event_handler("on_client_disconnected")
        async def _on_client_disconnected(transport, client):
            self.logger.info("Client disconnected, saving conversation and cleaning up")
            if caller_phone and caller_phone != "Unknown":
                await self._terminate_call(
                    "client disconnected", caller_phone, call_sid,
                    active_sessions, caller_info_storage,
                    save_conversation=True, immediate=True
                )
    
    async def _cleanup_on_pipeline_end(
        self,
        caller_phone: str,
        call_sid: str,
        active_sessions: Dict[str, Any],
        caller_info_storage: Dict[str, Any]
    ):
        """Clean up resources when pipeline ends"""
        try:
            if caller_phone and caller_phone != "Unknown":
                # Clean up stored caller info
                if call_sid and call_sid in caller_info_storage:
                    del caller_info_storage[call_sid]
                
                # Clean up active session
                if caller_phone in active_sessions:
                    del active_sessions[caller_phone]
                
                self.logger.info(f"Pipeline cleanup completed for caller: {caller_phone}")
        except Exception as e:
            self.logger.error(f"Error in pipeline cleanup: {e}")
