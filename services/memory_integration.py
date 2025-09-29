"""
Memory Integration Service for voice and WhatsApp services.

This service provides easy integration of memory capabilities
into your existing voice and WhatsApp services.
"""

import asyncio
import logging
from typing import List, Dict, Any, Optional
from services.batch_memory_processor import BatchMemoryProcessor
import os

logger = logging.getLogger(__name__)


class MemoryIntegration:
    """
    Memory integration service that can be used by voice and WhatsApp services
    to provide context-aware responses.
    """
    
    def __init__(self):
        self.processor = None
        self.initialized = False
    
    async def initialize(self):
        """Initialize the memory integration service"""
        if not self.initialized:
            self.processor = BatchMemoryProcessor(
                openai_api_key=os.getenv("OPENAI_API_KEY", ""),
                qdrant_settings={
                    "QDRANT_URL": os.getenv("QDRANT_URL", "http://localhost:6333"),
                    "QDRANT_API_KEY": os.getenv("QDRANT_API_KEY", ""),
                    "QDRANT_COLLECTION_NAME": os.getenv("QDRANT_COLLECTION_NAME", "koyo_memory"),
                    "QDRANT_VECTOR_SIZE": int(os.getenv("QDRANT_VECTOR_SIZE", "1536")),
                    "QDRANT_CONNECTION_TIMEOUT": int(os.getenv("QDRANT_CONNECTION_TIMEOUT", "30"))
                }
            )
            await self.processor.initialize()
            self.initialized = True
            logger.info("Memory integration service initialized")
    
    async def get_conversation_context(
        self, 
        user_id: str, 
        character_name: str, 
        current_message: str,
        limit: int = 5
    ) -> str:
        """
        Get relevant conversation context for a user-character pair.
        
        Args:
            user_id: User identifier
            character_name: Character name (e.g., "meher")
            current_message: Current user message
            limit: Maximum number of memories to retrieve
            
        Returns:
            str: Formatted conversation context
        """
        try:
            if not self.initialized:
                await self.initialize()
            
            # Get relevant memories
            memories = await self.processor.get_relevant_memories(
                user_id=user_id,
                character_id=character_name,
                current_message=current_message,
                limit=limit
            )
            
            if not memories:
                return ""
            
            # Format memories as context
            context_parts = []
            for memory in memories:
                context_parts.append(f"- {memory.message}")
            
            context = "Previous conversation context:\n" + "\n".join(context_parts)
            logger.debug(f"Retrieved {len(memories)} memories for user {user_id}")
            
            return context
            
        except Exception as e:
            logger.error(f"Failed to get conversation context: {e}")
            return ""
    
    async def enhance_system_prompt(
        self,
        user_id: str,
        character_name: str,
        current_message: str,
        base_system_prompt: str,
        limit: int = 5
    ) -> str:
        """
        Enhance a system prompt with relevant conversation context.
        
        Args:
            user_id: User identifier
            character_name: Character name
            current_message: Current user message
            base_system_prompt: Base system prompt
            limit: Maximum number of memories to include
            
        Returns:
            str: Enhanced system prompt with context
        """
        try:
            # Get conversation context
            context = await self.get_conversation_context(
                user_id=user_id,
                character_name=character_name,
                current_message=current_message,
                limit=limit
            )
            
            if not context:
                return base_system_prompt
            
            # Combine base prompt with context
            enhanced_prompt = f"{base_system_prompt}\n\n{context}"
            
            return enhanced_prompt
            
        except Exception as e:
            logger.error(f"Failed to enhance system prompt: {e}")
            return base_system_prompt
    
    async def get_memory_stats(self, user_id: str, character_name: str) -> Dict[str, Any]:
        """
        Get memory statistics for a user-character pair.
        
        Args:
            user_id: User identifier
            character_name: Character name
            
        Returns:
            Dict[str, Any]: Memory statistics
        """
        try:
            if not self.initialized:
                await self.initialize()
            
            # Count unsynced messages
            unsynced_count = await self.processor._count_unsynced_messages(user_id, character_name)
            
            return {
                "user_id": user_id,
                "character_name": character_name,
                "unsynced_messages": unsynced_count,
                "batch_ready": unsynced_count >= 25
            }
            
        except Exception as e:
            logger.error(f"Failed to get memory stats: {e}")
            return {"error": str(e)}


# Global memory integration instance
memory_integration = MemoryIntegration()
