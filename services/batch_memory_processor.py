"""
Batch Memory Processor for conversation-level memory extraction.

This service handles batch processing of Firebase messages to extract
semantic memories and store them in Qdrant.
"""

import asyncio
import logging
from datetime import datetime
import uuid
from typing import List, Dict, Any, Optional
import openai
from qdrant_memory_db.database_service import DatabaseService
from qdrant_memory_db.models.schemas import MemoryEntry
from database.firebase_service import firebase_service

logger = logging.getLogger(__name__)


class BatchMemoryProcessor:
    """
    Processes batches of Firebase messages to extract conversation-level memories
    and store them in Qdrant for semantic search.
    """
    
    def __init__(self, openai_api_key: str, qdrant_settings: Optional[Dict] = None):
        """
        Initialize the batch memory processor.
        
        Args:
            openai_api_key: OpenAI API key for embedding generation
            qdrant_settings: Optional Qdrant configuration
        """
        self.openai_client = openai.AsyncOpenAI(api_key=openai_api_key)
        self.db_service = DatabaseService(custom_settings=qdrant_settings)
        self.initialized = False
        self.logger = logging.getLogger(__name__)
    
    async def initialize(self):
        """Initialize the processor and Qdrant connection"""
        if not self.initialized:
            await self.db_service.initialize()
            self.initialized = True
            self.logger.info("Batch memory processor initialized")
    
    async def process_user_character_batch(
        self, 
        user_id: str, 
        character_name: str, 
        batch_size: int = 25
    ) -> bool:
        """
        Process a batch of unsynced messages for a specific user-character pair.
        
        Args:
            user_id: User identifier
            character_name: Character name (e.g., "meher")
            batch_size: Number of messages to process in batch
            
        Returns:
            bool: True if processing was successful
        """
        try:
            self.logger.info(f"Processing batch for user {user_id} and character {character_name}")
            
            # Get unsynced messages from Firebase
            unsynced_messages = await self._get_unsynced_messages(user_id, character_name, batch_size)
            
            if not unsynced_messages:
                self.logger.info(f"No unsynced messages found for user {user_id}")
                return True
            
            self.logger.info(f"Found {len(unsynced_messages)} unsynced messages")
            
            # Extract conversation memory
            conversation_memory = await self._extract_conversation_memory(unsynced_messages)
            
            if not conversation_memory:
                self.logger.warning(f"Failed to extract conversation memory for user {user_id}")
                return False
            
            # Store in Qdrant
            success = await self._store_conversation_memory(
                user_id=user_id,
                character_id=character_name,
                conversation_memory=conversation_memory,
                message_count=len(unsynced_messages)
            )
            
            if success:
                # Mark messages as synced in Firebase
                await self._mark_messages_as_synced(user_id, character_name, unsynced_messages)
                self.logger.info(f"Successfully processed batch for user {user_id}")
            
            return success
            
        except Exception as e:
            self.logger.error(f"Failed to process batch for user {user_id}: {e}")
            return False
    
    async def _get_unsynced_messages(
        self, 
        user_id: str, 
        character_name: str, 
        limit: int
    ) -> List[Dict[str, Any]]:
        """Get unsynced messages from Firebase for a user-character pair"""
        try:
            messages_ref = firebase_service.db.collection("users").document(user_id).collection("conversations").document(character_name).collection("messages")
            
            # Query for unsynced messages; avoid order_by to prevent requiring a composite index
            unsynced_query = messages_ref.where("sync", "==", False).limit(limit)
            unsynced_docs = unsynced_query.get()
            
            messages = []
            for doc in unsynced_docs:
                message_data = doc.to_dict()
                message_data["doc_id"] = doc.id
                messages.append(message_data)
            
            # Sort client-side by timestamp if present
            try:
                messages.sort(key=lambda x: x.get("timestamp"))
            except Exception:
                pass

            return messages
            
        except Exception as e:
            self.logger.error(f"Failed to get unsynced messages: {e}")
            return []
    
    async def _extract_conversation_memory(self, messages: List[Dict[str, Any]]) -> Optional[str]:
        """
        Extract semantic memory from a batch of conversation messages.
        
        Args:
            messages: List of message dictionaries from Firebase
            
        Returns:
            str: Extracted conversation memory or None if extraction failed
        """
        try:
            # Sort messages by timestamp
            sorted_messages = sorted(messages, key=lambda x: x["timestamp"])
            
            # Create conversation context
            conversation_context = self._build_conversation_context(sorted_messages)
            
            # Use OpenAI to extract semantic memory
            memory_prompt = f"""
            Analyze the following conversation and extract key semantic memories about the user.
            Focus on: user preferences, important information shared, emotional context, topics of interest.
            
            Conversation:
            {conversation_context}
            
            Extract 2-3 key memories that would be useful for future conversations.
            Format as clear, concise statements.
            """
            
            response = await self.openai_client.chat.completions.create(
                model="gpt-3.5-turbo",
                messages=[{"role": "user", "content": memory_prompt}],
                max_tokens=300,
                temperature=0.3
            )
            
            extracted_memory = response.choices[0].message.content.strip()
            self.logger.info(f"Extracted conversation memory: {extracted_memory[:100]}...")
            
            return extracted_memory
            
        except Exception as e:
            self.logger.error(f"Failed to extract conversation memory: {e}")
            return None
    
    def _build_conversation_context(self, messages: List[Dict[str, Any]]) -> str:
        """Build conversation context from messages"""
        context_parts = []
        
        for msg in messages:
            sender = "User" if msg["sender"] == "user" else "Assistant"
            content = msg["content"]
            timestamp = msg["timestamp"].strftime("%H:%M") if hasattr(msg["timestamp"], 'strftime') else str(msg["timestamp"])
            
            context_parts.append(f"{sender} ({timestamp}): {content}")
        
        return "\n".join(context_parts)
    
    async def _store_conversation_memory(
        self,
        user_id: str,
        character_id: str,
        conversation_memory: str,
        message_count: int
    ) -> bool:
        """Store conversation memory in Qdrant"""
        try:
            # Generate embedding for the conversation memory
            embedding = await self._generate_embedding(conversation_memory)
            
            # Create memory ID as UUID to satisfy Qdrant ID constraints
            memory_id = str(uuid.uuid4())
            
            # Store in Qdrant
            success = await self.db_service.create_memory(
                memory_id=memory_id,
                vector=embedding,
                user_id=user_id,
                character_id=character_id,
                role="conversation_summary",
                message=conversation_memory,
                timestamp=datetime.now().isoformat()
            )
            
            if success:
                self.logger.info(f"Stored conversation memory for user {user_id} with {message_count} messages")
            
            return success
            
        except Exception as e:
            self.logger.error(f"Failed to store conversation memory: {e}")
            return False
    
    async def _generate_embedding(self, text: str) -> List[float]:
        """Generate OpenAI embedding for text"""
        try:
            response = await self.openai_client.embeddings.create(
                model="text-embedding-3-small",
                input=text
            )
            return response.data[0].embedding
        except Exception as e:
            self.logger.error(f"Failed to generate embedding: {e}")
            return []
    
    async def _mark_messages_as_synced(self, user_id: str, character_name: str, messages: List[Dict[str, Any]]) -> bool:
        """Mark messages as synced in Firebase"""
        try:
            batch = firebase_service.db.batch()
            
            for msg in messages:
                doc_ref = (
                    firebase_service.db
                    .collection("users").document(user_id)
                    .collection("conversations").document(character_name)
                    .collection("messages").document(msg["doc_id"]) 
                )
                batch.update(doc_ref, {"sync": True})
            
            batch.commit()
            self.logger.info(f"Marked {len(messages)} messages as synced")
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to mark messages as synced: {e}")
            return False
    
    async def get_relevant_memories(
        self, 
        user_id: str, 
        character_id: str, 
        current_message: str, 
        limit: int = 5
    ) -> List[MemoryEntry]:
        """
        Get relevant memories for context in current conversation.
        
        Args:
            user_id: User identifier
            character_id: Character identifier
            current_message: Current user message
            limit: Maximum number of memories to return
            
        Returns:
            List[MemoryEntry]: Relevant memories
        """
        try:
            # Generate embedding for current message
            embedding = await self._generate_embedding(current_message)
            
            # Search Qdrant for relevant memories
            memories = await self.db_service.search_memories(
                query_vector=embedding,
                user_id=user_id,
                character_id=character_id,
                limit=limit
            )
            
            return memories
            
        except Exception as e:
            self.logger.error(f"Failed to get relevant memories: {e}")
            return []
    
    async def check_and_process_batch(self, user_id: str, character_name: str) -> bool:
        """
        Check if batch processing is needed and process if so.
        
        Args:
            user_id: User identifier
            character_name: Character name
            
        Returns:
            bool: True if processing was triggered
        """
        try:
            # Count unsynced messages
            unsynced_count = await self._count_unsynced_messages(user_id, character_name)
            
            if unsynced_count >= 25:  # Batch threshold
                self.logger.info(f"Triggering batch processing for user {user_id} ({unsynced_count} unsynced messages)")
                return await self.process_user_character_batch(user_id, character_name)
            
            return False
            
        except Exception as e:
            self.logger.error(f"Failed to check and process batch: {e}")
            return False
    
    async def _count_unsynced_messages(self, user_id: str, character_name: str) -> int:
        """Count unsynced messages for a user-character pair"""
        try:
            messages_ref = firebase_service.db.collection("users").document(user_id).collection("conversations").document(character_name).collection("messages")
            unsynced_query = messages_ref.where("sync", "==", False)
            unsynced_docs = unsynced_query.get()
            return len(unsynced_docs)
        except Exception as e:
            self.logger.error(f"Failed to count unsynced messages: {e}")
            return 0
