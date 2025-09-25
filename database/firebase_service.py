"""
Firebase Firestore service for managing conversation data.

This module provides functionality to save and manage conversation messages
and metadata in Firebase Firestore.
"""

import json
import logging
import os
from datetime import datetime
from typing import Optional, Dict, List, Any

import firebase_admin
from firebase_admin import credentials, firestore


class FirebaseService:
    """Service for managing Firebase Firestore operations"""
    
    def __init__(self):
        """Initialize the Firebase service"""
        self.db = None
        self.logger = logging.getLogger(__name__)
        self._initialize_firebase()
    
    def _initialize_firebase(self):
        """Initialize Firebase connection"""
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
            
            self.db = firestore.client()
            # Test Firebase connection
            test_doc = self.db.collection("_health_check").document("test")
            test_doc.set({"timestamp": datetime.now()}, merge=True)
            test_doc.delete()
            self.logger.info("Firebase initialized and tested successfully")
        except Exception as e:
            self.logger.error(f"Firebase initialization failed: {e}")
            self.db = None
            # Don't crash the app, but log the critical error
            self.logger.critical("Firebase is not available - message saving will fail")
    
    async def create_or_update_conversation_metadata(
        self, 
        user_id: str, 
        character_name: str, 
        call_sid: Optional[str] = None
    ):
        """Create or update conversation metadata document"""
        if self.db is None:
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
            conversation_ref = self.db.collection("users").document(user_id).collection("conversations").document(character_name)
            conversation_ref.set(conversation_metadata, merge=True)
            
        except Exception as e:
            self.logger.error(f"Failed to update conversation metadata: {e}")

    def extract_conversation_messages(self, llm_context) -> List[Dict[str, Any]]:
        """Extract conversation messages from LLM context"""
        messages = llm_context.get_messages()
        conversation_messages = []
        for msg in messages:
            role = msg.get("role")
            content = msg.get("content", "")
            if role in ["user", "assistant"] and content.strip():
                sender = "user" if role == "user" else "character"
                conversation_messages.append({
                    "sender": sender,
                    "content": content,
                    "timestamp": datetime.now()
                })
        return conversation_messages

    async def save_voice_messages_to_firebase_batch(
        self, 
        caller_phone: str, 
        messages: List[Dict[str, Any]], 
        call_sid: Optional[str] = None
    ):
        """Global helper function to save voice messages to Firebase using batch operations"""
        try:
            if not messages:
                return
                
            self.logger.info(f"Batch saving {len(messages)} voice messages to Firebase for {caller_phone}")
            
            # Create batch for faster operations
            batch = self.db.batch()
            character_name = "meher"
            
            # Update conversation metadata once
            await self.create_or_update_conversation_metadata(caller_phone, character_name, call_sid)
            
            # Add all messages to batch
            conversation_ref = self.db.collection("users").document(caller_phone).collection("conversations").document(character_name)
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
            self.logger.info(f"Batch saved {len(messages)} voice messages to Firebase for {caller_phone}")
            
        except Exception as e:
            self.logger.error(f"Failed to batch save voice messages to Firebase: {e}")

    async def save_message_to_firebase(
        self,
        user_id: str, 
        sender: str, 
        content: str, 
        timestamp: Optional[datetime] = None, 
        sync: bool = False,
        conversation_type: str = "text",  # "text" or "voice"
        call_sid: Optional[str] = None
    ) -> bool:
        """
        Save a message to Firebase Firestore in nested collection structure:
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
        if self.db is None:
            self.logger.warning("Firebase not initialized, skipping message save")
            return False
        
        try:
            if timestamp is None:
                timestamp = datetime.now()
            
            # Use "meher" as the character name for all conversations
            character_name = "meher"
            
            # Create or update conversation metadata
            await self.create_or_update_conversation_metadata(user_id, character_name, call_sid)
            
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
            doc_ref = self.db.collection("users").document(user_id).collection("conversations").document(character_name).collection("messages").add(message_data)
            self.logger.info(f"Saved {sender} message to Firebase for user {user_id} in conversation with {character_name}")
            
            # Update message count in conversation metadata
            try:
                conversation_ref = self.db.collection("users").document(user_id).collection("conversations").document(character_name)
                conversation_ref.update({"message_count": firestore.Increment(1)})
            except Exception as e:
                self.logger.warning(f"Failed to update message count: {e}")
            
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to save message to Firebase: {e}")
            return False


# Global Firebase service instance
firebase_service = FirebaseService()
