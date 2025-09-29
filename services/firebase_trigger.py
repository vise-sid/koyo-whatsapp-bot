"""
Firebase Trigger Service for real-time batch processing.

This service provides a simple HTTP endpoint that can be called
when messages are added to Firebase to trigger batch processing.
"""

import asyncio
import logging
from typing import Dict, Any
from fastapi import APIRouter, HTTPException
from services.batch_memory_processor import BatchMemoryProcessor
import os

logger = logging.getLogger(__name__)

# Create router for Firebase triggers
firebase_router = APIRouter(prefix="/firebase", tags=["firebase-trigger"])

# Global batch processor instance
batch_processor = None

async def get_batch_processor() -> BatchMemoryProcessor:
    """Get or create batch processor instance"""
    global batch_processor
    if batch_processor is None:
        batch_processor = BatchMemoryProcessor(
            openai_api_key=os.getenv("OPENAI_API_KEY", ""),
            qdrant_settings={
                "QDRANT_URL": os.getenv("QDRANT_URL", "http://localhost:6333"),
                "QDRANT_API_KEY": os.getenv("QDRANT_API_KEY", ""),
                "QDRANT_COLLECTION_NAME": os.getenv("QDRANT_COLLECTION_NAME", "chat_memory"),
                "QDRANT_VECTOR_SIZE": int(os.getenv("QDRANT_VECTOR_SIZE", "1536")),
                "QDRANT_CONNECTION_TIMEOUT": int(os.getenv("QDRANT_CONNECTION_TIMEOUT", "30"))
            }
        )
        await batch_processor.initialize()
    return batch_processor


@firebase_router.post("/message-created")
async def on_message_created(
    user_id: str,
    character_name: str,
    message_id: str = None
) -> Dict[str, Any]:
    """
    Triggered when a new message is created in Firebase.
    This endpoint should be called by Firebase Cloud Functions or webhooks.
    
    Args:
        user_id: User identifier
        character_name: Character name (e.g., "meher")
        message_id: Optional message ID for logging
    """
    try:
        logger.info(f"Message created trigger for user {user_id}, character {character_name}")
        
        processor = await get_batch_processor()
        
        # Check if batch processing is needed
        triggered = await processor.check_and_process_batch(user_id, character_name)
        
        return {
            "success": True,
            "user_id": user_id,
            "character_name": character_name,
            "message_id": message_id,
            "batch_triggered": triggered,
            "message": "Message created trigger processed successfully"
        }
        
    except Exception as e:
        logger.error(f"Error in message created trigger: {e}")
        raise HTTPException(status_code=500, detail=f"Error processing trigger: {str(e)}")


@firebase_router.post("/batch-process")
async def trigger_batch_process(
    user_id: str,
    character_name: str
) -> Dict[str, Any]:
    """
    Manually trigger batch processing for a user-character pair.
    """
    try:
        logger.info(f"Manual batch process trigger for user {user_id}, character {character_name}")
        
        processor = await get_batch_processor()
        
        # Process batch
        success = await processor.process_user_character_batch(user_id, character_name)
        
        return {
            "success": success,
            "user_id": user_id,
            "character_name": character_name,
            "message": "Batch processing completed" if success else "Batch processing failed"
        }
        
    except Exception as e:
        logger.error(f"Error in batch process trigger: {e}")
        raise HTTPException(status_code=500, detail=f"Error processing batch: {str(e)}")


@firebase_router.get("/health")
async def health_check() -> Dict[str, Any]:
    """
    Health check endpoint for Firebase triggers.
    """
    try:
        processor = await get_batch_processor()
        
        return {
            "status": "healthy",
            "service": "firebase-trigger-service",
            "processor_initialized": processor.initialized
        }
        
    except Exception as e:
        logger.error(f"Health check failed: {e}")
        return {
            "status": "unhealthy",
            "error": str(e)
        }
