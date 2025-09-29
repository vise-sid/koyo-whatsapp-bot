"""
Memory Trigger Service for real-time batch processing.

This service provides HTTP endpoints to trigger batch processing
and integrates with your existing FastAPI application.
"""

import asyncio
import logging
from typing import Dict, Any
from fastapi import APIRouter, HTTPException, BackgroundTasks
from services.batch_memory_processor import BatchMemoryProcessor
from database.firebase_service import firebase_service

logger = logging.getLogger(__name__)

# Create router for memory endpoints
memory_router = APIRouter(prefix="/memory", tags=["memory"])

# Global batch processor instance
batch_processor = None

async def get_batch_processor() -> BatchMemoryProcessor:
    """Get or create batch processor instance"""
    global batch_processor
    if batch_processor is None:
        import os
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


@memory_router.post("/check-batch/{user_id}/{character_name}")
async def check_and_process_batch(
    user_id: str, 
    character_name: str,
    background_tasks: BackgroundTasks
) -> Dict[str, Any]:
    """
    Check if batch processing is needed for a user-character pair.
    This endpoint should be called when a new message is added to Firebase.
    """
    try:
        processor = await get_batch_processor()
        
        # Check if batch processing is needed
        triggered = await processor.check_and_process_batch(user_id, character_name)
        
        return {
            "user_id": user_id,
            "character_name": character_name,
            "batch_triggered": triggered,
            "message": "Batch processing checked successfully"
        }
        
    except Exception as e:
        logger.error(f"Error checking batch for user {user_id}: {e}")
        raise HTTPException(status_code=500, detail=f"Error checking batch: {str(e)}")


@memory_router.post("/process-batch/{user_id}/{character_name}")
async def process_batch(
    user_id: str, 
    character_name: str,
    background_tasks: BackgroundTasks
) -> Dict[str, Any]:
    """
    Manually trigger batch processing for a user-character pair.
    """
    try:
        processor = await get_batch_processor()
        
        # Process batch in background
        background_tasks.add_task(
            processor.process_user_character_batch,
            user_id,
            character_name
        )
        
        return {
            "user_id": user_id,
            "character_name": character_name,
            "message": "Batch processing started in background"
        }
        
    except Exception as e:
        logger.error(f"Error processing batch for user {user_id}: {e}")
        raise HTTPException(status_code=500, detail=f"Error processing batch: {str(e)}")


@memory_router.get("/search/{user_id}/{character_name}")
async def search_memories(
    user_id: str,
    character_name: str,
    query: str,
    limit: int = 5
) -> Dict[str, Any]:
    """
    Search for relevant memories for a user-character pair.
    """
    try:
        processor = await get_batch_processor()
        
        memories = await processor.get_relevant_memories(
            user_id=user_id,
            character_id=character_name,
            current_message=query,
            limit=limit
        )
        
        return {
            "user_id": user_id,
            "character_name": character_name,
            "query": query,
            "memories": [
                {
                    "memory_id": mem.memory_id,
                    "role": mem.role,
                    "message": mem.message,
                    "timestamp": mem.timestamp,
                    "relevance_score": mem.relevance_score
                }
                for mem in memories
            ],
            "total_count": len(memories)
        }
        
    except Exception as e:
        logger.error(f"Error searching memories for user {user_id}: {e}")
        raise HTTPException(status_code=500, detail=f"Error searching memories: {str(e)}")


@memory_router.get("/stats/{user_id}/{character_name}")
async def get_memory_stats(
    user_id: str,
    character_name: str
) -> Dict[str, Any]:
    """
    Get memory statistics for a user-character pair.
    """
    try:
        processor = await get_batch_processor()
        
        # Count unsynced messages
        unsynced_count = await processor._count_unsynced_messages(user_id, character_name)
        
        # Get total message count
        messages_ref = firebase_service.db.collection("users").document(user_id).collection("conversations").document(character_name).collection("messages")
        total_docs = messages_ref.get()
        total_count = len(total_docs)
        
        return {
            "user_id": user_id,
            "character_name": character_name,
            "total_messages": total_count,
            "unsynced_messages": unsynced_count,
            "synced_messages": total_count - unsynced_count,
            "sync_percentage": round((total_count - unsynced_count) / total_count * 100, 2) if total_count > 0 else 0
        }
        
    except Exception as e:
        logger.error(f"Error getting memory stats for user {user_id}: {e}")
        raise HTTPException(status_code=500, detail=f"Error getting memory stats: {str(e)}")


@memory_router.get("/health")
async def health_check() -> Dict[str, Any]:
    """
    Health check endpoint for the memory service.
    """
    try:
        processor = await get_batch_processor()
        
        return {
            "status": "healthy",
            "service": "memory-trigger-service",
            "processor_initialized": processor.initialized
        }
        
    except Exception as e:
        logger.error(f"Health check failed: {e}")
        return {
            "status": "unhealthy",
            "error": str(e)
        }
