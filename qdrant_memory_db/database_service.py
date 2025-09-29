"""
Database service for Qdrant vector database
"""

import logging
import asyncio
from typing import List, Dict, Any, Optional
from qdrant_client import QdrantClient, models
from qdrant_client.http.exceptions import ResponseHandlingException

from app.core.config import settings
from app.core.exceptions import DatabaseError
from app.models.schemas import MemoryEntry

logger = logging.getLogger(__name__)


class DatabaseService:
    """Service for managing Qdrant vector database operations"""

    def __init__(self, custom_settings: Optional[Dict] = None):
        """
        Initialize database service.

        Args:
            custom_settings: Optional dictionary to override default settings
        """
        self.client: Optional[QdrantClient] = None

        # Use custom settings if provided
        if custom_settings:
            self.collection_name = custom_settings.get(
                "QDRANT_COLLECTION_NAME", settings.QDRANT_COLLECTION_NAME
            )
            self.vector_size = custom_settings.get(
                "QDRANT_VECTOR_SIZE", settings.QDRANT_VECTOR_SIZE
            )
            self.qdrant_url = custom_settings.get("QDRANT_URL", settings.QDRANT_URL)
            self.qdrant_api_key = custom_settings.get(
                "QDRANT_API_KEY", settings.QDRANT_API_KEY
            )
            self.timeout = custom_settings.get(
                "QDRANT_CONNECTION_TIMEOUT", settings.QDRANT_CONNECTION_TIMEOUT
            )
        else:
            self.collection_name = settings.QDRANT_COLLECTION_NAME
            self.vector_size = settings.QDRANT_VECTOR_SIZE
            self.qdrant_url = settings.QDRANT_URL
            self.qdrant_api_key = settings.QDRANT_API_KEY
            self.timeout = settings.QDRANT_CONNECTION_TIMEOUT

        self._initialized = False

    async def initialize(self):
        """Initialize database connection and collections"""
        try:
            logger.info("🔄 Initializing database service...")

            self.client = QdrantClient(
                url=self.qdrant_url, api_key=self.qdrant_api_key, timeout=self.timeout
            )

            await self._test_connection()
            await self._initialize_collection()

            self._initialized = True
            logger.info("✅ Database service initialized successfully")

        except Exception as e:
            logger.error(f"❌ Failed to initialize database service: {e}")
            raise DatabaseError(f"Database initialization failed: {str(e)}")

    async def _create_indexes(self):
        """Create payload indexes"""
        try:
            loop = asyncio.get_event_loop()

            indexes = [
                ("user_id", "keyword"),
                ("character_id", "keyword"),
                ("timestamp", "datetime"),
            ]

            for field_name, field_schema in indexes:
                try:
                    await loop.run_in_executor(
                        None,
                        self.client.create_payload_index,
                        self.collection_name,
                        field_name,
                        field_schema,
                    )
                    logger.debug(f"Created index for {field_name}")
                except ResponseHandlingException as e:
                    if "already exists" in str(e).lower():
                        logger.debug(f"Index {field_name} already exists")
                    else:
                        raise

        except Exception as e:
            logger.warning(f"⚠️ Index creation warning: {e}")
            # Don't fail initialization for index issues

    async def search_memories(
        self, query_vector: List[float], user_id: str, character_id: str, limit: int = 5
    ) -> List[MemoryEntry]:
        """Search for relevant memories"""
        if not self._initialized:
            raise DatabaseError("Database service not initialized")

        try:
            loop = asyncio.get_event_loop()

            search_results = await loop.run_in_executor(
                None,
                self.client.search,
                self.collection_name,
                query_vector,
                models.Filter(
                    must=[
                        models.FieldCondition(
                            key="user_id", match=models.MatchValue(value=user_id)
                        ),
                        models.FieldCondition(
                            key="character_id",
                            match=models.MatchValue(value=character_id),
                        ),
                    ]
                ),
                limit,
                True,  # with_payload
            )

            memories = []
            for result in search_results:
                payload = result.payload
                memories.append(
                    MemoryEntry(
                        memory_id=str(result.id),
                        role=payload["role"],
                        message=payload["message"],
                        timestamp=payload["timestamp"],
                        relevance_score=result.score,
                    )
                )

            logger.debug(f"Found {len(memories)} memories for user {user_id}")
            return memories

        except Exception as e:
            logger.error(f"❌ Memory search failed: {e}")
            raise DatabaseError(f"Memory search failed: {str(e)}")

    async def create_memory(
        self,
        memory_id: str,
        vector: List[float],
        user_id: str,
        character_id: str,
        role: str,
        message: str,
        timestamp: str,
    ) -> bool:
        """Create a new memory entry"""
        if not self._initialized:
            raise DatabaseError("Database service not initialized")

        try:
            loop = asyncio.get_event_loop()

            point = models.PointStruct(
                id=memory_id,
                vector=vector,
                payload={
                    "user_id": user_id,
                    "character_id": character_id,
                    "role": role,
                    "message": message,
                    "timestamp": timestamp,
                },
            )

            await loop.run_in_executor(
                None, self.client.upsert, self.collection_name, [point]
            )

            logger.debug(f"Created memory {memory_id} for user {user_id}")
            return True

        except Exception as e:
            logger.error(f"❌ Memory creation failed: {e}")
            raise DatabaseError(f"Memory creation failed: {str(e)}")

    async def update_memory(
        self,
        memory_id: str,
        vector: List[float],
        user_id: str,
        character_id: str,
        role: str,
        message: str,
        timestamp: str,
    ) -> bool:
        """Update an existing memory entry"""
        if not self._initialized:
            raise DatabaseError("Database service not initialized")

        try:
            loop = asyncio.get_event_loop()

            point = models.PointStruct(
                id=memory_id,
                vector=vector,
                payload={
                    "user_id": user_id,
                    "character_id": character_id,
                    "role": role,
                    "message": message,
                    "timestamp": timestamp,
                    "updated": True,
                },
            )

            await loop.run_in_executor(
                None, self.client.upsert, self.collection_name, [point]
            )

            logger.debug(f"Updated memory {memory_id} for user {user_id}")
            return True

        except Exception as e:
            logger.error(f"❌ Memory update failed: {e}")
            raise DatabaseError(f"Memory update failed: {str(e)}")

    async def health_check(self) -> Dict[str, Any]:
        """Check database health"""
        try:
            if not self._initialized:
                return {"status": "error", "message": "Not initialized"}

            loop = asyncio.get_event_loop()
            collections = await loop.run_in_executor(None, self.client.get_collections)

            collection_info = None
            for collection in collections.collections:
                if collection.name == self.collection_name:
                    collection_info = await loop.run_in_executor(
                        None, self.client.get_collection, self.collection_name
                    )
                    break

            return {
                "status": "healthy",
                "collection_exists": collection_info is not None,
                "collection_name": self.collection_name,
                "vector_size": self.vector_size,
                "points_count": collection_info.points_count if collection_info else 0,
            }

        except Exception as e:
            logger.error(f"❌ Database health check failed: {e}")
            return {"status": "error", "message": str(e)}

    async def close(self):
        """Close database connections"""
        try:
            if self.client:
                # Qdrant client doesn't need explicit closing
                self.client = None
                self._initialized = False
                logger.info("✅ Database service closed")
        except Exception as e:
            logger.error(f"❌ Error closing database service: {e}")
