"""
Database service for Qdrant vector database
"""

import logging
import asyncio
from typing import List, Dict, Any, Optional
import os
from qdrant_client import QdrantClient, models
from qdrant_client.http.exceptions import ResponseHandlingException

from .models.schemas import MemoryEntry


class DatabaseError(Exception):
    """Generic database error for Qdrant service"""
    pass

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

        # Resolve configuration from provided custom_settings or environment variables
        # Collection name is hardcoded by design to ensure consistency across environments
        env_collection = "koyo_memory"
        env_vector_size = int(os.getenv("QDRANT_VECTOR_SIZE", "1536"))
        env_url = os.getenv("QDRANT_URL", "http://localhost:6333")
        env_api_key = os.getenv("QDRANT_API_KEY", "")
        env_timeout = int(os.getenv("QDRANT_CONNECTION_TIMEOUT", "30"))

        custom = custom_settings or {}
        # Always use fixed collection name
        self.collection_name = "koyo_memory"
        self.vector_size = custom.get("QDRANT_VECTOR_SIZE", env_vector_size)
        self.qdrant_url = custom.get("QDRANT_URL", env_url)
        self.qdrant_api_key = custom.get("QDRANT_API_KEY", env_api_key)
        self.timeout = custom.get("QDRANT_CONNECTION_TIMEOUT", env_timeout)

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

    async def _test_connection(self) -> None:
        """Verify Qdrant connectivity by listing collections."""
        loop = asyncio.get_event_loop()
        try:
            await loop.run_in_executor(None, self.client.get_collections)
        except Exception as e:
            raise DatabaseError(f"Qdrant connection failed: {str(e)}")

    async def _initialize_collection(self) -> None:
        """Ensure the target collection exists with proper vector params and indexes."""
        loop = asyncio.get_event_loop()
        try:
            # Check if collection exists
            collections = await loop.run_in_executor(None, self.client.get_collections)
            names = {c.name for c in collections.collections or []}
            if self.collection_name not in names:
                # Create collection
                await loop.run_in_executor(
                    None,
                    self.client.create_collection,
                    self.collection_name,
                    models.VectorParams(size=self.vector_size, distance=models.Distance.COSINE),
                )
                logger.info(f"Created Qdrant collection '{self.collection_name}'")

            # Create payload indexes (best-effort)
            await self._create_indexes()
        except Exception as e:
            # Do not fail hard if index creation fails; only fail if collection creation failed
            raise DatabaseError(f"Collection initialization failed: {str(e)}")

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

            query_filter = models.Filter(
                must=[
                    models.FieldCondition(
                        key="user_id", match=models.MatchValue(value=user_id)
                    ),
                    models.FieldCondition(
                        key="character_id",
                        match=models.MatchValue(value=character_id),
                    ),
                ]
            )

            def _search():
                return self.client.search(
                    collection_name=self.collection_name,
                    query_vector=query_vector,
                    query_filter=query_filter,
                    limit=limit,
                    with_payload=True,
                )

            search_results = await loop.run_in_executor(None, _search)

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

            def _upsert():
                return self.client.upsert(collection_name=self.collection_name, points=[point])

            await loop.run_in_executor(None, _upsert)

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

            def _upsert():
                return self.client.upsert(collection_name=self.collection_name, points=[point])

            await loop.run_in_executor(None, _upsert)

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
