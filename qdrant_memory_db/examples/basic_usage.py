"""
Basic usage example for Qdrant Memory DB package.
"""

import asyncio
from qdrant_memory_db import DatabaseService


async def main():
    # Configure settings via environment variables or custom dict
    custom_config = {
        "QDRANT_URL": "http://localhost:6333",
        "QDRANT_COLLECTION_NAME": "chat_memories",
        "QDRANT_VECTOR_SIZE": 1536,
        "QDRANT_API_KEY": None,
        "QDRANT_CONNECTION_TIMEOUT": 30,
    }

    # Initialize service
    db_service = DatabaseService(custom_settings=custom_config)

    try:
        # Initialize connection
        await db_service.initialize()

        # Check health
        health = await db_service.health_check()
        print(f"Database health: {health}")

        # Create a memory
        success = await db_service.create_memory(
            memory_id="mem_001",
            vector=[0.1] * 1536,  # Your embedding vector
            user_id="user123",
            character_id="char456",
            role="user",
            message="Hello there!",
            timestamp="2025-09-26T11:48:00Z",
        )
        print(f"Memory created: {success}")

        # Search memories
        memories = await db_service.search_memories(
            query_vector=[0.1] * 1536,
            user_id="user123",
            character_id="char456",
            limit=5,
        )
        print(f"Found {len(memories)} memories")

    finally:
        await db_service.close()


if __name__ == "__main__":
    asyncio.run(main())
