"""
Data models and schemas for memory entries.
"""

from typing import Optional, List
from pydantic import BaseModel, Field


class MemoryEntry(BaseModel):
    """Memory entry data model."""

    memory_id: str = Field(..., description="Unique memory identifier")
    role: str = Field(..., description="Role (user/assistant)")
    message: str = Field(..., description="Message content")
    timestamp: str = Field(..., description="ISO timestamp")
    relevance_score: Optional[float] = Field(None, description="Search relevance score")


class MemorySearchResult(BaseModel):
    """Search result containing memories and metadata."""

    memories: List[MemoryEntry]
    total_count: int
    search_time_ms: float
    query_metadata: Optional[dict] = None


class MemoryCreateRequest(BaseModel):
    """Request model for creating memories."""

    user_id: str
    character_id: str
    role: str
    message: str
    vector: List[float]
    timestamp: Optional[str] = None

    class Config:
        json_schema_extra = {
            "example": {
                "user_id": "user123",
                "character_id": "char456",
                "role": "user",
                "message": "Hello, how are you?",
                "vector": [0.1, 0.2, 0.3],
                "timestamp": "2025-09-26T11:48:00Z",
            }
        }
