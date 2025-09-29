"""
Application configuration management
"""

from pydantic import field_validator
from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    """Application settings"""

    # Basic settings
    APP_NAME: str = "Koyo Chat API"
    DEBUG: bool = False
    HOST: str = "0.0.0.0"
    PORT: int = 8000

    # Database
    QDRANT_URL: str
    QDRANT_API_KEY: str
    QDRANT_VECTOR_SIZE: int = 1536
    QDRANT_COLLECTION_NAME: str = "chat_memory"
    QDRANT_CONNECTION_TIMEOUT: int = 30

    @field_validator("DEBUG", mode="before")
    def parse_debug(cls, v):
        if isinstance(v, str):
            return v.lower() in ("true", "1", "yes", "on")
        return v

    @field_validator("CORS_ORIGINS", mode="before")
    def parse_cors_origins(cls, v):
        if isinstance(v, str):
            return [origin.strip() for origin in v.split(",")]
        return v

    @field_validator("ALLOWED_HOSTS", mode="before")
    def parse_allowed_hosts(cls, v):
        if isinstance(v, str):
            return [host.strip() for host in v.split(",")]
        return v

    class Config:
        env_file = ".env"
        case_sensitive = True


settings = Settings()
