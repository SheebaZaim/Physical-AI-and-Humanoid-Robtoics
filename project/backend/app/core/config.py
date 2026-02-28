from pydantic_settings import BaseSettings
from typing import Optional, List


class Settings(BaseSettings):
    PROJECT_NAME: str = "Educational Book Platform API"
    API_V1_STR: str = "/api/v1"
    # Default to wildcard so the HF Spaces backend accepts requests from
    # any origin (Vercel frontend, local dev, etc.).  Override via the
    # BACKEND_CORS_ORIGINS environment variable / HF Space secret when you
    # want to restrict to specific domains, e.g.:
    #   BACKEND_CORS_ORIGINS=["https://your-project.vercel.app"]
    BACKEND_CORS_ORIGINS: List[str] = ["*"]

    # Database settings
    DATABASE_URL: Optional[str] = None

    # Auth settings
    SECRET_KEY: str
    ACCESS_TOKEN_EXPIRE_MINUTES: int = 60 * 24 * 8  # 8 days

    # OpenAI settings (for embeddings - backup)
    OPENAI_API_KEY: str

    # Gemini settings (for embeddings - FREE!)
    GEMINI_API_KEY: Optional[str] = None

    # OpenRouter settings (for chat completions)
    OPENROUTER_API_KEY: Optional[str] = None
    OPENROUTER_API_BASE: str = "https://openrouter.ai/api/v1"
    OPENROUTER_APP_NAME: str = "Physical-AI-Book-Platform"
    OPENROUTER_MODEL: str = "openai/gpt-4o"

    # Qdrant settings
    QDRANT_HOST: str = "localhost"
    QDRANT_PORT: int = 6333
    QDRANT_API_KEY: Optional[str] = None  # For Qdrant Cloud
    QDRANT_URL: Optional[str] = None  # For Qdrant Cloud

    class Config:
        env_file = ".env"


settings = Settings()
