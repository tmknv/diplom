"""
Для получения секретов из env файлика
"""
from functools import lru_cache
from pathlib import Path
from typing import Optional

from pydantic_settings import BaseSettings, SettingsConfigDict
from pydantic import Field


class AppSettings(BaseSettings):

    # LLM 
    # LLM_BASE_URL: str = Field(..., description="Базовый URL LLM")
    # LLM_MODEL: str = Field(..., description="Название модели")
    # LLM_API_KEY: Optional[str] = Field(None, description="API ключ(нету)")

    # # MinIO
    # MINIO_ENDPOINT: str = Field(..., description="Хост:порт MinIO")
    # MINIO_ACCESS_KEY: str = Field(..., description="MinIO access key")
    # MINIO_SECRET_KEY: str = Field(..., description="MinIO secret key")


    # Конфиг Pydantic Settings
    model_config = SettingsConfigDict(
        # env_file=str(Path(__file__).resolve().parent.parent.parent / ".env"),
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=True,
        extra="allow",  # можно хранить дополнительные параметры
    )


@lru_cache()
def get_settings() -> AppSettings:
    """
    Кэшируем настройки, чтобы не пересоздавались при каждом импорте.
    Использование: settings = get_settings()
    """
    return AppSettings()





if __name__ == "__main__":
    settings = get_settings()

    print(settings.LLM_BASE_URL)
    print(settings.RUN_MODE)

