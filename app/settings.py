from pydantic_settings import BaseSettings, SettingsConfigDict
import os

# выбор env файла в зависимости от типа запуска
ENV_FILE = ".env" if os.getenv("ENV") == "docker" else ".env.local"

class Settings(BaseSettings):
    qdrant_host: str
    qdrant_port: int
    ollama_host: str
    embedding_model: str
    llm_model: str
    collection_name: str
    redis_host: str
    redis_port: int
    db_path: str
    log_path: str

    model_config = SettingsConfigDict(
        env_file=ENV_FILE,
        env_file_encoding="utf-8"
    )


settings = Settings()
