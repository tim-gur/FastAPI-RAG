from pydantic_settings import BaseSettings, SettingsConfigDict

class Settings(BaseSettings):
    qdrant_host: str
    qdrant_port: int
    ollama_host: str
    embedding_model: str
    llm_model: str
    collection_name: str
    redis_host: str
    redis_port: int

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8"
    )


settings = Settings()
