from pydantic import Field, SecretStr
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore")

    CHUNK_SIZE: int = Field(default=1000)
    CHUNK_OVERLAP: int = Field(default=200)

    # Embedding Settings
    HF_TOKEN: SecretStr = SecretStr("hf_api_key")
    EMBEDDING_MODEL_NAME: str = Field(default="all-MiniLM-L6-v2")

    # LLM Settings
    GROQ_API_KEY: SecretStr = SecretStr("groq_api_key")
    LLM_MODEL_NAME: str = Field(default="meta-llama/llama-4-maverick-17b-128e-instruct")

    # Milvus Settings
    MILVUS_URI: str = Field(default="http://localhost:19530")
    MILVUS_TOKEN: str = Field(default="")
    MILVUS_COLLECTION_NAME: str = Field(default="eval_rag")
    MILVUS_EMBEDDING_DIM: int = Field(default=384)

    # Postgress Settings
    POSTGRES_USER: str | None = Field(default="postgres")
    POSTGRES_PASSWORD: SecretStr = SecretStr("postgres")
    POSTGRES_PORT: int = Field(default=5432)
    POSTGRES_DB: str | None = Field(default="postgres")
    POSTGRES_APPLICATION_NAME: str | None = Field(default="rag-ai-agent")
    POSTGRES_HOST: str = Field(default="localhost")
    POSTGRES_MIN_CONNECTIONS_PER_POOL: int = Field(default=1)
    POSTGRES_MAX_CONNECTIONS_PER_POOL: int = Field(default=10)


settings = Settings()
