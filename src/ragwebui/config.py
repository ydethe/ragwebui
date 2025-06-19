from pydantic_settings import BaseSettings, SettingsConfigDict


class Config(BaseSettings):
    model_config = SettingsConfigDict(
        case_sensitive=True, env_file=".env", env_file_encoding="utf-8", extra="allow"
    )

    LOGLEVEL: str
    QDRANT_URL: str
    QDRANT_QUERY_LIMIT: int
    QDRANT_API_KEY: str
    OPENAI_API_KEY: str
    COLLECTION_NAME: str
    DAV_ROOT: str
    EMBEDDING_MODEL: str
    EMBEDDING_MODEL_TRUST_REMOTE_CODE: bool
    CHUNK_SIZE: int
    CHUNK_OVERLAP: int
    OPEN_MODEL_PREF: str
    RAG_AUGMENTATION: int
    TORCH_NUM_THREADS: int


config = Config()
