"""Application configuration loaded from environment / .env file."""

from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    groq_api_key: str = ""
    groq_model: str = "llama-3.3-70b-versatile"
    max_attempts_per_word: int = 3
    default_words: list[str] = ["cat", "dog", "bird", "fish", "sun"]
    max_history_turns: int = 10
    log_level: str = "INFO"

    model_config = {"env_file": ".env", "env_file_encoding": "utf-8"}


settings = Settings()
