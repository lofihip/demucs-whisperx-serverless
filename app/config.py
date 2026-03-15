from functools import lru_cache
from pathlib import Path

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore")

    app_host: str = Field(default="0.0.0.0", alias="APP_HOST")
    app_port: int = Field(default=8000, alias="APP_PORT")
    app_log_level: str = Field(default="INFO", alias="APP_LOG_LEVEL")
    api_key: str = Field(default="", alias="API_KEY")
    max_upload_size_mb: int = Field(default=2048, alias="MAX_UPLOAD_SIZE_MB")

    work_root: Path = Field(default=Path("/workspace/runtime"), alias="WORK_ROOT")
    model_cache_dir: Path = Field(default=Path("/workspace/models"), alias="MODEL_CACHE_DIR")
    temp_root: Path = Field(default=Path("/workspace/tmp"), alias="TEMP_ROOT")

    gpu_device: str = Field(default="cuda", alias="GPU_DEVICE")
    gpu_serial_execution: bool = Field(default=True, alias="GPU_SERIAL_EXECUTION")
    preload_on_startup: bool = Field(default=True, alias="PRELOAD_ON_STARTUP")
    preload_timeout_seconds: int = Field(default=7200, alias="PRELOAD_TIMEOUT_SECONDS")

    demucs_default_model: str = Field(default="htdemucs", alias="DEMUCS_DEFAULT_MODEL")
    demucs_default_two_stems: str = Field(default="", alias="DEMUCS_DEFAULT_TWO_STEMS")

    whisperx_default_model: str = Field(default="large-v3", alias="WHISPERX_DEFAULT_MODEL")
    whisperx_default_language: str = Field(default="", alias="WHISPERX_DEFAULT_LANGUAGE")
    whisperx_default_batch_size: int = Field(default=8, alias="WHISPERX_DEFAULT_BATCH_SIZE")
    whisperx_compute_type: str = Field(default="float16", alias="WHISPERX_COMPUTE_TYPE")
    whisperx_preload_align_language: str = Field(default="en", alias="WHISPERX_PRELOAD_ALIGN_LANGUAGE")
    whisperx_enable_diarization: bool = Field(default=True, alias="WHISPERX_ENABLE_DIARIZATION")
    huggingface_token: str = Field(default="", alias="HUGGINGFACE_TOKEN")

    def ensure_directories(self) -> None:
        for path in (self.work_root, self.model_cache_dir, self.temp_root):
            path.mkdir(parents=True, exist_ok=True)

    @property
    def max_upload_size_bytes(self) -> int:
        return self.max_upload_size_mb * 1024 * 1024


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    settings = Settings()
    settings.ensure_directories()
    return settings
