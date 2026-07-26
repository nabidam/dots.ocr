"""Configuration loading for the dots.ocr API."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import yaml
from dotenv import load_dotenv
from pydantic import BaseModel, Field, PositiveInt, field_validator

DEFAULT_CONFIG_PATH = Path(__file__).resolve().parent.parent / "config.yaml"


class AppConfig(BaseModel):
    """HTTP service settings."""

    name: str = "dots-ocr-api"
    host: str = "0.0.0.0"
    port: PositiveInt = 8080
    log_level: str = "INFO"
    max_upload_size_mb: PositiveInt = 50
    max_concurrent_inferences: PositiveInt = 1


class VllmConfig(BaseModel):
    """OpenAI-compatible vLLM connection settings."""

    protocol: str = "http"
    host: str = "localhost"
    port: PositiveInt = 8000
    model_name: str = "model"
    api_key: str = "0"


class OcrConfig(BaseModel):
    """dots.ocr inference settings."""

    prompt_mode: str = "prompt_layout_all_en"
    temperature: float = Field(default=0.1, ge=0)
    top_p: float = Field(default=0.9, gt=0, le=1)
    max_completion_tokens: PositiveInt = 32768
    pdf_dpi: PositiveInt = 200

    @field_validator("prompt_mode")
    @classmethod
    def validate_prompt_mode(cls, value: str) -> str:
        supported_modes = {
            "prompt_layout_all_en",
            "prompt_layout_only_en",
            "prompt_ocr",
            "prompt_grounding_ocr",
            "prompt_web_parsing",
            "prompt_scene_spotting",
            "prompt_image_to_svg",
            "prompt_general",
        }
        if value not in supported_modes:
            raise ValueError(
                f"Unsupported prompt mode {value!r}; "
                f"choose one of {sorted(supported_modes)}"
            )
        return value


class LoggingConfig(BaseModel):
    """Rotating log file settings."""

    file: str = "logs/dots-ocr-api.log"
    max_bytes: PositiveInt = 10 * 1024 * 1024
    backup_count: int = Field(default=5, ge=1)


class Settings(BaseModel):
    """Complete API settings."""

    app: AppConfig = Field(default_factory=AppConfig)
    vllm: VllmConfig = Field(default_factory=VllmConfig)
    ocr: OcrConfig = Field(default_factory=OcrConfig)
    logging: LoggingConfig = Field(default_factory=LoggingConfig)

    @property
    def max_upload_size_bytes(self) -> int:
        """Return the configured upload limit in bytes."""

        return self.app.max_upload_size_mb * 1024 * 1024


def _read_yaml(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(f"Configuration file does not exist: {path}")
    with path.open("r", encoding="utf-8") as config_file:
        parsed = yaml.safe_load(config_file) or {}
    if not isinstance(parsed, dict):
        raise ValueError(f"Configuration root must be a mapping: {path}")
    return parsed


def _set_nested_value(config: dict[str, Any], section: str, key: str, value: Any) -> None:
    config.setdefault(section, {})[key] = value


def _apply_environment_overrides(config: dict[str, Any]) -> None:
    """Apply explicit env vars while keeping secrets out of config.yaml."""

    overrides: tuple[tuple[str, str, str, Any], ...] = (
        ("API_HOST", "app", "host", str),
        ("API_PORT", "app", "port", int),
        ("LOG_LEVEL", "app", "log_level", str),
        ("MAX_UPLOAD_SIZE_MB", "app", "max_upload_size_mb", int),
        ("MAX_CONCURRENT_INFERENCES", "app", "max_concurrent_inferences", int),
        ("VLLM_PROTOCOL", "vllm", "protocol", str),
        ("VLLM_HOST", "vllm", "host", str),
        ("VLLM_PORT", "vllm", "port", int),
        ("VLLM_MODEL_NAME", "vllm", "model_name", str),
        ("OCR_PROMPT_MODE", "ocr", "prompt_mode", str),
        ("OCR_TEMPERATURE", "ocr", "temperature", float),
        ("OCR_TOP_P", "ocr", "top_p", float),
        ("OCR_MAX_COMPLETION_TOKENS", "ocr", "max_completion_tokens", int),
        ("PDF_DPI", "ocr", "pdf_dpi", int),
        ("LOG_FILE", "logging", "file", str),
        ("LOG_MAX_BYTES", "logging", "max_bytes", int),
        ("LOG_BACKUP_COUNT", "logging", "backup_count", int),
    )
    for environment_name, section, key, converter in overrides:
        raw_value = os.getenv(environment_name)
        if raw_value is None:
            continue
        try:
            _set_nested_value(config, section, key, converter(raw_value))
        except (TypeError, ValueError) as exc:
            raise ValueError(
                f"Environment variable {environment_name} has an invalid value"
            ) from exc

    api_key = os.getenv("VLLM_API_KEY", os.getenv("API_KEY"))
    if api_key is not None:
        _set_nested_value(config, "vllm", "api_key", api_key)


def load_settings() -> Settings:
    """Load YAML defaults, then local .env and process environment overrides."""

    load_dotenv(Path(__file__).resolve().parent.parent / ".env")
    config_path = Path(os.getenv("DOTS_OCR_CONFIG", str(DEFAULT_CONFIG_PATH)))
    config = _read_yaml(config_path)
    _apply_environment_overrides(config)
    return Settings.model_validate(config)


__all__ = ["AppConfig", "LoggingConfig", "OcrConfig", "Settings", "VllmConfig", "load_settings"]
