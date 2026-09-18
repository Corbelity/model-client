"""Provider-agnostic, traced access to text, image and speech model APIs.

Quick start
-----------
    from corbelity.model_client import make_model_client

    client = make_model_client("anthropic", model="claude-sonnet-5")
    print(client.complete("You are terse.", "Name three prime numbers."))

Routing has no global local/cloud flag. The SERVICE NAME picks the provider, so every
call site says plainly which one it is using:

    anthropic     Anthropic API, direct
    openrouter    OpenRouter, cloud
    ollama-local  Ollama running on the local host
    ollama        Ollama Cloud
    huggingface   HuggingFace Inference (the only one that also does image and sound)

What this package will not do to your process
---------------------------------------------
It does not configure logging, read a .env file, or write to os.environ. Those are
application decisions. It reads only the environment variables named in each
ProviderSpec, and it attaches a NullHandler to its own logger so that importing it
produces no output and changes no global state.
"""
from __future__ import annotations

import logging

from .catalog import ModelCatalog, ModelInfo, builtin_catalog, load_catalog
from .client import INCOMPLETE_FINISH_REASONS, ModelClient, load_sdk
from .config import ModelConfig, get_default_config, set_default_config
from .errors import (
    MissingBaseUrlError,
    MissingCredentialsError,
    MissingDependencyError,
    ModelClientError,
    UnknownServiceError,
    UnsupportedModalityError,
)
from .factory import (
    client_class,
    known_services,
    make_model_client,
    provider_spec,
    resolve_service,
    supported_modalities,
)
from .media import (
    IMAGE,
    SOUND,
    SUPPORTED_IMAGE_MIMES,
    TEXT,
    ImageInput,
    LLMResult,
    MediaResult,
    ModelResult,
    sniff_audio_mime,
    sniff_image_mime,
    validate_images,
)
from .messages import History, Message, validate_history
from .registry import ProviderSpec, get_spec, register_provider
from .trace import JsonlTraceLogger, NullTrace, TraceSink, make_trace_logger, trace_file_path

__version__ = "0.1.0"

__all__ = [
    "IMAGE",
    "INCOMPLETE_FINISH_REASONS",
    "SOUND",
    "SUPPORTED_IMAGE_MIMES",
    "TEXT",
    "History",
    "ImageInput",
    "JsonlTraceLogger",
    "LLMResult",
    "MediaResult",
    "Message",
    "MissingBaseUrlError",
    "MissingCredentialsError",
    "MissingDependencyError",
    "ModelCatalog",
    "ModelClient",
    "ModelClientError",
    "ModelConfig",
    "ModelInfo",
    "ModelResult",
    "NullTrace",
    "ProviderSpec",
    "TraceSink",
    "UnknownServiceError",
    "UnsupportedModalityError",
    "__version__",
    "builtin_catalog",
    "client_class",
    "get_default_config",
    "get_spec",
    "known_services",
    "load_catalog",
    "load_sdk",
    "make_model_client",
    "make_trace_logger",
    "provider_spec",
    "register_provider",
    "resolve_service",
    "set_default_config",
    "sniff_audio_mime",
    "sniff_image_mime",
    "supported_modalities",
    "trace_file_path",
    "validate_history",
    "validate_images",
]

# A library adds a handler of last resort and nothing else. Without this, a warning
# emitted before the application configures logging prints "No handlers could be found";
# with anything more, the library would be dictating the application's log pipeline.
logging.getLogger(__name__).addHandler(logging.NullHandler())
