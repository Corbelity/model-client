"""Provider-agnostic, traced access to text, image and speech model APIs.

Quick start
-----------
    from corbelity.model_client import make_model_client

    client = make_model_client("anthropic", model="claude-sonnet-5")
    print(client.complete("You are terse.", "Name three prime numbers."))

Routing has no global local/cloud flag. The SERVICE NAME picks the provider, so every
call site says plainly which one it is using:

    anthropic     Anthropic API, direct
    openai        OpenAI API, direct -- text, image and sound
    gemini        Gemini, via Google's OpenAI-compatibility endpoint (text only)
    openrouter    OpenRouter, cloud
    ollama-local  Ollama running on the local host
    ollama        Ollama Cloud
    huggingface   HuggingFace Inference -- text, image and sound

Text is the common denominator; only openai and huggingface generate images and speech,
and each does so through its own SDK surface rather than through chat completions. Ask a
service for something it cannot produce and it raises UnsupportedModalityError before any
network call -- supported_modalities() answers the same question without even importing
the provider.

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
    TooManyImagesError,
    UnknownServiceError,
    UnsupportedFidelityError,
    UnsupportedImageInputError,
    UnsupportedModalityError,
    UnsupportedQualityError,
    UnsupportedSizeError,
)
from .factory import (
    available_services,
    catalog_for,
    client_class,
    known_services,
    make_model_client,
    provider_spec,
    resolve_service,
    supported_aspect_ratios,
    supported_image_qualities,
    supported_image_sizes,
    supported_input_fidelities,
    supported_modalities,
)
from .media import (
    IMAGE,
    SOUND,
    SUPPORTED_IMAGE_MIMES,
    TEXT,
    ImageInput,
    ImageOptions,
    LLMResult,
    MediaResult,
    ModelResult,
    aspect_ratios_of,
    parse_aspect_ratio,
    parse_size,
    sizes_for_ratio,
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
    "ImageOptions",
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
    "TooManyImagesError",
    "TraceSink",
    "UnknownServiceError",
    "UnsupportedFidelityError",
    "UnsupportedImageInputError",
    "UnsupportedModalityError",
    "UnsupportedQualityError",
    "UnsupportedSizeError",
    "__version__",
    "aspect_ratios_of",
    "available_services",
    "builtin_catalog",
    "catalog_for",
    "client_class",
    "get_default_config",
    "get_spec",
    "known_services",
    "load_catalog",
    "load_sdk",
    "make_model_client",
    "make_trace_logger",
    "parse_aspect_ratio",
    "parse_size",
    "provider_spec",
    "register_provider",
    "resolve_service",
    "set_default_config",
    "sizes_for_ratio",
    "sniff_audio_mime",
    "sniff_image_mime",
    "supported_aspect_ratios",
    "supported_image_qualities",
    "supported_input_fidelities",
    "supported_image_sizes",
    "supported_modalities",
    "trace_file_path",
    "validate_history",
    "validate_images",
]

# A library adds a handler of last resort and nothing else. Without this, a warning
# emitted before the application configures logging prints "No handlers could be found";
# with anything more, the library would be dictating the application's log pipeline.
logging.getLogger(__name__).addHandler(logging.NullHandler())
