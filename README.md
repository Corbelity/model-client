# corbelity-model-client

[![CI](https://github.com/Corbelity/model-client/actions/workflows/ci.yml/badge.svg)](https://github.com/Corbelity/model-client/actions/workflows/ci.yml)
[![Python 3.12+](https://img.shields.io/badge/python-3.12%2B-blue)](https://www.python.org/downloads/)
[![License: Apache 2.0](https://img.shields.io/badge/license-Apache%202.0-green)](LICENSE)

One way to call text, image and speech models across Anthropic, OpenRouter, Ollama and
HuggingFace — with identical logging and tracing on every call, whichever provider
answered.

```python
from corbelity.model_client import make_model_client

client = make_model_client("anthropic", model="claude-sonnet-5")
print(client.complete("You are terse.", "Name three prime numbers."))
```

---

## Why I built this

     Over the past six months I built a series of agents, tools, and test harnesses against AI models.
     In every one of them I wanted to run the same code against different models, local ones included,
     and compare what came back.

     That started as a config-driven switcher to paper over the differences in how each provider wants
     to be called. From there it became a pluggable client, interchangeable at runtime and instrumented
     so that every call produced the same log line and the same trace record no matter which provider
     answered. I was copying it into each new project.

     This is that code, pulled out once so it can be reused and maintained in one place. If you're
     evaluating models across providers, or want the same code path to run against a local Ollama
     box and a cloud API, it may save you the same work.

---

## Install

The core package has **no required dependencies**. Each provider SDK is an extra, so
installing this to talk to one provider never drags in the other three:

```bash
pip install "corbelity-model-client[anthropic]"
pip install "corbelity-model-client[openai]"
pip install "corbelity-model-client[openrouter]"
pip install "corbelity-model-client[gemini]"
pip install "corbelity-model-client[ollama]"
pip install "corbelity-model-client[huggingface]"
pip install "corbelity-model-client[all]"
```

Asking for a provider whose SDK is missing raises an error containing the exact install
command, rather than a `ModuleNotFoundError` naming a package you never asked for.

Requires Python 3.12 or newer.

## Providers

There is no global local/cloud switch. The **service name** picks the provider, so every
call site says plainly which one it is using.

| Service | What it is | Credential (first match wins) | Endpoint |
|---|---|---|---|
| `anthropic` | Anthropic API, direct | `ANTHROPIC_API_KEY` | fixed |
| `openai` | OpenAI API, direct | `OPENAI_API_KEY` | `OPENAI_BASE_URL` |
| `openrouter` | OpenRouter, cloud | `OPENROUTER_API_KEY` | `OPENROUTER_BASE_URL` |
| `gemini` | Gemini, via Google's OpenAI-compatible endpoint | `GEMINI_API_KEY`, `GOOGLE_API_KEY` | `GEMINI_BASE_URL` |
| `ollama-local` | Ollama on a local host | none — the box is trusted | `LOCAL_OLLAMA_URL`, `OLLAMA_HOST` (**required**) |
| `ollama` | Ollama Cloud | `OLLAMA_API_KEY` | `OLLAMA_CLOUD_URL` |
| `huggingface` | HuggingFace Inference | `HF_TOKEN`, `HUGGINGFACE_HUB_KEY`, `HUGGINGFACEHUB_API_TOKEN` | fixed |

`openai` and `huggingface` produce text, images and sound; the rest are text. Ask before
you build:

```python
from corbelity.model_client import supported_modalities

supported_modalities("huggingface")   # frozenset({'text', 'image', 'sound'})
```

That answers without importing an SDK or needing a credential, so a caller can reject an
impossible request before spending anything on it.

Common aliases fold onto the canonical names (`hf` → `huggingface`, `open_router` →
`openrouter`, `google` → `gemini`, `ollama-cloud` → `ollama`), and names are trimmed and
lower-cased.

### A note on Gemini

`gemini` reaches Google's models through their **OpenAI-compatibility endpoint**, which is
a shim — Google's own guidance is that if you aren't already using the OpenAI libraries,
you should call the Gemini API directly. It's here because it gets Gemini text working
through the existing dialect with no new dependency.

So `gemini` is text-only today. Image generation through the shim is unverified, and audio
generation runs over the Live API, which is a bidirectional streaming session rather than a
request/response call and so can't be wrapped honestly by `generate_speech()`. Both wait
for a native provider built on `google-genai`, which will arrive as a **separate service**
rather than a change to this one — so code written against the shim won't break.

## Usage

### Text, with history and images

```python
from corbelity.model_client import ImageInput, make_model_client

client = make_model_client("openrouter", model="openai/gpt-4o", temperature=0.2)

answer = client.complete(
    "You are a careful editor.",
    "What changed between these two drafts?",
    history=[
        {"role": "user", "content": "Here is the first draft."},
        {"role": "assistant", "content": "Noted."},
    ],
    images=[ImageInput.from_bytes(png_bytes, name="draft.png")],
)

print(client.total_tokens, client.finish_reason)
```

History must alternate, start with a `user` turn, and end with an `assistant` turn.
Nothing is auto-repaired: silently merging same-role turns would rewrite your context
without saying so. Images attach to the current turn only and never enter history. Both
are validated before the provider is touched, so a malformed request costs nothing.

### Images and speech

```python
result = make_model_client("huggingface", model="black-forest-labs/FLUX.1-dev").generate_image(
    "a corbel bracket, isometric, line art"
)
Path("out.png").write_bytes(result.data)   # result.mime_type tells you how to render it
```

Asking a provider for a modality it cannot produce raises `UnsupportedModalityError`
before any network call, rather than returning a provider-specific 400.

## Configuration

Resolution order, everywhere:

**per-call argument → injected `ModelConfig` → environment → package default**

```python
from corbelity.model_client import ModelConfig, make_model_client, set_default_config

config = ModelConfig(default_model="claude-sonnet-5", default_max_tokens=8192)
client = make_model_client("anthropic", config=config)   # or set_default_config(config)
```

| Variable | Default | Notes |
|---|---|---|
| `CORBELITY_DEFAULT_SERVICE` | `openrouter` | used only when a call names no service |
| `CORBELITY_DEFAULT_MODEL` | `anthropic/claude-opus-4.8` | |
| `CORBELITY_MAX_TOKENS` | `4096` | a cap, not a target |
| `CORBELITY_TEMPERATURE` | `0.7` | |
| `CORBELITY_TOP_P` | unset | left to the provider unless you set it |
| `CORBELITY_NUM_CTX` | `16384` | Ollama context window; ignored elsewhere |
| `CORBELITY_MODEL_CATALOG` | unset | path to a catalog that merges over the built-in one |
| `CORBELITY_SERVICES` | unset | comma-separated allow-list of services to list |

This package **does not** call `load_dotenv()`, **does not** write to `os.environ`, and
**does not** configure logging. Those are your application's decisions. It reads only the
variables above and the per-provider credentials, and attaches a `NullHandler` to its own
logger, so importing it produces no output and changes no global state. If you want `.env`
loading, do it in your entry point before building a client.

## Model catalog

Model facts churn on the providers' schedule, not on this package's release schedule, so
they live in JSON rather than in code. The built-in `models.json` is a starting set; your
own catalog merges over it by model id, so a file only carries what it changes:

```bash
export CORBELITY_MODEL_CATALOG=/etc/models.json
```

```python
from corbelity.model_client import load_catalog

for entry in load_catalog().for_service("openrouter"):
    print(entry.id, entry.accepts_images, entry.cost_per_1k_input)
```

The catalog is descriptive, not enforcing — calling a model that isn't listed works fine.
It exists so a UI can populate a dropdown, and so provider code can read a capability flag
instead of matching hardcoded model-name prefixes. Two such flags ship today:

- `supports_sampling` — some OpenAI-compatible models reject `temperature` and `top_p`
  with a 400 rather than ignoring them. It does **not** apply to Anthropic: the Messages
  API withdrew sampling parameters altogether, so that client never sends them and the
  flag would have nothing to govern.
- `max_tokens_param` — newer OpenAI models reject `max_tokens` and require
  `max_completion_tokens`. Set it to the name that model wants.

Both default to the permissive behaviour when a model isn't listed, so an unlisted model
still works.

### Showing only the services you can actually call

A user catalog merges over the built-in one, which means it can add a model or correct an
entry, but it can't remove one. If you only hold an OpenRouter key, you don't want four
providers' models in a dropdown you can't use.

That's a filter on the *listing*, not a change to the catalog:

```python
from corbelity.model_client import ModelConfig, available_services, catalog_for

config = ModelConfig(services=available_services())
for entry in catalog_for(config):
    print(entry.id)          # only services whose credentials are actually set
```

`available_services()` reports which services have their credentials and endpoints present
in the environment — with only `OPENROUTER_API_KEY` set it returns `("openrouter",)`, and
`ollama-local` appears when `LOCAL_OLLAMA_URL` is set, since a local host needs an address
rather than a key. It checks that a variable is **set**, never that the credential works;
verifying would mean a network call per provider at startup.

Nothing applies it for you. Silently hiding a provider because an environment variable is
missing is how you get "why did my model disappear?", so the filter is always an explicit
choice. Set it yourself if you'd rather:

```python
ModelConfig(services=("openrouter", "ollama-local"))
```

or from the environment:

```bash
export CORBELITY_SERVICES=openrouter,ollama-local
```

Aliases fold, so `("hf",)` and `("huggingface",)` mean the same thing. `None` (the default)
means no filter; an empty tuple means show nothing, which is a different thing.

**This narrows what you list, and nothing else.** It does not prevent a call to a
filtered-out service, and provider code still reads the full catalog — so the capability
flags for every model stay findable no matter what a UI happens to be showing.

## Tracing

Every provider and every modality funnels through one wrapper, so tracing plugs in once
and covers all of them. Inject any object with a matching `llm_call()` — nothing needs to
subclass or import anything:

```python
from corbelity.model_client import JsonlTraceLogger, make_model_client

trace = JsonlTraceLogger("logs/traces.jsonl")
client = make_model_client("anthropic", trace=trace)
```

Or build one from the environment (`TRACE_ENABLED=1`, `TRACE_FILE=...`), which returns
`None` when tracing is off so the disabled path costs nothing:

```python
from corbelity.model_client import make_trace_logger

client = make_model_client("anthropic", trace=make_trace_logger())
```

Each record holds the full prompts, the response, timing, token usage and finish reason.
Binary payloads — generated images and audio, and attached input images — are written
beside the trace as named files, with the record keeping `{name, mime_type, bytes}`; a
multi-megabyte PNG inlined as base64 would make the trace unreadable. Failed calls are
traced with their latency and error before the exception is re-raised, because the call
you can't reproduce is the one you most needed the trace for.

> **Traces contain full prompt and response text.** Treat the trace file and its artifacts
> directory as sensitive, keep them out of version control, and think before enabling
> tracing on anything handling third-party data. Tracing is off by default for this
> reason.

Tracing is never load-bearing: a tracer that raises is logged and swallowed, so a full
disk cannot turn a working model call into a failed one.

## Adding your own provider

```python
from corbelity.model_client import ProviderSpec, register_provider

register_provider(ProviderSpec(
    name="my-gateway",
    client_path="my_package.clients:MyGatewayClient",
    key_env=("MY_GATEWAY_KEY",),
    default_base_url="https://gateway.internal/v1",
    modalities=frozenset({"text"}),
))
```

Subclass `ModelClient`, implement `_build_client()` and `_invoke()`, and you inherit the
timing, logging, tracing, validation and error handling unchanged. See
[DESIGN.md](DESIGN.md) for why the seam is drawn there.

## Development

```bash
uv sync --group dev
uv run pytest          # live provider tests are excluded by default
uv run ruff check .
uv run mypy
uv run pytest -m live  # hits real endpoints; needs credentials and costs money
```

## Documentation

- [DESIGN.md](DESIGN.md) — why the library is shaped the way it is
- [CHANGELOG.md](CHANGELOG.md)
- [SECURITY.md](SECURITY.md)

## Status and support

Pre-1.0: the public API may change between minor versions, and breaking changes will be
called out in the changelog. Provided as-is under the Apache License 2.0, with no support
commitment or SLA. Issues and pull requests are welcome, but response time is
best-effort.

## License

Apache License 2.0 — see [LICENSE](LICENSE) and [NOTICE](NOTICE).

Provider names are the trademarks of their respective owners and are used only to identify
the service being called. Each provider SDK is an optional dependency under its own
license and is not redistributed here.
