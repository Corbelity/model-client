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
pip install "corbelity-model-client[gemini-native]"
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
| `gemini-native` | Gemini, via Google's own SDK (`google-genai`) | `GEMINI_API_KEY`, `GOOGLE_API_KEY` | `GEMINI_NATIVE_BASE_URL` |
| `ollama-local` | Ollama on a local host | none — the box is trusted | `LOCAL_OLLAMA_URL`, `OLLAMA_HOST` (**required**) |
| `ollama` | Ollama Cloud | `OLLAMA_API_KEY` | `OLLAMA_CLOUD_URL` |
| `huggingface` | HuggingFace Inference | `HF_TOKEN`, `HUGGINGFACE_HUB_KEY`, `HUGGINGFACEHUB_API_TOKEN` | fixed |

`openai` and `huggingface` produce text, images and sound; `gemini-native` produces text
and video; the rest are text. Ask before you build:

```python
from corbelity.model_client import supported_modalities

supported_modalities("huggingface")   # frozenset({'text', 'image', 'sound'})
```

That answers without importing an SDK or needing a credential, so a caller can reject an
impossible request before spending anything on it.

Common aliases fold onto the canonical names (`hf` → `huggingface`, `open_router` →
`openrouter`, `google` → `gemini`, `google-genai` → `gemini-native`, `ollama-cloud` → `ollama`), and names are trimmed and
lower-cased.

### A note on Gemini

There are two Gemini services, and they reach the same models by different routes.

**`gemini-native`** uses Google's own SDK, `google-genai`. This is the route Google
recommends, and the one where Gemini's other capabilities arrive. Text, with history and
image attachments, under the same contract as every other provider — and video, through
Veo (see [Video](#video)):

```python
client = make_model_client("gemini-native", model="gemini-3.8-flash")
client.complete("Be terse.", "Summarise this.", images=[ImageInput.from_bytes(png)])
client.last_result.output_reasoning_tokens   # tokens spent thinking, billed as output
```

Three things behave differently from the chat-completions providers, all on purpose:

- **`completion_tokens` includes thinking.** Gemini reports reasoning tokens separately
  from the answer's, but bills them as output, so the flat figure adds them back. The split
  is on `output_text_tokens` and `output_reasoning_tokens`.
- **Thinking spends the output cap.** A low `max_tokens` can be used up entirely by
  reasoning, which comes back as `finish_reason="MAX_TOKENS"` with empty text (and a
  warning). Raise the cap.
- **A refused prompt** returns no answer at all; it is reported as
  `finish_reason="prompt_blocked"` with empty text, and the block reason is logged.

The key is read from `GEMINI_API_KEY`, then `GOOGLE_API_KEY`, and passed to the SDK
explicitly. The SDK is pinned to the Gemini Developer API: it does not switch to Vertex AI
because `GOOGLE_GENAI_USE_VERTEXAI` happens to be set in the environment.

**`gemini`** goes through Google's **OpenAI-compatibility endpoint**, a shim that gets
Gemini text working through the chat-completions dialect with no extra dependency. It is
text-only and stays exactly as it was, so code written against it keeps working. One key
serves both services, so `available_services()` reports both when it is set.

The built-in catalog shows both routes, each with its own model:

| Model | Service | Route |
|---|---|---|
| `gemini-3.8-flash` | `gemini-native` | Google's own SDK |
| `gemini-3.5-flash-lite` | `gemini` | OpenAI-compatibility endpoint |
| `veo-3.1-generate-preview`, `veo-3.1-lite-generate-preview` | `gemini-native` | Google's own SDK (video has no compatibility route) |

A model id can appear only once in a catalog (it is both the key and the name sent to the
API), so the same model can't be listed under both. Routing follows the service name you
pass, never the catalog, so any Gemini model can go either way:

```python
make_model_client("gemini-native", model="gemini-3.8-flash")   # native SDK
make_model_client("gemini", model="gemini-3.8-flash")          # compatibility endpoint
```

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
result = make_model_client("openai", model="gpt-image-2.5-flare").generate_image(
    "a corbel bracket, isometric, line art"
)
Path("out.png").write_bytes(result.data)   # result.mime_type tells you how to render it
```

Asking a provider for a modality it cannot produce raises `UnsupportedModalityError`
before any network call, rather than returning a provider-specific 400.

### Reference images

`generate_image()` also takes **reference images** to condition the generation on — a
character sheet, a set, a prop — so that a face or a place stays the same between
generations:

```python
from corbelity.model_client import ImageInput, make_model_client

client = make_model_client("openai", model="gpt-image-2.5-flare")
result = client.generate_image(
    "same character, three-quarter view, dusk lighting",
    images=[ImageInput.from_bytes(sheet_bytes, name="hero-sheet.png")],
)
```

Same `ImageInput` type and same validation as the vision input to `complete()`. Omit
`images` and the call is exactly what it was before the parameter existed.

Not every service can take them, and a service that can may cap how many. Both are
declared on the provider spec, so you can ask without building a client:

```python
from corbelity.model_client import get_spec

get_spec("openai").image_input             # True
get_spec("openai").max_reference_images    # 16
```

A service without the capability raises `UnsupportedImageInputError`, and too many
references raises `TooManyImagesError` — both **before** any network call, so you are
never told about the limit after uploading several megabytes. Today only `openai`
supports references; on that path they switch the call from the images endpoint to the
edit endpoint, which is handled for you.

### Size, aspect ratio and quality

Framing and quality are settings, not prompt text:

```python
client.generate_image("a wide establishing shot", aspect_ratio="16:9", quality="medium")
client.generate_image("a poster",                 size="2048x2048", quality="max")
```

`aspect_ratio` is the framing decision as a director states it — the client resolves it to
a concrete resolution on that service. `size` names the pixels outright. They are
**mutually exclusive**; passing both raises rather than resolving a precedence puzzle.
`quality` is independent of both, and omitting any of them leaves the provider's own
default in place.

When a ratio maps to several resolutions, `aspect_ratio` picks the **smallest**. Pixels
drive cost, so the cheap end is the safer default to choose on your behalf — a storyboard
thumbnail at 4K is money spent on an image nobody will examine closely. Name a `size` when
you want the large one.

**Nothing is ever silently substituted.** A ratio the service cannot frame, a malformed
size, or an unknown quality level raises `UnsupportedSizeError` / `UnsupportedQualityError`
before any network call. A near-match would mean believing you have 16:9 frames and
finding out much later by looking at them.

Ask ahead, without a client or a credential:

```python
from corbelity.model_client import (
    supported_aspect_ratios, supported_image_backgrounds, supported_image_qualities,
    supported_image_sizes, supported_input_fidelities, supported_output_formats,
)

supported_aspect_ratios("openai")     # ('1:1', '3:2', '2:3', '16:9', '9:16')
supported_image_qualities("openai")   # ('auto', 'low', 'medium', 'high', 'xhigh', 'max')
supported_image_sizes("openai")       # the resolutions it offers, plus 'auto'
supported_input_fidelities("openai")  # ('low', 'high')
supported_image_backgrounds("openai") # ('transparent', 'opaque', 'auto')
supported_output_formats("openai")    # ('png', 'jpeg', 'webp')
```

### Reference adherence

How closely the reference images are followed is a setting too:

```python
client.generate_image(
    "the same character, three-quarter view",
    images=[sheet],
    input_fidelity="high",
)
```

It only means anything alongside `images`, so passing it without them raises. That is
deliberate rather than lenient: a setting quietly dropped for having nothing to act on
looks exactly like a model that ignored it, and there would be no way to tell which
happened. An empty `supported_input_fidelities()` means the service has no such control --
not that it refuses references.

### Output encoding

```python
client.generate_image("a prop, isolated", background="transparent")
client.generate_image("a storyboard frame", output_format="webp", output_compression=70)
```

Transparency lets a character or prop be composited over a background generated separately,
rather than committing to whole frames. `output_format` and `output_compression` are storage
and fidelity: webp at a sensible compression is a large saving on a thumbnail nobody
inspects closely, while a final render wants lossless.

Two combinations contradict themselves and raise:

```python
client.generate_image("x", background="transparent", output_format="jpeg")  # no alpha
client.generate_image("x", output_format="png", output_compression=70)      # lossless
```

Both are judged against the format actually in effect, so `output_compression=70` with no
format named is also refused when the service's default is lossless -- a setting that cannot
do anything is refused rather than accepted and dropped.

The `mime_type` on the result always describes the **bytes**, never what was requested. If
they disagree, the bytes are what you will render, and `result.output_format` records what
the service said it produced so the disagreement is findable. Requesting a format changes
only the fallback used when the bytes are unrecognisable.

`supported_image_sizes` is what a service *offers*, not everything it will take: where a
provider documents an arbitrary `WIDTHxHEIGHT` (OpenAI does), a well-formed size outside
that list is passed through untouched rather than refused. That is passthrough, not
approval — if the provider rejects it, you see the provider's refusal, which still beats
this client guessing.

### Token usage, and what it costs

Where a provider reports a breakdown of its input, `ModelResult` carries it alongside the
flat figures:

```python
result.prompt_tokens         # everything the provider counted as input
result.input_text_tokens     # … of which text
result.input_image_tokens    # … of which image
result.input_cached_tokens   # … of which cached

result.completion_tokens     # everything the provider counted as output
result.output_text_tokens    # … of which text
result.output_image_tokens   # … of which image
```

This matters for cost rather than curiosity: the components bill at different rates, so a
caller pricing a call from the flat figures alone overstates a cached call and understates
one carrying image input. Output tokens dominate the cost of a generated image, so the
output split is what tells you whether a per-image cost figure is exact or an
approximation. Anything a provider doesn't report stays `None`, and the flat figures are
unchanged. The trace record carries both splits too, which is where a cost monitor should
read them from.

### What was actually produced

For generated media, `MediaResult` also reports the settings the provider says it **used**
— never an echo of what was asked for:

```python
result.size           # "2048x1152", as the provider states it
result.quality        # "low" | "medium" | "high" | "xhigh" | "max"
result.output_format  # "png" | "webp" | "jpeg"
result.background     # "transparent" | "opaque"
result.created        # the provider's own timestamp
result.extra          # any response field this package does not model yet
```

That distinction is the point. A caller that asked for 16:9 and silently got 1:1 has no
other way to find out except by opening the file and measuring it, and anything recording
evidence about its own output needs to record what it got rather than what it requested.

`mime_type` stays separate from `output_format` on purpose: it describes the bytes in
hand, and if the two ever disagree, the bytes are what you will render.

`extra` exists so a field a provider adds later is observable without waiting for a
release here — scalars only, never the payload.

### Video

Video takes a minute or more to generate, so it is a **job** rather than a call:
`submit_video()` returns at once, and the video is collected later.

```python
from corbelity.model_client import ImageInput, make_model_client

client = make_model_client("gemini-native", model="veo-3.1-lite-generate-preview")
job = client.submit_video(
    "A slow pan across a harbour at sunrise",
    first_frame=ImageInput.from_bytes(png),      # optional: animate from this image
    duration_seconds=8, resolution="1080p", aspect_ratio="16:9",
)
store(job.to_ref().to_dict())     # the job runs, and is billed, whether or not you poll

status = job.poll()               # one status request; never blocks for long
video = job.wait(timeout_s=600)   # or poll until done
Path("clip.mp4").write_bytes(video.data)
```

`generate_video(...)` is submit-then-wait in one call, for scripts. In a server, submit
and poll instead, so no thread is held for minutes.

**A job outlives the process that started it.** `to_ref().to_dict()` is plain data an
application can store; `client.resume_video(stored)` picks the job up anywhere, even after
a restart. A reference from a different service or model is refused rather than polled
through the wrong provider.

**Running out of time is not a failure.** `wait()` raises `VideoTimeoutError`, which
carries the job's reference: the video is still being made and can be resumed. The other
endings are `VideoJobFailedError` (the provider gave up, with its reason),
`ContentFilteredError` (refused on safety grounds, with the reasons given), and
`VideoJobNotFoundError` (the provider no longer knows the job: Veo keeps them two days).

**Inputs are named by role**, because an image means something different in each:
`first_frame` (animate from it), `last_frame` (interpolate towards it, which needs a
`first_frame`), `references` (people or objects to keep consistent) and `extend` (continue
an earlier clip; pass its result directly). The built-in catalog switches on
`first_frame` and `last_frame` for Veo; `references` and `extend` follow once they are
verified against the live API, and are refused for those models until then.

**Everything is checked before anything is sent**, against the service and then against
what the catalog states about the model. A value the model does not offer is refused,
never swapped for a nearby one. The one thing filled in is a setting a model rule forces
to a single value that you left unset — Veo's 1080p and 4K are 8-second only, so
`resolution="1080p"` alone sends `duration_seconds=8`, and the trace records what was
asked for and what was sent, separately. The same judgement is available with no client,
credential or network call, so a UI can grey out what will not work:

```python
from corbelity.model_client import VideoInputs, VideoOptions, resolve_video_request

resolve_video_request("gemini-native", "veo-3.1-lite-generate-preview",
                      VideoInputs(first_frame=frame), VideoOptions(resolution="1080p"))
# returns resolution='1080p', duration_seconds=8 -- or raises what a submit would
```

Veo on the Gemini API, specifically:

- **Audio is always produced.** `generate_audio` is refused (the Gemini API has no such
  parameter), and so is `seed` (the same: it exists only on Google's enterprise route).
- **The finished video is kept for two days.** `result.source_uri` is the provider's
  handle to it, which is what extending it needs.
- **The download goes only to the endpoint you configured**, with your key. The file id is
  read from the response; the host it names is never contacted.

> **Not yet reported:** the duration, dimensions and frame rate the video actually has.
> `MediaResult` reports what a provider states it produced, and Veo states none of these;
> reading them from the MP4 itself is planned. Until then, measure the file if it matters.

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
| `CORBELITY_VIDEO_POLL_INTERVAL` | `10` | seconds between polls in `wait()` |
| `CORBELITY_VIDEO_WAIT_TIMEOUT` | `600` | seconds `wait()` holds before `VideoTimeoutError` |

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

The one place a stated fact *is* enforced is a model's `video` block: what it says a model
cannot do (a resolution, a duration, a combination of inputs) is refused before anything
is sent. Veo's own rejection of a bad combination is a generic "unsupported request" that
does not say what conflicted, so the library's refusal is the only place a caller learns
what to change. A model without a block is passed through for the provider to judge, so an
unlisted video model still works too. See [Video](#video).

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

A video job writes **two** records, joined by the operation id: one when it is submitted
(with the input frames as role-named artifacts, `-first.png`, `-last.png`, ...) and one
when its end is first seen (with the video). Polls are not traced. An application that
already stores its videos can pass `JsonlTraceLogger(..., video_artifacts=False)` to keep
the records but skip writing video payloads.

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
timing, logging, tracing, validation and error handling unchanged.

A provider that generates images also implements
`_invoke_image(prompt, images, options)`, where `options` is an `ImageOptions` carrying the
requested `size` and `quality`. Omit any setting that is `None` rather than substituting a
default of your own, and declare what the service can actually do on its spec —
`image_input`, `max_reference_images`, `image_sizes`, `image_custom_size` and
`image_qualities` — so the base class refuses impossible requests before they reach you.

See [DESIGN.md](DESIGN.md) for why the seam is drawn there.

## Development

```bash
uv sync --group dev
uv run pytest          # live provider tests are excluded by default
uv run ruff check .
uv run mypy
uv run pytest -m live  # hits real endpoints; needs credentials and costs money
```

Run the `live` tests after any provider SDK major bump: the rest of the suite fakes every
provider, so it cannot see an SDK change at all. They need a credential and, for
HuggingFace, a model your enabled inference providers actually serve — override a default
that stops being routed with `HF_LIVE_TEXT_MODEL`, `HF_LIVE_IMAGE_MODEL` or
`HF_LIVE_SOUND_MODEL`. A missing credential or an unroutable model skips rather than fails,
so read the counts: a skip is not a pass.

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
