# Design notes

Why this library is shaped the way it is. The README covers what it does; this covers the
decisions behind it, including the ones that cost something.

The problem it solves is narrow: **calling several model providers and being able to tell,
afterwards, exactly what happened.** Not an agent framework, not a gateway, not a serving
stack. Everything below follows from taking that scope seriously and refusing to widen it.

---

## 1. One template method owns observability

The public entry points — `complete()`, `generate_image()`, `generate_speech()` — do not
call providers. They call `_run()`, which times the call, records it, and delegates the
actual provider work to a `_invoke*()` method on the subclass.

```
complete()                 validate, build the request record
  └─ _run(modality, fn)    time it, log it, trace it, catch and re-raise
       └─ _invoke()        the ONLY provider-specific code
```

The reason is arithmetic. With four providers and three modalities, putting timing and
logging in each provider means up to twelve places that must agree on what a call record
looks like — and they will not stay agreeing. Here there is one. Add a fifth provider and
it arrives fully instrumented without its author writing a line of logging.

The consequences that matter:

- **One INFO line per call**, identical in shape across providers: service, model,
  modality, prompt and completion tokens, finish reason, latency.
- **Failures are traced before the exception is re-raised**, with their latency and the
  request that caused them. A failed call you cannot reproduce is the one you most needed
  the record for, and provider exceptions rarely carry the request.
- **Tracing is injected, never imported.** `_run()` takes a `TraceSink`; the request path
  has no dependency on any concrete tracer.

The cost: a provider cannot customise the observability of its own calls. That is the
point, and it is the right trade for this scope — but it is a constraint, not a freebie.

## 2. Observability is never load-bearing

A tracer that raises is caught, logged at WARNING, and swallowed. A full disk must not
turn a working model call into a failed one.

This is the rule that decides several smaller questions. Trace writes catch `OSError`
around the file handle. Artifact writes that fail degrade to metadata rather than losing
the whole record. `NullTrace` exists so callers who want a non-`None` sink have one,
though passing `trace=None` is marginally cheaper because the `if self._trace is not None`
guard skips the call entirely.

The inverse rule matters too: **observability never changes behaviour.** Nothing in the
trace path touches the request, and a call produces the same result traced or not.

## 3. The provider seam is drawn at two methods

A provider implements `_build_client()` (construct the SDK object) and `_invoke()`
(make the call, return a normalized result). Everything else — validation, timing,
logging, tracing, credential resolution, modality checks, error translation — is inherited.

Where several providers speak the same wire protocol, the shared half lives in an
intermediate base and the leaves carry only a `SPEC`. Three services use the OpenAI
chat-completions dialect through the same SDK — OpenRouter, OpenAI directly, and Gemini's
compatibility endpoint — so `OpenAICompatibleClient` holds the request and response
handling and each leaf is about a dozen lines. The two Ollama clients share a base the
same way. The test is protocol, not vendor: a provider gets its own `_invoke()` when it
speaks differently, not when it bills differently.

Modality is a separate axis from dialect. `OpenAIClient` inherits text from that base but
implements `_invoke_image()` and `_invoke_speech()` itself, because images and speech come
from entirely different SDK surfaces (`images.generate`, `audio.speech.create`) with their
own response shapes. Sharing the text path does not imply sharing the others.

Two design rules keep that seam honest:

**Results are normalized, not passed through.** `_invoke()` returns an `LLMResult` or
`MediaResult` carrying text or bytes plus prompt/completion/total tokens and a finish
reason. If providers' native response objects leaked upward, every caller would need a
provider-shaped branch and the uniform log line would be impossible. Token extraction is
best-effort via `get_field()`, which reads a name as either a dict key or an attribute:
the SDKs return dicts, pydantic models and namespaces, and a field that is simply absent
should read as "not counted", never raise.

**`_invoke()`'s parameters are not defaulted.** `history` and `images` are required
positional arguments even though both are usually empty. A subclass that silently dropped
an attachment would answer confidently about a picture the model never saw. A `TypeError`
at import time is strictly better than that.

## 4. The service name is the routing decision

There is no `local=True` flag and no "is this a local URL?" inference. `ollama-local` and
`ollama` are separate registered services with separate classes, differing only in how the
SDK client is built.

A boolean flag has to be threaded through every layer that might set it, and at the call
site `make_model_client(local=use_local)` tells you nothing about where the call went.
`make_model_client("ollama-local")` tells you exactly. The cost is one extra class for
what is a few lines of difference; the benefit is that no call site is ambiguous and no
code has to infer intent from a URL.

Aliases (`hf`, `open_router`, `ollama-cloud`) fold onto canonical names at the registry
boundary, so tolerance for spelling lives in one place rather than in scattered
comparisons.

## 5. Provider identity is data

`ProviderSpec` holds everything about a provider that is not request handling: canonical
name, credential environment variables, endpoint variables and default, whether an
endpoint is mandatory, supported modalities, aliases, the pip extra, and the implementing
class as a `"module:ClassName"` string resolved on first use.

That last indirection is load-bearing in two ways:

- It removes a circular import. Providers import the base class, which imports the
  registry; if the registry imported the providers, the cycle would have to be broken
  with a function-local import somewhere unpleasant.
- It lets `supported_modalities("huggingface")` answer **without importing anything**. A
  caller can reject an impossible request before installing an SDK, finding a credential,
  or spending a network round trip.

Credential variables are a **tuple, tried in order**. That is how `HF_TOKEN` (what the
HuggingFace libraries read) and the older names this code used are all honoured. The
prototype achieved the same thing by assigning one name onto another in `os.environ` at
import time — correct in an application, disqualifying in a library. A lookup order gets
the identical result without writing to global state that other libraries also read.

Third-party providers register through the same mechanism the built-ins use. Nothing in
the registry is privileged.

## 6. Model facts are data, and go stale on someone else's schedule

Which models exist, what they cost, whether they accept images, whether they tolerate a
`temperature` parameter — none of that changes when this package changes. It changes when
a vendor ships. So it lives in `models.json`, loaded through `importlib.resources` (not
`__file__` path arithmetic, so it works from a wheel or a zipapp), and a user catalog
merges over the built-in one by model id.

The concrete case: some current Anthropic models **removed** sampling parameters and
return a 400 rather than a warning if you send `temperature`. The prototype carried a
hardcoded tuple of model-name prefixes. Now the catalog's `supports_sampling` flag is
authoritative when stated, and the prefix match is only a fallback. A model released after
this package ships is a JSON edit, not a release.

The second flag makes the same point from a different direction. Newer OpenAI models
reject `max_tokens` and require `max_completion_tokens` — not a capability difference but
a **spelling** one, and one that will keep happening as providers revise their APIs. So
`max_tokens_param` names the key rather than encoding a rule: the shared
chat-completions client builds its payload with whatever the catalog says this model
wants, defaulting to `max_tokens`. A prefix list would have needed editing on OpenAI's
schedule; this needs editing on the catalog's.

The general shape worth keeping: when providers differ in a way that is **data** (a name,
a flag, an endpoint), put it in data. Reserve code for differences in **behaviour**.

The catalog is **descriptive, not enforcing**. Calling an unlisted model works fine.
Nothing here gates a request — it exists so a UI can populate a dropdown and so provider
code can read a flag instead of hardcoding a list.

### Listing is filtered; calling is not

The merge in the previous paragraph can add an entry and can correct one, but it cannot
remove one. That leaves a real gap: someone holding a single provider's key does not want
four providers' models in a dropdown, and no merge rule will take them out.

The fix is a filter on the *listing path* rather than on the catalog. `catalog_for(config)`
narrows to `config.services`; `available_services()` reports which services have their
credentials and endpoints present, so the common case is one line:

```python
ModelConfig(services=available_services())
```

Two constraints make this safe, and both are worth stating because both are easy to
violate later while tidying up:

**Provider code reads the unfiltered catalog.** `_accepts_sampling_params()` calls
`load_catalog()`, not `catalog_for()`. A capability flag must be findable for any model
the caller actually names, whatever a UI happens to be listing at the time. Narrowing a
display must never change how a call behaves — the moment those two call sites are merged
"for consistency", filtering a service out of a dropdown starts sending `temperature` to a
model that rejects it. `tests/test_service_filter.py` has a test whose only job is to fail
if that happens.

**Filtering is never automatic.** `available_services()` supplies an answer; nothing
applies it. A library that silently hid a provider because an environment variable was
unset would produce a question — "why is my model gone?" — whose answer lives in someone
else's code. One explicit line keeps the behaviour where the reader can see it.

The same reasoning explains what `available_services()` deliberately does *not* do: it
checks that a credential is **set**, never that it works. Verification would mean a network
call per provider at startup, and a provider that is briefly down is not a provider you
should stop offering.

## 7. Validation refuses to repair

History must strictly alternate, begin with a `user` turn, and end with an `assistant`
turn. The last rule is what guarantees that `history + [current user turn]` is still
alternating.

Malformed history is **rejected, never fixed**. Merging two consecutive user turns would
silently rewrite the caller's context and change what the model was asked — a failure that
shows up as a subtly wrong answer rather than an error. Every message names the offending
index, so a UI can point at the card.

This is also where provider disagreement gets normalized. Anthropic hard-rejects malformed
histories; OpenRouter mostly tolerates them. Validating here means the behaviour is the
same everywhere instead of provider-dependent, which is the difference between a rule you
can learn and a surprise you discover in production.

Validation runs **before** the provider is touched, so a malformed request costs nothing.

### The sniffer default is deliberately wrong-looking

`sniff_image_mime()` returns `application/octet-stream` for anything it does not
recognise — a *non-image* type, from an image sniffer. That looks like a bug and is the
opposite.

A fetched URL's `Content-Type` is frequently absent or generic, so the type has to come
from magic bytes. If an unrecognised blob defaulted to, say, `image/png`, the error would
surface as an opaque provider-side 400. Defaulting to a non-image type instead routes it
into `validate_images()`, which rejects it with a message naming exactly which types are
allowed.

(The audio sniffer defaults to `audio/wav`, because there the input is a model's own
output in a known-good container, not an arbitrary blob. Different situation, different
default.)

WEBP gets its own check because its tag sits at byte offset 8, after `RIFF` and a length —
and a bare `RIFF` prefix is just as likely to be a WAV file. A naive prefix match gets this
wrong in a way that only shows up on one file format.

**What validation deliberately does *not* do:** enforce counts or size caps. Those are
policy and belong at the HTTP boundary where policy already lives. A script sending one
40 MB scan is doing something legitimate that a UI's limits should not forbid.

## 8. Configuration is accepted, never reached for

A library never reads the world; it accepts the world.

`ModelConfig` is an immutable dataclass with a single documented resolution order:
**per-call argument → injected config → environment → package default.** Four words of
documentation that answer every "where did this temperature come from?" question.

What this package refuses to do, all of which the prototype did and all of which are
correct in an application:

- **No `load_dotenv()`.** Reading a `.env` file is a decision about a process, and a
  library that makes it silently reaches outside its own boundary.
- **No `os.environ` writes.** See §5.
- **No logging configuration.** No `basicConfig`, no handlers, no level setting on other
  libraries' loggers — only a `NullHandler` on its own. A library that configures the root
  logger hijacks its host application's log pipeline, and the bug report arrives eventually.
- **No import-time side effects at all.** `get_default_config()` builds from the
  environment lazily, on first use.

`ModelConfig.from_env()` takes an optional mapping so tests never touch `os.environ`, and
it raises on a malformed numeric value at construction rather than three seconds into a
provider round trip.

One historical note worth keeping: the prototype's config class was decorated
`@dataclass(frozen=True)` but had **no annotated fields**, so `@dataclass` generated an
`__init__` taking no arguments and the object could not be constructed with overrides at
all. The defensive `getattr(settings, "...", None)` calls scattered through the client were
compensating for that. Annotating the fields is most of what made configuration injectable.

## 9. Optional dependencies, imported lazily

Every provider SDK is an extra, and each is imported inside `_build_client()` rather than
at module top. So:

- installing this to call one provider does not drag in the other three;
- `import corbelity.model_client` succeeds with **zero** SDKs installed, and capability
  questions still answer;
- a broken release of any single SDK cannot break importing this package.

`load_sdk()` turns a missing optional dependency into an error carrying the exact pip
command. A bare `ModuleNotFoundError` naming a package the user never asked for makes the
extras system look like a bug rather than a feature.

CI installs **without** extras in one job precisely to keep this honest.

## 10. Configuration errors are resolved before SDK imports

Both Ollama clients resolve their host before importing the SDK. The order is deliberate:
with the import first, a missing `LOCAL_OLLAMA_URL` raises `MissingBaseUrlError` on a
machine that has the extra installed and `MissingDependencyError` on one that doesn't.

Same fault, two different errors, depending on an unrelated install. That is the kind of
environment-dependent behaviour that is miserable to reproduce from a bug report, and
avoiding it costs nothing.

## 11. Error types derive from the built-ins they replace

Everything raises from `ModelClientError`, so an application can catch one type. But
errors that were previously raised as built-ins also derive from those built-ins:
`MissingCredentialsError` is a `ValueError`, `MissingDependencyError` is an `ImportError`.

Web layers routinely map `ValueError` to a 400. Introducing a named type without that
second base would be a silent behaviour change at the HTTP boundary — the kind that turns
a clean 400 into a 500 and is noticed in production rather than in review.

Pure input-validation failures stay as plain `ValueError`. They describe a caller mistake,
not a library condition, and nothing is gained by making callers import a type to catch
them.

## 12. Degraded responses are loud

An empty response, or one whose `finish_reason` indicates truncation or filtering, logs a
WARNING. A truncated answer that looks like a complete one is the single most expensive
failure mode in this domain: it produces a confident, wrong result that nothing flags.

The set of "did not finish cleanly" reasons is provider-specific and lives in one frozen
set with the vocabularies documented beside it:

| Provider | Clean | Not clean |
|---|---|---|
| OpenAI / OpenRouter | `stop` | `length`, `content_filter`, `error` |
| Anthropic | `end_turn` | `max_tokens`, `refusal` |
| Ollama | `stop` | `length` |
| HuggingFace / TGI | `stop`, `eos_token` | `length` |

`tool_use`, `tool_calls` and `stop_sequence` are normal completions and are intentionally
absent. `error` is OpenRouter failing mid-stream: it returns the partial text it had with
no usage block, so this finish reason is the *only* signal that the answer is cut short.

## 13. What is deliberately not here

Naming the non-goals is part of the design, because each one is a thing someone will
reasonably ask for.

- **Retries and backoff.** Every provider SDK already has a retry policy, and stacking a
  second one on top produces surprising latency and duplicate spend. Configure the SDK's.
- **Streaming.** The interface is request/response, and every call is measured end to end.
  Streaming changes both, so it would be a second interface rather than a flag on this one.
- **Async.** Same reasoning. A sync interface used from a threadpool is adequate for the
  workloads this was built for; an async variant would be a parallel class hierarchy, not
  a parameter.
- **Cost calculation.** The catalog carries per-token prices, but nothing multiplies them
  out. Prices change without notice and a stale number quoted confidently is worse than no
  number.
- **Prompt templating, agents, tool loops, RAG.** Different libraries. This one is the
  bottom layer they would sit on.
- **Response caching.** Cache keys for model calls are subtle (temperature, history,
  attachments all participate) and the right policy is application-specific.

## 14. Testing strategy

The seam described in §3 is what makes this testable without a network: a `ProviderSpec`
plus a subclass is enough to stand a provider up. The suite covers validation, MIME
sniffing, configuration resolution, credential ordering, the catalog, the registry, and
the observability contract — including that a tracer which raises does not break the call,
and that a failure is traced before being re-raised.

If a test ever needs more than a spec and a subclass to fake a provider, the seam has
leaked and that is the bug to fix.

Live provider tests are marked `live` and excluded by default. They cost money, need
credentials, and fail for reasons that have nothing to do with this code.
