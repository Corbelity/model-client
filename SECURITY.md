# Security Policy

## Reporting a vulnerability

Please report security issues **privately**, not as a public GitHub issue.

- Preferred: [GitHub private vulnerability reporting](https://github.com/Corbelity/model-client/security/advisories/new)
- Or email: **info@corbelity.com**

Please include what you were running (version, Python version, which provider extras),
what you observed, and the smallest reproduction you have. If you have a fix in mind, say
so — but please don't open a public pull request for a security issue before we've agreed
how to handle it.

**What to expect.** This is a small project maintained alongside other work; there is no
SLA. A realistic expectation is acknowledgement within about a week, and a fix or a
decision within thirty days for anything confirmed. You'll be credited in the changelog
unless you'd rather not be. Please give us a chance to ship a fix before disclosing
publicly.

## Supported versions

Pre-1.0, only the latest released version is supported. Fixes ship in a new release rather
than as patches to older ones.

## Scope

**In scope:** anything in this package that leaks credentials, executes untrusted input,
writes outside the paths it was given, or lets model or provider output escape the
boundaries described below.

**Out of scope:** vulnerabilities in the provider SDKs or provider services themselves
(report those to the vendor); the content models generate; and misconfiguration of an
application that uses this library.

## Security properties worth knowing

These are the things most likely to bite you, and the assumptions this package makes.

### Traces contain your prompts

Trace records hold the **full** prompt and response text, and the artifacts directory holds
every image and audio payload sent or produced. Nothing is truncated — a trimmed trace
cannot reproduce a call, which is the whole reason the trace exists.

Consequences:

- Tracing is **off by default**. Enabling it is a deliberate choice.
- Keep trace files and their artifacts directory out of version control, out of shared
  drives, and out of any log shipper you haven't reviewed.
- If you run this against third-party data, treat the trace directory with the same
  controls as the data itself.

### Credentials

- API keys are read from the environment variables named in each `ProviderSpec`, or passed
  explicitly as `api_key=`. They are never written to a trace record and never logged.
- A per-instance `api_key=` is how a multi-user front end passes a session key through
  without mutating `os.environ`. **If you build such a front end, make sure that key does
  not reach your own application logs** — this package won't put it there, but your
  request logging might.
- This package never writes to `os.environ`, so it cannot leak a credential into another
  library's view of the environment.
- Missing-credential errors name the environment variables consulted; they never echo a
  value.

### Untrusted input

- Conversation history and image attachments are validated before any provider call:
  strict role alternation, no extra keys, string content only, and an allow-list of image
  MIME types. Malformed input raises `ValueError` rather than reaching the provider.
- Image MIME types are determined by **magic bytes**, not by a caller-supplied
  content type or filename extension. An unrecognised image defaults to a non-image type
  so it is rejected by validation rather than guessed at.
- No count or size caps are enforced on attachments. That is policy and belongs at your
  HTTP boundary — a script legitimately sending one 40 MB scan should not be limited by a
  UI's rules. **If you accept uploads from the internet, impose your own limits.**
- Model output is returned as-is and is never evaluated, deserialized, or executed. If you
  parse it as JSON or render it as HTML, that is your boundary to defend.

### Filesystem and network

- The only paths this package writes are the trace file and artifacts directory you
  configure. It creates no other files.
- The only hosts it contacts are the provider endpoints you configure, via the providers'
  own SDKs. It performs no other network access, and no telemetry of any kind.
- Provider SDKs are imported lazily and only when their provider is used.
