# Contributing

Thanks for looking. This is a small library maintained alongside other work, so the
sections below are mostly about setting expectations honestly — what tends to get merged,
what doesn't, and how long you should expect to wait.

## Before you write code

**Open an issue first for anything substantial.** A bug report needs no preamble, and a
one-line fix can come straight in as a pull request. But for a new feature, a new
provider, or a refactor, please describe it in an issue before building it. The worst
outcome here is you spending an evening on something that was never going to be merged —
and that is entirely avoidable with a short conversation.

Please also read [DESIGN.md](DESIGN.md), particularly the "What is deliberately not here"
section. Several reasonable-sounding features are absent on purpose, and that document
explains why.

## What tends to get merged

- Bug fixes, with a test that fails before the fix and passes after it.
- Provider fixes when an SDK or API changes shape — these are the most valuable
  contributions this project gets, because they're the ones that break silently.
- Catalog updates in `models.json` for new or retired models.
- Documentation corrections, including in code comments.
- Improved error messages, especially ones that name the specific thing to fix.

## What probably doesn't

- **New providers in this package.** The registry exists so you don't need one:
  `register_provider()` accepts a `ProviderSpec` from any package, and your provider
  inherits all the timing, logging, tracing and validation unchanged. Publish it as your
  own package. A provider moves into this one only if there's clear demand and I can
  commit to maintaining it, which means keeping up with an SDK I may not use.
- **Features listed as non-goals** in DESIGN.md §13 (retries, streaming, async, cost
  calculation, prompt templating, caching). Each is absent for a stated reason. If you
  think a reason is wrong, that's a genuinely interesting issue to open — but make the
  argument before writing the code.
- **Widening the scope.** This library calls models and records what happened. Agents,
  tool loops, and RAG belong in a layer above it.
- **Pure style changes.** Formatting is `ruff`'s business. Reorganizing files or renaming
  things for taste creates review load without changing behaviour.
- **Relaxing validation.** History validation refuses to auto-repair malformed input, and
  `sniff_image_mime` deliberately defaults to a non-image type. Both look like bugs and
  are not; see DESIGN.md §7 before changing either.

## Development setup

```bash
git clone https://github.com/Corbelity/model-client.git
cd model-client
uv sync --group dev --extra all
```

Then, before you open a pull request:

```bash
uv run ruff check .
uv run mypy
uv run pytest
```

All three run in CI and all three must pass. CI additionally installs the package with
**no** provider extras and runs the suite again, which verifies that the package imports
and answers capability questions with zero SDKs present. If your change adds a top-level
import of a provider SDK, that job is what will catch it.

Requires Python 3.12 or newer.

### Tests

New behaviour needs a test. The suite runs without network access, credentials, or
provider SDKs, and it should stay that way: a `ProviderSpec` plus a `ModelClient` subclass
is enough to stand up a fake provider, and `tests/test_client.py` shows how. If a test
seems to need more than that, the seam between the base class and the providers has
probably leaked, and that's worth raising in the pull request.

Tests that hit real endpoints are marked `@pytest.mark.live` and excluded by default. They
cost money and need credentials, so CI never runs them. Run them yourself with
`uv run pytest -m live` if your change touches provider request or response handling.

### Code style

`ruff` handles formatting and lint; don't hand-tune to taste.

The one convention worth stating: **comments explain why, not what.** The existing code
documents the non-obvious reasons — why WEBP needs a check at byte offset 8, why an empty
`images` list is omitted from a request rather than sent, why observability failures are
swallowed. Those comments are the most valuable thing in the source, and a pull request
that removes one to save a line will be asked to put it back.

Public API changes need a docstring and a `CHANGELOG.md` entry under `[Unreleased]`.

## Sign your commits (DCO)

Contributions are accepted under the
[Developer Certificate of Origin](https://developercertificate.org/): a short statement
that you wrote the contribution or otherwise have the right to submit it under this
project's license. There is no CLA to sign.

Add a sign-off line to each commit:

```bash
git commit -s -m "Fix WEBP detection for files with a leading RIFF chunk"
```

which appends:

```
Signed-off-by: Your Name <your.email@example.com>
```

Forgot on the last commit? `git commit --amend -s`. Across several? `git rebase --signoff`
over the range.

## Pull requests

- One logical change per pull request. A fix bundled with a refactor is hard to review and
  harder to revert.
- Describe what changes and **why**. A link to the issue is enough for the what.
- Note anything you couldn't verify — an SDK you don't have credentials for, a provider
  you couldn't test against. That's useful information, not a weakness in the submission.
- Expect follow-up questions. Being asked why you chose an approach isn't a rejection.

**AI-assisted contributions are fine** — this library exists to be used with these tools.
But you're accountable for what you submit: read it, run it, and be able to explain why it
works. A patch the submitter can't explain is worse than no patch, because reviewing it
costs more than writing it would have.

## Response times

This is maintained in gaps between other work. Realistically: a week or so for a first
response, sometimes longer. A pull request sitting unreviewed means I haven't got to it,
not that it's been rejected — feel free to bump it after a couple of weeks.

Security issues follow a different path: see [SECURITY.md](SECURITY.md), and please don't
open a public issue for one.

## License

By contributing, you agree that your contributions are licensed under the
[Apache License 2.0](LICENSE), the same license that covers this project.
