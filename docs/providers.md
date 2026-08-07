# Providers, routes, and cost control

## Credentials

Generation can use Hugging Face Inference Providers, OpenRouter, or both. Credentials are read from:

- `HF_TOKEN` or the standard Hugging Face login cache;
- `OPENROUTER_API_KEY`.

No credential is written to the workspace. Reading, judging, and analyzing an existing prepared run do not require provider credentials.

## Preflight and pinned routes

With the default `--backend auto` policy, preflight uses the preferred backend stored in each model's versioned configuration and falls back to the other backend only if the preferred backend is unavailable or all of its candidate providers fail. A missing credential disables only that backend. `--backend hf` and `--backend openrouter` remain whole-run overrides. A repeatable `--prefer-backend MODEL=hf|openrouter` option can override the automatic order for individual models during preflight.

The default panel prefers Hugging Face for `gemma-3-12b`, `llama-4-scout`, `ministral-3-14b-2512`, and `mistral-large-3-2512`. It prefers OpenRouter for `llama-3.3-70b-instruct`, `qwen3-30b-a3b-instruct-2507`, and `qwen3-235b-a22b-instruct-2507`. The versioned OpenRouter provider order is `groq`, `crusoe`, `novita`, `deepinfra` for Llama 3.3; `wandb`, `nebius`, `alibaba`, `siliconflow` for Qwen3 30B; and `alibaba`, `parasail`, `novita`, `deepinfra` for Qwen3 235B. For Qwen3 235B, DeepInfra is deliberately last because the observed Hugging Face/DeepInfra route repeatedly returned HTTP 504 and reduced adaptive concurrency to one request. Endpoint discovery still removes providers that are not currently exposed.

Hugging Face candidates come from the Hub inference-provider mapping. OpenRouter candidates come from the current per-model endpoints API, with the model panel's provider list retained as a deterministic fallback if endpoint discovery is unavailable. Every candidate is probed before selection. Attempts, latency, response status, token usage, and reported cost are recorded in `config/preflight.json` and the SQLite database. The preflight provenance also records the backend order actually used for every model.

A successful preflight selects one exact route for each model. `config/routes.json` is bound to the experiment manifest. Generation never switches a cell silently to another provider after preflight.

## Output ceilings and truncation recovery

Initial request ceilings are model-specific. They were derived from an observed smoke run with additional headroom and rounded to 256-token boundaries. The source profile is packaged as `instruction_duplication/smoke-profile-2026-08-05.json`.

The initial ceilings are deliberately lower than provider capability because they are also used for realistic cost reservations. If a completed provider response reports a truncation finish reason, the attempt is retained and the same cell is retried with a ceiling 50% larger, rounded to the next 256-token boundary. This repeats only as needed and never exceeds the configured model/provider capability.

An expanded truncation retry has a distinct logical idempotency key because its request payload differs from the earlier attempt. Transport retries at the same output ceiling reuse the same logical idempotency key.

## Budget reservations

Before dispatching a request, the runner reserves the route-priced worst-case cost for that attempt's output ceiling. A completed attempt reconciles the reservation against provider-reported cost when available, or a conservative local estimate otherwise.

An explicit HTTP 429 response is accounted as zero generation spend and releases its reservation because the provider rejected the request for rate limiting. Timeouts, network failures, and other transient failures remain conservatively accounted because provider-side execution may have started. Refusals and truncations retain their realized or conservatively estimated cost.

A provider can report a final charge above the reservation only after the request has completed. In that case the attempt is retained and generation stops rather than dispatching additional work beyond the cumulative cap.

## Transient retries, concurrency, and cooldowns

Generation uses independent per-model workers, a global request limit, shared provider gates, and an adaptive limit for each exact pinned route.

A transient provider failure does three things:

1. records the attempt and leaves the cell retryable;
2. establishes or extends the provider-wide cooldown using `Retry-After` when available and deterministic backoff otherwise;
3. reduces concurrency for the affected exact route.

Retryable cells are collected and run again in later sweeps. `--retries N` means up to `N` additional transient retry sweeps after the initial sweep. This avoids spending all retry opportunities immediately while a provider is still throttling the route. Sustained successful requests gradually restore route concurrency. A success does not cancel an already active provider cooldown.

The pinned provider is unchanged throughout these retries. Provider throttling therefore affects throughput and completion, not the experimental route identity. Progress output names constrained exact routes as `model@backend/provider` and shows `limit=current/unconstrained`, active requests, and any remaining shared-provider cooldown, so a provider bottleneck can be identified while the run is still in progress.

`instruction-duplication status --workspace <path>` reports stored progress and accounted cost. After transient retry rounds are exhausted, `retryable` cells proceed to intention-to-treat analysis as generation failures. The all-in-one `reproduce` workflow stops before final judging and analysis only if cells remain `pending`, `running`, or `budget_blocked`.
