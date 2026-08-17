# Providers, routes, and cost control

## Credentials

Generation can use Hugging Face Inference Providers, OpenRouter, or both. Credentials are read from:

- `HF_TOKEN` or the standard Hugging Face login cache;
- `OPENROUTER_API_KEY`.

No credential is written to the workspace. Reading, judging, and analyzing an existing prepared run do not require provider credentials.

## Preflight and pinned routes

With the default `--backend auto` policy, preflight uses the preferred backend stored in each model's versioned configuration and falls back to the other backend only if the preferred backend is unavailable or all of its candidate providers fail. A missing credential disables only that backend. `--backend hf` and `--backend openrouter` remain whole-run overrides. A repeatable `--prefer-backend MODEL=hf|openrouter` option can override the automatic order for individual models during preflight.

The default panel prefers Hugging Face for `gemma-3-12b`, `llama-4-scout`, `ministral-3-14b-2512`, and `mistral-large-3-2512`. It prefers OpenRouter for `llama-3.3-70b-instruct`, `qwen3-30b-a3b-instruct-2507`, and `qwen3-235b-a22b-instruct-2507`. The versioned OpenRouter provider order is `groq`, `crusoe`, `novita`, `deepinfra` for Llama 3.3; `wandb`, `nebius`, `alibaba`, `siliconflow` for Qwen3 30B; and `alibaba`, `parasail`, `novita`, `deepinfra` for Qwen3 235B. For Qwen3 235B, DeepInfra is deliberately last because the observed Hugging Face/DeepInfra route repeatedly returned HTTP 504 and reduced adaptive concurrency to one request. Endpoint discovery still removes providers that are not currently exposed.

Hugging Face candidates come from the Hub inference-provider mapping. OpenRouter candidates come from the current per-model endpoints API, with the model panel's provider list retained as a deterministic fallback if endpoint discovery is unavailable. Every candidate receives two sequential functional probes followed by a simultaneous capacity batch of `min(--per-model-concurrency, route maximum)` requests. Capacity probes are not retried: every slot must succeed at once, otherwise that candidate is rejected and the next fixed provider/backend candidate is considered. Functional probes may retry a transient error twice with the unchanged request and logical idempotency key.

Each successful probe validates the provider/request contract: `finish_reason=stop`, visible non-empty content, no exposed reasoning field, no refusal or truncation, and (for OpenRouter) the requested provider binding. Provider-reported usage is accounting telemetry rather than a second truncation signal: some hosted routes have reported billed token counts above the request limit on otherwise complete `stop` responses. Experimental instruction-following behavior—including malformed protocol output or a missing/unparseable final answer—is retained as provenance but does not reject a route, because it is an experimental outcome rather than a provider-capability requirement. Retry delays honor `Retry-After` and otherwise use deterministic exponential backoff. Explicit HTTP 429 rejections do not consume the experiment cost cap. Attempts, latency, response status, token usage, reported cost, capacity target, and achieved simultaneous load are recorded in `config/preflight.json` and the SQLite database. Every preflight invocation has a fresh qualification-run identifier; retries inside one probe keep the same logical idempotency key, while a later invocation gets a new provider idempotency key and distinct physical attempt keys. The preflight provenance also records the backend order actually used for every model.

A successful preflight selects one exact route for each model. `config/routes.json` is bound to the experiment manifest. Generation never switches a cell silently to another provider after preflight, and rerunning preflight cannot change provider identity after generation attempts exist.

## Output ceilings and truncation handling

Initial request ceilings are model-specific and deliberately loop-bounding rather than provider-maximal. Version 3.0.5 retains the 3.0.4 caps; Mistral Large 3 remains at the restored 2,304-token ceiling rather than the regressed 1,792-token value. The later 2026-08-10 240-cell sample had an ordinary `finish_reason=stop` completion at 1,663 tokens (p99 1,504.35), making 1,792 too close to the observed tail. The packaged `instruction_duplication/smoke-profile-2026-08-05.json` retains the original legacy-smoke provenance and records this later sample as supplemental cross-protocol ceiling evidence. A clean 3.0.6 smoke with the frozen judge-v2 measurement layer is appropriate before confirmatory inference; a partial 3.0.4 smoke can be resumed for transport debugging without changing its pinned routes.

The ceilings are deliberately lower than provider capability because they are also used for realistic cost reservations. A provider response whose finish reason indicates output-length truncation is terminal for that cell. It is retained as `truncated` and receives the predefined intention-to-treat treatment; the runner does **not** issue a larger-ceiling replacement draw. This avoids conditional resampling when hosted routes are not empirically deterministic even at temperature zero with a requested seed. Transport failures and provider-level non-success finish reasons may still be retried with the unchanged payload and the same logical idempotency key. A complete `finish_reason=stop` response is retained even when provider-reported usage exceeds the requested ceiling, because observed hosted-provider billing/tokenization anomalies can make that telemetry larger than the generation limit. The realized cost is still committed conservatively.

## Budget reservations

Before dispatching a request, the runner reserves the route-priced worst-case cost for that attempt's output ceiling. A completed attempt reconciles the reservation against provider-reported cost when available, or a conservative local estimate otherwise.

An explicit HTTP 429 response is accounted as zero generation spend and releases its reservation because the provider rejected the request for rate limiting. Timeouts, network failures, and other transient failures remain conservatively accounted because provider-side execution may have started. Refusals and truncations retain their realized or conservatively estimated cost.

A provider can report a final charge above the reservation only after the request has completed. A valid paid response is retained. The actual charge is committed, and subsequent reservations are blocked once the cumulative cap is exceeded; already in-flight requests are still recorded. When a provider reports zero cost but nonzero token usage, accounting uses the larger of reported cost and the route-priced token estimate.

## Transient retries, concurrency, and cooldowns

Generation uses independent per-model workers, a global request limit, shared provider gates, and an adaptive limit for each exact pinned route.

A transient provider failure does three things:

1. records the attempt and leaves the cell retryable;
2. establishes or extends the provider-wide cooldown using `Retry-After` when available and deterministic backoff otherwise;
3. reduces concurrency for the affected exact route.

Retryable cells are collected and run again in later sweeps. `--retries N` means up to `N` additional transient retry sweeps after the initial sweep. This avoids spending all retry opportunities immediately while a provider is still throttling the route. A concentrated 429 burst is not a run-level failure: only the affected exact route is adaptively throttled and cooled down, while unrelated models continue. Sustained successful requests gradually restore route concurrency. A success does not cancel an already active provider cooldown.

HTTP 429 responses additionally feed an exact-route circuit. Four 429 responses within 60 seconds open a route configured for four or more workers; routes configured below four workers require three. Once open, no new request is dispatched on that route during the invocation. In-flight attempts finish and are persisted, already rejected cells remain `retryable`, untouched cells remain `pending`, and generation exits with a resumable route-capacity error instead of reducing a long tail to one request at a time. A single isolated 429 follows the ordinary retry path and does not open the circuit.

For OpenRouter errors, attempt provenance records an upstream provider when the structured error metadata names one, records OpenRouter only when the response explicitly identifies it, and otherwise records `unknown-via-openrouter`; it does not guess.

The pinned provider is unchanged throughout these retries. Provider throttling therefore affects throughput and completion, not the experimental route identity. Version 3.0.5 (retained in 3.0.6) restores the earlier Llama 3.3 OpenRouter preference order with DeepInfra before Groq for newly pinned routes; an existing 3.0.4 workspace keeps its stored route unchanged. Progress output names constrained exact routes as `model@backend/provider` and shows `limit=current/unconstrained`, active requests, and any remaining shared-provider cooldown, so a provider bottleneck can be identified while the run is still in progress.

`instruction-duplication status --workspace <path>` reports stored progress and accounted cost. After ordinary transient retry rounds are exhausted, `retryable` cells proceed to intention-to-treat analysis as generation failures. A route-capacity circuit is an operational stop, so the all-in-one workflow does not judge or analyze that invocation; rerun later with the same pinned route or use a fresh workspace if a different provider is required.
