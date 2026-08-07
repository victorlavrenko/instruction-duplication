"""Deterministic provider discovery, probing, and route provenance."""

from __future__ import annotations

import asyncio
import json
import time
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from datetime import UTC, datetime
from types import MappingProxyType
from typing import Literal
from urllib.parse import quote

import httpx
from huggingface_hub import model_info

from .answer_utils import extract_answer
from .json_types import (
    FrozenJsonObject,
    JsonArray,
    JsonObject,
    freeze_json_object,
    is_object_sequence,
    json_object,
    object_value,
)
from .judge import extract_protocol_final_answer, parse_protocol
from .models import Model
from .provider import (
    SAFETY_REASONS,
    TRANSIENT_STATUS,
    TRUNCATION_REASONS,
    ProviderResponse,
    Route,
    estimate_cost,
    headers,
    hf_token,
    idempotency_key,
    openrouter_token,
    parse_response,
    realized_cost,
    reasoning_field,
    request_payload,
    retry_delay,
    route_api_key,
    route_url,
)
from .records import AttemptRecord, GenerationCell
from .types import AttemptStatus

type BackendPreference = Literal["auto", "hf", "openrouter"]
type LiveBackend = Literal["huggingface", "openrouter"]
type PreflightProgressStatus = Literal[
    "discovering",
    "probing",
    "rejected",
    "selected",
    "discovery_failed",
]
_ALLOWED_HF_TASKS = {"conversational", "text-generation", "image-text-to-text"}


@dataclass(frozen=True, slots=True)
class PreflightProgress:
    """One concise route-selection event suitable for CLI progress reporting."""

    model_id: str
    backend: LiveBackend
    provider: str | None
    status: PreflightProgressStatus
    detail: str | None = None


type PreflightProgressSink = Callable[[PreflightProgress], None]


@dataclass(frozen=True, slots=True)
class ProbeResult:
    """The selected response and attempt records for one route probe."""

    content: str
    attempts: tuple[AttemptRecord, ...]
    returned_provider: str | None


@dataclass(frozen=True, slots=True, init=False)
class PreflightResult:
    """Pinned routes and complete candidate-probe provenance."""

    routes: Mapping[str, Route]
    provenance: FrozenJsonObject
    attempts: tuple[AttemptRecord, ...]

    def __init__(
        self,
        routes: Mapping[str, Route],
        provenance: object,
        attempts: Sequence[AttemptRecord],
    ) -> None:
        object.__setattr__(self, "routes", MappingProxyType(dict(routes)))
        object.__setattr__(
            self,
            "provenance",
            freeze_json_object(provenance, path="preflight.provenance"),
        )
        object.__setattr__(self, "attempts", tuple(attempts))


class PreflightRunError(RuntimeError):
    """Preflight failure carrying partial candidate and attempt provenance."""

    def __init__(
        self,
        message: str,
        provenance: JsonObject,
        attempts: Sequence[AttemptRecord],
    ) -> None:
        super().__init__(message)
        self.provenance: FrozenJsonObject = freeze_json_object(
            provenance,
            path="preflight.error.provenance",
        )
        self.attempts = tuple(attempts)


class RouteSelectionError(RuntimeError):
    """No candidate route worked for one model."""

    def __init__(self, model_id: str, candidates: Sequence[JsonObject]) -> None:
        self.model_id = model_id
        self.candidates: tuple[FrozenJsonObject, ...] = tuple(
            freeze_json_object(candidate, path=f"preflight.candidates[{index}]")
            for index, candidate in enumerate(candidates)
        )
        super().__init__(
            f"no working fixed route for {model_id}: "
            + "; ".join(str(row) for row in self.candidates)
        )


@dataclass(slots=True)
class PreflightBudget:
    """Sequential cumulative budget guard shared by all route probes."""

    cap: float
    committed: float = 0.0

    def authorize(self, reservation: float) -> None:
        if self.committed + reservation > self.cap + 1e-12:
            raise PreflightBudgetExceeded(
                "preflight cost cap would be exceeded: "
                f"${self.committed + reservation:.6f} > ${self.cap:.6f}"
            )

    def commit(self, actual: float) -> None:
        self.committed += actual
        if self.committed > self.cap + 1e-9:
            raise PreflightBudgetExceeded(
                "provider cost exceeded its preflight reservation: "
                f"${self.committed:.6f} > ${self.cap:.6f}"
            )


class PreflightBudgetExceeded(RuntimeError):
    """Raised before dispatch when the cumulative preflight cap is exhausted."""


class ProbeFailure(RuntimeError):
    """A failed route probe that retains all billable attempt records."""

    def __init__(self, message: str, attempts: Sequence[AttemptRecord]) -> None:
        super().__init__(message)
        self.attempts = tuple(attempts)


class ProbeValidationError(RuntimeError):
    """A non-retryable parsed response that is unsuitable for the experiment."""

    def __init__(self, status: AttemptStatus, message: str) -> None:
        super().__init__(message)
        self.status = status


type AttemptSink = Callable[[Sequence[AttemptRecord]], None]


def discover_hf_providers(model: Model, token: str, timeout_seconds: float) -> list[str]:
    """Return sorted live chat-capable providers from the current Hub schema."""
    info = model_info(
        model.hf_id,
        expand=["inferenceProviderMapping"],
        token=token,
        timeout=timeout_seconds,
    )
    live = {
        details.provider
        for details in info.inference_provider_mapping or []
        if details.provider
        and details.status.casefold() == "live"
        and (not details.task or details.task.casefold() in _ALLOWED_HF_TASKS)
    }
    return sorted(live, key=str.casefold)


def _backend_order(
    preference: BackendPreference,
    model: Model,
    override: LiveBackend | None = None,
) -> list[LiveBackend]:
    if preference == "hf":
        return ["huggingface"]
    if preference == "openrouter":
        return ["openrouter"]
    preferred: LiveBackend = override or model.preferred_backend
    fallback: LiveBackend = "openrouter" if preferred == "huggingface" else "huggingface"
    return [preferred, fallback]


def _ordered_providers(discovered: Sequence[str], preferred: Sequence[str]) -> list[str]:
    by_key = {name.casefold(): name for name in discovered if name.strip()}
    selected: list[str] = []
    seen: set[str] = set()
    for name in preferred:
        key = name.casefold()
        current = by_key.get(key)
        if current is not None and key not in seen:
            selected.append(current)
            seen.add(key)
    for name in sorted(by_key.values(), key=str.casefold):
        key = name.casefold()
        if key not in seen:
            selected.append(name)
            seen.add(key)
    return selected


def _openrouter_endpoints_url(model: Model) -> str:
    author, separator, slug = model.openrouter_id.partition("/")
    if not separator or not author or not slug:
        raise ValueError(f"invalid OpenRouter model id: {model.openrouter_id!r}")
    return (
        "https://openrouter.ai/api/v1/models/"
        f"{quote(author, safe='')}/{quote(slug, safe='')}/endpoints"
    )


async def discover_openrouter_providers(
    model: Model,
    client: httpx.AsyncClient,
) -> list[str]:
    """Return current OpenRouter endpoint providers in deterministic preference order."""
    response = await client.get(_openrouter_endpoints_url(model))
    response.raise_for_status()
    decoded: object = response.json()
    root = json_object(decoded, path="OpenRouter endpoint response")
    data = object_value(root.get("data"), name="OpenRouter endpoint response data")
    endpoints = data.get("endpoints")
    if not is_object_sequence(endpoints):
        raise ValueError("OpenRouter endpoint response data.endpoints must be a list")
    providers: list[str] = []
    for index, endpoint in enumerate(endpoints):
        row = object_value(endpoint, name=f"OpenRouter endpoint[{index}]")
        tag = row.get("tag")
        if isinstance(tag, str) and tag.strip():
            providers.append(tag.strip().split("/", 1)[0])
    return _ordered_providers(providers, model.openrouter_providers)


def _route(model: Model, backend: LiveBackend, provider: str) -> Route:
    routed_model = f"{model.hf_id}:{provider}" if backend == "huggingface" else model.openrouter_id
    return Route(
        backend=backend,
        provider=provider,
        model=routed_model,
        input_usd_per_million=model.input_usd_per_million,
        output_usd_per_million=model.output_usd_per_million,
        max_concurrency=model.max_concurrency,
        pricing_source="versioned-model-panel-v2",
        determinism_verified=False,
    )


def _probe_error_state(
    exc: Exception,
    explicit: AttemptStatus,
) -> tuple[AttemptStatus, bool]:
    if isinstance(exc, ProbeValidationError):
        return exc.status, False
    if explicit is not AttemptStatus.FAILED:
        return explicit, False
    if isinstance(exc, httpx.TimeoutException):
        return AttemptStatus.TIMEOUT, True
    if isinstance(exc, httpx.NetworkError):
        return AttemptStatus.NETWORK_ERROR, True
    if isinstance(exc, httpx.HTTPStatusError):
        return AttemptStatus.HTTP_ERROR, exc.response.status_code in TRANSIENT_STATUS
    if isinstance(exc, (ValueError, TypeError, KeyError)):
        return AttemptStatus.INVALID_RESPONSE, False
    return AttemptStatus.FAILED, False


def _probe_attempt_record(
    *,
    model: Model,
    route: Route,
    probe_index: int,
    attempt_number: int,
    logical_key: str,
    reservation: float,
    status: AttemptStatus,
    raw: JsonObject | None,
    response: httpx.Response | None,
    parsed: ProviderResponse | None,
    error: str | None,
    started_wall: str,
    started: float,
) -> AttemptRecord:
    accounted = reservation
    reported: float | None = None
    input_tokens: int | None = None
    output_tokens: int | None = None
    finish: str | None = None
    if parsed is not None:
        reported = parsed.reported_cost_usd
        input_tokens = parsed.input_tokens
        output_tokens = parsed.output_tokens
        finish = parsed.finish_reason
        accounted = realized_cost(route, parsed, reservation)
    return AttemptRecord(
        request_key=(
            f"preflight:{model.id}:{route.backend}:{route.provider}:"
            f"probe-{probe_index}:attempt-{attempt_number}"
        ),
        phase="preflight",
        cell_id=None,
        model_id=model.id,
        attempt_number=attempt_number,
        backend=route.backend,
        provider=route.provider,
        routed_model=route.model,
        status=status,
        idempotency_key=logical_key,
        requested_max_tokens=model.request_output_tokens,
        reservation_usd=reservation,
        reported_cost_usd=reported,
        accounted_cost_usd=accounted,
        input_tokens=input_tokens,
        output_tokens=output_tokens,
        http_status=response.status_code if response is not None else None,
        finish_reason=finish,
        latency_seconds=time.monotonic() - started,
        error=error,
        raw_response_json=json.dumps(raw, ensure_ascii=False) if raw else None,
        started_at=started_wall,
        completed_at=datetime.now(UTC).isoformat(),
    )


def _validate_probe_response(
    parsed: ProviderResponse,
    raw: JsonObject,
    model: Model,
    cell: GenerationCell,
    route: Route,
) -> tuple[str, str | None]:
    reasoning = reasoning_field(raw, parsed.content)
    if reasoning:
        raise ProbeValidationError(
            AttemptStatus.REASONING_EXPOSED,
            f"endpoint exposed reasoning: {reasoning}",
        )
    if parsed.finish_reason in TRUNCATION_REASONS:
        raise ProbeValidationError(
            AttemptStatus.TRUNCATED,
            f"response truncated at empirical ceiling {model.request_output_tokens}",
        )
    if parsed.finish_reason in SAFETY_REASONS or parsed.refusal:
        raise ProbeValidationError(
            AttemptStatus.REFUSED,
            parsed.refusal or f"blocked: {parsed.finish_reason}",
        )
    answer = (
        extract_answer(parsed.content, cell.choices)
        if cell.condition_id == "zero"
        else extract_protocol_final_answer(parsed.content, cell.choices)
    )
    if answer.status != "parsed" or answer.option is None:
        raise ProbeValidationError(
            AttemptStatus.INVALID_RESPONSE,
            f"probe has no parseable final answer: {answer.status}",
        )
    returned_provider = parsed.provider
    if (
        returned_provider
        and route.backend == "openrouter"
        and route.provider.casefold() not in returned_provider.casefold()
    ):
        raise ProbeValidationError(
            AttemptStatus.INVALID_RESPONSE,
            f"requested provider {route.provider!r}, response reported {returned_provider!r}",
        )
    return parsed.content, returned_provider


def _calibrate_route(candidate: Route, attempts: Sequence[AttemptRecord]) -> Route:
    """Calibrate a route conservatively from provider-reported probe billing."""
    factors: list[float] = []
    for attempt in attempts:
        if (
            attempt.reported_cost_usd is None
            or attempt.input_tokens is None
            or attempt.output_tokens is None
        ):
            continue
        baseline = (
            attempt.input_tokens * candidate.input_usd_per_million
            + attempt.output_tokens * candidate.output_usd_per_million
        ) / 1_000_000
        if baseline > 0 and attempt.reported_cost_usd > 0:
            factors.append(attempt.reported_cost_usd / baseline)
    if factors:
        factor = max(1.0, max(factors) * 1.10)
        source = f"provider-reported-probe-calibration-v1;factor={factor:.6f}"
    else:
        factor = 1.0
        source = "versioned-conservative-model-panel-v2"
    return Route(
        backend=candidate.backend,
        provider=candidate.provider,
        model=candidate.model,
        input_usd_per_million=candidate.input_usd_per_million * factor,
        output_usd_per_million=candidate.output_usd_per_million * factor,
        max_concurrency=candidate.max_concurrency,
        pricing_source=source,
        determinism_verified=False,
    )


async def _probe_route(
    client: httpx.AsyncClient,
    model: Model,
    cell: GenerationCell,
    route: Route,
    *,
    transport_retries: int = 2,
    probe_index: int = 1,
    budget: PreflightBudget | None = None,
    attempt_sink: AttemptSink | None = None,
) -> ProbeResult:
    """Probe one route at the empirical request ceiling; never retry truncation."""
    payload = request_payload(model, cell, route)
    reservation = estimate_cost(route, payload) * 1.10
    request_id = f"preflight:{model.id}:{route.backend}:{route.provider}:{probe_index}"
    logical_key = idempotency_key(request_id, "preflight")
    attempts: list[AttemptRecord] = []
    for attempt_number in range(1, transport_retries + 2):
        if budget is not None:
            budget.authorize(reservation)
        started_wall = datetime.now(UTC).isoformat()
        started = time.monotonic()
        raw: JsonObject | None = None
        response: httpx.Response | None = None
        parsed: ProviderResponse | None = None
        status = AttemptStatus.FAILED
        error: str | None = None
        retryable = False
        caught: Exception | None = None
        success_content: str | None = None
        returned_provider: str | None = None
        try:
            response = await client.post(
                route_url(route),
                json=payload,
                headers={"Idempotency-Key": logical_key},
            )
            response.raise_for_status()
            decoded_response: object = response.json()
            raw = json_object(decoded_response, path="provider response")
            parsed = parse_response(raw)
            success_content, returned_provider = _validate_probe_response(
                parsed, raw, model, cell, route
            )
            status = AttemptStatus.COMPLETED
        except Exception as exc:
            caught = exc
            error = str(exc)
            status, retryable = _probe_error_state(exc, status)
        record = _probe_attempt_record(
            model=model,
            route=route,
            probe_index=probe_index,
            attempt_number=attempt_number,
            logical_key=logical_key,
            reservation=reservation,
            status=status,
            raw=raw,
            response=response,
            parsed=parsed,
            error=error,
            started_wall=started_wall,
            started=started,
        )
        attempts.append(record)
        if attempt_sink is not None:
            attempt_sink((record,))
        if budget is not None:
            budget.commit(record.accounted_cost_usd)
        if success_content is not None:
            return ProbeResult(success_content, tuple(attempts), returned_provider)
        if not retryable or attempt_number > transport_retries:
            raise ProbeFailure(error or "preflight failed", attempts) from caught
        if caught is None:
            raise RuntimeError("retryable probe failure lacked an exception")
        await asyncio.sleep(retry_delay(caught, attempt_number - 1, logical_key))
    raise RuntimeError("unreachable")


def _empty_clients() -> dict[LiveBackend, httpx.AsyncClient]:
    return {}


def _empty_attempts() -> list[AttemptRecord]:
    return []


@dataclass(slots=True)
class PreflightSession:
    """Own clients, candidate provenance, and attempt accounting for route selection."""

    backend_preference: BackendPreference
    model_backends: Mapping[str, LiveBackend]
    representative_cell: GenerationCell
    timeout_seconds: float
    clients: dict[LiveBackend, httpx.AsyncClient] = field(default_factory=_empty_clients)
    attempts: list[AttemptRecord] = field(default_factory=_empty_attempts)
    budget: PreflightBudget | None = None
    attempt_sink: AttemptSink | None = None
    progress_sink: PreflightProgressSink | None = None

    def report(
        self,
        model: Model,
        backend: LiveBackend,
        status: PreflightProgressStatus,
        *,
        provider: str | None = None,
        detail: str | None = None,
    ) -> None:
        """Emit progress without coupling route selection to a console implementation."""
        if self.progress_sink is not None:
            self.progress_sink(PreflightProgress(model.id, backend, provider, status, detail))

    async def client_for(self, route: Route) -> httpx.AsyncClient:
        """Create one client per backend and reuse it across candidate providers."""
        if route.backend == "fake":
            raise ValueError("fake routes do not use an HTTP client")
        if route.backend in self.clients:
            return self.clients[route.backend]
        key = route_api_key(route)
        if not key:
            raise RuntimeError(f"credential missing for {route.backend}")
        self.clients[route.backend] = httpx.AsyncClient(
            timeout=httpx.Timeout(self.timeout_seconds),
            headers=headers(route.backend, key),
            limits=httpx.Limits(max_connections=16, max_keepalive_connections=8),
        )
        return self.clients[route.backend]

    async def providers_for(self, model: Model, backend: LiveBackend) -> list[str]:
        """Return deterministic provider candidates for one backend."""
        if backend == "openrouter":
            if not openrouter_token():
                raise RuntimeError("OPENROUTER_API_KEY is not set")
            provider = model.openrouter_providers[0] if model.openrouter_providers else "discovery"
            client = await self.client_for(_route(model, "openrouter", provider))
            try:
                discovered = await discover_openrouter_providers(model, client)
            except (httpx.HTTPError, ValueError):
                discovered = []
            return discovered or list(model.openrouter_providers)
        token = hf_token()
        if not token:
            raise RuntimeError("HF_TOKEN is not set and no cached Hugging Face token was found")
        return await asyncio.to_thread(discover_hf_providers, model, token, self.timeout_seconds)

    async def verify_candidate(
        self,
        model: Model,
        candidate: Route,
    ) -> tuple[Route, JsonObject]:
        """Probe a candidate twice and return a pinned route plus provenance row."""
        client = await self.client_for(candidate)
        first = await _probe_route(
            client,
            model,
            self.representative_cell,
            candidate,
            probe_index=1,
            budget=self.budget,
            attempt_sink=self.attempt_sink,
        )
        self.attempts.extend(first.attempts)
        second = await _probe_route(
            client,
            model,
            self.representative_cell,
            candidate,
            probe_index=2,
            budget=self.budget,
            attempt_sink=self.attempt_sink,
        )
        self.attempts.extend(second.attempts)
        deterministic = first.content == second.content
        protocol_observations: JsonArray = []
        if self.representative_cell.condition_id != "zero":
            for content in (first.content, second.content):
                protocol = parse_protocol(content, self.representative_cell.choices)
                protocol_observations.append(
                    json_object(
                        {
                            "xml_valid": protocol.xml_valid,
                            "structure_valid": protocol.structure_valid,
                            "errors": list(protocol.errors),
                        },
                        path="preflight.protocol_observation",
                    )
                )
        calibrated = _calibrate_route(candidate, first.attempts + second.attempts)
        selected = Route(
            backend=calibrated.backend,
            provider=calibrated.provider,
            model=calibrated.model,
            input_usd_per_million=calibrated.input_usd_per_million,
            output_usd_per_million=calibrated.output_usd_per_million,
            max_concurrency=calibrated.max_concurrency,
            pricing_source=calibrated.pricing_source,
            determinism_verified=deterministic,
        )
        row: JsonObject = {
            "backend": candidate.backend,
            "provider": candidate.provider,
            "result": "selected",
            "determinism_verified": deterministic,
            "returned_provider": second.returned_provider,
            "pricing_source": selected.pricing_source,
            "input_usd_per_million": selected.input_usd_per_million,
            "output_usd_per_million": selected.output_usd_per_million,
            "protocol_observations": protocol_observations,
        }
        return selected, row

    async def select_model(self, model: Model) -> tuple[Route, list[JsonObject]]:
        """Try providers in model-specific backend order and retain full provenance."""
        candidate_rows: list[JsonObject] = []
        backends = _backend_order(
            self.backend_preference,
            model,
            self.model_backends.get(model.id),
        )
        for backend in backends:
            self.report(model, backend, "discovering")
            try:
                providers = await self.providers_for(model, backend)
            except Exception as exc:
                self.report(model, backend, "discovery_failed", detail=str(exc))
                candidate_rows.append(
                    {
                        "backend": backend,
                        "provider": None,
                        "result": "discovery_failed",
                        "error": str(exc),
                    }
                )
                continue
            for provider in providers:
                candidate = _route(model, backend, provider)
                self.report(model, backend, "probing", provider=provider)
                try:
                    selected, row = await self.verify_candidate(model, candidate)
                except PreflightBudgetExceeded:
                    raise
                except ProbeFailure as exc:
                    self.attempts.extend(exc.attempts)
                    self.report(
                        model,
                        backend,
                        "rejected",
                        provider=provider,
                        detail=str(exc),
                    )
                    candidate_rows.append(
                        {
                            "backend": backend,
                            "provider": provider,
                            "result": "rejected",
                            "error": str(exc),
                        }
                    )
                    continue
                except Exception as exc:
                    self.report(
                        model,
                        backend,
                        "rejected",
                        provider=provider,
                        detail=str(exc),
                    )
                    candidate_rows.append(
                        {
                            "backend": backend,
                            "provider": provider,
                            "result": "rejected",
                            "error": str(exc),
                        }
                    )
                    continue
                self.report(
                    model,
                    backend,
                    "selected",
                    provider=provider,
                    detail=(
                        "deterministic" if selected.determinism_verified else "non-identical probes"
                    ),
                )
                candidate_rows.append(row)
                return selected, candidate_rows
        raise RouteSelectionError(model.id, candidate_rows)

    async def close(self) -> None:
        """Close every client created during preflight."""
        await asyncio.gather(*(client.aclose() for client in self.clients.values()))


def _validate_backend_credentials(backend: BackendPreference) -> None:
    if backend == "hf" and not hf_token():
        raise RuntimeError("HF_TOKEN is not set and no cached Hugging Face token was found")
    if backend == "openrouter" and not openrouter_token():
        raise RuntimeError("OPENROUTER_API_KEY is not set")
    if backend == "auto" and not (hf_token() or openrouter_token()):
        raise RuntimeError("set HF_TOKEN, OPENROUTER_API_KEY, or both")


def _validate_model_backends(
    models: Sequence[Model],
    backend: BackendPreference,
    model_backends: Mapping[str, LiveBackend] | None,
) -> dict[str, LiveBackend]:
    overrides: dict[str, LiveBackend] = dict(model_backends) if model_backends is not None else {}
    if overrides and backend != "auto":
        raise ValueError("per-model backend preferences require --backend auto")
    known = {model.id for model in models}
    unknown = sorted(set(overrides) - known)
    if unknown:
        raise ValueError("unknown model ids in backend preferences: " + ", ".join(unknown))
    return overrides


async def check_routes(
    models: Sequence[Model],
    representative_cell: GenerationCell,
    *,
    timeout_seconds: float = 90,
    fake: bool = False,
    backend: BackendPreference = "auto",
    model_backends: Mapping[str, LiveBackend] | None = None,
    max_cost: float | None = None,
    committed_cost: float = 0.0,
    attempt_sink: AttemptSink | None = None,
    progress_sink: PreflightProgressSink | None = None,
) -> PreflightResult:
    """Find, verify twice, and pin one route per model with full provenance."""
    if fake:
        routes = {
            model.id: Route("fake", "fake", model.id, 0.0, 0.0, model.max_concurrency, "fake", True)
            for model in models
        }
        return PreflightResult(
            routes,
            {"backend_policy": "fake", "models": {}},
            (),
        )
    _validate_backend_credentials(backend)
    overrides = _validate_model_backends(models, backend, model_backends)
    budget = None if max_cost is None else PreflightBudget(max_cost, committed_cost)
    session = PreflightSession(
        backend,
        MappingProxyType(overrides),
        representative_cell,
        timeout_seconds,
        budget=budget,
        attempt_sink=attempt_sink,
        progress_sink=progress_sink,
    )
    routes: dict[str, Route] = {}
    provenance: JsonObject = {}
    failure: Exception | None = None
    try:
        for model in models:
            selected, candidate_rows = await session.select_model(model)
            routes[model.id] = selected
            provenance[model.id] = json_object(
                {"candidates": candidate_rows}, path=f"preflight.{model.id}"
            )["candidates"]
    except RouteSelectionError as exc:
        provenance[exc.model_id] = json_object(
            {"candidates": list(exc.candidates)}, path=f"preflight.{exc.model_id}"
        )["candidates"]
        failure = exc
    except Exception as exc:
        failure = exc
    finally:
        await session.close()
    backend_orders = json_object(
        {model.id: _backend_order(backend, model, overrides.get(model.id)) for model in models},
        path="preflight.backend_order",
    )
    document: JsonObject = {
        "backend_policy": backend,
        "backend_order": backend_orders,
        "representative_cell_id": representative_cell.cell_id,
        "status": "failed" if failure else "completed",
        "error": str(failure) if failure else None,
        "models": provenance,
    }
    if failure is not None:
        raise PreflightRunError(str(failure), document, session.attempts) from failure
    return PreflightResult(routes, document, tuple(session.attempts))
