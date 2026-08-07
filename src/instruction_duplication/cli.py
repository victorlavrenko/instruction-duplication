"""Command-line workflows for the instruction-duplication experiment."""

from __future__ import annotations

import argparse
import asyncio
import datetime as dt
import json
import logging
import math
import sys
from collections.abc import Mapping, Sequence
from pathlib import Path

from . import __version__
from .datasets_loader import (
    DEFAULT_DATASETS,
    DEFAULT_SEED,
    select_questions,
    source_descriptor,
)
from .environment import runtime_environment
from .generate import (
    BUDGET_SAFETY_FACTOR,
    GenerationProgress,
    GenerationProgressSink,
    RouteBottleneck,
    route_parallelism_summary,
    run_pending,
)
from .io_utils import read_json, sha256_json, write_json, write_jsonl, write_text
from .json_types import JsonObject, is_object_sequence, json_object, object_value, string_mapping
from .judge import judge
from .lexical import LEXICAL_VERSION, build_reference
from .manifest import build_manifest
from .models import Model, select_models
from .preflight import (
    BackendPreference,
    LiveBackend,
    PreflightProgress,
    PreflightProgressSink,
    PreflightRunError,
    check_routes,
)
from .provider import Route, estimate_cost, normalize_routes, request_payload
from .records import JudgmentWrite
from .report import render_report
from .stats import build_analysis
from .storage import Database
from .types import Question
from .workspace import Workspace

LOGGER = logging.getLogger("instruction_duplication")


def _progress(message: str) -> None:
    """Write concise operational progress without mixing it with report output."""
    print(message, file=sys.stderr, flush=True)


def _short_error(message: str, *, limit: int = 140) -> str:
    """Reduce provider exceptions to one readable console line."""
    first_line = message.splitlines()[0].strip() if message else "unknown error"
    if " for url " in first_line:
        first_line = first_line.split(" for url ", 1)[0]
    if len(first_line) <= limit:
        return first_line
    return first_line[: limit - 1].rstrip() + "…"


def _backend_label(backend: str) -> str:
    if backend == "huggingface":
        return "HF"
    if backend == "openrouter":
        return "OpenRouter"
    return backend


def _preflight_progress(event: PreflightProgress) -> None:
    backend = _backend_label(event.backend)
    prefix = f"[preflight] {event.model_id}: {backend}"
    if event.status == "discovering":
        _progress(f"{prefix}: discovering providers")
        return
    if event.status == "probing":
        _progress(f"{prefix}/{event.provider}: probing")
        return
    if event.status == "rejected":
        _progress(f"{prefix}/{event.provider}: rejected — {_short_error(event.detail or '')}")
        return
    if event.status == "discovery_failed":
        _progress(f"{prefix}: unavailable — {_short_error(event.detail or '')}")
        return
    detail = "verified twice"
    if event.detail == "deterministic":
        detail += ", identical responses"
    elif event.detail:
        detail += f", {event.detail}"
    _progress(f"{prefix}/{event.provider}: selected ({detail})")


def _generation_progress(event: GenerationProgress) -> None:
    percent = 100.0 * event.processed / event.total if event.total else 100.0
    suffix = ""
    if event.failed or event.budget_blocked:
        suffix = f", failed={event.failed}, budget-blocked={event.budget_blocked}"
    rate = (
        f"{event.rate_per_second * 60:.1f} cells/min"
        if event.rate_per_second is not None
        else "calculating"
    )
    eta = _format_duration(event.eta_seconds)
    _progress(
        f"[generation] {event.processed}/{event.total} ({percent:.0f}%), "
        f"completed={event.completed}{suffix}, rate={rate}, ETA={eta}"
    )
    if event.bottlenecks:
        details = "; ".join(_format_bottleneck(item) for item in event.bottlenecks)
        _progress(f"[generation] bottlenecks: {details}")


def _format_bottleneck(bottleneck: RouteBottleneck) -> str:
    """Render one exact throttled route without hiding the model behind provider totals."""
    backend = _backend_label(bottleneck.backend)
    detail = (
        f"{bottleneck.model_id}@{backend}/{bottleneck.provider} "
        f"limit={bottleneck.current_limit}/{bottleneck.unconstrained_limit} "
        f"active={bottleneck.active}"
    )
    if bottleneck.cooldown_seconds > 0:
        detail += f" cooldown={_format_duration(bottleneck.cooldown_seconds)}"
    return detail


def _format_duration(seconds: float | None) -> str:
    """Render a compact ETA suitable for one-line progress output."""
    if seconds is None or not math.isfinite(seconds) or seconds < 0:
        return "calculating"
    rounded = math.ceil(seconds)
    hours, remainder = divmod(rounded, 3600)
    minutes, secs = divmod(remainder, 60)
    if hours:
        return f"{hours}h {minutes:02d}m"
    if minutes:
        return f"{minutes}m {secs:02d}s"
    return f"{secs}s"


def _report_pinned_routes(routes: dict[str, Route]) -> None:
    for model_id, route in routes.items():
        _progress(f"[routes] {model_id}: {_backend_label(route.backend)}/{route.provider}")


def _positive_int(value: str) -> int:
    parsed = int(value)
    if parsed < 1:
        raise argparse.ArgumentTypeError("must be a positive integer")
    return parsed


def _nonnegative_int(value: str) -> int:
    parsed = int(value)
    if parsed < 0:
        raise argparse.ArgumentTypeError("must be a non-negative integer")
    return parsed


def _positive_float(value: str) -> float:
    parsed = float(value)
    if not math.isfinite(parsed) or parsed <= 0:
        raise argparse.ArgumentTypeError("must be finite and positive")
    return parsed


def _confidence(value: str) -> float:
    parsed = float(value)
    if not math.isfinite(parsed) or not 0 < parsed < 1:
        raise argparse.ArgumentTypeError("must be between zero and one")
    return parsed


def _backend_preference(value: object) -> BackendPreference:
    """Validate an argparse backend value as the supported literal type."""
    if value == "auto":
        return "auto"
    if value == "hf":
        return "hf"
    if value == "openrouter":
        return "openrouter"
    raise ValueError(f"unsupported backend preference: {value!r}")


def _model_backend_preferences(
    values: Sequence[str],
    models: Sequence[Model],
    backend: BackendPreference,
) -> dict[str, LiveBackend]:
    """Parse repeatable MODEL=BACKEND preferences for automatic preflight."""
    if values and backend != "auto":
        raise ValueError("--prefer-backend requires --backend auto")
    known = {model.id for model in models}
    result: dict[str, LiveBackend] = {}
    for value in values:
        model_id, separator, raw_backend = value.partition("=")
        if not separator or not model_id or not raw_backend:
            raise ValueError("--prefer-backend must use MODEL=hf or MODEL=openrouter")
        if model_id not in known:
            raise ValueError(f"unknown model id in --prefer-backend: {model_id}")
        if model_id in result:
            raise ValueError(f"duplicate --prefer-backend for model: {model_id}")
        if raw_backend == "hf":
            result[model_id] = "huggingface"
        elif raw_backend == "openrouter":
            result[model_id] = "openrouter"
        else:
            raise ValueError("--prefer-backend must use MODEL=hf or MODEL=openrouter")
    return result


def _workspace(args: argparse.Namespace) -> Workspace:
    return Workspace(Path(args.workspace).resolve())


def _models(ws: Workspace) -> list[Model]:
    value = read_json(ws.models)
    if not is_object_sequence(value):
        raise ValueError("models.json must contain a list")
    return [
        Model.from_dict(object_value(row, name=f"models[{index}]"))
        for index, row in enumerate(value)
    ]


def _requested_selection(args: argparse.Namespace) -> JsonObject:
    return {
        "datasets": list(args.datasets),
        "questions_per_dataset": int(args.questions_per_dataset),
        "seed": int(args.seed),
        "algorithm": "normalize-deduplicate-shuffle-v2",
    }


def _validate_resume_request(ws: Workspace, args: argparse.Namespace) -> None:
    manifest = ws.require_prepared()
    source = source_descriptor(
        args.datasets,
        Path(args.input_jsonl) if args.input_jsonl else None,
    )
    selection = _requested_selection(args)
    selected_models = select_models(args.models)
    models = [model.to_dict() for model in selected_models]
    changed: list[str] = []
    if json_object(manifest.source, path="manifest.source") != source:
        changed.append("source")
    if json_object(manifest.selection, path="manifest.selection") != selection:
        changed.append("selection/seed")
    if manifest.models_hash != sha256_json(models):
        changed.append("model panel/configuration/order")
    if changed:
        raise RuntimeError(
            "existing workspace does not match the requested experiment: "
            + ", ".join(changed)
            + "; use a new workspace"
        )
    with Database(ws.database) as db:
        db.validate_plan(manifest.question_ids, [model.id for model in selected_models])


def _prepare_workspace(args: argparse.Namespace) -> tuple[int, int, int]:
    ws = _workspace(args)
    ws.create()
    if ws.manifest.exists():
        _validate_resume_request(ws, args)
        manifest = ws.load_manifest()
        question_count = len(manifest.question_ids)
        model_count = len(manifest.models)
        return question_count, model_count, question_count * model_count * 8
    partial = [
        path.name
        for path in (
            ws.questions,
            ws.models,
            ws.dataset_audit,
            ws.environment,
            ws.lexical_reference,
            ws.database,
        )
        if path.exists()
    ]
    if partial:
        raise RuntimeError(
            "workspace contains an incomplete preparation ("
            + ", ".join(partial)
            + "); remove the workspace contents or use a new workspace"
        )

    questions, audit = select_questions(
        dataset_names=args.datasets,
        questions_per_dataset=args.questions_per_dataset,
        seed=args.seed,
        input_jsonl=Path(args.input_jsonl) if args.input_jsonl else None,
    )
    models = select_models(args.models)
    question_rows = [question.to_dict() for question in questions]
    model_rows = [model.to_dict() for model in models]
    environment = runtime_environment()
    manifest = build_manifest(
        source=json_object(audit["source"], path="dataset audit.source"),
        selection=json_object(audit["selection"], path="dataset audit.selection"),
        questions=question_rows,
        models=models,
        environment=environment,
    )

    # Every ordinary file is atomically replaced; manifest is written last as the
    # commit marker that makes the workspace resumable.
    write_jsonl(ws.questions, question_rows)
    write_json(ws.models, model_rows)
    write_json(ws.dataset_audit, audit)
    write_json(ws.environment, environment)
    write_json(ws.lexical_reference, build_reference(questions))
    try:
        with Database(ws.database) as db:
            cells = db.prepare(questions, [model.id for model in models])
    except BaseException:
        ws.database.unlink(missing_ok=True)
        raise
    write_json(ws.manifest, manifest.to_dict())
    LOGGER.info(
        "prepared questions=%d models=%d cells=%d manifest=%s",
        len(questions),
        len(models),
        cells,
        manifest.identity_hash,
    )
    return len(questions), len(models), cells


def _route_document(ws: Workspace, models: list[Model]) -> tuple[dict[str, Route], JsonObject]:
    value = read_json(ws.routes)
    if not isinstance(value, dict) or value.get("schema_version") != 2:
        raise RuntimeError("routes.json is absent or incompatible; rerun preflight")
    manifest = ws.load_manifest()
    if value.get("manifest_identity_hash") != manifest.identity_hash:
        raise RuntimeError("routes were generated for a different experiment manifest")
    raw = value.get("routes")
    if not isinstance(raw, dict):
        raise ValueError("routes.json routes must be an object")
    if value.get("routes_hash") != sha256_json(raw):
        raise RuntimeError("routes.json failed its integrity hash")
    return normalize_routes(raw, models), value


def _pin_routes(
    ws: Workspace,
    *,
    timeout: float,
    fake: bool,
    backend: BackendPreference,
    model_backends: Mapping[str, LiveBackend] | None = None,
    max_cost: float | None = None,
    progress_sink: PreflightProgressSink | None = None,
) -> dict[str, Route]:
    manifest = ws.require_prepared()
    models = _models(ws)
    with Database(ws.database) as db:
        representative = db.representative_cell()
        try:
            result = asyncio.run(
                check_routes(
                    models,
                    representative,
                    timeout_seconds=timeout,
                    fake=fake,
                    backend=backend,
                    model_backends=model_backends,
                    max_cost=max_cost,
                    committed_cost=db.spend(),
                    attempt_sink=db.record_attempts,
                    progress_sink=progress_sink,
                )
            )
        except PreflightRunError as exc:
            write_json(ws.preflight_log, exc.provenance)
            raise
    raw_routes = {model_id: route.to_dict() for model_id, route in result.routes.items()}
    route_document = {
        "schema_version": 2,
        "manifest_identity_hash": manifest.identity_hash,
        "routes_hash": sha256_json(raw_routes),
        "routes": raw_routes,
    }
    write_json(ws.preflight_log, result.provenance)
    write_json(ws.routes, route_document)
    return dict(result.routes)


def _estimate_cap(db: Database, models: list[Model], routes: dict[str, Route]) -> float:
    model_by_id = {model.id: model for model in models}
    one_pass = 0.0
    for cell in db.pending(list(model_by_id)):
        model = model_by_id[cell.model_id]
        route = routes[model.id]
        payload = request_payload(model, cell, route)
        one_pass += estimate_cost(route, payload) * BUDGET_SAFETY_FACTOR
    # Request ceilings already exceed every attached smoke-run per-model maximum by
    # at least 25%; the additional 15% is operational headroom, not an assumption
    # that responses reach provider capability limits.
    return max(0.01, db.spend() + one_pass * 1.15)


def _run_generation(
    args: argparse.Namespace,
    ws: Workspace,
    *,
    progress_sink: GenerationProgressSink | None = None,
) -> dict[str, int | float]:
    ws.require_prepared()
    models = _models(ws)
    routes, _ = _route_document(ws, models)
    with Database(ws.database) as db:
        max_cost = args.max_cost
        if max_cost is None:
            max_cost = _estimate_cap(db, models, routes)
            _progress(f"[generation] automatic cumulative cost cap: ${max_cost:.4f}")
        effective, model_limits, _ = route_parallelism_summary(
            models, routes, args.concurrency, args.per_model_concurrency
        )
        worker_count = sum(model_limits.values())
        limits = ", ".join(f"{model_id}={limit}" for model_id, limit in model_limits.items())
        _progress(
            f"[generation] parallelism: global={args.concurrency}, "
            f"per-model={args.per_model_concurrency}, model-workers={worker_count}, "
            f"effective request maximum={effective} ({limits})"
        )
        return asyncio.run(
            run_pending(
                db,
                models,
                routes,
                concurrency=args.concurrency,
                per_model_concurrency=args.per_model_concurrency,
                max_cost=max_cost,
                timeout_seconds=args.timeout,
                retries=args.retries,
                fake=args.fake,
                progress_sink=progress_sink,
            )
        )


def _judge_workspace(ws: Workspace) -> int:
    ws.require_prepared(require_runtime_environment=False)
    reference = read_json(ws.lexical_reference)
    if not isinstance(reference, dict) or reference.get("lexical_version") != LEXICAL_VERSION:
        raise RuntimeError("lexical reference is incompatible")
    total = 0
    with Database(ws.database) as db:
        targets = db.judgment_targets(lexical_version=LEXICAL_VERSION)
        batch: list[JudgmentWrite] = []
        for row in targets:
            question = Question(
                id=str(row["question_id"]),
                dataset=str(row["dataset"]),
                stem=str(row["stem"]),
                choices=string_mapping(row["choices"], name="judgment_target.choices"),
                gold=str(row["gold"]),
                gold_text=str(row["gold_text"]),
                gold_source=str(row["gold_source"]),
                gold_raw=str(row["gold_raw"]),
                source_split=str(row["source_split"]),
            )
            judgment = judge(
                question,
                str(row["status"]),
                row["content"],
                str(row["condition_id"]),
                reference,
            )
            batch.append(
                JudgmentWrite(
                    cell_id=str(row["cell_id"]),
                    content_hash=str(row["content_hash"]),
                    judged_at=dt.datetime.now(dt.UTC).isoformat(),
                    judgment=json_object(judgment, path="judge result"),
                    question_id=question.id,
                )
            )
            if len(batch) >= 500:
                db.put_judgments(batch, lexical_version=LEXICAL_VERSION)
                total += len(batch)
                batch.clear()
        if batch:
            db.put_judgments(batch, lexical_version=LEXICAL_VERSION)
            total += len(batch)
    return total


def _analyze_workspace(
    ws: Workspace,
    *,
    permutations: int,
    bootstraps: int,
    confidence_level: float,
) -> JsonObject:
    manifest = ws.require_prepared(require_runtime_environment=False)
    with Database(ws.database) as db:
        rows = list(db.iter_analysis_rows())
        missing = sum(row.judgment is None for row in rows)
        if missing:
            raise RuntimeError(f"{missing} cells lack current judgments; run judge first")
        analysis = build_analysis(
            rows,
            permutations=permutations,
            bootstraps=bootstraps,
            confidence_level=confidence_level,
        )
        analysis["manifest_identity_hash"] = manifest.identity_hash
        analysis["environment_hash"] = manifest.environment_hash
        analysis["software_version"] = __version__
        write_json(ws.analysis, analysis)
        write_text(ws.report, render_report(analysis))
        write_jsonl(ws.cells_export, db.iter_rows())
        write_jsonl(ws.attempts_export, db.attempts())
    return analysis


def _require_reproduce_generation_ready_for_analysis(ws: Workspace) -> None:
    """Require every planned cell to have reached an analyzable generation state."""
    with Database(ws.database) as db:
        counts = db.counts()

    blocking = {
        status: counts.get(status, 0)
        for status in ("pending", "running", "budget_blocked")
        if counts.get(status, 0)
    }
    if blocking:
        detail = ", ".join(f"{status}={count}" for status, count in blocking.items())
        raise RuntimeError(
            "generation has cells that were not fully attempted: "
            + detail
            + "; rerun reproduce or increase the cost cap before analysis"
        )

    retryable = counts.get("retryable", 0)
    if retryable:
        label = "cell remains" if retryable == 1 else "cells remain"
        _progress(
            f"[generation] {retryable} {label} retryable after the configured retry rounds; "
            "proceeding with ITT analysis and scoring them as generation failures"
        )


def command_prepare(args: argparse.Namespace) -> None:
    ws = _workspace(args)
    with ws.lock():
        questions, models, cells = _prepare_workspace(args)
    print(f"Prepared {questions} questions, {models} models, and {cells} cells.")


def command_preflight(args: argparse.Namespace) -> None:
    ws = _workspace(args)
    backend = _backend_preference(args.backend)
    with ws.lock():
        ws.require_prepared()
        models = _models(ws)
        model_backends = _model_backend_preferences(args.prefer_backend, models, backend)
        routes = _pin_routes(
            ws,
            timeout=args.timeout,
            fake=args.fake,
            backend=backend,
            model_backends=model_backends,
            max_cost=args.max_cost,
            progress_sink=_preflight_progress,
        )
    print(json.dumps({key: route.to_dict() for key, route in routes.items()}, indent=2))


def command_run(args: argparse.Namespace) -> None:
    ws = _workspace(args)
    with ws.lock():
        summary = _run_generation(args, ws, progress_sink=_generation_progress)
    print(json.dumps(summary, indent=2, sort_keys=True))


def command_judge(args: argparse.Namespace) -> None:
    ws = _workspace(args)
    with ws.lock():
        count = _judge_workspace(ws)
    print(f"Wrote or refreshed {count} judgments.")


def command_analyze(args: argparse.Namespace) -> None:
    ws = _workspace(args)
    with ws.lock():
        analysis = _analyze_workspace(
            ws,
            permutations=args.permutations,
            bootstraps=args.bootstraps,
            confidence_level=args.confidence_level,
        )
    print(render_report(analysis), end="")


def command_status(args: argparse.Namespace) -> None:
    ws = _workspace(args)
    manifest = ws.require_prepared(require_runtime_environment=False)
    with Database(ws.database) as db:
        output = {
            "manifest_identity_hash": manifest.identity_hash,
            "counts": db.counts(),
            "accounted_spend_usd": db.spend(),
            "reported_spend_usd": db.reported_spend(),
            "attempts": sum(1 for _ in db.attempts()),
            "routes": read_json(ws.routes) if ws.routes.exists() else None,
        }
    print(json.dumps(output, indent=2, sort_keys=True))


def command_reproduce(args: argparse.Namespace) -> None:
    ws = _workspace(args)
    requested_models = select_models(args.models)
    requested_questions = len(args.datasets) * int(args.questions_per_dataset)
    requested_cells = requested_questions * len(requested_models) * 8
    per_dataset_label = "question" if args.questions_per_dataset == 1 else "questions"
    question_label = "question" if requested_questions == 1 else "questions"
    model_label = "model" if len(requested_models) == 1 else "models"
    cell_label = "cell" if requested_cells == 1 else "cells"
    _progress(
        f"Instruction Duplication {__version__}: {args.questions_per_dataset} {per_dataset_label} "
        f"per dataset, {requested_questions} {question_label} total, {len(requested_models)} "
        f"{model_label}, 8 conditions, {requested_cells} {cell_label}."
    )
    _progress(f"Workspace: {ws.directory}")
    if args.max_cost is not None:
        _progress(f"Cumulative cost cap: ${args.max_cost:.2f}")

    with ws.lock():
        prepared = ws.manifest.exists()
        _progress(
            "[1/5] using the existing prepared workspace"
            if prepared
            else "[1/5] preparing the workspace"
        )
        _prepare_workspace(args)
        ws.require_prepared()
        prepared_models = _models(ws)
        backend = _backend_preference(args.backend)
        model_backends = _model_backend_preferences(
            args.prefer_backend,
            prepared_models,
            backend,
        )
        if ws.routes.exists():
            if model_backends:
                raise RuntimeError(
                    "--prefer-backend cannot change an already pinned route; "
                    "use a fresh workspace or rerun preflight explicitly"
                )
            _progress("[2/5] using pinned provider routes")
            pinned_routes, _ = _route_document(ws, prepared_models)
            _report_pinned_routes(pinned_routes)
        else:
            if args.fake:
                _progress("[2/5] using deterministic fake routes")
            else:
                _progress(
                    "[2/5] preflight: resolving fixed routes "
                    "(per-model preferred backend, then fallback)"
                )
            _pin_routes(
                ws,
                timeout=args.preflight_timeout,
                fake=args.fake,
                backend=backend,
                model_backends=model_backends,
                max_cost=args.max_cost,
                progress_sink=_preflight_progress,
            )
        _progress("[3/5] generating or resuming responses")
        summary = _run_generation(args, ws, progress_sink=_generation_progress)
        _require_reproduce_generation_ready_for_analysis(ws)
        _progress("[4/5] applying deterministic judgments")
        refreshed = _judge_workspace(ws)
        _progress(f"[4/5] judgments written or refreshed: {refreshed}")
        _progress("[5/5] analyzing and exporting results")
        analysis = _analyze_workspace(
            ws,
            permutations=args.permutations,
            bootstraps=args.bootstraps,
            confidence_level=args.confidence_level,
        )
    LOGGER.info("generation=%s judgments_refreshed=%d", summary, refreshed)
    print(render_report(analysis), end="")


def _add_workspace(command: argparse.ArgumentParser) -> None:
    command.add_argument("--workspace", default="run")


def _add_selection(command: argparse.ArgumentParser) -> None:
    command.add_argument("questions_per_dataset", nargs="?", type=_positive_int, default=100)
    command.add_argument("--datasets", nargs="+", default=list(DEFAULT_DATASETS))
    command.add_argument("--seed", type=int, default=DEFAULT_SEED)
    command.add_argument("--input-jsonl")
    command.add_argument("--models", nargs="*")


def _add_generation(command: argparse.ArgumentParser) -> None:
    command.add_argument("--max-cost", type=_positive_float)
    command.add_argument("--concurrency", type=_positive_int, default=64)
    command.add_argument("--per-model-concurrency", type=_positive_int, default=8)
    command.add_argument("--timeout", type=_positive_float, default=600.0)
    command.add_argument(
        "--retries",
        type=_nonnegative_int,
        default=3,
        help="additional retry rounds for transient provider failures",
    )
    command.add_argument("--fake", action="store_true", help=argparse.SUPPRESS)


def _add_analysis(command: argparse.ArgumentParser) -> None:
    command.add_argument("--permutations", type=_positive_int, default=50_000)
    command.add_argument("--bootstraps", type=_positive_int, default=10_000)
    command.add_argument("--confidence-level", type=_confidence, default=0.95)


def build_parser() -> argparse.ArgumentParser:
    root = argparse.ArgumentParser(
        prog="instruction-duplication",
        description="Run the instruction-duplication experiment.",
    )
    root.add_argument("--version", action="version", version=f"%(prog)s {__version__}")
    root.add_argument("--verbose", action="store_true")
    root.add_argument("--debug", action="store_true")
    commands = root.add_subparsers(dest="command", required=True)

    reproduce = commands.add_parser(
        "reproduce",
        help="prepare, preflight, generate, judge, and analyze",
    )
    _add_workspace(reproduce)
    _add_selection(reproduce)
    _add_generation(reproduce)
    _add_analysis(reproduce)
    reproduce.add_argument("--preflight-timeout", type=_positive_float, default=90.0)
    reproduce.add_argument("--backend", choices=("auto", "hf", "openrouter"), default="auto")
    reproduce.add_argument(
        "--prefer-backend",
        action="append",
        default=[],
        metavar="MODEL=BACKEND",
        help="override one model's automatic backend preference (hf or openrouter)",
    )
    reproduce.set_defaults(handler=command_reproduce)

    prepare = commands.add_parser("prepare", help="prepare an immutable factorial plan")
    _add_workspace(prepare)
    _add_selection(prepare)
    prepare.set_defaults(handler=command_prepare)

    preflight = commands.add_parser("preflight", help="probe and pin provider routes")
    _add_workspace(preflight)
    preflight.add_argument("--timeout", type=_positive_float, default=90.0)
    preflight.add_argument("--backend", choices=("auto", "hf", "openrouter"), default="auto")
    preflight.add_argument(
        "--prefer-backend",
        action="append",
        default=[],
        metavar="MODEL=BACKEND",
        help="override one model's automatic backend preference (hf or openrouter)",
    )
    preflight.add_argument("--max-cost", type=_positive_float)
    preflight.add_argument("--fake", action="store_true", help=argparse.SUPPRESS)
    preflight.set_defaults(handler=command_preflight)

    run = commands.add_parser("run", help="generate or resume pending cells")
    _add_workspace(run)
    _add_generation(run)
    run.set_defaults(handler=command_run)

    status = commands.add_parser("status", help="show plan status, spend, attempts, and routes")
    _add_workspace(status)
    status.set_defaults(handler=command_status)

    judge_parser = commands.add_parser("judge", help="apply current deterministic measurements")
    _add_workspace(judge_parser)
    judge_parser.set_defaults(handler=command_judge)

    analyze = commands.add_parser("analyze", help="compute versioned contrasts and exports")
    _add_workspace(analyze)
    _add_analysis(analyze)
    analyze.set_defaults(handler=command_analyze)
    return root


def main(argv: list[str] | None = None) -> None:
    parser = build_parser()
    args = parser.parse_args(argv)
    logging.basicConfig(
        level=logging.DEBUG if args.debug else logging.INFO if args.verbose else logging.WARNING,
        format="%(levelname)s %(name)s: %(message)s",
    )
    try:
        args.handler(args)
    except KeyboardInterrupt:
        raise SystemExit(130) from None
    except ValueError as exc:
        if args.debug:
            raise
        LOGGER.error("%s", exc)
        raise SystemExit(2) from None
    except RuntimeError as exc:
        if args.debug:
            raise
        LOGGER.error("%s", exc)
        raise SystemExit(3) from None
    except Exception:
        LOGGER.exception("unexpected internal error")
        raise SystemExit(70) from None
