"""
Dream orchestrator for the specialist-based architecture.

This module coordinates the full dream cycle:
0. [Optional] Surprisal sampling: Pre-filter observations by geometric surprisal
1. Run deduction specialist (self-directed exploration, creates deductive observations)
2. Run induction specialist (self-directed exploration, creates inductive observations)

Specialists are self-directed agents that explore the observation space and create
higher-level observations. When surprisal sampling finds interesting observations,
they're passed as hints, but specialists are free to follow the evidence wherever it leads.
"""

from __future__ import annotations

import logging
import re
import time
import uuid
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any

import sentry_sdk
from sqlalchemy import func, select

from src import crud, models
from src.config import settings
from src.dependencies import tracked_db
from src.dreamer.specialists import SPECIALISTS, SpecialistResult
from src.dreamer.surprisal import SurprisalScore  # type: ignore
from src.exceptions import SpecialistExecutionError, SurprisalError
from src.schemas import DreamType
from src.telemetry.events import DreamRunEvent, emit
from src.telemetry.logging import (
    accumulate_metric,
    log_performance_metrics,
)
from src.utils.config_helpers import get_configuration
from src.utils.queue_payload import DreamPayload

logger = logging.getLogger(__name__)

_DREAM_META_MARKERS = (
    "dreamer",
    "schedule dream",
    "schedule_dream",
    "queue",
    "204",
    "deriver",
    "baseline",
    "consolidation",
)
_CONTENT_NORMALIZATION_TABLE = str.maketrans(
    {
        char: " "
        for char in (
            r"""!"#$%&'()*+,-./:;<=>?@[\]^_`{|}~，。！？、；：‘’“”（）【】《》〈〉「」『』〔〕［］｛｝﹝﹞…—～·"""
        )
    }
)


@dataclass
class DreamResult:
    """Result of a dream cycle for telemetry reporting."""

    # Run identification
    run_id: str
    specialists_run: list[str]

    # Specialist outcomes
    deduction_success: bool
    induction_success: bool

    # Surprisal sampling
    surprisal_enabled: bool
    surprisal_conclusion_count: int

    # Aggregate metrics
    total_iterations: int
    total_duration_ms: float
    input_tokens: int
    output_tokens: int


@dataclass
class DreamCollectionSnapshot:
    """Minimal collection state needed to judge whether a dream converged."""

    active_ids: set[str]
    explicit_count: int
    explicit_unique_norm_count: int
    peer_card: tuple[str, ...]
    derived_docs: list[tuple[str, str, str]]


def _normalize_observation_content(content: str) -> str:
    """Normalize observation text for coarse duplicate accounting."""
    normalized = content.casefold().translate(_CONTENT_NORMALIZATION_TABLE)
    return re.sub(r"\s+", " ", normalized).strip()


def _is_meta_dream_observation(content: str) -> bool:
    """Filter dream self-reporting that should not count as useful consolidation."""
    normalized = _normalize_observation_content(content)
    return any(marker in normalized for marker in _DREAM_META_MARKERS)


async def _snapshot_collection(
    workspace_name: str,
    observer: str,
    observed: str,
) -> DreamCollectionSnapshot:
    """Capture active documents and peer card state for before/after comparison."""
    async with tracked_db("dream.snapshot") as db:
        stmt = select(models.Document.id, models.Document.level, models.Document.content).where(
            models.Document.workspace_name == workspace_name,
            models.Document.observer == observer,
            models.Document.observed == observed,
            models.Document.deleted_at.is_(None),
        )
        rows = (await db.execute(stmt)).all()
        peer_card = await crud.get_peer_card(
            db,
            workspace_name=workspace_name,
            observer=observer,
            observed=observed,
        )

    active_ids = {row.id for row in rows}
    explicit_contents = [row.content for row in rows if row.level == "explicit"]
    normalized_explicit = {
        _normalize_observation_content(content)
        for content in explicit_contents
        if _normalize_observation_content(content)
    }
    derived_docs = [
        (row.id, row.level, row.content) for row in rows if row.level != "explicit"
    ]
    return DreamCollectionSnapshot(
        active_ids=active_ids,
        explicit_count=len(explicit_contents),
        explicit_unique_norm_count=len(normalized_explicit),
        peer_card=tuple(peer_card or []),
        derived_docs=derived_docs,
    )


async def run_dream(
    workspace_name: str,
    observer: str,
    observed: str,
    session_name: str | None = None,
) -> DreamResult | None:
    """
    Run a full dream cycle with optional surprisal-based sampling.

    The dream cycle runs specialists sequentially:
    0. [Optional] Surprisal sampling: Pre-filter observations by geometric surprisal
    1. Deduction specialist: Creates deductive observations from explicit facts
    2. Induction specialist: Creates inductive observations from patterns

    Uses short-lived DB sessions to avoid holding connections during LLM calls.

    Args:
        workspace_name: Workspace identifier
        observer: Observer peer name
        observed: Observed peer name
        session_name: Session identifier if specified
    """
    if not settings.DREAM.ENABLED:
        return None

    run_id = str(uuid.uuid4())[:8]
    task_name = f"dream_orchestrator_{run_id}"
    start_time = time.perf_counter()

    logger.info(
        f"[{run_id}] Starting dream cycle for {workspace_name}/{observer}/{observed}"
    )

    # Short-lived DB session for config resolution
    async with tracked_db("dream.config") as db:
        if session_name is not None:
            session = await crud.get_session(
                db, workspace_name=workspace_name, session_name=session_name
            )
        else:
            session = None

        workspace = await crud.get_workspace(db, workspace_name=workspace_name)
        configuration = get_configuration(None, session, workspace)
    if not configuration.dream.enabled:
        logger.info(
            f"[{run_id}] Dreams disabled for {workspace_name}/{session_name}, skipping dream"
        )
        return None

    # Track specialist outcomes
    deduction_success = False
    induction_success = False
    surprisal_observation_count = 0
    deduction_result: SpecialistResult | None = None
    induction_result: SpecialistResult | None = None

    # Phase 0: Surprisal-based sampling (if enabled)
    # Specialists are self-directed by default - hints are optional suggestions
    exploration_hints: list[str] | None = None

    if settings.DREAM.SURPRISAL.ENABLED:
        logger.info(f"[{run_id}] Phase 0: Computing surprisal scores")
        try:
            from src.dreamer.surprisal import sample_observations_with_surprisal

            high_surprisal_obs = await sample_observations_with_surprisal(
                workspace_name=workspace_name,
                observer=observer,
                observed=observed,
            )

            logger.info(
                f"[{run_id}] Surprisal: Found {len(high_surprisal_obs)} high-surprisal observations"
            )
            surprisal_observation_count = len(high_surprisal_obs)
            accumulate_metric(
                task_name, "surprisal_observations", len(high_surprisal_obs), "count"
            )

            if len(high_surprisal_obs) > 0:
                # Use high-surprisal observations as hints for exploration
                exploration_hints = _create_queries_from_surprisal(high_surprisal_obs)
                logger.info(
                    f"[{run_id}] ✨ SURPRISAL HINTS: Suggesting {len(exploration_hints)} "
                    + "high-surprisal topics for specialists to investigate"
                )
                logger.info(
                    f"[{run_id}] Targeting observations with surprisal range: "
                    + f"{high_surprisal_obs[-1].surprisal:.3f} to {high_surprisal_obs[0].surprisal:.3f}"
                )
            else:
                logger.info(
                    f"[{run_id}] No high-surprisal observations - specialists will explore freely"
                )

        except SurprisalError as e:
            logger.error(f"[{run_id}] Surprisal sampling failed: {e}", exc_info=True)
            accumulate_metric(task_name, "surprisal_error", str(e), "blob")
            # Specialists will explore freely without hints

    # Phase 1: Run deduction specialist (manages its own DB sessions)
    logger.info(f"[{run_id}] Phase 1: Running deduction specialist")
    deduction_specialist = SPECIALISTS["deduction"]
    try:
        deduction_result = await deduction_specialist.run(
            workspace_name=workspace_name,
            observer=observer,
            observed=observed,
            session_name=session_name,
            hints=exploration_hints,
            configuration=configuration,
            parent_run_id=run_id,
        )
        logger.info(
            f"[{run_id}] Deduction completed: {deduction_result.content[:200]}..."
        )
        accumulate_metric(
            task_name, "deduction_result", deduction_result.content, "blob"
        )
        deduction_success = deduction_result.success
    except SpecialistExecutionError as e:
        logger.error(f"[{run_id}] Deduction specialist failed: {e}", exc_info=True)
        accumulate_metric(task_name, "deduction_error", str(e), "blob")

    # Phase 2: Run induction specialist (after deduction so it can see new deductive obs)
    logger.info(f"[{run_id}] Phase 2: Running induction specialist")
    induction_specialist = SPECIALISTS["induction"]
    try:
        induction_result = await induction_specialist.run(
            workspace_name=workspace_name,
            observer=observer,
            observed=observed,
            session_name=session_name,
            hints=exploration_hints,
            configuration=configuration,
            parent_run_id=run_id,
        )
        logger.info(
            f"[{run_id}] Induction completed: {induction_result.content[:200]}..."
        )
        accumulate_metric(
            task_name, "induction_result", induction_result.content, "blob"
        )
        induction_success = induction_result.success
    except SpecialistExecutionError as e:
        logger.error(f"[{run_id}] Induction specialist failed: {e}", exc_info=True)
        accumulate_metric(task_name, "induction_error", str(e), "blob")

    # Log final metrics
    duration_ms = (time.perf_counter() - start_time) * 1000
    accumulate_metric(task_name, "total_duration", duration_ms, "ms")

    logger.info(f"[{run_id}] Dream cycle completed in {duration_ms:.0f}ms")
    log_performance_metrics("dream_orchestrator", run_id)

    # Aggregate metrics from specialist results
    total_iterations = (deduction_result.iterations if deduction_result else 0) + (
        induction_result.iterations if induction_result else 0
    )
    total_input_tokens = (deduction_result.input_tokens if deduction_result else 0) + (
        induction_result.input_tokens if induction_result else 0
    )
    total_output_tokens = (
        deduction_result.output_tokens if deduction_result else 0
    ) + (induction_result.output_tokens if induction_result else 0)

    # Emit DreamRunEvent with aggregated metrics
    emit(
        DreamRunEvent(
            run_id=run_id,
            workspace_name=workspace_name,
            session_name=session_name,
            observer=observer,
            observed=observed,
            specialists_run=["deduction", "induction"],
            deduction_success=deduction_success,
            induction_success=induction_success,
            surprisal_enabled=settings.DREAM.SURPRISAL.ENABLED,
            surprisal_conclusion_count=surprisal_observation_count,
            total_iterations=total_iterations,
            total_input_tokens=total_input_tokens,
            total_output_tokens=total_output_tokens,
            total_duration_ms=duration_ms,
        )
    )

    return DreamResult(
        run_id=run_id,
        specialists_run=["deduction", "induction"],
        deduction_success=deduction_success,
        induction_success=induction_success,
        surprisal_enabled=settings.DREAM.SURPRISAL.ENABLED,
        surprisal_conclusion_count=surprisal_observation_count,
        total_iterations=total_iterations,
        total_duration_ms=duration_ms,
        input_tokens=total_input_tokens,
        output_tokens=total_output_tokens,
    )


def _create_queries_from_surprisal(
    high_surprisal_obs: list[SurprisalScore],
) -> list[str]:
    """
    Create search queries from high-surprisal observations.

    Strategy: Use observation content as semantic search queries.
    Truncate if too long (>200 chars).

    Args:
        high_surprisal_obs: List of SurprisalScore objects

    Returns:
        List of query strings (max 10)
    """
    queries: list[Any] = []
    for score in high_surprisal_obs:
        content = score.observation.content
        if len(content) > 200:
            content = content[:200] + "..."
        queries.append(content)
    return queries[:10]  # Limit to 10 queries


@sentry_sdk.trace
async def process_dream(
    payload: DreamPayload,
    workspace_name: str,
) -> None:
    """
    Process a dream task by performing collection maintenance operations.

    Args:
        payload: The dream task payload containing workspace, peer, and dream type information
    """
    logger.info(
        f"""
(っ- ‸ - ς)ᶻ z 𐰁 ᶻ z 𐰁 ᶻ z 𐰁\n
DREAM: {payload.dream_type} documents for {workspace_name}/{payload.observer}/{payload.observed}\n
𐰁 z ᶻ 𐰁 z ᶻ 𐰁 z ᶻ(っ- ‸ - ς)"""
    )

    try:
        match payload.dream_type:
            case DreamType.OMNI:
                before_state = await _snapshot_collection(
                    workspace_name,
                    payload.observer,
                    payload.observed,
                )
                result = await run_dream(
                    workspace_name=workspace_name,
                    observer=payload.observer,
                    observed=payload.observed,
                    session_name=payload.session_name,
                )

                # Log completion (telemetry event already emitted in run_dream)
                if result is not None:
                    after_state = await _snapshot_collection(
                        workspace_name,
                        payload.observer,
                        payload.observed,
                    )
                    deleted_document_ids = before_state.active_ids - after_state.active_ids
                    new_non_meta_derived_docs = [
                        (doc_id, level, content)
                        for doc_id, level, content in after_state.derived_docs
                        if doc_id not in before_state.active_ids
                        and not _is_meta_dream_observation(content)
                    ]
                    peer_card_changed = before_state.peer_card != after_state.peer_card
                    before_duplicate_count = (
                        before_state.explicit_count
                        - before_state.explicit_unique_norm_count
                    )
                    after_duplicate_count = (
                        after_state.explicit_count - after_state.explicit_unique_norm_count
                    )
                    should_advance_baseline = bool(
                        deleted_document_ids
                        or peer_card_changed
                        or new_non_meta_derived_docs
                    )

                    logger.info(
                        f"Dream completed: run_id={result.run_id}, "
                        + f"iterations={result.total_iterations}, "
                        + f"duration={result.total_duration_ms:.0f}ms"
                    )
                    logger.info(
                        "[%s] dream materialization: deleted=%s new_non_meta_derived=%s "
                        + "peer_card_changed=%s explicit_before=%s explicit_after=%s "
                        + "unique_before=%s unique_after=%s baseline=%s",
                        result.run_id,
                        len(deleted_document_ids),
                        len(new_non_meta_derived_docs),
                        peer_card_changed,
                        before_state.explicit_count,
                        after_state.explicit_count,
                        before_state.explicit_unique_norm_count,
                        after_state.explicit_unique_norm_count,
                        "advance" if should_advance_baseline else "skip",
                    )

                    # Both guard fields advance together only on successful consolidation.
                    if should_advance_baseline:
                        now_iso = datetime.now(timezone.utc).isoformat()
                        async with tracked_db("dream.guard_pair_write") as db:
                            collection = await crud.get_collection(
                                db,
                                workspace_name,
                                observer=payload.observer,
                                observed=payload.observed,
                                with_for_update=True,
                            )
                            count_stmt = select(func.count(models.Document.id)).where(
                                models.Document.workspace_name == workspace_name,
                                models.Document.observer == payload.observer,
                                models.Document.observed == payload.observed,
                                models.Document.level == "explicit",
                                models.Document.deleted_at.is_(None),
                            )
                            current_explicit_count = int(await db.scalar(count_stmt) or 0)
                            dream_meta = dict(collection.internal_metadata.get("dream", {}))
                            dream_meta["last_dream_at"] = now_iso
                            dream_meta["last_dream_document_count"] = current_explicit_count
                            await crud.update_collection_internal_metadata(
                                db,
                                workspace_name,
                                payload.observer,
                                payload.observed,
                                update_data={"dream": dream_meta},
                            )
                    else:
                        logger.info(
                            "[%s] Skipping dream baseline advance: no material consolidation "
                            + "change detected",
                            result.run_id,
                        )

    except Exception as e:
        logger.error(
            f"Error processing dream task {payload.dream_type} for {payload.observer}/{payload.observed}: {str(e)}",
            exc_info=True,
        )
        if settings.SENTRY.ENABLED:
            sentry_sdk.capture_exception(e)
        # Don't re-raise - we want to mark the dream task as processed even if it fails
