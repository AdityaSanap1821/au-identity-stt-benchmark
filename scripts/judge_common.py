"""Shared helpers for the semantic WER judge experiment scripts."""

import asyncio
import json
import sqlite3
from pathlib import Path

from rich.console import Console
from rich.progress import BarColumn, MofNCompleteColumn, Progress, TextColumn, TimeElapsedColumn

from stt_benchmark.evaluation.semantic_wer import SemanticWEREvaluator
from stt_benchmark.models import ServiceName

# USD per million tokens: (input, output). Cache reads cost 0.1x input and
# 5-minute cache writes 1.25x input.
PRICES = {
    "claude-sonnet-5-5": (2.00, 10.00),
    "claude-sonnet-4-5-20250929": (3.00, 15.00),
}

console = Console()


def parse_judge(spec: str) -> tuple[str, str | None]:
    """Parse ``model:effort``; an effort of ``none`` means no thinking at temperature 0."""
    model, _, effort = spec.partition(":")
    if not effort:
        raise ValueError(
            f"judge spec {spec!r} must be model:effort (effort 'none' for no thinking)"
        )
    return model, None if effort == "none" else effort


def load_pairs(db_path: Path, services: list[str] | None = None) -> list[dict]:
    """Load every transcription that has ground truth, with its stored WER result if any."""
    known = {s.value for s in ServiceName}
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    rows = conn.execute(
        """
        SELECT r.sample_id, r.service_name, r.model_name, r.transcription,
               gt.text AS reference, w.wer AS stored_wer,
               w.substitutions + w.deletions + w.insertions AS stored_error_count,
               w.reference_words AS stored_reference_words,
               w.errors AS stored_errors
        FROM results r
        JOIN ground_truth gt ON gt.sample_id = r.sample_id
        LEFT JOIN wer_metrics w ON w.sample_id = r.sample_id
            AND w.service_name = r.service_name AND w.model_name = r.model_name
        WHERE r.transcription IS NOT NULL AND trim(r.transcription) != ''
            AND trim(gt.text) != ''
        ORDER BY r.service_name, r.sample_id
        """
    ).fetchall()
    conn.close()
    pairs = []
    for row in rows:
        if row["service_name"] not in known or (services and row["service_name"] not in services):
            continue
        pair = dict(row)
        pair["stored_errors"] = json.loads(pair["stored_errors"]) if pair["stored_errors"] else []
        pairs.append(pair)
    return pairs


def pair_key(pair: dict) -> str:
    return f"{pair['service_name']}|{pair['model_name']}|{pair['sample_id']}"


def judgment_cost(evaluator: SemanticWEREvaluator) -> float:
    """Total USD spent by an evaluator so far."""
    price_in, price_out = PRICES.get(evaluator.model, PRICES["claude-sonnet-5-5"])
    u = evaluator.usage
    return (
        u["input_tokens"] * price_in
        + u["cache_read_input_tokens"] * price_in * 0.1
        + u["cache_creation_input_tokens"] * price_in * 1.25
        + u["output_tokens"] * price_out
    ) / 1_000_000


async def run_judgments(
    model: str,
    effort: str | None,
    pairs: list[dict],
    repeats: int,
    log_path: Path,
    db_path: Path,
) -> tuple[dict[str, list[dict | None]], SemanticWEREvaluator, int]:
    """Judge each pair ``repeats`` times, appending every judgment to a JSONL log.

    Judgments already in the log are reused, so an interrupted run picks up
    where it stopped. Returns each pair's runs (None for a failed run) keyed
    by :func:`pair_key`, the evaluator, and the number of judgments made now
    rather than read from the log.
    """
    evaluator = SemanticWEREvaluator(model=model, effort=effort, db_path=db_path)
    done: dict[tuple[str, int], dict | None] = {}
    if log_path.exists():
        for line in log_path.read_text().splitlines():
            entry = json.loads(line)
            # Failed judgments are retried.
            if entry["judge"] == evaluator.judge and entry["result"] is not None:
                done[(entry["key"], entry["repeat"])] = entry["result"]

    todo = [(p, i) for p in pairs for i in range(repeats) if (pair_key(p), i) not in done]
    if todo:
        try:
            await evaluator.warm_cache()
        except Exception as e:
            console.print(f"[yellow]Cache warm-up skipped: {e}[/yellow]")

    log_path.parent.mkdir(parents=True, exist_ok=True)
    with (
        log_path.open("a") as log,
        Progress(
            TextColumn(evaluator.judge.split(" prompt=")[0]),
            BarColumn(),
            MofNCompleteColumn(),
            TimeElapsedColumn(),
            console=console,
        ) as progress,
    ):
        task = progress.add_task("judging", total=len(pairs) * repeats, completed=len(done))

        async def judge_once(pair: dict, repeat: int) -> None:
            outcome = await evaluator.evaluate_with_retry(
                pair["reference"], pair["transcription"], filename=pair["sample_id"]
            )
            result = None
            if outcome is not None:
                r, _ = outcome
                result = {
                    "wer": r["wer"],
                    "error_count": r["substitutions"] + r["deletions"] + r["insertions"],
                    "reference_words": r["reference_words"],
                    "errors": r.get("errors", []),
                }
            done[(pair_key(pair), repeat)] = result
            log.write(
                json.dumps(
                    {
                        "judge": evaluator.judge,
                        "key": pair_key(pair),
                        "repeat": repeat,
                        "result": result,
                    }
                )
                + "\n"
            )
            log.flush()
            progress.advance(task)

        await asyncio.gather(*(judge_once(p, i) for p, i in todo))

    runs = {pair_key(p): [done.get((pair_key(p), i)) for i in range(repeats)] for p in pairs}
    return runs, evaluator, len(todo)
