#!/usr/bin/env python3
"""Compare semantic WER judge configurations before re-scoring the benchmark.

Picks a random set of (sample, service) transcriptions, judges each one
several times per effort level, and reports how much the judge disagrees
with itself between runs, how far it lands from the stored scores, what
fraction of runs fail, and what a full re-score would cost. Nothing is
written to the results database.

Usage:
    uv run python scripts/judge_consistency.py
    uv run python scripts/judge_consistency.py --pairs 100 --repeats 3 --efforts medium high
"""

import argparse
import asyncio
import json
import random
import sqlite3
import statistics
from datetime import UTC, datetime
from pathlib import Path

from rich.console import Console
from rich.progress import BarColumn, MofNCompleteColumn, Progress, TextColumn, TimeElapsedColumn
from rich.table import Table

from stt_benchmark.config import get_config
from stt_benchmark.evaluation.semantic_wer import DEFAULT_JUDGE_MODEL, SemanticWEREvaluator
from stt_benchmark.models import ServiceName

# Claude Sonnet 5.5 prices in USD per million tokens (5-minute cache writes).
PRICE_PER_MTOK = {
    "input_tokens": 2.00,
    "output_tokens": 10.00,
    "cache_read_input_tokens": 0.20,
    "cache_creation_input_tokens": 2.50,
}

console = Console()


def load_pairs(db_path: Path) -> list[dict]:
    """Load every transcription that has ground truth, with its stored WER if any."""
    services = {s.value for s in ServiceName}
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    rows = conn.execute(
        """
        SELECT r.sample_id, r.service_name, r.model_name, r.transcription,
               gt.text AS reference, w.wer AS stored_wer
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
    return [dict(row) for row in rows if row["service_name"] in services]


async def judge_pairs(effort: str, pairs: list[dict], repeats: int, db_path: Path) -> dict:
    """Judge every pair ``repeats`` times at one effort level."""
    evaluator = SemanticWEREvaluator(model=DEFAULT_JUDGE_MODEL, effort=effort, db_path=db_path)
    await evaluator.warm_cache()

    with Progress(
        TextColumn(f"effort={effort}"),
        BarColumn(),
        MofNCompleteColumn(),
        TimeElapsedColumn(),
        console=console,
    ) as progress:
        task = progress.add_task("judging", total=len(pairs) * repeats)

        async def judge_once(pair: dict) -> dict | None:
            outcome = await evaluator.evaluate_with_retry(
                pair["reference"], pair["transcription"], filename=pair["sample_id"]
            )
            progress.advance(task)
            if outcome is None:
                return None
            result, _ = outcome
            return {
                "wer": result["wer"],
                "errors": result["substitutions"] + result["deletions"] + result["insertions"],
                "reference_words": result["reference_words"],
            }

        runs = await asyncio.gather(
            *(asyncio.gather(*(judge_once(pair) for _ in range(repeats))) for pair in pairs)
        )

    return {"judge": evaluator.judge, "runs": [list(r) for r in runs], "usage": evaluator.usage}


def summarize(pairs: list[dict], judged: dict, total_pairs: int) -> dict:
    """Reduce one effort level's runs to comparison statistics."""
    runs = judged["runs"]
    num_runs = sum(len(r) for r in runs)
    failed = sum(run is None for r in runs for run in r)
    complete = [(pair, r) for pair, r in zip(pairs, runs, strict=True) if None not in r]

    medians, spreads, identical, stored_diffs = [], [], 0, []
    errors_total = words_total = 0
    for pair, r in complete:
        by_wer = sorted(r, key=lambda run: run["wer"])
        median = by_wer[len(by_wer) // 2]
        medians.append(median["wer"])
        errors_total += median["errors"]
        words_total += median["reference_words"]
        spreads.append(by_wer[-1]["wer"] - by_wer[0]["wer"])
        identical += len({run["errors"] for run in r}) == 1
        if pair["stored_wer"] is not None:
            stored_diffs.append(median["wer"] - pair["stored_wer"])

    usage = judged["usage"]
    succeeded = max(num_runs - failed, 1)
    cost = sum(usage[k] * price / 1_000_000 for k, price in PRICE_PER_MTOK.items())
    cost_per_judgment = cost / succeeded

    return {
        "judge": judged["judge"],
        "failed_runs": f"{failed}/{num_runs}",
        "mean_wer": statistics.mean(medians) if medians else None,
        "pooled_wer": errors_total / words_total if words_total else None,
        "identical_repeats": identical / len(complete) if complete else None,
        "mean_repeat_spread": statistics.mean(spreads) if spreads else None,
        "max_repeat_spread": max(spreads) if spreads else None,
        "mean_abs_diff_vs_stored": (
            statistics.mean(abs(d) for d in stored_diffs) if stored_diffs else None
        ),
        "mean_diff_vs_stored": statistics.mean(stored_diffs) if stored_diffs else None,
        "output_tokens_per_judgment": usage["output_tokens"] / succeeded,
        "cost_per_judgment": cost_per_judgment,
        "full_rescore_cost_1x": cost_per_judgment * total_pairs,
        "full_rescore_cost_3x": cost_per_judgment * total_pairs * 3,
        "medians": medians,
    }


def fmt(value, kind: str) -> str:
    if value is None:
        return "-"
    if kind == "pct":
        return f"{value:.2%}"
    if kind == "usd":
        return f"${value:,.2f}"
    if kind == "int":
        return f"{value:,.0f}"
    return str(value)


def print_summary(summaries: dict[str, dict], repeats: int) -> None:
    table = Table(title=f"Judge comparison ({repeats} runs per transcription)")
    table.add_column("Metric", style="cyan")
    for effort in summaries:
        table.add_column(f"effort={effort}", justify="right")

    rows = [
        ("Failed runs", "failed_runs", "str"),
        ("Mean WER (median run)", "mean_wer", "pct"),
        ("Pooled WER (median run)", "pooled_wer", "pct"),
        ("Identical error counts across runs", "identical_repeats", "pct"),
        ("Mean WER spread across runs", "mean_repeat_spread", "pct"),
        ("Max WER spread across runs", "max_repeat_spread", "pct"),
        ("Mean |diff| vs stored score", "mean_abs_diff_vs_stored", "pct"),
        ("Mean diff vs stored score", "mean_diff_vs_stored", "pct"),
        ("Output tokens per judgment", "output_tokens_per_judgment", "int"),
        ("Cost per judgment", "cost_per_judgment", "usd"),
        ("Full re-score, 1 run each", "full_rescore_cost_1x", "usd"),
        ("Full re-score, 3 runs each", "full_rescore_cost_3x", "usd"),
    ]
    for label, key, kind in rows:
        table.add_row(label, *(fmt(s[key], kind) for s in summaries.values()))
    console.print(table)

    efforts = list(summaries)
    for i, a in enumerate(efforts):
        for b in efforts[i + 1 :]:
            ma, mb = summaries[a]["medians"], summaries[b]["medians"]
            if len(ma) == len(mb) and ma:
                diff = statistics.mean(abs(x - y) for x, y in zip(ma, mb, strict=True))
                console.print(f"Mean |diff| between {a} and {b}: {diff:.2%}")


async def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--pairs", type=int, default=100, help="Transcriptions to judge")
    parser.add_argument("--repeats", type=int, default=3, help="Runs per transcription")
    parser.add_argument(
        "--efforts", nargs="+", default=["medium", "high"], help="Effort levels to compare"
    )
    parser.add_argument("--seed", type=int, default=0, help="Seed for picking transcriptions")
    parser.add_argument("--test", action="store_true", help="Use test_results.db")
    parser.add_argument("--output", type=Path, help="Where to write the raw runs as JSON")
    args = parser.parse_args()

    config = get_config()
    db_path = config.data_dir / "test_results.db" if args.test else config.results_db

    all_pairs = load_pairs(db_path)
    pairs = random.Random(args.seed).sample(all_pairs, min(args.pairs, len(all_pairs)))
    console.print(
        f"Judging {len(pairs)} of {len(all_pairs)} transcriptions,"
        f" {args.repeats} runs each, efforts: {', '.join(args.efforts)}\n"
    )

    judged, summaries = {}, {}
    for effort in args.efforts:
        judged[effort] = await judge_pairs(effort, pairs, args.repeats, db_path)
        summaries[effort] = summarize(pairs, judged[effort], len(all_pairs))

    console.print()
    print_summary(summaries, args.repeats)

    output = args.output or config.data_dir / (
        f"judge_consistency_{datetime.now(UTC):%Y%m%dT%H%M%SZ}.json"
    )
    output.write_text(
        json.dumps(
            {
                "pairs": [
                    {k: pair[k] for k in ("sample_id", "service_name", "model_name", "stored_wer")}
                    for pair in pairs
                ],
                "efforts": {
                    effort: {
                        "judge": judged[effort]["judge"],
                        "runs": judged[effort]["runs"],
                        "usage": dict(judged[effort]["usage"]),
                        "summary": {k: v for k, v in summaries[effort].items() if k != "medians"},
                    }
                    for effort in args.efforts
                },
            },
            indent=2,
        )
    )
    console.print(f"\nRaw runs written to {output}")


if __name__ == "__main__":
    asyncio.run(main())
