#!/usr/bin/env python3
"""Measure how consistently semantic WER judges score the same transcription.

Judges a set of transcriptions several times with each judge and reports how
often the runs agree, how much judge noise that adds to a service's mean WER,
and how closely each judge matches the stored scores. Half the transcriptions
are ones the stored judge found errors in, since those are the ones judges
disagree on. Nothing is written to the results database; every judgment is
logged to ``runs.jsonl`` in the output directory, and re-running the same
command resumes from it.

Usage:
    uv run python scripts/judge_consistency.py
    uv run python scripts/judge_consistency.py --pairs 200 --repeats 5 \\
        --judges claude-sonnet-5-5:medium claude-sonnet-5-5:high
"""

import argparse
import asyncio
import json
import math
import random
import statistics
from pathlib import Path

from judge_common import console, judgment_cost, load_pairs, pair_key, parse_judge, run_judgments
from rich.table import Table

from stt_benchmark.config import get_config

DEFAULT_JUDGES = [
    "claude-sonnet-4-5-20250929:none",
    "claude-sonnet-5-5:medium",
    "claude-sonnet-5-5:high",
]
SERVICE_SAMPLES = 1000


def pick_pairs(all_pairs: list[dict], count: int, seed: int) -> list[dict]:
    """Pick ``count`` pairs, half with stored errors and half without."""
    rng = random.Random(seed)
    with_errors = [p for p in all_pairs if (p["stored_error_count"] or 0) > 0]
    without = [p for p in all_pairs if p["stored_error_count"] == 0]
    half = count // 2
    return rng.sample(with_errors, min(half, len(with_errors))) + rng.sample(
        without, min(count - half, len(without))
    )


def summarize(pairs: list[dict], runs: dict, error_share: float) -> dict:
    """Reduce one judge's runs to consistency statistics, per stratum."""
    strata = {"with errors": [], "without errors": []}
    failed = total = 0
    for pair in pairs:
        r = runs[pair_key(pair)]
        total += len(r)
        failed += sum(run is None for run in r)
        if None in r:
            continue
        stratum = "with errors" if pair["stored_error_count"] else "without errors"
        by_wer = sorted(r, key=lambda run: run["wer"])
        median = by_wer[len(by_wer) // 2]
        strata[stratum].append(
            {
                "identical": len({run["error_count"] for run in r}) == 1,
                "variance": statistics.pvariance([run["wer"] for run in r]),
                "matches_stored": median["error_count"] == pair["stored_error_count"],
                "abs_diff_stored": abs(median["wer"] - pair["stored_wer"]),
            }
        )

    summary = {"failed_runs": f"{failed}/{total}"}
    for name, items in strata.items():
        summary[name] = {
            "n": len(items),
            "identical": statistics.mean(i["identical"] for i in items) if items else None,
            "sd": math.sqrt(statistics.mean(i["variance"] for i in items)) if items else None,
            "matches_stored": (
                statistics.mean(i["matches_stored"] for i in items) if items else None
            ),
            "abs_diff_stored": (
                statistics.mean(i["abs_diff_stored"] for i in items) if items else None
            ),
        }
    # Judge noise on a service's mean WER, weighting each stratum by its share
    # of all transcriptions.
    sd_with, sd_without = summary["with errors"]["sd"], summary["without errors"]["sd"]
    if sd_with is not None and sd_without is not None:
        variance = error_share * sd_with**2 + (1 - error_share) * sd_without**2
        summary["service_mean_noise_95"] = 1.96 * math.sqrt(variance / SERVICE_SAMPLES)
    else:
        summary["service_mean_noise_95"] = None
    return summary


def pct(value) -> str:
    return "-" if value is None else f"{value:.1%}"


def print_summary(summaries: dict[str, dict], repeats: int) -> None:
    table = Table(title=f"Judge consistency ({repeats} runs per transcription)")
    table.add_column("Metric", style="cyan")
    for judge in summaries:
        table.add_column(judge, justify="right")

    def row(label, get):
        table.add_row(label, *(get(s) for s in summaries.values()))

    row("Failed runs", lambda s: s["failed_runs"])
    for stratum in ("with errors", "without errors"):
        row(f"[bold]Stored {stratum}[/bold] (n)", lambda s, k=stratum: str(s[k]["n"]))
        row("  Same error count every run", lambda s, k=stratum: pct(s[k]["identical"]))
        row("  WER std dev across runs", lambda s, k=stratum: pct(s[k]["sd"]))
        row("  Error count matches stored", lambda s, k=stratum: pct(s[k]["matches_stored"]))
        row("  Mean |WER diff| vs stored", lambda s, k=stratum: pct(s[k]["abs_diff_stored"]))
    row(
        f"Noise in a {SERVICE_SAMPLES:,}-sample mean (95%)",
        lambda s: (
            "-" if s["service_mean_noise_95"] is None else f"±{s['service_mean_noise_95']:.2%}"
        ),
    )
    row(
        "Cost per judgment",
        lambda s: "-" if s["cost_per_judgment"] is None else f"${s['cost_per_judgment']:.4f}",
    )
    console.print(table)


async def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--pairs", type=int, default=100, help="Transcriptions to judge")
    parser.add_argument("--repeats", type=int, default=3, help="Runs per transcription")
    parser.add_argument(
        "--judges",
        nargs="+",
        default=DEFAULT_JUDGES,
        help="Judges as model:effort; effort 'none' means no thinking at temperature 0",
    )
    parser.add_argument("--seed", type=int, default=0, help="Seed for picking transcriptions")
    parser.add_argument("--test", action="store_true", help="Use test_results.db")
    parser.add_argument("--output-dir", type=Path, help="Where to write runs and the summary")
    args = parser.parse_args()

    config = get_config()
    db_path = config.data_dir / "test_results.db" if args.test else config.results_db
    output_dir = args.output_dir or config.data_dir / "judge_experiments" / (
        f"consistency_n{args.pairs}_r{args.repeats}_s{args.seed}"
    )

    all_pairs = [p for p in load_pairs(db_path) if p["stored_error_count"] is not None]
    error_share = statistics.mean((p["stored_error_count"] or 0) > 0 for p in all_pairs)
    pairs = pick_pairs(all_pairs, args.pairs, args.seed)
    console.print(
        f"Judging {len(pairs)} transcriptions ({error_share:.0%} of all stored scores have"
        f" errors), {args.repeats} runs each\n"
    )

    summaries = {}
    for spec in args.judges:
        model, effort = parse_judge(spec)
        runs, evaluator, new_judgments = await run_judgments(
            model, effort, pairs, args.repeats, output_dir / "runs.jsonl", db_path
        )
        summary = summarize(pairs, runs, error_share)
        summary["cost_per_judgment"] = (
            judgment_cost(evaluator) / new_judgments if new_judgments else None
        )
        summaries[evaluator.judge.split(" repeats=")[0]] = summary

    console.print()
    print_summary(summaries, args.repeats)
    (output_dir / "summary.json").write_text(json.dumps(summaries, indent=2))
    console.print(f"\nRuns and summary written to {output_dir}")


if __name__ == "__main__":
    asyncio.run(main())
