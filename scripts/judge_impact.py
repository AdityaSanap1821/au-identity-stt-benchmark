#!/usr/bin/env python3
"""Measure how a new semantic WER judge changes the benchmark's results.

Re-judges every transcription for the chosen services with the new judge and
compares the outcome with the stored scores: each service's mean and pooled
WER, with a 95% confidence interval on the change, the ranking, and the gaps
between neighbouring services. It also writes the transcriptions where the
two judges disagree most to ``disagreements.md`` for manual review.

Nothing is written to the results database; every judgment is logged to
``runs.jsonl`` in the output directory, and re-running the same command
resumes from it.

With ``--baseline``, nothing is judged: the scores stored in the results
database are compared against those in a copy of it saved before a re-score.

Usage:
    uv run python scripts/judge_impact.py --services meta,azure,deepgram
    uv run python scripts/judge_impact.py --services meta,azure --judge claude-sonnet-5-5:high
    uv run python scripts/judge_impact.py --services meta,azure --baseline results_backup.db
"""

import argparse
import asyncio
import json
import random
import statistics
from pathlib import Path

from judge_common import console, judgment_cost, load_pairs, pair_key, parse_judge, run_judgments
from rich.table import Table

from stt_benchmark.config import get_config

BOOTSTRAP_ITERATIONS = 2000


def bootstrap_ci(diffs: list[float], seed: int = 0) -> tuple[float, float]:
    """95% bootstrap confidence interval for the mean of paired differences."""
    rng = random.Random(seed)
    n = len(diffs)
    means = sorted(sum(rng.choices(diffs, k=n)) / n for _ in range(BOOTSTRAP_ITERATIONS))
    return means[int(0.025 * BOOTSTRAP_ITERATIONS)], means[int(0.975 * BOOTSTRAP_ITERATIONS) - 1]


def service_stats(pairs: list[dict]) -> dict:
    """Stored and new WER for one service's judged transcriptions."""
    stored = [p["stored_wer"] for p in pairs]
    new = [p["new"]["wer"] for p in pairs]
    low, high = bootstrap_ci([n - s for n, s in zip(new, stored, strict=True)])
    return {
        "n": len(pairs),
        "stored_mean": statistics.mean(stored),
        "new_mean": statistics.mean(new),
        "diff_ci": (low, high),
        "stored_pooled": sum(p["stored_error_count"] for p in pairs)
        / sum(p["stored_reference_words"] for p in pairs),
        "new_pooled": sum(p["new"]["error_count"] for p in pairs)
        / sum(p["new"]["reference_words"] for p in pairs),
        "changed": statistics.mean(
            p["new"]["error_count"] != p["stored_error_count"] for p in pairs
        ),
    }


def gap_stats(a: list[dict], b: list[dict]) -> dict:
    """Mean WER gap between two services over the samples both have, per judge."""
    b_by_sample = {p["sample_id"]: p for p in b}
    shared = [(p, b_by_sample[p["sample_id"]]) for p in a if p["sample_id"] in b_by_sample]
    new_diffs = [pb["new"]["wer"] - pa["new"]["wer"] for pa, pb in shared]
    return {
        "stored": statistics.mean(pb["stored_wer"] - pa["stored_wer"] for pa, pb in shared),
        "new": statistics.mean(new_diffs),
        "new_ci": bootstrap_ci(new_diffs),
    }


def errors_text(errors: list[dict]) -> str:
    """Render a judge's error list; stored and new judges name the fields differently."""
    parts = []
    for e in errors:
        kind = e.get("error_type") or e.get("type") or "?"
        ref = e.get("reference_word", e.get("reference"))
        hyp = e.get("hypothesis_word", e.get("hypothesis"))
        parts.append(f"{kind}: {ref or '∅'} → {hyp or '∅'}")
    return "; ".join(parts) or "none"


def write_disagreements(pairs: list[dict], count: int, path: Path) -> None:
    ranked = sorted(
        (p for p in pairs if p["new"]["error_count"] != p["stored_error_count"]),
        key=lambda p: abs(p["new"]["error_count"] - p["stored_error_count"]),
        reverse=True,
    )[:count]
    lines = [f"# Largest judge disagreements ({len(ranked)})", ""]
    for p in ranked:
        lines += [
            f"## {p['service_name']} / {p['sample_id']}",
            "",
            f"- **Reference:** {p['reference']}",
            f"- **Hypothesis:** {p['transcription']}",
            f"- **Stored judge** ({p['stored_error_count']} errors): {errors_text(p['stored_errors'])}",
            f"- **New judge** ({p['new']['error_count']} errors): {errors_text(p['new']['errors'])}",
            "",
        ]
    path.write_text("\n".join(lines))


def print_report(by_service: dict[str, list[dict]], title: str) -> None:
    stats = {name: service_stats(pairs) for name, pairs in by_service.items()}
    stored_rank = {n: i for i, n in enumerate(sorted(stats, key=lambda n: stats[n]["stored_mean"]))}
    new_order = sorted(stats, key=lambda n: stats[n]["new_mean"])

    table = Table(title=title)
    for column in (
        "Service",
        "n",
        "Mean WER stored",
        "Mean WER new",
        "Change (95% CI)",
        "Pooled stored",
        "Pooled new",
        "Samples changed",
        "Rank",
    ):
        table.add_column(column, justify="left" if column == "Service" else "right")
    for rank, name in enumerate(new_order):
        s = stats[name]
        low, high = s["diff_ci"]
        table.add_row(
            name,
            str(s["n"]),
            f"{s['stored_mean']:.2%}",
            f"{s['new_mean']:.2%}",
            f"{s['new_mean'] - s['stored_mean']:+.2%} ({low:+.2%}, {high:+.2%})",
            f"{s['stored_pooled']:.2%}",
            f"{s['new_pooled']:.2%}",
            f"{s['changed']:.1%}",
            f"{stored_rank[name] + 1} → {rank + 1}",
        )
    console.print(table)

    gaps = Table(title="Gap to the next service (mean WER, same samples)")
    for column in ("Services", "Stored gap", "New gap (95% CI)"):
        gaps.add_column(column, justify="left" if column == "Services" else "right")
    for a, b in zip(new_order, new_order[1:], strict=False):
        g = gap_stats(by_service[a], by_service[b])
        low, high = g["new_ci"]
        gaps.add_row(
            f"{a} → {b}", f"{g['stored']:+.2%}", f"{g['new']:+.2%} ({low:+.2%}, {high:+.2%})"
        )
    console.print(gaps)


def compare_databases(
    baseline_path: Path, db_path: Path, services: list[str], review: int, output_dir: Path
) -> None:
    """Compare the scores stored in a baseline database copy against the current ones.

    Transcriptions that differ between the two databases (re-collected since
    the copy was made) are left out, since their scores aren't comparable.
    """
    current = {pair_key(p): p for p in load_pairs(db_path, services)}
    judged = []
    for p in load_pairs(baseline_path, services):
        c = current.get(pair_key(p))
        if (
            c is None
            or c["transcription"] != p["transcription"]
            or None in (p["stored_wer"], c["stored_wer"])
            or float("inf") in (p["stored_wer"], c["stored_wer"])
        ):
            continue
        new = {
            "wer": c["stored_wer"],
            "error_count": c["stored_error_count"],
            "reference_words": c["stored_reference_words"],
            "errors": c["stored_errors"],
        }
        judged.append(dict(p, new=new))
    by_service = {s: [p for p in judged if p["service_name"] == s] for s in services}
    missing = [s for s, pairs in by_service.items() if not pairs]
    if missing:
        raise SystemExit(f"No comparable scores for: {', '.join(missing)}")

    console.print(f"Comparing {len(judged)} transcriptions across {len(services)} services\n")
    print_report(by_service, f"{baseline_path.name} vs current scores")
    output_dir.mkdir(parents=True, exist_ok=True)
    write_disagreements(judged, review, output_dir / "disagreements.md")
    console.print(f"\nDisagreements written to {output_dir}")


async def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--services", required=True, help="Comma-separated services to re-judge")
    parser.add_argument(
        "--judge", default="claude-sonnet-5-5:medium", help="New judge as model:effort"
    )
    parser.add_argument("--review", type=int, default=40, help="Disagreements to write out")
    parser.add_argument(
        "--limit", type=int, help="Judge only the first N transcriptions per service (trial runs)"
    )
    parser.add_argument(
        "--baseline",
        type=Path,
        help="Compare against scores stored in this database copy instead of judging",
    )
    parser.add_argument("--test", action="store_true", help="Use test_results.db")
    parser.add_argument("--output-dir", type=Path, help="Where to write runs and the report")
    args = parser.parse_args()

    config = get_config()
    db_path = config.data_dir / "test_results.db" if args.test else config.results_db
    model, effort = parse_judge(args.judge)
    services = [s.strip() for s in args.services.split(",")]
    name = (
        f"impact_vs_{args.baseline.stem}"
        if args.baseline
        else f"impact_{args.judge.replace(':', '_')}"
    )
    output_dir = args.output_dir or config.data_dir / "judge_experiments" / name

    if args.baseline:
        compare_databases(args.baseline, db_path, services, args.review, output_dir)
        return

    pairs = [
        p
        for p in load_pairs(db_path, services)
        if p["stored_wer"] is not None and p["stored_wer"] != float("inf")
    ]
    if args.limit:
        pairs = [
            p for s in services for p in [q for q in pairs if q["service_name"] == s][: args.limit]
        ]
    missing = set(services) - {p["service_name"] for p in pairs}
    if missing:
        raise SystemExit(f"No stored scores for: {', '.join(sorted(missing))}")
    console.print(f"Judging {len(pairs)} transcriptions across {len(services)} services\n")

    runs, evaluator, new_judgments = await run_judgments(
        model, effort, pairs, 1, output_dir / "runs.jsonl", db_path
    )
    failed = [p for p in pairs if runs[pair_key(p)][0] is None]
    judged = [dict(p, new=runs[pair_key(p)][0]) for p in pairs if runs[pair_key(p)][0]]
    by_service = {s: [p for p in judged if p["service_name"] == s] for s in services}

    console.print()
    if failed:
        console.print(f"[yellow]{len(failed)} transcriptions failed and are excluded[/yellow]")
    if new_judgments:
        console.print(f"Cost of this run: ${judgment_cost(evaluator):.2f}\n")
    print_report(by_service, f"Stored judge vs {evaluator.judge.split(' repeats=')[0]}")

    write_disagreements(judged, args.review, output_dir / "disagreements.md")
    (output_dir / "failed.json").write_text(json.dumps([pair_key(p) for p in failed], indent=2))
    console.print(f"\nRuns, failures, and disagreements written to {output_dir}")


if __name__ == "__main__":
    asyncio.run(main())
