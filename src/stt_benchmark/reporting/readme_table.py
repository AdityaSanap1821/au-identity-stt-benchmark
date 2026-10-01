"""Helpers for the README "Results Summary" table.

The table between the ``RESULTS_TABLE`` marker comments in ``README.md`` is the
single source of truth for the published benchmark numbers. This module is the
one place that knows that table's column layout — it formats rows, parses the
table back into data, and upserts individual rows.

It is pure string/registry logic with **no database dependency** (the registry
import is lazy), so both the plot script — which *reads* the table to render the
Pareto charts — and the ``stt-benchmark update-readme`` command — which *writes*
a contributor's row into it — can share exactly one definition of the format.
"""

from __future__ import annotations

import re
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from pathlib import Path

    from stt_benchmark.services import ServiceDefinition

README_TABLE_START = "<!-- RESULTS_TABLE:START -->"
README_TABLE_END = "<!-- RESULTS_TABLE:END -->"

# Header + separator lines. format_row() emits data rows beneath these.
TABLE_HEADER = (
    "| Vendor | Model | Transcripts | Perfect | WER Mean | Pooled WER "
    "| TTFS Median | TTFS P95 | TTFS P99 |",
    "|--------|-------|-------------|---------|----------|------------"
    "|-------------|----------|----------|",
)

# Metric dict keys carried for each service, in display units (ms / %).
METRIC_KEYS = (
    "ttfb_median",
    "ttfb_p95",
    "ttfb_p99",
    "pooled_wer",
    "success_rate",
    "perfect_pct",
    "wer_mean",
)
# Optional key: whether the row was scored by the previous WER judge.
OPTIONAL_KEYS = ("previous_judge",)

# Marks WER cells scored by the previous judge; explained under the table.
PREVIOUS_JUDGE_MARK = "†"

# "0.80%", optionally followed by the previous-judge mark.
_WER_CELL = re.compile(rf"^([\d.]+)%\s*({PREVIOUS_JUDGE_MARK})?$")


def _format_wer(value: float, previous_judge: bool) -> str:
    return f"{value:.2f}%{PREVIOUS_JUDGE_MARK if previous_judge else ''}"


def _parse_wer(cell: str) -> tuple[float, bool]:
    match = _WER_CELL.match(cell)
    if not match:
        raise ValueError(f"not a WER cell: {cell!r}")
    value, mark = match.groups()
    return float(value), bool(mark)


def format_row(vendor: str, model_label: str, m: dict) -> str:
    """Render one ``| ... |`` data row.

    ``m`` holds the METRIC_KEYS in ms / %, and optionally the OPTIONAL_KEYS.
    """
    previous_judge = m.get("previous_judge", False)
    return (
        f"| {vendor} | {model_label} "
        f"| {m['success_rate']:.1f}% | {m['perfect_pct']:.1f}% "
        f"| {_format_wer(m['wer_mean'], previous_judge)} "
        f"| {_format_wer(m['pooled_wer'], previous_judge)} "
        f"| {m['ttfb_median']:.0f}ms | {m['ttfb_p95']:.0f}ms | {m['ttfb_p99']:.0f}ms |"
    )


def sort_key(vendor: str, is_current: bool, model_label: str) -> tuple:
    """Table ordering: by vendor, current model first, then model label."""
    return (vendor.lower(), not is_current, model_label.lower())


def build_table(rows: list[tuple[ServiceDefinition, dict]]) -> str:
    """Build the full table markdown from ``(definition, metrics)`` rows.

    Vendor/Model come from the registry definition (so they never drift from the
    services config); metrics come from the caller. Rows are grouped by vendor,
    current model first.
    """
    ordered = sorted(rows, key=lambda r: sort_key(r[0].vendor, r[0].is_current, r[0].model_label))
    lines = [*TABLE_HEADER]
    for definition, m in ordered:
        lines.append(format_row(definition.vendor, definition.model_label, m))
    return "\n".join(lines)


def parse_table_rows(content: str) -> list[dict]:
    """Parse the README results table into per-row dicts.

    Reads the block between the ``RESULTS_TABLE`` markers when present, otherwise
    scans the whole text for pipe-delimited rows. The header row, the
    ``|---|---|`` separator, and any row whose metric cells aren't numeric are
    skipped. Each returned dict has ``vendor``, ``model``, the METRIC_KEYS
    (values in ms / %), and the OPTIONAL_KEYS (``False`` when absent).
    """
    if README_TABLE_START in content and README_TABLE_END in content:
        _, _, rest = content.partition(README_TABLE_START)
        block, _, _ = rest.partition(README_TABLE_END)
    else:
        block = content

    def _num(cell: str) -> float:
        return float(cell.replace("%", "").replace("ms", "").strip())

    rows: list[dict] = []
    for line in block.splitlines():
        line = line.strip()
        if not line.startswith("|"):
            continue
        cells = [c.strip() for c in line.strip("|").split("|")]
        if len(cells) < 9:
            continue
        vendor, model = cells[0], cells[1]
        # Skip the header row and the |---|---| separator row.
        if vendor.lower() == "vendor" or set(vendor) <= set("-: "):
            continue
        try:
            wer_mean, mean_mark = _parse_wer(cells[4])
            pooled_wer, pooled_mark = _parse_wer(cells[5])
            rows.append(
                {
                    "vendor": vendor,
                    "model": model,
                    "success_rate": _num(cells[2]),
                    "perfect_pct": _num(cells[3]),
                    "wer_mean": wer_mean,
                    "pooled_wer": pooled_wer,
                    "ttfb_median": _num(cells[6]),
                    "ttfb_p95": _num(cells[7]),
                    "ttfb_p99": _num(cells[8]),
                    "previous_judge": mean_mark or pooled_mark,
                }
            )
        except ValueError:
            # A row whose metric cells aren't numeric isn't a data row; skip it.
            continue
    return rows


def update_readme_table(table_md: str, readme_path: Path) -> bool:
    """Replace the results table in README.md between the marker comments.

    Returns True if the README was updated, False if the file or markers are
    missing.
    """
    if not readme_path.exists():
        return False
    content = readme_path.read_text()
    if README_TABLE_START not in content or README_TABLE_END not in content:
        return False

    before, _, rest = content.partition(README_TABLE_START)
    _, _, after = rest.partition(README_TABLE_END)
    new_content = f"{before}{README_TABLE_START}\n{table_md}\n{README_TABLE_END}{after}"
    readme_path.write_text(new_content)
    return True


def upsert_readme_rows(
    readme_path: Path, new_rows: list[tuple[ServiceDefinition, dict]]
) -> list[str]:
    """Insert or replace individual rows in the README results table.

    Each entry in ``new_rows`` is ``(definition, metrics)``. A new row is matched
    to an existing one by ``(vendor, model_label)`` (case-insensitive); matches
    are replaced in place and brand-new entries are appended. Every other row is
    left exactly as it was — this is a *targeted* update, never a full rebuild,
    so rows contributed by others (or hand-added rows with no registry entry) are
    preserved. The merged set is re-sorted with the same key as ``build_table``.

    Returns the ``"Vendor — Model"`` labels that were written.
    """
    if not readme_path.exists():
        raise FileNotFoundError(readme_path)
    content = readme_path.read_text()
    if README_TABLE_START not in content or README_TABLE_END not in content:
        raise ValueError(f"RESULTS_TABLE markers not found in {readme_path}; cannot upsert rows.")

    # Existing rows, keyed by (vendor, model) for replace-or-append.
    by_key = {(r["vendor"].lower(), r["model"].lower()): r for r in parse_table_rows(content)}

    written: list[str] = []
    for definition, m in new_rows:
        row = {"vendor": definition.vendor, "model": definition.model_label}
        row.update({k: m[k] for k in METRIC_KEYS})
        row.update({k: m.get(k) for k in OPTIONAL_KEYS})
        by_key[(definition.vendor.lower(), definition.model_label.lower())] = row
        written.append(f"{definition.vendor} — {definition.model_label}")

    # Re-sort the merged set. Look up is_current from the registry where a row
    # maps to a known service; default True for foreign/manually-added rows.
    from stt_benchmark.services import STT_SERVICES

    is_current = {
        (d.vendor.lower(), d.model_label.lower()): d.is_current for d in STT_SERVICES.values()
    }
    merged = sorted(
        by_key.values(),
        key=lambda r: sort_key(
            r["vendor"], is_current.get((r["vendor"].lower(), r["model"].lower()), True), r["model"]
        ),
    )

    body = "\n".join([*TABLE_HEADER, *(format_row(r["vendor"], r["model"], r) for r in merged)])
    update_readme_table(body, readme_path)
    return written
