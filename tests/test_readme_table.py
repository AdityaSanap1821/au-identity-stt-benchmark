"""Tests for the README results-table helpers (single source of truth)."""

from types import SimpleNamespace

from stt_benchmark.reporting.readme_table import (
    README_TABLE_END,
    README_TABLE_START,
    format_row,
    parse_table_rows,
    upsert_readme_rows,
)


def _metrics(**overrides):
    base = {
        "success_rate": 99.8,
        "perfect_pct": 76.5,
        "wer_mean": 1.71,
        "pooled_wer": 1.62,
        "ttfb_median": 247.0,
        "ttfb_p95": 298.0,
        "ttfb_p99": 326.0,
    }
    base.update(overrides)
    return base


def _readme(rows: str) -> str:
    return (
        "# Title\n\nSome intro.\n\n"
        f"{README_TABLE_START}\n"
        "| Vendor | Model | Transcripts | Perfect | WER Mean | Pooled WER "
        "| TTFS Median | TTFS P95 | TTFS P99 |\n"
        "|--------|-------|-------------|---------|----------|------------"
        "|-------------|----------|----------|\n"
        f"{rows}"
        f"{README_TABLE_END}\n\n## After\n"
    )


def test_format_row_units_and_precision():
    row = format_row("Deepgram", "nova-3-general", _metrics())
    assert row == (
        "| Deepgram | nova-3-general | 99.8% | 76.5% | 1.71% | 1.62% | 247ms | 298ms | 326ms |"
    )


def test_parse_table_rows_skips_header_and_separator():
    content = _readme(
        "| Deepgram | nova-3-general | 99.8% | 76.5% | 1.71% | 1.62% | 247ms | 298ms | 326ms |\n"
        "| AWS | N/A | 100.0% | 77.4% | 1.68% | 1.75% | 1136ms | 1527ms | 1897ms |\n"
    )
    rows = parse_table_rows(content)
    assert [r["vendor"] for r in rows] == ["Deepgram", "AWS"]
    assert rows[0]["ttfb_p95"] == 298.0
    assert rows[1]["model"] == "N/A"


def test_parse_format_round_trips():
    line = "| Deepgram | nova-3-general | 99.8% | 76.5% | 1.71% | 1.62% | 247ms | 298ms | 326ms |"
    [parsed] = parse_table_rows(_readme(line + "\n"))
    assert format_row(parsed["vendor"], parsed["model"], parsed) == line


def test_upsert_inserts_in_sorted_position(tmp_path):
    readme = tmp_path / "README.md"
    readme.write_text(
        _readme(
            "| AWS | N/A | 100.0% | 77.4% | 1.68% | 1.75% | 1136ms | 1527ms | 1897ms |\n"
            "| Soniox | stt-rt-v4 | 99.8% | 84.1% | 1.25% | 1.29% | 249ms | 281ms | 310ms |\n"
        )
    )
    defn = SimpleNamespace(vendor="Deepgram", model_label="nova-3-general")
    written = upsert_readme_rows(readme, [(defn, _metrics())])

    assert written == ["Deepgram — nova-3-general"]
    vendors = [r["vendor"] for r in parse_table_rows(readme.read_text())]
    # Deepgram sorts between AWS and Soniox.
    assert vendors == ["AWS", "Deepgram", "Soniox"]


def test_upsert_replaces_existing_row(tmp_path):
    readme = tmp_path / "README.md"
    readme.write_text(
        _readme(
            "| Deepgram | nova-3-general | 99.8% | 76.5% | 1.71% | 1.62% | 247ms | 298ms | 326ms |\n"
        )
    )
    defn = SimpleNamespace(vendor="Deepgram", model_label="nova-3-general")
    upsert_readme_rows(readme, [(defn, _metrics(ttfb_median=999.0, pooled_wer=2.22))])

    rows = parse_table_rows(readme.read_text())
    assert len(rows) == 1
    assert rows[0]["ttfb_median"] == 999.0
    assert rows[0]["pooled_wer"] == 2.22


def test_upsert_preserves_foreign_rows(tmp_path):
    readme = tmp_path / "README.md"
    readme.write_text(
        _readme(
            "| ContributorCo | special-model | 95.0% | 50.0% | 5.00% | 6.00% "
            "| 500ms | 600ms | 700ms |\n"
        )
    )
    defn = SimpleNamespace(vendor="Deepgram", model_label="nova-3-general")
    upsert_readme_rows(readme, [(defn, _metrics())])

    vendors = {r["vendor"] for r in parse_table_rows(readme.read_text())}
    assert "ContributorCo" in vendors  # foreign row not dropped
    assert "Deepgram" in vendors


def test_upsert_is_idempotent(tmp_path):
    readme = tmp_path / "README.md"
    readme.write_text(
        _readme("| AWS | N/A | 100.0% | 77.4% | 1.68% | 1.75% | 1136ms | 1527ms | 1897ms |\n")
    )
    defn = SimpleNamespace(vendor="Deepgram", model_label="nova-3-general")
    upsert_readme_rows(readme, [(defn, _metrics())])
    once = readme.read_text()
    upsert_readme_rows(readme, [(defn, _metrics())])
    assert readme.read_text() == once


def test_previous_judge_mark_round_trips():
    line = (
        "| NVIDIA | Nemotron 3.0 ASR (en) | 100.0% | 76.1% | 1.90%† | 1.95%† "
        "| 221ms | 238ms | 252ms |"
    )
    [parsed] = parse_table_rows(_readme(line + "\n"))
    assert parsed["previous_judge"] is True
    assert parsed["wer_mean"] == 1.90
    assert format_row(parsed["vendor"], parsed["model"], parsed) == line


def test_upsert_replaces_previous_judge_row(tmp_path):
    readme = tmp_path / "README.md"
    readme.write_text(
        _readme(
            "| Deepgram | nova-3-general | 99.8% | 76.5% | 1.71%† | 1.62%† | 247ms | 298ms | 326ms |\n"
        )
    )
    defn = SimpleNamespace(vendor="Deepgram", model_label="nova-3-general")
    upsert_readme_rows(readme, [(defn, _metrics(wer_mean=1.32, pooled_wer=1.37))])

    [row] = parse_table_rows(readme.read_text())
    assert row["previous_judge"] is False
    assert row["wer_mean"] == 1.32
