# Scripts

## Pareto Frontier Plot

`pareto-frontier-plot.py` generates the README's latency/accuracy charts: scatter plots of TTFS latency vs Semantic WER with a Pareto frontier overlay, and a per-service latency distribution (median/P95/P99) chart.

### Usage

```bash
# Plot all services using defaults
python scripts/pareto-frontier-plot.py

# Plot specific services
python scripts/pareto-frontier-plot.py -s deepgram assemblyai soniox

# Use a config file
python scripts/pareto-frontier-plot.py -c scripts/plot-config.json

# Config file with CLI overrides
python scripts/pareto-frontier-plot.py -c scripts/plot-config.json --latency p95
```

### CLI Options

| Flag | Description | Default |
|------|-------------|---------|
| `-o`, `--output` | Output file path or directory | `assets/` |
| `--charts` | Chart types to generate (space-separated): `pareto`, `range` | `pareto range` |
| `-l`, `--latency` | Latency metrics for the Pareto charts (space-separated): `median`, `p95`, `p99` | `median` |
| `-s`, `--services` | Services to include (space-separated) | all available |
| `-c`, `--config` | Path to a JSON config file | none |
| `--show` | Display the plot interactively | off |

CLI arguments always take precedence over config file values.

### Config File

A JSON file that stores plot settings for repeatable generation. See `plot-config.json` for a working example.

```json
{
  "services": ["deepgram", "assemblyai", "soniox"],
  "display_names": {
    "deepgram": "Deepgram",
    "assemblyai": "AssemblyAI",
    "soniox": "Soniox"
  },
  "latency": ["median", "p95"],
  "output": "assets/",
  "show": false
}
```

| Key | Type | Description |
|-----|------|-------------|
| `services` | list of strings | Which services to include in the plot |
| `display_names` | dict | Maps service keys to display labels on the plot |
| `charts` | list of strings | Chart types to generate: `["pareto", "range"]` |
| `latency` | string or list | Latency metrics for the Pareto charts: `"p95"` or `["median", "p95"]` |
| `output` | string | Output file path or directory |
| `show` | boolean | Display the plot interactively |
| `label_offsets` | dict | Per-metric hand-placed label positions for dense regions, e.g. `{"median": {"Deepgram": [8, 2, "left"]}}` — `[dx, dy, alignment]` with offsets in points from the dot. Overrides the script's built-in defaults for that metric; labels not listed are placed automatically. |

All keys are optional. Omitted keys fall back to defaults.

## Judge Consistency

`judge_consistency.py` compares semantic WER judge settings before a full re-score. It picks a random set of transcriptions, judges each one several times per effort level, and reports failed runs, how often repeated runs agree, the difference from the stored scores, and the projected cost of a full re-score. It writes the raw runs to a JSON file and leaves the results database untouched.

```bash
# 100 transcriptions, 3 runs each, at medium and high effort
uv run python scripts/judge_consistency.py

uv run python scripts/judge_consistency.py --pairs 50 --repeats 5 --efforts low medium
```

| Flag | Description | Default |
|------|-------------|---------|
| `--pairs` | Transcriptions to judge | `100` |
| `--repeats` | Runs per transcription | `3` |
| `--efforts` | Effort levels to compare (space-separated) | `medium high` |
| `--seed` | Seed for picking transcriptions | `0` |
| `--test` | Use `test_results.db` | off |
| `--output` | Path for the raw runs JSON | `stt_benchmark_data/judge_consistency_<timestamp>.json` |
