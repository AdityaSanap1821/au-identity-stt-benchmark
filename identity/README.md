# Australian identity details over the phone: a small STT experiment

Two days of experiments on one question: how well do streaming speech-to-text services capture the details
a voice agent uses to check who it is talking to (names, spelled names, emails, dates of birth, Australian
addresses) on phone-quality audio? And what happens when you give the speech model the customer's record as hints?

It extends Pipecat's [stt-benchmark](https://github.com/pipecat-ai/stt-benchmark) (this repo is a fork): same
streaming harness and vendor services, plus an identity test set and field-level scoring. Results are in
[RESULTS.md](RESULTS.md).

## What's here

| File | What it does |
|---|---|
| `build.py` | Builds the test set: 150 scripted items + 70 full-flow clips (each caller's spelled surname and date of birth), all invented identities. Voices from `edge-tts` (12 English accents), then an 8 kHz mu-law phone line, half with added noise |
| `run.py` | Streams every clip through one vendor and one hint condition (`none`, `generic`, `record`) using the benchmark's `BenchmarkRunner`. Resumable; dropped sessions are retried |
| `textnorm.py` | Field checks: "as written" vs "recoverable after normalising" (number words, `dot`/`at`, spelled letters, Australian DD/MM dates, unit/street numbers) |
| `score.py` | Field accuracy per vendor and condition, impostor leakage, real-vs-synthetic comparison |
| `matcher.py` | Verification done in code on the transcripts: fuzzy name matching (Jaro-Winkler, Soundex), spelling fallback, date of birth, and a full-flow decision per caller, against genuine and wrong-party callers |
| `real_voice.py` | Recording script for real-voice clips, and ingest: splits long recordings at pauses, aligns lines to the script, applies the same phone line |
| `RECORDING_SCRIPT.md` | The 125 lines read for the real-voice set |

The only change outside this folder is an optional `stt_factory` argument on `BenchmarkRunner.benchmark_sample`,
so each clip can carry its own keyword hints.

## Running it

```bash
uv sync
uv run python identity/build.py
uv run python identity/run.py --vendor deepgram --condition none
uv run python identity/score.py
uv run python identity/matcher.py
```

Real-voice set (optional): record the lines in `RECORDING_SCRIPT.md` into `identity_data/real/`, then

```bash
uv run python identity/real_voice.py ingest
uv run python identity/real_voice.py check
uv run python identity/run.py --vendor deepgram --condition none --set real
uv run python identity/matcher.py --real
```

`check` re-transcribes each split clip locally and flags any that start with leftover audio from the previous line.

Keys go in `.env` (see `env.example`). `--gpu` runs local Whisper on CUDA (needs `nvidia-cublas-cu12` and
`nvidia-cudnn-cu12` in the environment). Generated audio, transcripts and the real-voice recordings stay in
`identity_data/`, which is not committed.

## Limits

- **Synthetic voices flatter the vendors.** The real-voice set is one speaker reading names unfamiliar to them,
  so its name and address numbers are pessimistic for customers saying their own names.
- **Small samples:** 23 genuine and 12 wrong-party callers, 25–35 clips per field. Differences under about
  15–20 points may be noise.
- **Each clip is transcribed on its own.** A live agent has the question as context and can tune settings per
  stage (turn hold for spelling, a stage prompt); none of that was tested.
- **Simulated phone line,** not real carrier audio. Default vendor settings, one run each.
- The real-voice long recordings were split automatically; two date lines where the reader said a different
  date from the script are scored as said.
