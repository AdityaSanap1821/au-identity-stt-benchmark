"""Stream every identity clip through one vendor under one hint condition.

    uv run python identity/run.py --vendor deepgram --condition none [--limit 5]

Vendors: assemblyai, deepgram, speechmatics, gemini, whisper
Conditions: none | generic (domain vocabulary) | record (the customer's details on file)

Results append to identity_data/results/<vendor>__<condition>.jsonl; reruns skip finished ids,
so a crash or Ctrl-C loses nothing. Run vendors in separate terminals to go in parallel.
"""

import argparse
import asyncio
import json
import os
import sys
from pathlib import Path

from loguru import logger
from pipecat.transcriptions.language import Language

sys.path.insert(0, str(Path(__file__).parent))
from build import AUDIO, DATA, hints, load_items  # noqa: E402

from stt_benchmark.models import AudioSample, ServiceName  # noqa: E402
from stt_benchmark.pipeline.benchmark_runner import BenchmarkRunner  # noqa: E402
from stt_benchmark.services import _get_env  # noqa: E402


def assemblyai(terms: list[str]):
    from pipecat.services.assemblyai.stt import AssemblyAISTTService

    # same settings as the repo's universal-3-6-pro entry, plus keyterms
    return AssemblyAISTTService(
        api_key=_get_env("ASSEMBLYAI_API_KEY"),
        settings=AssemblyAISTTService.Settings(
            model="universal-3-6-pro", min_turn_silence=50, vad_threshold=0.2,
            language_codes=[Language.EN], keyterms_prompt=terms or None,
        ),
        vad_force_turn_endpoint=True,
    )


def deepgram(terms: list[str]):
    from pipecat.services.deepgram.stt import DeepgramSTTService

    # Sydney endpoint: the onshore option an Australian regulated deployment would use
    return DeepgramSTTService(
        api_key=_get_env("DEEPGRAM_API_KEY"),
        base_url="api.au.deepgram.com",
        settings=DeepgramSTTService.Settings(
            model="nova-3-general", language=Language.EN, smart_format=True,
            profanity_filter=False, keyterm=terms or None,
        ),
    )


def speechmatics(terms: list[str]):
    from pipecat.services.speechmatics.stt import SpeechmaticsSTTService, TurnDetectionMode

    # same as the repo's speechmatics_agent_stt entry (linden-1), plus custom vocabulary
    return SpeechmaticsSTTService(
        api_key=_get_env("SPEECHMATICS_API_KEY"),
        base_url=os.getenv("SPEECHMATICS_RT_URL", "wss://eu2.rt.speechmatics.com/v2/agent"),
        settings=SpeechmaticsSTTService.Settings(
            language=Language.EN, turn_detection_mode=TurnDetectionMode.EXTERNAL,
            additional_vocab=[SpeechmaticsSTTService.AdditionalVocabEntry(content=t) for t in terms],
        ),
    )


def gemini(terms: list[str]):
    from pipecat.services.google.gemini_live.stt import GeminiSTTService

    # same as the repo's google_gemini_3_5_transcribe_live entry, plus phrase hints
    return GeminiSTTService(
        api_key=_get_env("GOOGLE_API_KEY"),
        settings=GeminiSTTService.Settings(
            model="gemini-3.5-transcribe-live", languages=[Language.EN_US],
            adaptation_phrases=terms or None,
        ),
    )


_whisper_cache: dict = {}


def whisper(terms: list[str]):
    from pipecat.services.whisper.stt import Model, WhisperSTTService

    class CachedWhisper(WhisperSTTService):
        # the runner builds a service per clip; load the 1.6 GB model once
        def _load(self):
            key = (self._settings.model, self._device)
            if key not in _whisper_cache:
                super()._load()
                _whisper_cache[key] = self._model
            self._model = _whisper_cache[key]

    # plain folder instead of the HF cache: the cache's symlinks need admin rights on Windows
    local = DATA / "models" / "faster-whisper-large-v3-turbo"
    if not (local / "model.bin").exists():
        from huggingface_hub import snapshot_download

        snapshot_download(Model.LARGE_V3_TURBO.value, local_dir=local)

    return CachedWhisper(
        device=WHISPER_DEVICE,
        compute_type="float16" if WHISPER_DEVICE == "cuda" else "int8",
        settings=WhisperSTTService.Settings(
            model=str(local), language=Language.EN, hotwords=" ".join(terms) or None,
        ),
    )


VENDORS = {
    "assemblyai": (ServiceName.ASSEMBLYAI_UNIVERSAL_3_6_PRO, assemblyai),
    "deepgram": (ServiceName.DEEPGRAM, deepgram),
    "speechmatics": (ServiceName.SPEECHMATICS_AGENT_STT, speechmatics),
    "gemini": (ServiceName.GOOGLE_GEMINI_3_5_TRANSCRIBE_LIVE, gemini),
    "whisper": (ServiceName.WHISPER, whisper),
}
WHISPER_DEVICE = "cpu"  # GPU needs CUDA 12 cuBLAS/cuDNN installed; use --gpu if you have them


async def main(vendor: str, condition: str, limit: int | None, ids: set[str] | None, which: str) -> None:
    service_name, factory = VENDORS[vendor]
    out = DATA / "results" / f"{vendor}__{condition}.jsonl"
    out.parent.mkdir(exist_ok=True)
    # a clip counts as done only if it produced a transcript; connection drops get retried on rerun
    rows = [json.loads(line) for line in out.open(encoding="utf-8")] if out.exists() else []
    done = {r["id"] for r in rows if r["transcript"] is not None and not r["error"]}

    items = [it for it in load_items().values() if which in ("all", it["set"])]
    items = [it for it in items if it["id"] not in done and (ids is None or it["id"] in ids)]
    items = items[:limit] if limit else items
    print(f"{vendor}/{condition}: {len(items)} to run ({len(done)} already done)")

    runner = BenchmarkRunner()
    for n, it in enumerate(items, 1):
        pcm = AUDIO / f"{it['id']}.pcm"
        sample = AudioSample(sample_id=it["id"], audio_path=str(pcm),
                             duration_seconds=pcm.stat().st_size / 32000, dataset_index=n)
        terms = hints(it, condition)
        r = await runner.benchmark_sample(sample, service_name, model=f"{vendor}:{condition}",
                                          stt_factory=lambda t=terms: factory(t))
        row = {"id": it["id"], "vendor": vendor, "condition": condition,
               "transcript": r.transcription, "ttfs": r.ttfb_seconds, "error": r.error}
        with out.open("a", encoding="utf-8") as f:
            f.write(json.dumps(row) + "\n")
        status = "ERR " + r.error[:60] if r.error else (r.transcription or "")[:70]
        print(f"[{n}/{len(items)}] {it['id']}: {status}")
        await asyncio.sleep(0.1)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--vendor", required=True, choices=VENDORS)
    ap.add_argument("--condition", default="none", choices=["none", "generic", "record"])
    ap.add_argument("--limit", type=int)
    ap.add_argument("--ids", help="comma-separated item ids")
    ap.add_argument("--set", default="all", choices=["all", "synthetic", "real"], dest="which")
    ap.add_argument("--gpu", action="store_true", help="run Whisper on CUDA")
    ap.add_argument("-v", "--verbose", action="store_true")
    a = ap.parse_args()
    sys.stdout.reconfigure(errors="replace")  # vendors sometimes return non-Latin script; don't crash printing it
    if not a.verbose:
        logger.remove()
        logger.add(sys.stderr, level="WARNING")
    if a.gpu:
        # CUDA libs from `uv pip install nvidia-cublas-cu12 "nvidia-cudnn-cu12==9.*"` (not in pyproject)
        import nvidia

        for lib in ("cublas", "cudnn"):
            bin_dir = Path(nvidia.__path__[0]) / lib / "bin"
            os.add_dll_directory(str(bin_dir))
            os.environ["PATH"] = f"{bin_dir}{os.pathsep}{os.environ['PATH']}"
        WHISPER_DEVICE = "cuda"
    asyncio.run(main(a.vendor, a.condition, a.limit, set(a.ids.split(",")) if a.ids else None, a.which))
