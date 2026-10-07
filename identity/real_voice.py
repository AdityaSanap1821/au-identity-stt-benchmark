"""Real-voice recordings: generate the reading script, later ingest the recordings.

    uv run python identity/real_voice.py script   # writes identity/RECORDING_SCRIPT.md
    uv run python identity/real_voice.py ingest   # identity_data/real/* -> phone-quality clips + items_real.jsonl

Every line is the exact script of an existing synthetic clip, so real and synthetic pair up by id.
"""

import json
import re
import subprocess
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
from build import DATA  # noqa: E402

SCRIPT = Path(__file__).parent / "RECORDING_SCRIPT.md"

# Rough guide for names a reader might not know. Real callers say their own names correctly.
SAY = {
    "Linh": "LING", "Wei": "WAY", "Priya": "PREE-ya", "Siobhan": "shi-VAWN", "Dimitrios": "dee-MEE-tree-os",
    "Giuseppe": "joo-ZEP-peh", "Krzysztof": "KSHISH-tof", "Jarrah": "JA-rah", "Ngozi": "en-GO-zee",
    "Thanh": "TAHN", "Ioannis": "yo-AH-nis", "Jiwoo": "JEE-woo", "Hoang": "HWANG", "Tomasz": "TOH-mash",
    "Eilidh": "AY-lee", "Kiri": "KIH-ree",
    "Nguyen": "WIN (or NGWEN)", "Zhang": "JAHNG", "Raghunathan": "RAH-goo-NAH-thun", "O'Sullivan": "oh-SULL-ih-vun",
    "Papadopoulos": "pa-pa-DOP-oo-los", "Esposito": "es-po-ZEE-to", "Wojciechowski": "voy-cheh-HOF-skee",
    "Tjungurrayi": "JOONG-gu-rai", "Okonkwo": "oh-KONG-kwo", "McGrath": "muh-GRAH", "Smyth": "SMITH",
    "Pham": "FAHM", "Venkataraman": "VEN-kuh-tuh-RAH-mun", "Karagiannis": "ka-ra-YAN-is", "Haddad": "ha-DAD",
    "Le": "LAY", "Nowak": "NOH-vak", "Macleod": "muh-KLOWD", "Ngata": "NAH-tah", "Al-Hassan": "al-HAS-sun",
    "Nugent": "NEW-jent", "McGraw": "muh-GRAW", "Fan": "FAN", "Chang": "CHANG", "Ragunath": "RAH-goo-nath",
    "Okonjo": "oh-KON-jo", "Espinoza": "es-pi-NO-za", "Hadid": "ha-DEED", "Novak": "NOH-vak",
    "McLeod": "muh-KLOWD", "Chew": "CHOO", "Chua": "CHOO-ah", "Rahman": "RAH-mun", "Bianchi": "bee-AN-kee",
    "Park": "PARK", "Vu": "VOO",
    "Woolloomooloo": "wool-uh-muh-LOO", "Woolloongabba": "wool-un-GAB-uh", "Goondiwindi": "gun-duh-WIN-dee",
    "Indooroopilly": "IN-duh-roo-PILL-ee", "Wahroonga": "wuh-RUNG-guh", "Joondalup": "JOON-duh-lup",
    "Mooroolbark": "MOO-rul-bark", "Murrumbeena": "mur-um-BEE-nuh", "Toongabbie": "toon-GAB-ee",
    "Kalgoorlie": "kal-GOOR-lee", "Warrnambool": "WOR-num-bool", "Wangaratta": "wang-uh-RAT-uh",
    "Coolangatta": "koo-lun-GAT-uh", "Maroochydore": "muh-ROO-chee-dor", "Bungarribee": "bung-GA-ri-bee",
    "Coolabah": "KOO-luh-bah", "Kurrajong": "KUR-uh-jong", "Wollondilly": "wol-un-DIL-ee",
    "Yarrawonga": "yarra-WONG-uh", "Merindah": "muh-RIN-duh", "Warrigal": "WOR-i-gul", "Kirribilli": "kirra-BILL-ee",
    "Boronia": "buh-ROH-nee-uh", "Banksia": "BANK-see-uh", "Illawarra": "illa-WORRA", "Tuggeranong": "TUG-uh-ruh-nong",
    "Belconnen": "bel-KON-un", "Glenorchy": "glen-OR-kee", "Mandurah": "MAN-dyuh-ruh", "Kogarah": "KOG-uh-ruh",
    "Parramatta": "parra-MAT-uh", "Cabramatta": "kabra-MAT-uh", "Dandenong": "DAN-duh-nong",
}

# Emails: one of each spoken style, then fill. Addresses: the hardest place names.
EMAIL_IDS = ["email_000", "email_001", "email_002", "email_003", "email_004",
             "email_005", "email_006", "email_007", "email_008", "email_009"]
HARD_SUBURBS = ["Woolloomooloo", "Woolloongabba", "Goondiwindi", "Indooroopilly", "Wahroonga", "Joondalup",
                "Mooroolbark", "Toongabbie", "Kalgoorlie", "Warrnambool"]


def guide(script: str) -> str:
    hits = [f"{w} = {p}" for w, p in SAY.items() if re.search(rf"(?<![\w']){re.escape(w)}(?![\w'])", script)]
    return "; ".join(hits)


def write_script() -> None:
    items = {it["id"]: it for it in map(json.loads, (DATA / "items.jsonl").open(encoding="utf-8"))}
    callers = [it for it in items.values() if it["field"] == "name"]
    addresses = [it for it in items.values() if it["field"] == "address"
                 and any(s.lower() == it["expected"]["suburb"] for s in HARD_SUBURBS)]

    sections = [
        ("A", "Spelled surnames: SHORT FILES, one per line",
         "Record each line as its own file named by its code (A01, A02 ...). Spell at a natural pace, "
         "the way you would to a call-centre agent.",
         [items[f"flow_spelled_{c['id']}"] for c in callers]),
        ("E", "Emails: SHORT FILES, one per line",
         "Own file per line (E01, E02 ...). Read it the way it's written ('dot', 'at', 'all one word').",
         [items[i] for i in EMAIL_IDS]),
        ("N", "Names: ONE LONG RECORDING",
         "One file named N.m4a (any format is fine). Read every line in order. "
         "Leave about 3 seconds of silence between lines, and no long pause inside a line.",
         callers),
        ("D", "Dates of birth: ONE LONG RECORDING", "One file named D.m4a. Same rules: 3 seconds between lines.",
         [items[f"flow_dob_{c['id']}"] for c in callers]),
        ("R", "Addresses: ONE LONG RECORDING", "One file named R.m4a. Same rules: 3 seconds between lines.",
         addresses),
    ]

    lines = [
        "# Real-voice recording script",
        "",
        f"{sum(len(s[3]) for s in sections)} lines. Set aside about 40 minutes. All identities are made up.",
        "",
        "**How to record**",
        "- Any recorder works (phone voice memo, Windows Sound Recorder). Any format is fine.",
        "- Quiet room, phone or laptop at normal talking distance. Speak like you're on a call: natural pace, not slow and careful.",
        "- Your own accent is exactly what we want. Don't imitate anyone.",
        "- Use the pronunciation guide for names you don't know. Real callers know how to say their own name.",
        "- Mistake? For short files, just re-record. In long recordings, stop and re-record that whole section (simplest), or tell me the line you repeated.",
        "- Put every file in: `identity_data/real/`",
        "",
    ]
    manifest = {}
    for code, title, how, group in sections:
        lines += [f"## {code}. {title}", "", how, "", "| # | Read this | How to say tricky words |", "|---|---|---|"]
        for n, it in enumerate(group, 1):
            key = f"{code}{n:02d}"
            manifest[key] = it["id"]
            lines.append(f"| {key} | {it['script']} | {guide(it['script'])} |")
        lines.append("")
    SCRIPT.write_text("\n".join(lines), encoding="utf-8")
    (DATA / "real_manifest.json").write_text(json.dumps(manifest, indent=1), encoding="utf-8")
    print(f"{len(manifest)} lines -> {SCRIPT}")


REAL = DATA / "real"
SR = 16000


def decode(path: Path) -> np.ndarray:

    raw = subprocess.run(["ffmpeg", "-loglevel", "error", "-i", str(path), "-ac", "1", "-ar", str(SR),
                          "-f", "s16le", "-"], capture_output=True, check=True).stdout
    return np.frombuffer(raw, dtype=np.int16)


def speech_frames(x: np.ndarray, frame: int = 320) -> np.ndarray:
    """True per 20 ms frame where level is clearly above this recording's own noise floor."""

    n = len(x) // frame
    db = 20 * np.log10(np.sqrt((x[: n * frame].astype(np.float64).reshape(n, frame) ** 2).mean(1)) + 1e-9)
    floor = np.percentile(db, 15)
    return db > floor + 12


# a chunk whose text starts like this begins a new line (lenient: Whisper heard "my date of work",
# "my data of birth", "I am left at" in these recordings)
OPENERS = {"N": r"^my nam", "D": r"^my dat", "R": r"^i (live|leave|lived|am|liv)"}


def _canon(text: str) -> str:
    """Comparable form of a line: ordinals and number words -> digits ("thirteenth" and "13th" -> "13")."""
    from textnorm import ORD_WORD, numbers_to_digits, words

    # Whisper's stock sign-offs on short chunks ("Thank you.", "I'll see you next time.") aren't speech
    text = re.sub(r"(?i)\b(thank you|thanks for watching|i'?ll see you next time|bye|uh-huh)\b[.!]?", " ", text)
    toks = [ORD_WORD.get(t, t) for t in words(text, breaks=True) if t not in ("st", "nd", "rd", "th")]
    return " ".join(numbers_to_digits(toks))


def _align(got: list[str], exp: list[str]) -> list[int | None]:
    """Monotonic alignment of detected lines to expected lines by text similarity.

    Each expected line gets at most one detected line; leftovers (false starts, first takes) are
    dropped; a tiny bonus for later detections makes re-takes win ties.
    """
    from difflib import SequenceMatcher

    sim = [[SequenceMatcher(None, g, e).ratio() for e in exp] for g in got]
    n, m = len(got), len(exp)
    best = [[0.0] * (m + 1) for _ in range(n + 1)]
    move = [[""] * (m + 1) for _ in range(n + 1)]
    for i in range(n + 1):
        for j in range(m + 1):
            if i == 0 and j == 0:
                continue
            opts = []
            if i and j and sim[i - 1][j - 1] >= 0.5:
                opts.append((best[i - 1][j - 1] + sim[i - 1][j - 1] + 1e-4 * i, "match"))
            if i:
                opts.append((best[i - 1][j], "skip_det"))
            if j:
                opts.append((best[i][j - 1], "skip_exp"))
            best[i][j], move[i][j] = max(opts)
    assign: list[int | None] = [None] * m
    i, j = n, m
    while i or j:
        mv = move[i][j]
        if mv == "match":
            assign[j - 1] = i - 1
            i, j = i - 1, j - 1
        elif mv == "skip_det":
            i -= 1
        else:
            j -= 1
    return assign, sim


def split_long(path: Path, scripts: list[str]) -> list[tuple[int, int] | None]:
    """Cut a long recording into its scripted lines. Returns a sample range per line, or None if missing.

    1. Split the audio at every pause >= 0.3 s, measured from the sound itself (exact cut points).
    2. Transcribe each short chunk on its own with local Whisper (CPU; audio never leaves the machine).
    3. A chunk that starts with the line opener ("My name...", "My date...", "I live...") starts a
       new line; following chunks belong to it.
    4. Align those lines to the script in order (re-takes -> later take; false starts dropped; a line
       with no confident match is left out rather than mislabelled).
    Cuts land mid-pause, so no audio is shared between clips. Whisper's word timestamps (which
    drifted by up to a line on these recordings) aren't used at all.
    """
    from faster_whisper import WhisperModel

    x = decode(path)
    model = WhisperModel(str(DATA / "models" / "faster-whisper-large-v3-turbo"), device="cpu", compute_type="int8")
    sp = speech_frames(x)  # 20 ms frames
    chunks, start, silent, last = [], None, 0, 0
    for i, on in enumerate(sp):
        if on:
            start = i if start is None else start
            silent, last = 0, i
        elif start is not None:
            silent += 1
            if silent >= 15:  # 300 ms
                chunks.append((start, last + 1))
                start = None
    if start is not None:
        chunks.append((start, last + 1))
    chunks = [c for c in chunks if c[1] - c[0] >= 5]  # ignore clicks under 100 ms

    def heard(a: int, b: int) -> str:
        seg = x[max(0, (a - 5) * 320) : (b + 5) * 320].astype("float32") / 32768
        segs, _ = model.transcribe(seg, language="en", temperature=0.0, condition_on_previous_text=False)
        return " ".join(s.text.strip() for s in segs)

    texts = [heard(a, b) for a, b in chunks]
    opener = re.compile(OPENERS[path.stem])
    groups: list[list] = []  # [first chunk, last chunk, text]
    for k, t in enumerate(texts):
        if not groups or opener.match(" ".join(re.findall(r"[a-z0-9]+", t.lower()))):
            groups.append([k, k, t])
        else:
            groups[-1][1] = k
            groups[-1][2] += " " + t

    assign, sim = _align([_canon(g[2]) for g in groups], [_canon(t) for t in scripts])
    used = {k for k in assign if k is not None}
    for k, g in enumerate(groups):
        if k not in used:
            print(f"  dropped {chunks[g[0]][0] * 0.02:6.1f}s  {g[2][:70]}")
    for j, k in enumerate(assign):
        if k is None:
            print(f"  MISSING line {j + 1}: {scripts[j][:70]}")
        elif sim[k][j] < 0.8:
            print(f"  low-confidence line {j + 1} ({sim[k][j]:.2f}): heard '{groups[k][2][:50]}' for '{scripts[j][:50]}'")
    print(f"{path.name}: {sum(a is not None for a in assign)}/{len(scripts)} lines matched "
          f"({len(chunks)} chunks, {len(groups)} line starts)")

    def mid_before(c: int) -> int:  # middle of the pause before chunk c
        return 0 if c == 0 else ((chunks[c - 1][1] + chunks[c][0]) // 2) * 320

    def mid_after(c: int) -> int:  # middle of the pause after chunk c
        return len(x) if c == len(chunks) - 1 else ((chunks[c][1] + chunks[c + 1][0]) // 2) * 320

    return [None if k is None else (mid_before(groups[k][0]), mid_after(groups[k][1])) for k in assign]


def to_phone_pcm(samples: np.ndarray, out: Path, noise: bool) -> None:
    """Same phone line as build.py: 300 ms lead-in, optional pink noise, 8 kHz mu-law, back to 16 kHz."""

    sp = speech_frames(samples)
    on = np.flatnonzero(sp)
    if len(on):  # trim long leading/trailing silence from the short files, keep 200 ms
        samples = samples[max(0, (on[0] - 10) * 320) : (on[-1] + 10) * 320]
    graph = "[0:a]adelay=300,aresample=24000"
    if noise:
        graph += "[a];anoisesrc=color=pink:amplitude=0.015:sample_rate=24000[n];[a][n]amix=inputs=2:duration=first:normalize=0"
    mulaw = out.with_suffix(".wav")
    subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-f", "s16le", "-ar", str(SR), "-ac", "1", "-i", "-",
                    "-filter_complex", graph, "-ar", "8000", "-ac", "1", "-c:a", "pcm_mulaw", str(mulaw)],
                   input=samples.astype(np.int16).tobytes(), check=True)
    subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-i", str(mulaw), "-ar", str(SR), "-ac", "1",
                    "-f", "s16le", str(out)], check=True)
    mulaw.unlink()


def real_id(orig: str) -> str:
    """real_name_000, flow_spelled_real_name_000 (so matcher.py's per-caller lookups still line up)."""
    for prefix in ("flow_spelled_", "flow_dob_"):
        if orig.startswith(prefix):
            return prefix + "real_" + orig[len(prefix):]
    return "real_" + orig


def ingest() -> None:
    from build import AUDIO

    items = {it["id"]: it for it in map(json.loads, (DATA / "items.jsonl").open(encoding="utf-8"))}
    manifest = json.loads((DATA / "real_manifest.json").read_text(encoding="utf-8"))
    by_section: dict[str, list[str]] = {}
    for key in manifest:
        by_section.setdefault(key[0], []).append(key)

    clips: dict[str, np.ndarray] = {}
    for section, keys in by_section.items():
        if section in "NDR":
            src = next(REAL.glob(f"{section}.*"))
            x = decode(src)
            ranges = split_long(src, [items[manifest[k]]["script"] for k in keys])
            for key, rng in zip(keys, ranges, strict=True):
                if rng:
                    clips[key] = x[rng[0] : rng[1]]
        else:
            for key in keys:
                found = list(REAL.glob(f"{key}.*"))
                if not found:
                    raise SystemExit(f"missing recording {key}")
                clips[key] = decode(found[0])

    real_items = []
    for key, orig_id in manifest.items():
        if key not in clips:
            continue  # line missing from the recording
        it = dict(items[orig_id])
        it.update(id=real_id(orig_id), set="real", pair=orig_id, script_key=key, voice="real")
        if "caller" in it:
            it["caller"] = real_id(it["caller"])
        to_phone_pcm(clips[key], AUDIO / f"{it['id']}.pcm", it["noise"])
        real_items.append(it)
    with (DATA / "items_real.jsonl").open("w", encoding="utf-8") as f:
        for it in real_items:
            f.write(json.dumps(it) + "\n")
    # listenable copies of the split long-recording lines, to spot-check the cuts
    prev = DATA / "preview" / "real"
    prev.mkdir(parents=True, exist_ok=True)
    for key in [k for k in ("N01", "N21", "N35", "D01", "D23", "D35", "R01", "R10") if k in clips]:
        subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-f", "s16le", "-ar", str(SR), "-ac", "1", "-i", "-",
                        str(prev / f"{key}.wav")], input=clips[key].tobytes(), check=True)
    secs = sum(len(c) for c in clips.values()) / SR
    print(f"{len(real_items)} real clips ({secs / 60:.1f} min of speech) -> items_real.jsonl")


def check() -> None:
    """Transcribe each cut clip locally and flag any that don't start with the line's opening phrase
    or don't end with the script's last number/word (a boundary bleed)."""
    from build import AUDIO
    from faster_whisper import WhisperModel

    model = WhisperModel(str(DATA / "models" / "faster-whisper-large-v3-turbo"), device="cpu", compute_type="int8")
    bad = 0
    for it in map(json.loads, (DATA / "items_real.jsonl").open(encoding="utf-8")):
        if it["script_key"][0] not in "NDR":
            continue
        x = np.frombuffer((AUDIO / f"{it['id']}.pcm").read_bytes(), dtype=np.int16).astype("float32") / 32768
        segs, _ = model.transcribe(x, language="en", temperature=0.0, condition_on_previous_text=False)
        heard = " ".join(seg.text.strip() for seg in segs)
        first = re.findall(r"[a-z0-9]+", heard.lower())[:1]
        starts_ok = first in (["my"], ["i"])
        flag = "" if starts_ok else "  <-- starts with leftover audio"
        bad += not starts_ok
        print(f"{it['script_key']}  {heard[:95]}{flag}")
    print(f"clips with leftover audio at the start: {bad}")


if __name__ == "__main__":
    if sys.argv[1:] == ["script"]:
        write_script()
    elif sys.argv[1:] == ["ingest"]:
        ingest()
    elif sys.argv[1:] == ["check"]:
        check()
    else:
        print(__doc__)
