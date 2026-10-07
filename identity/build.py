"""Build the Australian identity test set: scripted items -> TTS audio -> phone-quality PCM.

All identities are invented. Suburbs/postcodes are public geography; names are random pairings.

    uv run python identity/build.py            # build everything (skips audio that exists)
    uv run python identity/build.py --dry-run  # write items.jsonl only

Output (git-ignored): identity_data/items.jsonl, identity_data/audio/<id>.pcm (16 kHz s16le mono,
passed through 8 kHz mu-law first so it sounds like a phone line).
"""

import argparse
import asyncio
import json
import random
import subprocess
from pathlib import Path

DATA = Path("identity_data")
AUDIO = DATA / "audio"
SEED = 7

# Voices weighted toward Australian English; the rest cover accents common among Australian callers.
VOICES = [
    "en-AU-NatashaNeural", "en-AU-WilliamMultilingualNeural",
    "en-AU-NatashaNeural", "en-AU-WilliamMultilingualNeural",
    "en-IN-NeerjaNeural", "en-IN-PrabhatNeural", "en-HK-SamNeural", "en-HK-YanNeural",
    "en-PH-RosaNeural", "en-SG-WayneNeural", "en-NG-AbeoNeural", "en-GB-SoniaNeural",
    "en-NZ-MollyNeural", "en-ZA-LukeNeural",
]

GIVEN = ["Linh", "Wei", "Priya", "Siobhan", "Dimitrios", "Giuseppe", "Aisha", "Krzysztof", "Jarrah",
         "Ngozi", "Hamish", "Catherine", "Thanh", "Rajesh", "Ioannis", "Luca", "Fatima", "Jiwoo",
         "Hoang", "Tomasz", "Eilidh", "Kiri", "Mohammed"]
SURNAMES = ["Nguyen", "Zhang", "Raghunathan", "O'Sullivan", "Papadopoulos", "Esposito", "Rahman",
            "Wojciechowski", "Tjungurrayi", "Okonkwo", "McGrath", "Smyth", "Pham", "Venkataraman",
            "Karagiannis", "Bianchi", "Haddad", "Park", "Le", "Nowak", "Macleod", "Ngata", "Al-Hassan"]
# (name on record, name the impostor says). Homophones can't be told apart by ear at all.
IMPOSTORS = [("Nguyen", "Nugent", False), ("Smyth", "Smith", True), ("McGrath", "McGraw", False),
             ("Pham", "Fan", False), ("Zhang", "Chang", False), ("Raghunathan", "Ragunath", False),
             ("Okonkwo", "Okonjo", False), ("Esposito", "Espinoza", False), ("Haddad", "Hadid", False),
             ("Nowak", "Novak", True), ("Macleod", "McLeod", True), ("Chua", "Chew", False)]
SPELL_NAMES = ["Nguyen", "Wojciechowski", "Raghunathan", "Featherstone", "Tjungurrayi", "Okonkwo",
               "Pham", "Zhang", "Haddad", "Esposito", "Karagiannis", "Venkataraman", "Bianchi",
               "Macleod", "Smyth", "McGrath", "Ngata", "Rahman", "Chua", "Nowak", "Papadopoulos",
               "Vu", "Le", "Park", "Walker"]
NATO = {"m": "Mike", "s": "Sierra", "p": "Papa", "b": "Bravo", "v": "Victor", "n": "November",
        "d": "Delta", "t": "Tango", "f": "Foxtrot", "c": "Charlie", "g": "Golf", "k": "Kilo"}
DOMAINS = [("gmail.com", "gmail dot com"), ("outlook.com", "outlook dot com"),
           ("hotmail.com", "hotmail dot com"), ("bigpond.com", "bigpond dot com"),
           ("bigpond.net.au", "bigpond dot net dot au"), ("yahoo.com.au", "yahoo dot com dot au"),
           ("icloud.com", "icloud dot com"), ("optusnet.com.au", "optusnet dot com dot au")]
STREETS = ["Wattle", "Banksia", "Kurrajong", "Boronia", "Coolabah", "Warrigal", "Bungarribee",
           "Yarrawonga", "Wollondilly", "Merindah", "Kirribilli", "Macquarie", "Railway", "Church",
           "Oxford", "Station", "Jacaranda", "Illawarra"]
TYPES = ["Street", "Road", "Avenue", "Crescent", "Parade", "Place", "Close", "Drive", "Terrace", "Lane"]
SUBURBS = [  # (suburb, state code, spoken state, postcode)
    ("Parramatta", "nsw", "New South Wales", "2150"), ("Cabramatta", "nsw", "New South Wales", "2166"),
    ("Bankstown", "nsw", "N S W", "2200"), ("Woolloomooloo", "nsw", "New South Wales", "2011"),
    ("Wahroonga", "nsw", "N S W", "2076"), ("Wagga Wagga", "nsw", "New South Wales", "2650"),
    ("Kurri Kurri", "nsw", "N S W", "2327"), ("Wollongong", "nsw", "New South Wales", "2500"),
    ("Toongabbie", "nsw", "New South Wales", "2146"), ("Coolangatta", "qld", "Queensland", "4225"),
    ("Toowoomba", "qld", "Queensland", "4350"), ("Indooroopilly", "qld", "Q L D", "4068"),
    ("Woolloongabba", "qld", "Queensland", "4102"), ("Maroochydore", "qld", "Queensland", "4558"),
    ("Goondiwindi", "qld", "Queensland", "4390"), ("Footscray", "vic", "Victoria", "3011"),
    ("Dandenong", "vic", "Victoria", "3175"), ("Mooroolbark", "vic", "Victoria", "3138"),
    ("Murrumbeena", "vic", "Victoria", "3163"), ("Wangaratta", "vic", "Victoria", "3677"),
    ("Warrnambool", "vic", "Victoria", "3280"), ("Joondalup", "wa", "Western Australia", "6027"),
    ("Mandurah", "wa", "W A", "6210"), ("Kalgoorlie", "wa", "Western Australia", "6430"),
    ("Glenelg", "sa", "South Australia", "5045"), ("Elizabeth", "sa", "South Australia", "5112"),
    ("Launceston", "tas", "Tasmania", "7250"), ("Glenorchy", "tas", "Tasmania", "7010"),
    ("Palmerston", "nt", "Northern Territory", "0830"), ("Belconnen", "act", "A C T", "2617"),
    ("Tuggeranong", "act", "A C T", "2900"), ("Mildura", "vic", "Victoria", "3500"),
    ("Kogarah", "nsw", "New South Wales", "2217"), ("Ipswich", "qld", "Queensland", "4305"),
    ("Geelong", "vic", "Victoria", "3220"),
]
GENERIC_HINTS = (
    TYPES + ["Esplanade", "Highway", "Court", "Circuit", "Unit", "slash"]
    + ["gmail", "outlook", "hotmail", "bigpond", "yahoo", "icloud", "optusnet", "dot com", "dot com dot au", "dot net dot au", "at"]
    + ["New South Wales", "Victoria", "Queensland", "Western Australia", "South Australia", "Tasmania", "Northern Territory", "ACT"]
)

ONES = "zero one two three four five six seven eight nine ten eleven twelve thirteen fourteen fifteen sixteen seventeen eighteen nineteen".split()
TENS_W = "_ _ twenty thirty forty fifty sixty seventy eighty ninety".split()
ORD = {1: "first", 2: "second", 3: "third", 4: "fourth", 5: "fifth", 6: "sixth", 7: "seventh", 8: "eighth",
       9: "ninth", 10: "tenth", 11: "eleventh", 12: "twelfth", 13: "thirteenth", 14: "fourteenth",
       15: "fifteenth", 16: "sixteenth", 17: "seventeenth", 18: "eighteenth", 19: "nineteenth",
       20: "twentieth", 30: "thirtieth"}
MONTH_NAMES = "January February March April May June July August September October November December".split()


def say(n: int) -> str:
    """0..999 in words."""
    if n < 20:
        return ONES[n]
    if n < 100:
        return TENS_W[n // 10] + ("" if n % 10 == 0 else "-" + ONES[n % 10])
    rest = n % 100
    return ONES[n // 100] + " hundred" + ("" if rest == 0 else " and " + say(rest))


def say_ordinal(n: int) -> str:
    if n in ORD:
        return ORD[n]
    return TENS_W[n // 10] + "-" + ORD[n % 10]


def say_year(y: int) -> str:
    if y >= 2000:
        return "two thousand" + ("" if y == 2000 else " and " + say(y - 2000))
    return say(y // 100) + " " + (say(y % 100) if y % 100 >= 10 else "oh " + say(y % 100))


def say_digits(s: str) -> str:
    return " ".join("zero" if c == "0" else ONES[int(c)] for c in s)


def spell(name: str, rng: random.Random) -> str:
    letters = [c.upper() for c in name if c.isalpha()]
    parts, i = [], 0
    while i < len(letters):
        if i + 1 < len(letters) and letters[i] == letters[i + 1] and rng.random() < 0.7:
            parts.append(f"double {letters[i]}")
            i += 2
            continue
        c = letters[i]
        parts.append(f"{c} for {NATO[c.lower()]}" if c.lower() in NATO and rng.random() < 0.25 else c)
        i += 1
    return ", ".join(parts)


def build_items() -> list[dict]:
    rng = random.Random(SEED)
    items: list[dict] = []

    def add(field: str, script: str, expected, record=None, **extra):
        items.append({"id": f"{field}_{sum(i['field'] == field for i in items):03d}", "field": field,
                      "script": script, "expected": expected, "record": record or {}, **extra})

    # 1. Full name, genuine caller (23)
    for g, s in zip(GIVEN, rng.sample(SURNAMES, len(SURNAMES)), strict=False):
        add("name", f"My name is {g} {s}.", {"given": g, "surname": s},
            {"given": g, "surname": s}, impostor=False)
    # 2. Full name, impostor: says a different (similar-sounding) surname than the one on record (12)
    for rec, said, homophone in IMPOSTORS:
        g = rng.choice(GIVEN)
        add("name", f"My name is {g} {said}.", {"given": g, "surname": said},
            {"given": g, "surname": rec}, impostor=True, homophone=homophone)
    # 3. Spelled surname (25)
    for s in SPELL_NAMES:
        add("spelled", f"That's {spell(s, rng)}.", {"surname": s}, {"surname": s}, rate="-15%")
    # 4. Email (30)
    for _ in range(30):
        g, s = rng.choice(GIVEN), rng.choice(SPELL_NAMES)
        dom, dom_spoken = rng.choice(DOMAINS)
        style = rng.choice(["dot", "dot", "num", "underscore", "spelled"])
        n = rng.randint(10, 99)
        gl, sl = g.lower(), s.lower()
        if style == "dot":
            email, spoken = f"{gl}.{sl}@{dom}", f"{g} dot {s} at {dom_spoken}"
        elif style == "num":
            email, spoken = f"{gl}{sl}{n}@{dom}", f"{g} {s} {say(n)} at {dom_spoken}, all one word"
        elif style == "underscore":
            email, spoken = f"{gl}_{sl}@{dom}", f"{g} underscore {s} at {dom_spoken}"
        else:
            email, spoken = f"{gl[0]}.{sl}@{dom}", f"{g[0].upper()} dot, {', '.join(sl.upper())}, at {dom_spoken}"
        add("email", f"My email is {spoken}.", {"email": email},
            {"email": email, "surname": s, "given": g}, style=style)
    # 5. Date of birth (25)
    for _ in range(25):
        y, mo, d = rng.randint(1950, 2004), rng.randint(1, 12), rng.randint(1, 28)
        style = rng.choice(["ordinal_of", "ordinal", "digits", "month_first"])
        if style == "ordinal_of":
            spoken = f"the {say_ordinal(d)} of {MONTH_NAMES[mo - 1]}, {say_year(y)}"
        elif style == "ordinal":
            spoken = f"{say_ordinal(d)} {MONTH_NAMES[mo - 1]} {say_year(y)}"
        elif style == "digits":
            spoken = f"{say_digits(f'{d:02d}')}, {say_digits(f'{mo:02d}')}, {say(y % 100) if y % 100 >= 10 else 'oh ' + say(y % 100)}"
        else:
            spoken = f"{MONTH_NAMES[mo - 1]} {say_ordinal(d)}, {say_year(y)}"
        add("dob", f"My date of birth is {spoken}.", {"dob": f"{y:04d}-{mo:02d}-{d:02d}"}, style=style)
    # 6. Address (35)
    for suburb, state, state_spoken, pc in SUBURBS:
        street, typ = rng.choice(STREETS), rng.choice(TYPES)
        number = str(rng.randint(10, 180))  # >= 10 so a unit digit can't merge into it
        unit = str(rng.randint(1, 9)) if rng.random() < 0.4 else None
        num_spoken = say(int(number))
        if unit:
            num_spoken = rng.choice([f"unit {say(int(unit))}, {num_spoken}", f"{say(int(unit))} slash {num_spoken}"])
        if rng.random() < 0.6 or pc[0] == "0" or (pc[2] == "0" and pc[3] != "0"):
            pc_spoken = say_digits(pc)  # "two one five zero"
        else:  # "twenty-one fifty", "twenty-two hundred"
            pc_spoken = f"{say(int(pc[:2]))} {'hundred' if pc[2:] == '00' else say(int(pc[2:]))}"
        add("address", f"I live at {num_spoken} {street} {typ}, {suburb}, {state_spoken}, {pc_spoken}.",
            {"unit": unit, "number": number, "street": street.lower(), "type": typ.lower(),
             "suburb": suburb.lower(), "state": state, "postcode": pc},
            {"street": street, "suburb": suburb})

    for i, it in enumerate(items):
        it["voice"] = VOICES[(i * 5 + rng.randint(0, 3)) % len(VOICES)]
        it["noise"] = rng.random() < 0.5
    return items + build_flow_items(items)


def build_flow_items(items: list[dict]) -> list[dict]:
    """Full-flow callers: each name caller also spells their surname and gives their DOB, same voice.

    Genuine callers have the record's details. Wrong-party callers (the "impostors") are different
    people who honestly give their OWN surname and date of birth. Separate RNG so the original
    150 items stay byte-identical.
    """
    rng = random.Random(SEED + 1)
    flow = []
    for it in [i for i in items if i["field"] == "name"]:
        said = it["expected"]["surname"]
        y, mo, d = rng.randint(1950, 2004), rng.randint(1, 12), rng.randint(1, 28)
        dob = f"{y:04d}-{mo:02d}-{d:02d}"
        # the wrong party's own DOB differs from the customer's on record
        rec_dob = dob if not it.get("impostor") else f"{y - rng.randint(3, 20):04d}-{rng.randint(1, 12):02d}-{rng.randint(1, 28):02d}"
        common = {"caller": it["id"], "voice": it["voice"], "noise": it["noise"], "impostor": it.get("impostor", False),
                  "homophone": it.get("homophone", False)}
        flow.append({"id": f"flow_spelled_{it['id']}", "field": "flow_spelled", "script": f"That's {spell(said, rng)}.",
                     "expected": {"surname": said}, "record": {"surname": it["record"]["surname"]}, "rate": "-15%", **common})
        flow.append({"id": f"flow_dob_{it['id']}", "field": "flow_dob",
                     "script": f"My date of birth is the {say_ordinal(d)} of {MONTH_NAMES[mo - 1]}, {say_year(y)}.",
                     "expected": {"dob": dob}, "record": {"dob": rec_dob}, **common})
    return flow


def load_items() -> dict[str, dict]:
    """Synthetic items plus real-voice items (identity_data/items_real.jsonl) if recorded."""
    items = {}
    for name in ("items.jsonl", "items_real.jsonl"):
        path = DATA / name
        if path.exists():
            for line in path.open(encoding="utf-8"):
                it = json.loads(line)
                it.setdefault("set", "synthetic")
                items[it["id"]] = it
    # Reading slips in the real recordings (reader said a different value than the script, confirmed by
    # all 5 vendors hearing the same thing). The test measures hearing, so score what was said.
    # {"D16": {"dob": "2004-01-17"}} keyed by recording-script line.
    fixes_path = DATA / "real_corrections.json"
    if fixes_path.exists():
        fixes = json.loads(fixes_path.read_text(encoding="utf-8"))
        for it in items.values():
            if it.get("script_key") in fixes:
                it["expected"] = {**it["expected"], **fixes[it["script_key"]]}
                if not it.get("impostor"):  # a genuine caller's record holds what they actually said
                    it["record"] = {**it["record"], **fixes[it["script_key"]]}
    return items


def hints(item: dict, condition: str) -> list[str]:
    if condition == "none":
        return []
    if condition == "generic":
        return GENERIC_HINTS
    rec = item["record"]  # "record": what an agent would know from the account on file
    return [v for v in (rec.get("given"), rec.get("surname"), rec.get("email"), rec.get("street"), rec.get("suburb")) if v]


async def synth(item: dict) -> None:
    import edge_tts

    pcm = AUDIO / f"{item['id']}.pcm"
    if pcm.exists():
        return
    mp3 = AUDIO / f"{item['id']}.mp3"
    await edge_tts.Communicate(item["script"], item["voice"], rate=item.get("rate", "+0%")).save(str(mp3))
    # phone line: 300 ms lead-in, optional pink noise, 8 kHz mu-law, then back to 16 kHz PCM for the harness
    graph = "[0:a]adelay=300,aresample=24000"
    if item["noise"]:
        graph += "[a];anoisesrc=color=pink:amplitude=0.015:sample_rate=24000[n];[a][n]amix=inputs=2:duration=first:normalize=0"
    mulaw = AUDIO / f"{item['id']}.wav"
    subprocess.run(
        ["ffmpeg", "-y", "-loglevel", "error", "-i", str(mp3), "-filter_complex", graph,
         "-ar", "8000", "-ac", "1", "-c:a", "pcm_mulaw", str(mulaw)],
        check=True,
    )
    subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-i", str(mulaw), "-ar", "16000", "-ac", "1",
                    "-f", "s16le", str(pcm)], check=True)
    mp3.unlink()
    mulaw.unlink()


async def main(dry_run: bool) -> None:
    AUDIO.mkdir(parents=True, exist_ok=True)
    items = build_items()
    with open(DATA / "items.jsonl", "w", encoding="utf-8") as f:
        for it in items:
            f.write(json.dumps(it) + "\n")
    print(f"{len(items)} items -> {DATA / 'items.jsonl'}")
    if dry_run:
        return
    sem = asyncio.Semaphore(6)

    async def one(it):
        async with sem:
            await synth(it)

    await asyncio.gather(*(one(it) for it in items))
    secs = sum((AUDIO / f"{it['id']}.pcm").stat().st_size for it in items) / 32000
    print(f"audio ready: {secs / 60:.1f} min total")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--dry-run", action="store_true")
    asyncio.run(main(ap.parse_args().dry_run))
