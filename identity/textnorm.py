"""Normalise transcripts of spoken identity details and check them field by field.

Two verdicts per field:
- strict: the expected value appears in the transcript as delivered (light cleanup only).
- lenient: the expected value is recoverable after our own normaliser
  (number words -> digits, "at"/"dot" -> symbols, spelled letters collapsed).
strict fails + lenient passes  => a formatting problem, not a hearing problem.
lenient fails                  => the vendor misheard (or split/dropped) it.

Run `uv run python identity/textnorm.py` for the self-check.
"""

import re
from difflib import SequenceMatcher

UNITS = {
    "zero": 0, "oh": 0, "o": 0, "nought": 0, "one": 1, "two": 2, "three": 3, "four": 4,
    "five": 5, "six": 6, "seven": 7, "eight": 8, "nine": 9,
}
TEENS = {
    "ten": 10, "eleven": 11, "twelve": 12, "thirteen": 13, "fourteen": 14, "fifteen": 15,
    "sixteen": 16, "seventeen": 17, "eighteen": 18, "nineteen": 19,
}
TENS = {
    "twenty": 20, "thirty": 30, "forty": 40, "fifty": 50, "sixty": 60, "seventy": 70,
    "eighty": 80, "ninety": 90,
}
ORD_WORD = {
    "first": "one", "second": "two", "third": "three", "fourth": "four", "fifth": "five",
    "sixth": "six", "seventh": "seven", "eighth": "eight", "ninth": "nine", "tenth": "ten",
    "eleventh": "eleven", "twelfth": "twelve", "thirteenth": "thirteen", "fourteenth": "fourteen",
    "fifteenth": "fifteen", "sixteenth": "sixteen", "seventeenth": "seventeen",
    "eighteenth": "eighteen", "nineteenth": "nineteen", "twentieth": "twenty", "thirtieth": "thirty",
}
MONTHS = {
    m: i + 1
    for i, m in enumerate(
        "january february march april may june july august september october november december".split()
    )
}
MONTHS.update({m[:3]: n for m, n in list(MONTHS.items())})
MONTHS["sept"] = 9
LETTER_NAMES = {
    "ay": "a", "bee": "b", "be": "b", "see": "c", "sea": "c", "cee": "c", "dee": "d", "ee": "e",
    "eff": "f", "ef": "f", "gee": "g", "aitch": "h", "haitch": "h", "eye": "i", "jay": "j",
    "kay": "k", "el": "l", "ell": "l", "em": "m", "en": "n", "pee": "p", "queue": "q", "cue": "q",
    "ar": "r", "are": "r", "ess": "s", "es": "s", "tee": "t", "tea": "t", "you": "u", "vee": "v",
    "ex": "x", "why": "y", "wye": "y", "zed": "z", "zee": "z",
}
STREET_TYPES = {
    "st": "street", "rd": "road", "ave": "avenue", "av": "avenue", "cres": "crescent",
    "cr": "crescent", "pde": "parade", "pl": "place", "tce": "terrace", "hwy": "highway",
    "cl": "close", "ct": "court", "dr": "drive", "ln": "lane", "esp": "esplanade", "cct": "circuit",
}
STATES = {
    "new south wales": "nsw", "victoria": "vic", "queensland": "qld", "western australia": "wa",
    "south australia": "sa", "tasmania": "tas", "northern territory": "nt",
    "australian capital territory": "act", "n s w": "nsw", "q l d": "qld", "a c t": "act",
    "w a": "wa", "s a": "sa", "n t": "nt",
}


def words(text: str, breaks: bool = False) -> list[str]:
    """Lowercase tokens; keeps digits, splits on everything else (hyphens, commas, slashes).

    breaks=True also emits "," for commas/slashes/full stops so digit runs don't merge across them.
    """
    pattern = r"[a-z]+|\d+|[,/;.]" if breaks else r"[a-z]+|\d+"
    return [t if t.isalnum() else "," for t in re.findall(pattern, text.lower().replace("'", ""))]


def numbers_to_digits(tokens: list[str]) -> list[str]:
    """Replace spoken numbers with digit strings, keeping other tokens.

    "double seven" -> "77"; runs of single digits concatenate ("two one five zero" -> "2150",
    "zero three" -> "03"); "eighty five" -> "85"; "one hundred and twelve" -> "112".
    "oh"/"o" only count as zero next to another digit, so a stray "oh" stays a word.
    """
    out: list[tuple[str, bool]] = []  # (token, is a single spoken digit that may join a run)
    i = 0
    n = len(tokens)

    def is_num_word(t: str) -> bool:
        return t in UNITS or t in TEENS or t in TENS or t.isdigit()

    while i < n:
        t = tokens[i]
        nxt = tokens[i + 1] if i + 1 < n else ""
        if t in ("double", "triple") and (nxt in UNITS or (nxt.isdigit() and len(nxt) == 1)):
            d = str(UNITS.get(nxt, nxt))
            out.append((d * (2 if t == "double" else 3), True))
            i += 2
        elif t in ("oh", "o") and not (
            (i > 0 and is_num_word(tokens[i - 1]) and tokens[i - 1] not in ("oh", "o"))
            or (is_num_word(nxt) and nxt not in ("oh", "o"))
        ):
            out.append((t, False))
            i += 1
        elif t in TENS:
            v = TENS[t]
            i += 1
            if nxt in UNITS and nxt not in ("oh", "o", "zero"):
                v += UNITS[nxt]
                i += 1
            out.append((str(v), False))
        elif t in TEENS:
            out.append((str(TEENS[t]), False))
            i += 1
        elif t in UNITS or (t.isdigit() and len(t) == 1):
            out.append((str(UNITS.get(t, t)), True))
            i += 1
        else:
            out.append((t, False))
            i += 1
            continue
        # "X hundred (and) Y", "X thousand (and) Y"
        while i < n and tokens[i] in ("hundred", "thousand") and out[-1][0].isdigit():
            mult = 100 if tokens[i] == "hundred" else 1000
            base = int(out.pop()[0]) * mult
            i += 1
            if i < n and tokens[i] == "and":
                i += 1
            if i < n and is_num_word(tokens[i]):
                two = i + 1 < n and tokens[i] in TENS and tokens[i + 1] in UNITS
                rest = int(numbers_to_digits(tokens[i : i + (2 if two else 1)])[0])
                if rest < mult:
                    base += rest
                    i += 2 if two else 1
            out.append((str(base), False))

    # merge runs of single spoken digits: "two one five zero" -> "2150"; "eighty five" stays "85"
    merged: list[str] = []
    prev_joinable = False
    for tok, joinable in out:
        if joinable and prev_joinable:
            merged[-1] += tok
        else:
            merged.append(tok)
        prev_joinable = joinable
    return [m for m in merged if m != ","]


def norm_name(s: str) -> str:
    return re.sub(r"[^a-z]", "", s.lower())


def letter_runs(text: str) -> str:
    """Collapse spelled letters into a string: "N, G, U" / "N-G-U" / "M for Mike" -> "ngu".

    Letter-name words ("en", "gee") only count inside a run of at least two letters,
    so ordinary words like "you" or "are" don't leak in.
    """
    # vendors write spelled letters as caps blocks ("NG for golf", "TJUNGU"): split them
    text = re.sub(r"\b[A-Z]{2,}\b", lambda m: " ".join(m.group(0)), text)
    toks = words(text)
    letters: list[str | None] = []
    i = 0
    while i < len(toks):
        t = toks[i]
        if len(t) == 1 and t.isalpha():
            letters.append(t)
            if i + 2 < len(toks) and toks[i + 1] == "for":
                i += 2  # "m for mike"
            elif i + 3 < len(toks) and toks[i + 1] == "as" and toks[i + 2] == "in":
                i += 3  # "m as in mary"
        elif t == "double" and i + 1 < len(toks) and len(toks[i + 1]) == 1 and toks[i + 1].isalpha():
            letters.extend([toks[i + 1], toks[i + 1]])
            i += 1
        elif t in LETTER_NAMES:
            letters.append("?" + LETTER_NAMES[t])  # tentative
        else:
            letters.append(None)
        i += 1
    # keep tentative letters only when a neighbour is a letter too
    out = []
    for j, ch in enumerate(letters):
        if ch is None:
            out.append(" ")
        elif ch.startswith("?"):
            nb = [letters[k] for k in (j - 1, j + 1) if 0 <= k < len(letters)]
            out.append(ch[1] if any(x is not None for x in nb) else " ")
        else:
            out.append(ch)
    return "".join(out)


def check_name(transcript: str, expected: str) -> dict:
    """expected: a single name (given or surname). Near-miss ratio shows what fuzzy matching could recover."""
    exp = norm_name(expected)
    toks = [norm_name(t) for t in re.split(r"[\s,.;:!?]+", transcript)]
    toks = [t for t in toks if t]
    strict = exp in toks
    # names like O'Brien or Van Der Berg may be split or joined
    joined = "".join(toks)
    lenient = strict or exp in joined or exp in letter_runs(transcript).replace(" ", "")
    best = max((SequenceMatcher(None, exp, t).ratio() for t in toks), default=0.0)
    return {"strict": strict, "lenient": lenient, "near_miss": round(best, 3)}


def check_spelled(transcript: str, expected: str) -> dict:
    exp = norm_name(expected)
    runs = letter_runs(transcript).replace(" ", "")
    # strict: the vendor itself produced the joined word or a clean letter sequence
    compact = re.sub(r"[^a-z]", "", transcript.lower())
    strict = exp in [norm_name(t) for t in transcript.split()] or exp in compact and exp in runs
    return {"strict": strict, "lenient": exp in runs or exp in compact}


EMAIL_WORDS = {"at": "@", "dot": ".", "underscore": "_", "dash": "-", "hyphen": "-", "period": ".", "full stop": "."}


def normalise_email(text: str) -> str:
    t = text.lower().replace("full stop", "dot")
    t = re.sub(r"\.(?=\S)", " dot ", t)  # "icloud.com" -> dot; a full stop before a space is just punctuation
    toks = numbers_to_digits(words(t.replace("@", " at ").replace("_", " underscore ")))
    toks = [EMAIL_WORDS.get(x, x) for x in toks]
    # spelled letters inside the local part ("n g u y e n") collapse naturally once spaces go
    return "".join(LETTER_NAMES.get(x, x) if len(x) > 1 and x in ("ay", "bee", "dee", "ee", "gee", "jay", "kay", "pee", "tee", "vee", "zed") else x for x in toks)


def check_email(transcript: str, expected: str) -> dict:
    exp = expected.lower()
    raw = re.sub(r"\s+", "", transcript.lower())
    strict = exp in raw
    lenient = strict or exp in normalise_email(transcript)
    return {"strict": strict, "lenient": lenient}


def parse_dob(text: str) -> tuple[int | None, int | None, int | None]:
    """Best-effort (day, month, year) from a transcript, assuming Australian day-first order."""
    t = text.lower()
    m = re.search(r"(\d{1,2})\s*[/.-]\s*(\d{1,2})\s*[/.-]\s*(\d{2,4})", t)
    if m:
        return int(m.group(1)), int(m.group(2)), _year(int(m.group(3)))
    toks = words(t, breaks=True)
    # ordinals -> cardinal words first, so "twenty seventh" becomes "twenty seven" -> 27
    toks = [ORD_WORD.get(x, x) for x in toks]
    toks = [re.sub(r"(st|nd|rd|th)$", "", x) if re.fullmatch(r"\d+(st|nd|rd|th)", x) else x for x in toks]
    toks = numbers_to_digits(toks)
    month = next((MONTHS[x] for x in toks if x in MONTHS), None)
    nums = [x for x in toks if x.isdigit()]
    # join "19" "85" -> "1985"
    joined: list[str] = []
    for x in nums:
        if joined and joined[-1] in ("19", "20") and len(x) == 2:
            joined[-1] += x
        else:
            joined.append(x)
    day = year = None
    if month is not None:
        day = next((int(x) for x in joined if 1 <= int(x) <= 31 and len(x) <= 2), None)
        rest = [x for x in joined if not (day is not None and int(x) == day and len(x) <= 2)]
        year = _year(int(rest[-1])) if rest else None
    elif len(joined) >= 3:
        day, month, year = int(joined[0]), int(joined[1]), _year(int(joined[2]))
    elif len(joined) == 1 and len(joined[0]) in (6, 8):  # "03041985"
        d = joined[0]
        day, month, year = int(d[:2]), int(d[2:4]), _year(int(d[4:]))
    elif len(joined) == 2 and len(joined[0]) == 4:  # "zero one zero four sixty two" -> "0104", "62"
        day, month, year = int(joined[0][:2]), int(joined[0][2:]), _year(int(joined[1]))
    return day, month, year


def _year(y: int) -> int:
    if y < 100:
        return 1900 + y if y > 30 else 2000 + y
    return y


def check_dob(transcript: str, expected: str) -> dict:
    """expected: 'YYYY-MM-DD'.

    strict: reads correctly as Australian day/month. us_order: came out month-first (01/26/1951).
    lenient also accepts a month-first date when it can't be misread (day > 12); 01/04 can't be saved.
    """
    y, mo, d = (int(x) for x in expected.split("-"))
    got = parse_dob(transcript)
    strict = got == (d, mo, y)
    us_order = not strict and got == (mo, d, y)
    return {"strict": strict, "lenient": strict or (us_order and d > 12), "us_order": us_order,
            "parsed": "-".join("?" if v is None else str(v) for v in got)}


def normalise_address(text: str) -> str:
    t = " " + " ".join(numbers_to_digits(words(text, breaks=True))) + " "
    for k, v in STATES.items():
        t = t.replace(f" {k} ", f" {v} ")
    toks = [STREET_TYPES.get(x, x) for x in t.split()]
    return " " + " ".join(toks) + " "


def check_address(transcript: str, expected: dict) -> dict:
    """expected keys: unit (optional), number, street, type, suburb, state, postcode."""
    norm = normalise_address(transcript)
    digits = [x for x in norm.split() if x.isdigit()]
    res = {}
    if expected.get("unit"):
        res["unit"] = expected["unit"] in digits
    res["number"] = expected["number"] in digits
    if expected.get("unit"):
        # unit before number, adjacent ("3/14" or "unit 3, 14")
        try:
            k = digits.index(expected["unit"])
            res["unit_order"] = digits[k + 1 : k + 2] == [expected["number"]]
        except ValueError:
            res["unit_order"] = False
    for key in ("street", "type", "suburb", "state"):
        res[key] = f" {expected[key].lower()} " in norm
    res["postcode"] = expected["postcode"] in digits or expected["postcode"] in "".join(digits)
    res["all"] = all(res.values())
    return res


if __name__ == "__main__":
    assert numbers_to_digits(words("two one five zero")) == ["2150"]
    assert numbers_to_digits(words("zero eight three zero")) == ["0830"]
    assert numbers_to_digits(words("double seven")) == ["77"]
    assert numbers_to_digits(words("eighty-five")) == ["85"]
    assert numbers_to_digits(words("one hundred and twelve")) == ["112"]
    assert numbers_to_digits(words("oh I see")) == ["oh", "i", "see"]
    assert numbers_to_digits(words("unit two, one", breaks=True)) == ["unit", "2", "1"]

    assert letter_runs("It's N, G, U, Y, E, N.").replace(" ", "") == "nguyen"
    assert "mcgrath" in letter_runs("M for Mike, C, G, R, A, T, H").replace(" ", "")
    assert "nguyen" in letter_runs("en gee you why ee en").replace(" ", "")
    assert check_spelled("That's N-G-U-Y-E-N", "Nguyen")["lenient"]
    assert not check_spelled("That's N-G-U-Y-A-N", "Nguyen")["lenient"]
    assert check_spelled("That's NG for golf, u. Y e? N for November.", "Nguyen")["lenient"]

    assert check_name("My name is Siobhan O'Brien", "O'Brien")["strict"]
    r = check_name("My name is Shivon Obrian", "Siobhan")
    assert not r["lenient"] and r["near_miss"] > 0.5

    e = check_email("j dot nguyen eighty two at gmail dot com", "j.nguyen82@gmail.com")
    assert e == {"strict": False, "lenient": True}, e
    assert check_email("j.nguyen82@gmail.com", "j.nguyen82@gmail.com")["strict"]
    assert check_email("My email is l dot TJUNGU. R. RAY. I. At icloud.com.", "l.tjungurrayi@icloud.com")["lenient"]
    assert not check_email("j dot win eighty two at gmail dot com", "j.nguyen82@gmail.com")["lenient"]

    assert check_dob("the third of April, nineteen eighty-five", "1985-04-03")["lenient"]
    assert check_dob("3rd April 1985", "1985-04-03")["lenient"]
    assert check_dob("03/04/1985", "1985-04-03")["lenient"]
    assert check_dob("zero three, zero four, eighty five", "1985-04-03")["lenient"]
    assert not check_dob("04/03/1985", "1985-04-03")["lenient"]  # ambiguous US order = unrecoverable
    r = check_dob("the 01/26/1951.", "1951-01-26")
    assert not r["strict"] and r["us_order"] and r["lenient"], r
    assert check_dob("zero one zero four sixty two", "1962-04-01")["strict"]
    assert check_dob("one one one zero ninety eight", "1998-10-11")["strict"]
    assert check_dob("twenty seventh July nineteen eighty four", "1984-07-27")["strict"]
    assert check_dob("October twenty seventh, nineteen ninety two", "1992-10-27")["strict"]

    a = {"unit": "3", "number": "14", "street": "wattle", "type": "street", "suburb": "parramatta", "state": "nsw", "postcode": "2150"}
    assert check_address("Unit three, fourteen Wattle Street, Parramatta, New South Wales, two one five zero", a)["all"]
    assert check_address("3/14 Wattle St, Parramatta NSW 2150", a)["all"]
    assert not check_address("3/40 Wattle St, Parramatta NSW 2150", a)["all"]
    print("textnorm self-check passed")
