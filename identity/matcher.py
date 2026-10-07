"""Verify identity in code: compare an HONEST transcript (no record hints) with the record on file.

For each field, measures two rates at several strictness settings:
- genuine pass: the real customer is accepted (higher is better)
- impostor pass: someone else is accepted (lower is better)

Impostors: for surnames, the 12 scripted callers who say a similar/identical-sounding name;
for other fields, every other customer's record ("random impostor": a much easier test).
The "record-hinted + exact" row is the approach from the main run, for comparison.

    uv run python identity/matcher.py            # writes identity_data/matcher_report.md
    uv run python identity/matcher.py --real     # same, real-voice clips -> matcher_report_real.md
    uv run python identity/matcher.py --selftest
"""

import json
import re
import sys
from collections import defaultdict
from pathlib import Path

from tabulate import tabulate

sys.path.insert(0, str(Path(__file__).parent))
from build import DATA, load_items  # noqa: E402
from textnorm import (  # noqa: E402
    letter_runs,
    norm_name,
    normalise_address,
    normalise_email,
    parse_dob,
)

VENDORS = ["assemblyai", "deepgram", "speechmatics", "gemini", "whisper"]
SET = "real" if "--real" in sys.argv else "synthetic"


def jaro_winkler(a: str, b: str) -> float:
    """0..1 similarity that rewards a shared start (Martha/Marhta 0.961)."""
    if a == b:
        return 1.0
    if not a or not b:
        return 0.0
    window = max(len(a), len(b)) // 2 - 1
    am, bm = [False] * len(a), [False] * len(b)
    m = 0
    for i, ca in enumerate(a):
        for j in range(max(0, i - window), min(len(b), i + window + 1)):
            if not bm[j] and b[j] == ca:
                am[i] = bm[j] = True
                m += 1
                break
    if not m:
        return 0.0
    ta = [c for c, f in zip(a, am, strict=True) if f]
    tb = [c for c, f in zip(b, bm, strict=True) if f]
    t = sum(x != y for x, y in zip(ta, tb, strict=True)) / 2
    jaro = (m / len(a) + m / len(b) + (m - t) / m) / 3
    prefix = 0
    for x, y in zip(a[:4], b[:4], strict=False):
        if x != y:
            break
        prefix += 1
    return jaro + prefix * 0.1 * (1 - jaro)


def soundex(name: str) -> str:
    """Classic 4-char sound code: Smith and Smyth -> S530."""
    name = norm_name(name)
    if not name:
        return ""
    codes = {c: str(d) for d, letters in enumerate(["aeiouyhw", "bfpv", "cgjkqsxz", "dt", "l", "mn", "r"]) for c in letters}
    out, prev = name[0].upper(), codes.get(name[0])
    for c in name[1:]:
        d = codes.get(c, "")
        if d != "0" and d != prev:
            out += d
        if c not in "hw":
            prev = d
    return (out + "000")[:4]


def candidates(transcript: str) -> list[str]:
    """Name-like candidates: single tokens and adjacent pairs ("o sullivan", "al hassan")."""
    toks = [norm_name(t) for t in re.split(r"[\s,.;:!?]+", transcript)]
    toks = [t for t in toks if t]
    return toks + [a + b for a, b in zip(toks, toks[1:], strict=False)]


def name_score(transcript: str, record: str) -> tuple[float, bool]:
    """Best Jaro-Winkler against the record surname, and whether any candidate sounds the same."""
    rec = norm_name(record)
    cands = candidates(transcript) or [""]
    sim = max(jaro_winkler(c, rec) for c in cands)
    same_sound = any(soundex(c) == soundex(rec) for c in cands if len(c) > 1)
    return sim, same_sound


def window_sim(text: str, target: str) -> float:
    """Best similarity of `target` against any same-length slice of `text` (+-2 chars)."""
    best = 0.0
    for L in range(max(1, len(target) - 2), len(target) + 3):
        for i in range(0, max(1, len(text) - L + 1)):
            best = max(best, jaro_winkler(text[i : i + L], target))
    return best


def email_score(transcript: str, record: str) -> float:
    return window_sim(normalise_email(transcript), record.lower())


def dob_match(transcript: str, record: str) -> bool:
    y, mo, d = (int(x) for x in record.split("-"))
    got = parse_dob(transcript)
    return got == (d, mo, y) or (got == (mo, d, y) and d > 12)  # unambiguous US order is recoverable


def address_score(transcript: str, rec: dict) -> tuple[bool, float]:
    """(numbers all exact, min similarity of street and suburb names)."""
    norm = normalise_address(transcript)
    digits = [x for x in norm.split() if x.isdigit()]
    nums_ok = rec["number"] in digits and (rec["postcode"] in digits or rec["postcode"] in "".join(digits))
    if rec.get("unit"):
        k = digits.index(rec["unit"]) if rec["unit"] in digits else -1
        nums_ok = nums_ok and k >= 0 and digits[k + 1 : k + 2] == [rec["number"]]
    toks = [t for t in norm.split() if not t.isdigit()]
    grams = toks + [a + b for a, b in zip(toks, toks[1:], strict=False)]
    sims = [max((jaro_winkler(g, norm_name(rec[k])) for g in grams), default=0.0) for k in ("street", "suburb")]
    return nums_ok, min(sims)


def load() -> tuple[dict, dict]:
    items = {k: it for k, it in load_items().items() if it["set"] == SET}
    res: dict = defaultdict(dict)  # (vendor, condition) -> id -> transcript
    for path in (DATA / "results").glob("*.jsonl"):
        for r in map(json.loads, path.open(encoding="utf-8")):
            if r["transcript"] is not None and not r["error"]:
                res[(r["vendor"], r["condition"])][r["id"]] = r["transcript"]
    return items, res


def rate(xs: list[bool]) -> str:
    return f"{sum(xs)}/{len(xs)} ({100 * sum(xs) / len(xs):.0f}%)" if xs else "-"


def main() -> None:
    items, res = load()
    out = []

    # ---- Surname: genuine callers vs targeted impostors ----
    rules = [
        ("exact", lambda s, snd: s == 1.0),
        ("similarity >= 0.95", lambda s, snd: s >= 0.95),
        ("similarity >= 0.90", lambda s, snd: s >= 0.90),
        ("similarity >= 0.85", lambda s, snd: s >= 0.85),
        ("sounds alike OR >= 0.90", lambda s, snd: snd or s >= 0.90),
    ]
    names = [it for it in items.values() if it["field"] == "name"]
    t = []
    for v in VENDORS:
        hinted = res[(v, "record")]
        g = [it for it in names if not it.get("impostor") and it["id"] in hinted]
        imp = [it for it in names if it.get("impostor") and not it.get("homophone") and it["id"] in hinted]
        hom = [it for it in names if it.get("homophone") and it["id"] in hinted]

        def passes(group, trans, rule):
            return [rule(*name_score(trans[it["id"]], it["record"]["surname"])) for it in group]

        exact = rules[0][1]
        t.append([v, "record-hinted transcript, exact", rate(passes(g, hinted, exact)),
                  rate(passes(imp, hinted, exact)), rate(passes(hom, hinted, exact))])
        honest = res[(v, "none")]
        for label, rule in rules:
            t.append(["", f"honest transcript, {label}", rate(passes(g, honest, rule)),
                      rate(passes(imp, honest, rule)), rate(passes(hom, honest, rule))])
    out.append("## Surname check: genuine callers vs impostors who say a similar name\n\n"
               "Similar names: Nugent for Nguyen, McGraw for McGrath, Chang for Zhang (9 callers). "
               "Homophones: Smith/Smyth, Novak/Nowak, McLeod/Macleod (3 callers).\n\n"
               + tabulate(t, ["vendor", "approach", "genuine pass", "similar-name impostor pass",
                              "homophone impostor pass"], tablefmt="github"))

    # ---- Other fields: genuine pass vs random impostor (another customer's record) ----
    def field_table(field, title, rules, score):
        rows = []
        group = [it for it in items.values() if it["field"] == field]
        for v in VENDORS:
            honest = res[(v, "none")]
            g = [it for it in group if it["id"] in honest]
            for i, (label, rule) in enumerate(rules):
                gen = [rule(score(honest[it["id"]], it)) for it in g]
                imp = [rule(score(honest[it["id"]], o)) for it in g for o in group if o["id"] != it["id"]]
                rows.append([v if i == 0 else "", label, rate(gen), rate(imp)])
        out.append(f"## {title}\n\n" + tabulate(rows, ["vendor", "rule", "genuine pass",
                                                       "random impostor pass"], tablefmt="github"))

    field_table("email", "Email (honest transcript, normalised)",
                [("exact", lambda s: s == 1.0), ("similarity >= 0.95", lambda s: s >= 0.95),
                 ("similarity >= 0.90", lambda s: s >= 0.90)],
                lambda tr, it: email_score(tr, it["record"]["email"]))
    field_table("dob", "Date of birth (honest transcript, normalised, Australian order)",
                [("exact date", lambda ok: ok)], lambda tr, it: dob_match(tr, it["expected"]["dob"]))
    field_table("address", "Address (numbers exact; street and suburb names fuzzy)",
                [("names exact", lambda r: r[0] and r[1] == 1.0), ("names >= 0.90", lambda r: r[0] and r[1] >= 0.90),
                 ("names >= 0.85", lambda r: r[0] and r[1] >= 0.85)],
                lambda tr, it: address_score(tr, it["expected"]))
    field_table("spelled", "Spelled surname (letters collapsed)",
                [("exact", lambda s: s == 1.0), ("similarity >= 0.90", lambda s: s >= 0.90)],
                lambda tr, it: window_sim(letter_runs(tr).replace(" ", ""), norm_name(it["record"]["surname"])))

    out.append(flow_table(items, res))

    report = "\n\n".join(out) + "\n"
    (DATA / ("matcher_report.md" if SET == "synthetic" else f"matcher_report_{SET}.md")).write_text(report, encoding="utf-8")
    print(report)


def flow_table(items: dict, res: dict) -> str:
    """One verified/not-verified decision per caller: name -> (spelling fallback) -> date of birth."""
    callers = [it for it in items.values() if it["field"] == "name"]

    def spelled_ok(tr: str, rec: str, exact: bool) -> bool:
        sim = window_sim(letter_runs(tr).replace(" ", ""), norm_name(rec))
        return sim == 1.0 if exact else sim >= 0.90

    policies = {
        "A. record-hinted name (exact) + DOB": {"hinted": True, "spell": None, "dob": True},
        "B. honest name (exact) + DOB": {"hinted": False, "spell": None, "dob": True},
        "C1. honest name >=0.95, else spelling (exact letters), + DOB": {"hinted": False, "spell": "exact", "dob": True},
        "C2. honest name >=0.95, else spelling (>=0.90 similar), + DOB": {"hinted": False, "spell": "fuzzy", "dob": True},
        "D. C1 without DOB (name only)": {"hinted": False, "spell": "exact", "dob": False},
    }
    rows = []
    for v in VENDORS:
        honest, hinted = res[(v, "none")], res[(v, "record")]
        for i, (label, p) in enumerate(policies.items()):
            gen, wrong, hom, spelled_used = [], [], [], []
            for c in callers:
                ids = (c["id"], f"flow_spelled_{c['id']}", f"flow_dob_{c['id']}")
                if not all(x in honest for x in ids) or c["id"] not in hinted:
                    continue
                rec = c["record"]["surname"]
                name_tr = hinted[c["id"]] if p["hinted"] else honest[c["id"]]
                sim = name_score(name_tr, rec)[0]
                name_ok = sim == 1.0 if p["spell"] is None else sim >= 0.95
                used_spelling = not name_ok and p["spell"] is not None
                if used_spelling:
                    name_ok = spelled_ok(honest[ids[1]], rec, p["spell"] == "exact")
                ok = name_ok and (not p["dob"] or dob_match(honest[ids[2]], items[ids[2]]["record"]["dob"]))
                if not c.get("impostor"):
                    gen.append(ok)
                    spelled_used.append(used_spelling)
                elif c.get("homophone"):
                    hom.append(ok)
                else:
                    wrong.append(ok)
            rows.append([v if i == 0 else "", label, rate(gen), rate(spelled_used) if p["spell"] else "-",
                         rate(wrong), rate(hom)])
    return ("## Full flow: one decision per caller\n\n"
            "Wrong party = a different person with a similar name who gives their OWN surname spelling and "
            "date of birth (Nugent for Nguyen). Homophone wrong party = Smith for Smyth, Novak for Nowak, "
            "McLeod for Macleod.\n\n"
            + tabulate(rows, ["vendor", "policy", "genuine verified", "genuine needed spelling",
                              "wrong party verified (similar)", "wrong party verified (homophone)"],
                       tablefmt="github"))


def selftest() -> None:
    assert abs(jaro_winkler("martha", "marhta") - 0.961) < 0.001
    assert jaro_winkler("nguyen", "nguyen") == 1.0
    assert soundex("Smith") == soundex("Smyth") == "S530"
    assert soundex("Robert") == "R163" and soundex("Ashcraft") == "A261"
    assert name_score("My name is Aisha Sullivan", "O'Sullivan")[0] > 0.9
    assert name_score("My name is Wei Fan", "Pham")[0] < 0.8
    assert email_score("hamish dot novak at gmail dot com", "hamish.nowak@gmail.com") > 0.9
    assert dob_match("01/26/1951", "1951-01-26") and not dob_match("04/01/1985", "1985-04-01")
    rec = {"unit": "3", "number": "46", "street": "coolabah", "suburb": "bankstown", "postcode": "2200"}
    assert address_score("3/46 Koolabah Place, Bankstown NSW 2200", rec)[0]
    assert not address_score("346 Coolabah Place, Bankstown NSW 2200", rec)[0]
    print("matcher self-check passed")


if __name__ == "__main__":
    selftest() if "--selftest" in sys.argv else main()
