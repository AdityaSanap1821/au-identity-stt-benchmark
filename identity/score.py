"""Score every results file field by field and print markdown tables.

    uv run python identity/score.py

Writes identity_data/scores.csv (one row per clip x vendor x condition) and
identity_data/report.md (the tables printed below).
"""

import csv
import json
import statistics
import sys
from collections import defaultdict
from pathlib import Path

from tabulate import tabulate

sys.path.insert(0, str(Path(__file__).parent))
from build import DATA, load_items  # noqa: E402
from textnorm import check_address, check_dob, check_email, check_name, check_spelled  # noqa: E402

VENDOR_ORDER = ["assemblyai", "deepgram", "speechmatics", "gemini", "whisper"]
COND_ORDER = ["none", "generic", "record"]


def score_row(item: dict, transcript: str) -> dict:
    exp = item["expected"]
    f = {"flow_spelled": "spelled", "flow_dob": "dob"}.get(item["field"], item["field"])
    if f == "name":
        g, s = check_name(transcript, exp["given"]), check_name(transcript, exp["surname"])
        row = {"ok": s["lenient"], "ok_strict": s["strict"], "given_ok": g["lenient"], "near_miss": s["near_miss"]}
        if item.get("impostor"):
            # did the transcript produce the name ON RECORD even though the caller said a different one?
            row["record_leak"] = check_name(transcript, item["record"]["surname"])["lenient"]
        return row
    if f == "spelled":
        r = check_spelled(transcript, exp["surname"])
        return {"ok": r["lenient"], "ok_strict": r["strict"]}
    if f == "email":
        r = check_email(transcript, exp["email"])
        return {"ok": r["lenient"], "ok_strict": r["strict"]}
    if f == "dob":
        r = check_dob(transcript, exp["dob"])
        return {"ok": r["lenient"], "ok_strict": r["strict"], "us_order": r["us_order"], "parsed": r["parsed"]}
    r = check_address(transcript, exp)
    return {"ok": r["all"], **{f"addr_{k}": v for k, v in r.items() if k != "all"}}


def pct(xs: list[bool]) -> str:
    return f"{sum(xs)}/{len(xs)} ({100 * sum(xs) / len(xs):.0f}%)" if xs else "-"


def main() -> None:
    items = load_items()
    rows, no_result, real_rows = [], defaultdict(int), []
    for path in sorted((DATA / "results").glob("*.jsonl")):
        latest = {r["id"]: r for r in map(json.loads, path.open(encoding="utf-8"))}  # reruns override
        for r in latest.values():
            it = items[r["id"]]
            if r["transcript"] is None or r["error"]:
                if it["set"] == "synthetic" and not it["field"].startswith("flow_"):
                    no_result[(r["vendor"], r["condition"])] += 1  # connection/timeout: not accuracy
                continue
            if it["set"] == "real":
                if r["condition"] == "none":
                    real_rows.append((it, r["vendor"], score_row(it, r["transcript"])["ok"]))
                continue  # real clips: paired table below
            if it["field"].startswith("flow_"):
                continue  # full-flow clips are scored by matcher.py
            s = score_row(it, r["transcript"])
            rows.append({"id": r["id"], "field": it["field"], "vendor": r["vendor"], "condition": r["condition"],
                         "voice": it["voice"], "noise": it["noise"], "impostor": it.get("impostor", False),
                         "homophone": it.get("homophone", False), "ttfs": r["ttfs"], "error": r["error"],
                         "transcript": r["transcript"], "script": it["script"], **s})
    if not rows:
        print("no results yet")
        return

    with (DATA / "scores.csv").open("w", newline="", encoding="utf-8") as f:
        keys = sorted({k for r in rows for k in r}, key=lambda k: list(rows[0]).index(k) if k in rows[0] else 99)
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        w.writerows(rows)

    groups = defaultdict(list)
    for r in rows:
        groups[(r["vendor"], r["condition"])].append(r)
    for k in no_result:
        groups.setdefault(k, [])
    order = sorted(groups, key=lambda k: (VENDOR_ORDER.index(k[0]), COND_ORDER.index(k[1])))
    out = []

    # 1. Field accuracy (genuine callers only; impostors are scored separately)
    t = []
    for k in order:
        g = [r for r in groups[k] if not r["impostor"]]
        by = lambda f, key="ok", g=g: [r[key] for r in g if r["field"] == f and key in r]  # noqa: E731
        lat = [r["ttfs"] for r in g if r["ttfs"] is not None]
        close = [r["ok"] or float(r["near_miss"]) >= 0.8 for r in g if r["field"] == "name"]
        t.append([*k, pct(by("name")), pct(close), pct(by("spelled")), pct(by("email", "ok_strict")), pct(by("email")),
                  pct(by("dob", "ok_strict")), pct(by("dob", "us_order")), pct(by("address")), f"{statistics.median(lat) * 1000:.0f}" if lat else "-",
                  no_result[k]])
    out.append("## Field accuracy (genuine callers)\n\n" + tabulate(
        t, ["vendor", "hints", "surname exact", "surname close (>=80% similar)", "spelled surname", "email as written", "email recoverable",
            "DOB (AU order)", "DOB came out US order", "full address", "median ms to final", "no result (connection)"], tablefmt="github"))

    # 2. Address components (where addresses break)
    comps = ["addr_unit", "addr_number", "addr_unit_order", "addr_street", "addr_type", "addr_suburb", "addr_state", "addr_postcode"]
    t = [[*k, *(pct([r[c] for r in groups[k] if r["field"] == "address" and c in r]) for c in comps)] for k in order]
    out.append("## Address components\n\n" + tabulate(t, ["vendor", "hints", *(c[5:] for c in comps)], tablefmt="github"))

    # 3. Impostors: does hinting with the record make the system "hear" the record?
    t = []
    for k in order:
        imp = [r for r in groups[k] if r["impostor"] and not r["homophone"]]
        hom = [r for r in groups[k] if r["impostor"] and r["homophone"]]
        t.append([*k, pct([r.get("record_leak", False) for r in imp]), pct([r["ok"] for r in imp]),
                  pct([float(r["near_miss"]) >= 0.8 for r in imp]),
                  pct([r.get("record_leak", False) for r in hom])])
    out.append("## Impostor callers (said a different surname from the record)\n\n"
               "`record surname appeared` = transcript contains the name on file, not the name spoken.\n\n" + tabulate(
                   t, ["vendor", "hints", "record surname appeared (similar names)", "spoken surname correct",
                       "spoken surname close (>=80%)", "record surname appeared (homophones)"], tablefmt="github"))

    # 4. By accent (no hints, all fields)
    voices = sorted({r["voice"] for r in rows})
    t = [[v.replace("Neural", "").replace("Multilingual", ""),
          *(pct([r["ok"] for r in groups.get((vn, "none"), []) if r["voice"] == v and not r["impostor"]]) for vn in VENDOR_ORDER)]
         for v in voices]
    out.append("## By voice (no hints)\n\n" + tabulate(t, ["voice", *VENDOR_ORDER], tablefmt="github"))

    out.append(paired_table(items, real_rows))

    report = "\n\n".join(out) + "\n"
    (DATA / "report.md").write_text(report, encoding="utf-8")
    print(report)


def paired_table(items: dict, real_rows: list) -> str:
    """Same script, same vendor, no hints: real voice vs synthetic voice, field by field."""
    if not real_rows:
        return "## Real voice vs synthetic voice\n\nNo real-voice results yet."
    synth = {}
    for path in (DATA / "results").glob("*__none.jsonl"):
        for r in map(json.loads, path.open(encoding="utf-8")):
            it = items.get(r["id"])
            if it and it["set"] == "synthetic" and r["transcript"] is not None and not r["error"]:
                synth[(r["vendor"], r["id"])] = score_row(it, r["transcript"])["ok"]
    field_of = {"flow_spelled": "spelled", "flow_dob": "dob"}
    t = []
    for v in VENDOR_ORDER:
        cells = [v]
        for f in ("name", "spelled", "email", "dob", "address"):
            pairs = [(synth[(v, it["pair"])], ok) for it, vv, ok in real_rows
                     if vv == v and field_of.get(it["field"], it["field"]) == f and (v, it["pair"]) in synth]
            if pairs:
                s, rl = sum(p[0] for p in pairs), sum(p[1] for p in pairs)
                cells.append(f"{100 * s / len(pairs):.0f}% -> {100 * rl / len(pairs):.0f}% (n={len(pairs)})")
            else:
                cells.append("-")
        t.append(cells)
    return ("## Real voice vs synthetic voice (same lines, no hints)\n\nEach cell: synthetic accuracy -> real "
            "accuracy on the identical script. Names include wrong-party lines (scored on the name spoken).\n\n"
            + tabulate(t, ["vendor", "surname", "spelled surname", "email", "date of birth", "full address"],
                       tablefmt="github"))


if __name__ == "__main__":
    main()
