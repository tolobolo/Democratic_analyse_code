"""Browser UI for the voting simulator.

Run:  python3 ui.py
Opens http://127.0.0.1:8000 in your browser. Ctrl+C to stop.
Set UI_PORT=8765 to use another port.
All voting logic lives in election.py; this file only renders a form and the results.
"""
from __future__ import annotations

import html
import os
import random
import threading
import webbrowser
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import parse_qs

from election import (
    GRADES_DESC,
    LEVELLING,
    LEVELLING_SEATS,
    STORTING_SEATS,
    THRESHOLD,
    Grade,
    count_all,
    parse_split,
    seats_in,
    simulate_districts,
)

HOST, PORT = "127.0.0.1", int(os.environ.get("UI_PORT", "8000"))

GRADE_COLOUR = {
    Grade.EXCELLENT: "#1b7f3b",
    Grade.VERY_GOOD: "#5cb85c",
    Grade.GOOD: "#a8d08d",
    Grade.ACCEPTABLE: "#f0d66b",
    Grade.POOR: "#f0a04b",
    Grade.REJECT: "#d9534f",
}

DEFAULTS = {
    "candidates": "Far-left\nCenter\nFar-right",
    "voters": "10000",
    "seed": "42",
    "sperregrense": "on",
    "split": "1",
    "split_custom": "",
}

PAGE = """<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<title>Majority Judgment vs Plurality vs Storting</title>
<style>
  body {{ font-family: system-ui, sans-serif; max-width: 1000px; margin: 2rem auto; padding: 0 1rem; color: #222; }}
  h1 {{ font-size: 1.5rem; }}
  form {{ display: grid; grid-template-columns: 1fr 1fr; gap: 1rem 2rem; background: #f6f6f6; padding: 1rem 1.5rem; border-radius: 8px; }}
  fieldset {{ border: 1px solid #ccc; border-radius: 6px; }}
  label {{ display: block; margin: .5rem 0 .2rem; font-weight: 600; }}
  input[type=number], textarea {{ width: 100%; box-sizing: border-box; padding: .3rem; }}
  textarea {{ height: 6rem; font-family: monospace; }}
  input[type=range] {{ width: 80%; vertical-align: middle; }}
  .hint {{ display: block; font-size: .85rem; color: #666; font-weight: normal; margin-bottom: .3rem; }}
  .intro {{ background: #eef3ff; padding: .8rem 1rem; border-radius: 6px; }}
  .run {{ grid-column: 1 / -1; }}
  button {{ font-size: 1.1rem; padding: .5rem 2rem; background: #2563eb; color: white; border: 0; border-radius: 6px; cursor: pointer; }}
  .error {{ background: #fde2e2; border: 1px solid #d9534f; padding: .8rem; border-radius: 6px; margin-top: 1rem; }}
  table {{ border-collapse: collapse; margin: .5rem 0 1rem; width: 100%; }}
  th, td {{ border: 1px solid #ddd; padding: .35rem .6rem; text-align: right; }}
  th:first-child, td:first-child {{ text-align: left; }}
  .bar {{ display: flex; height: 18px; width: 220px; border-radius: 3px; overflow: hidden; }}
  .bar div {{ height: 100%; }}
  .legend span {{ display: inline-block; margin-right: 1rem; }}
  .legend i {{ display: inline-block; width: 12px; height: 12px; margin-right: .3rem; vertical-align: middle; }}
  .verdict {{ font-size: 1.2rem; font-weight: 700; margin: 1rem 0; }}
  .rule {{ font-size: .9rem; color: #555; }}
  .hidden {{ display: none; }}
</style>
</head>
<body>
<h1>Majority Judgment vs Plurality vs Storting</h1>
<p class="intro">The page makes up a random group of voters and spreads them over Norway's 19 Storting districts
by population. Every voter picks one favourite party (Excellent) and gives every other party a random grade from
Very Good down to Reject, with no hidden logic, because real people do not follow a formula. The same ballots
are then counted three ways, district by district, and every system hands out the 169 mandates.
<b>Majority Judgment</b> ranks the parties in each district by median grade.
<b>Plurality</b> ranks them by votes (each voter's favourite).
In both, the district's mandates go to the winner, or are shared by the split rule you choose.
<b>Storting</b> uses Norway's real district count: modified Sainte-Lagu&euml; with the 2025 seat numbers.
Each district gives up one mandate; those 19 are the <b>Utjevning</b> mandates, handled by the sperregrense switch.
Change the settings and press Run to see where the systems agree or disagree.</p>

<form method="post" action="/">
  <fieldset>
    <legend>Rules</legend>
    <label>Sperregrense (4% threshold)</label>
    <span class="hint"><b>On:</b> the 19 Utjevning mandates form a national pool counted on the national vote totals,
    and only parties with at least 4% of the national vote can win from it. This is close to Norway's levelling seats.<br>
    <b>Off:</b> Utjevning becomes a 20th district with its own freshly made voters and 19 mandates, counted exactly
    like every other district. No threshold anywhere.</span>
    <label><input type="radio" name="sperregrense" value="on" {sperre_on}> On</label>
    <label><input type="radio" name="sperregrense" value="off" {sperre_off}> Off</label>

    <label>Mandate split per district</label>
    <span class="hint">How a district's mandates are shared under Majority Judgment and plurality.
    The Storting count always uses Norway's own rule and ignores this.</span>
    <label><input type="radio" name="split" value="1" {split_1}> Winner takes all</label>
    <label><input type="radio" name="split" value="2,1,1" {split_211}> Half to the winner, a quarter each to 2nd and 3rd</label>
    <label><input type="radio" name="split" value="3,2,1" {split_321}> Most / ok / least to the top three (3:2:1)</label>
    <label><input type="radio" name="split" value="custom" {split_custom_checked}> Custom weights:
      <input type="text" name="split_custom" value="{split_custom}" placeholder="e.g. 4,2,1,1" style="width:8rem"></label>
    <span class="hint">Weights in ranking order, parties beyond the list get nothing. Rounded by largest remainder.</span>
  </fieldset>

  <fieldset>
    <legend>Candidates and voters</legend>
    <label>Parties / candidates</label>
    <span class="hint">One name per line.</span>
    <textarea name="candidates">{candidates}</textarea>
    <label>Number of voters</label>
    <span class="hint">How many random voters to create, spread over the districts by population
    (Oslo gets about ten times as many as Finnmark). More voters give steadier results.</span>
    <input type="number" name="voters" min="1" value="{voters}">
    <label>Random seed</label>
    <span class="hint">Starting number for the random generator. Same seed = exactly the same voters every run,
    so you can change one setting and see what it alone did. Change the seed to get a fresh electorate.</span>
    <input type="number" name="seed" value="{seed}">
  </fieldset>

  <div class="run"><button type="submit">Run election</button></div>
</form>

{result}

</body>
</html>
"""


def parse_candidates(text: str) -> list[str]:
    names = []
    for line in text.splitlines():
        name = line.strip()
        if not name:
            continue
        if name in names:
            raise ValueError(f"duplicate candidate '{name}'")
        names.append(name)
    if len(names) < 2:
        raise ValueError("enter at least two candidates")
    return names


def run(form: dict[str, str]) -> str:
    voters = int(form["voters"])
    if voters < 1:
        raise ValueError("number of voters must be at least 1")
    seed = int(form["seed"])
    rng = random.Random(seed)
    names = parse_candidates(form["candidates"])
    weights = parse_split(form["split_custom"] if form["split"] == "custom" else form["split"])
    sperre = form["sperregrense"] != "off"
    e = count_all(simulate_districts(names, voters, rng, sperre), names, rng, weights, sperre)
    n = e.n_voters
    S = STORTING_SEATS
    split_txt = ",".join(map(str, weights))
    if sperre:
        mode = (f"Sperregrense on: the {LEVELLING_SEATS} {LEVELLING} mandates are a national pool for parties with at least "
                f"{THRESHOLD:.0%} of the vote: {html.escape(', '.join(e.eligible))}.")
    else:
        mode = (f"Sperregrense off: {LEVELLING} is a 20th district with {len(e.district_ballots[LEVELLING])} voters "
                f"and {LEVELLING_SEATS} mandates.")
    setup = f"{n} voters over 19 districts, seed {seed}, split {split_txt}. {mode}"

    out = [f'<p class="rule">{setup}</p>']

    winner = e.agree()
    if winner:
        out.append(f'<p class="verdict">All three systems make {html.escape(winner)} the largest party.</p>')
    else:
        out.append('<p class="verdict">' + ", ".join(
            f"{k}: {html.escape(' = '.join(v))}" for k, v in e.picks().items()) + ".</p>")

    def seat_bar(seats: int, top: int, colour: str) -> str:
        return f'<div class="bar"><div style="width:{100 * seats / (top or 1):.2f}%;background:{colour}"></div></div>'

    out.append("<h2>Majority Judgment</h2>")
    out.append('<p class="rule">In each district the grades of every party are sorted best to worst and the middle one '
               "is the <b>median</b>. Highest median ranks first. If two parties share a median, the one with more "
               "grades above the median ranks higher. If still tied, the worst grade is removed from both and it "
               f"starts over. The district's mandates then go by the split rule ({split_txt}). The table shows the "
               "grades over all voters, the national median, <b>p</b> (share above the median), <b>q</b> (share below) "
               "and the total seats won.</p>")
    out.append('<p class="legend">' + "".join(
        f'<span><i style="background:{GRADE_COLOUR[g]}"></i>{g.label}</span>' for g in GRADES_DESC) + "</p>")
    head = "".join(f"<th>{g.label}</th>" for g in GRADES_DESC)
    out.append(f"<table><tr><th>Party</th><th>Grades</th>{head}<th>Median</th><th>p</th><th>q</th>"
               "<th>Seats</th><th>Seat share</th><th></th></tr>")
    top_mj = max(e.mj.seats.values())
    for r in sorted(e.mj_national, key=lambda r: -e.mj.seats[r.name]):
        bar = "".join(
            f'<div style="width:{100 * r.counts[g] / n:.2f}%;background:{GRADE_COLOUR[g]}" title="{g.label}: {r.counts[g]}"></div>'
            for g in GRADES_DESC)
        cells = "".join(f"<td>{r.counts[g]}</td>" for g in GRADES_DESC)
        sseats = e.mj.seats[r.name]
        out.append(f"<tr><td>{html.escape(r.name)}</td><td><div class=\"bar\">{bar}</div></td>"
                   f"{cells}<td><b>{r.majority.label}</b></td><td>{r.p:.2f}</td><td>{r.q:.2f}</td>"
                   f"<td><b>{sseats}</b></td><td>{sseats / S:.1%}</td><td>{seat_bar(sseats, top_mj, '#1b7f3b')}</td></tr>")
    out.append("</table>")
    out.append(f'<p>Largest party: <b>{html.escape(" = ".join(e.mj.largest))}</b></p>')

    out.append("<h2>Plurality</h2>")
    out.append('<p class="rule">The same ballots, counted the old way: each voter gets one vote for the party they '
               "graded highest. In each district the parties are ranked by votes and the mandates go by the split "
               f"rule ({split_txt}). When a voter gave the same top grade to several parties one is picked at random. "
               f"That happened on {e.ambiguous} of {n} ballots.</p>")
    out.append("<table><tr><th>Party</th><th>Votes</th><th>Vote share</th><th>Seats</th><th>Seat share</th><th></th></tr>")
    top_pl = max(e.plurality.seats.values())
    for c in sorted(names, key=lambda c: -e.plurality.seats[c]):
        sseats = e.plurality.seats[c]
        out.append(f"<tr><td>{html.escape(c)}</td><td>{e.votes[c]}</td><td>{e.votes[c] / n:.1%}</td>"
                   f"<td><b>{sseats}</b></td><td>{sseats / S:.1%}</td><td>{seat_bar(sseats, top_pl, '#2563eb')}</td></tr>")
    out.append("</table>")
    out.append(f'<p>Largest party: <b>{html.escape(" = ".join(e.plurality.largest))}</b></p>')

    out.append("<h2>Storting (Norwegian system)</h2>")
    out.append('<p class="rule">The same votes as plurality, counted district by district the way Norway does it: '
               "each party's votes are divided by 1.4, then 3, 5, 7, ... and the district's mandates go one by one "
               "to the highest result. This is modified Sainte-Lagu&euml; (not D'Hondt, which divides by 1, 2, 3). "
               "The 1.4 makes the first seat harder to win for small parties. Uses the official 2025 seat count per "
               "district. The 19 levelling seats and the 4% threshold are not modelled.</p>")
    out.append("<table><tr><th>Party</th><th>Votes</th><th>Vote share</th><th>Seats</th><th>Seat share</th><th></th></tr>")
    top_st = max(e.storting.seats.values())
    for c in sorted(names, key=lambda c: -e.storting.seats[c]):
        sseats = e.storting.seats[c]
        out.append(f"<tr><td>{html.escape(c)}</td><td>{e.votes[c]}</td><td>{e.votes[c] / n:.1%}</td>"
                   f"<td><b>{sseats}</b></td><td>{sseats / S:.1%}</td><td>{seat_bar(sseats, top_st, '#7c3aed')}</td></tr>")
    out.append("</table>")
    out.append(f'<p>Largest party: <b>{html.escape(" = ".join(e.storting.largest))}</b></p>')

    out.append("<h2>Districts</h2>")
    out.append('<p class="rule">Who comes first in each district under each system, with seats won there in brackets. '
               f"Each district counts one mandate less than its official number; the {LEVELLING} row holds those "
               f"{LEVELLING_SEATS}.</p>")
    out.append("<table><tr><th>District</th><th>Mandates</th><th>Voters</th><th>Majority Judgment</th>"
               "<th>Plurality</th><th>Storting</th></tr>")

    def top(res, d):
        best = max(res.by_district[d].values())
        return html.escape(" = ".join(c for c in names if res.by_district[d][c] == best)) + f" ({best})"

    for d in e.districts:
        voters_txt = str(len(e.district_ballots[d])) if d in e.district_ballots else "national votes"
        out.append(f"<tr><td>{html.escape(d)}</td><td>{seats_in(d)}</td><td>{voters_txt}</td>"
                   f"<td>{top(e.mj, d)}</td><td>{top(e.plurality, d)}</td><td>{top(e.storting, d)}</td></tr>")
    out.append("</table>")
    return "\n".join(out)


def render(form: dict[str, str], result: str = "") -> bytes:
    split = form["split"]
    page = PAGE.format(
        candidates=html.escape(form["candidates"]),
        voters=html.escape(form["voters"]),
        seed=html.escape(form["seed"]),
        sperre_on="checked" if form["sperregrense"] != "off" else "",
        sperre_off="checked" if form["sperregrense"] == "off" else "",
        split_1="checked" if split == "1" else "",
        split_211="checked" if split == "2,1,1" else "",
        split_321="checked" if split == "3,2,1" else "",
        split_custom_checked="checked" if split == "custom" else "",
        split_custom=html.escape(form["split_custom"]),
        result=result,
    )
    return page.encode()


class Handler(BaseHTTPRequestHandler):
    def _send(self, body: bytes) -> None:
        self.send_response(200)
        self.send_header("Content-Type", "text/html; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self) -> None:
        self._send(render(DEFAULTS))

    def do_POST(self) -> None:
        length = int(self.headers.get("Content-Length", 0))
        data = parse_qs(self.rfile.read(length).decode(), keep_blank_values=True)
        form = {k: data.get(k, [v])[0] for k, v in DEFAULTS.items()}
        try:
            result = run(form)
        except ValueError as e:
            result = f'<div class="error">Invalid input: {html.escape(str(e))}</div>'
        self._send(render(form, result))

    def log_message(self, fmt, *args):  # quieter console
        pass


def main() -> None:
    server = ThreadingHTTPServer((HOST, PORT), Handler)
    url = f"http://{HOST}:{PORT}"
    print(f"Voting simulator running at {url}  (Ctrl+C to stop)")
    threading.Timer(0.5, webbrowser.open, args=(url,)).start()
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nstopped")


if __name__ == "__main__":
    main()
