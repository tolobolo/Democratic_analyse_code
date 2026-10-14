"""Majority Judgment vs plurality voting on simulated ballots.

Every voter picks one favourite party (Excellent) and gives every other party a
random grade from Very Good down to Reject (random.choice, no hidden logic).

Voters are spread over Norway's 19 Storting districts by population. Each district
gives up one mandate (150 stay in the districts), and every system is counted per
district so that all three hand out the 169 mandates:
  Majority Judgment  district ranking by median grade, mandates by --split
  Plurality          district ranking by votes (the favourite), mandates by --split
  Storting (Norway)  modified Sainte-Lague (1.4, 3, 5, 7, ...) per district, real seat counts

The 19 remaining mandates ("Utjevning"):
  --sperregrense     (default) a national pool counted on national votes, open only
                     to parties with at least 4% of the national vote
  --no-sperregrense  a 20th district with its own simulated voters, counted like the rest

--split 1       winner takes all of the district's mandates (default)
--split 2,1,1   half to the winner, a quarter each to second and third
--split 3,2,1   most / ok / least to the top three, nothing to the rest

Run:  python election.py --voters 10000 --split 2,1,1 --no-sperregrense
      python election.py --candidates Ap Hoyre Frp MDG
"""
from __future__ import annotations

import argparse
import random
from collections import Counter
from functools import cmp_to_key
from dataclasses import dataclass
from enum import Enum


class Grade(Enum):
    """Grades have no numeric value. Their order is the declaration order, best first."""

    EXCELLENT = "Excellent"
    VERY_GOOD = "Very Good"
    GOOD = "Good"
    ACCEPTABLE = "Acceptable"
    POOR = "Poor"
    REJECT = "Reject"

    @property
    def label(self) -> str:
        return self.value

    def beats(self, other: "Grade") -> bool:
        return GRADES_DESC.index(self) < GRADES_DESC.index(other)

    def loses_to(self, other: "Grade") -> bool:
        return GRADES_DESC.index(self) > GRADES_DESC.index(other)


GRADES_DESC = list(Grade)  # best to worst


def sort_best_first(grades: list[Grade]) -> list[Grade]:
    return sorted(grades, key=GRADES_DESC.index)

Ballot = dict[str, Grade]


# ---------------------------------------------------------------- shared helpers

def grades_for(ballots: list[Ballot], candidate: str) -> list[Grade]:
    """All N grades for a candidate. A missing grade counts as REJECT."""
    return [b.get(candidate, Grade.REJECT) for b in ballots]


# ---------------------------------------------------------------- majority judgment

def majority_grade(grades: list[Grade]) -> Grade:
    """Sort best to worst and take the middle grade.

    Even count: if the two middle grades are the same, that is the median.
    Otherwise remove the worst grade and try again.
    """
    if not grades:
        raise ValueError("no grades")
    ordered = sort_best_first(grades)
    while True:
        n = len(ordered)
        if n % 2 == 1:
            return ordered[n // 2]
        if ordered[n // 2 - 1] == ordered[n // 2]:
            return ordered[n // 2]
        ordered.pop()  # drop the worst grade, try again


@dataclass
class MJResult:
    name: str
    counts: dict[Grade, int]
    majority: Grade
    p: float  # share strictly above the majority grade
    q: float  # share strictly below the majority grade
    _grades: list[Grade]

    def gauge(self) -> tuple[Grade, float, float]:
        return self.majority, self.p, self.q


def compare(grades_a: list[Grade], grades_b: list[Grade]) -> int:
    """Negative if A ranks above B, positive if B ranks above A, 0 if a full tie.

    1. Higher majority grade wins.
    2. Same majority grade: more grades above the median wins.
    3. Still tied: remove the worst grade from both and start over.
    """
    a = sort_best_first(grades_a)
    b = sort_best_first(grades_b)
    while a and b:
        ma, mb = majority_grade(a), majority_grade(b)
        if ma != mb:
            return -1 if ma.beats(mb) else 1
        above_a = sum(1 for g in a if g.beats(ma))
        above_b = sum(1 for g in b if g.beats(mb))
        if above_a != above_b:
            return -1 if above_a > above_b else 1
        a.pop()  # remove the worst grade
        b.pop()
    return 0


def mj_rank(ballots: list[Ballot], candidates: list[str]) -> list[MJResult]:
    """Candidates ranked best first by Majority Judgment."""
    results = []
    for name in candidates:
        grades = grades_for(ballots, name)
        maj = majority_grade(grades)
        n = len(grades)
        counts = Counter(grades)
        results.append(
            MJResult(
                name=name,
                counts={g: counts.get(g, 0) for g in GRADES_DESC},
                majority=maj,
                p=sum(1 for g in grades if g.beats(maj)) / n,
                q=sum(1 for g in grades if g.loses_to(maj)) / n,
                _grades=grades,
            )
        )
    results.sort(key=cmp_to_key(lambda x, y: compare(x._grades, y._grades)))
    return results


def mj_winner(ballots: list[Ballot], candidates: list[str]) -> str:
    return mj_rank(ballots, candidates)[0].name


# ---------------------------------------------------------------- plurality

@dataclass
class PluralityResult:
    counts: dict[str, int]
    winners: list[str]  # more than one name means a tie
    ambiguous: int  # ballots where several candidates shared the top grade


def vote_from_ballot(ballot: Ballot, candidates: list[str], rng: random.Random) -> tuple[str, bool]:
    """Top-graded candidate. Random among ties, flagged as ambiguous."""
    best = sort_best_first([ballot.get(c, Grade.REJECT) for c in candidates])[0]
    top = [c for c in candidates if ballot.get(c, Grade.REJECT) == best]
    return rng.choice(top), len(top) > 1


def collect_votes(ballots: list[Ballot], candidates: list[str], rng: random.Random) -> tuple[dict[str, int], int]:
    """One vote per ballot for its top-graded party. Shared by plurality and Storting."""
    counts = {c: 0 for c in candidates}
    ambiguous = 0
    for b in ballots:
        choice, amb = vote_from_ballot(b, candidates, rng)
        counts[choice] += 1
        ambiguous += amb
    return counts, ambiguous


def plurality_from_votes(counts: dict[str, int], ambiguous: int = 0) -> PluralityResult:
    top = max(counts.values())
    winners = [c for c in counts if counts[c] == top]
    return PluralityResult(counts=counts, winners=winners, ambiguous=ambiguous)


def plurality_tally(ballots: list[Ballot], candidates: list[str], rng: random.Random) -> PluralityResult:
    counts, ambiguous = collect_votes(ballots, candidates, rng)
    return plurality_from_votes(counts, ambiguous)


# ---------------------------------------------------------------- storting (norway)

STORTING_SEATS = 169
FIRST_DIVISOR = 1.4


@dataclass
class SeatResult:
    votes: dict[str, int]
    seats: dict[str, int]
    largest: list[str]  # party or parties with the most seats
    ambiguous: int


def sainte_lague(votes: dict[str, int], seats: int, first_divisor: float = FIRST_DIVISOR) -> dict[str, int]:
    """Modified Sainte-Lague as used for the Norwegian Storting.

    Each party's votes are divided by first_divisor (1.4), then 3, 5, 7, ...
    Seats go one at a time to the party with the highest current quotient.
    Not D'Hondt (which divides by 1, 2, 3, ...). The 1.4 makes the first seat
    harder to win for small parties.
    """
    won = {p: 0 for p in votes}
    for _ in range(seats):
        quotient = {p: votes[p] / (first_divisor if won[p] == 0 else 2 * won[p] + 1) for p in votes}
        best = max(quotient.values())
        tied = [p for p in votes if quotient[p] == best]
        winner = min(tied, key=lambda p: (-votes[p], p))  # tie: more votes, then name
        won[winner] += 1
    return won


def storting_from_votes(counts: dict[str, int], ambiguous: int = 0, seats: int = STORTING_SEATS) -> SeatResult:
    won = sainte_lague(counts, seats)
    top = max(won.values())
    largest = [p for p in counts if won[p] == top]
    return SeatResult(votes=counts, seats=won, largest=largest, ambiguous=ambiguous)


def storting_tally(ballots: list[Ballot], candidates: list[str], rng: random.Random,
                   seats: int = STORTING_SEATS) -> SeatResult:
    counts, ambiguous = collect_votes(ballots, candidates, rng)
    return storting_from_votes(counts, ambiguous, seats)


# ---------------------------------------------------------------- simulation

def random_ballot(candidates: list[str], rng: random.Random) -> Ballot:
    """One random favourite gets Excellent; every other party a random grade below that.

    The favourite is the voter's single vote under plurality and Storting.
    No numbers, no hidden logic: real people do not follow a formula.
    """
    fav = rng.choice(candidates)
    others = GRADES_DESC[1:]  # Very Good ... Reject
    return {c: Grade.EXCELLENT if c == fav else rng.choice(others) for c in candidates}


def simulate(candidates: list[str], n_voters: int, rng: random.Random) -> list[Ballot]:
    return [random_ballot(candidates, rng) for _ in range(n_voters)]


# ---------------------------------------------------------------- districts

# Official 2025 Storting seat counts and population 1.1.2024 (regjeringen.no,
# "Mandatfordeling 2024, med virkning for stortingsvalget i 2025").
DISTRICTS: dict[str, tuple[int, int]] = {  # name: (mandates, population)
    "Østfold": (9, 312152),
    "Akershus": (20, 728803),
    "Oslo": (20, 717710),
    "Hedmark": (7, 202048),
    "Oppland": (6, 174256),
    "Buskerud": (8, 269819),
    "Vestfold": (7, 256432),
    "Telemark": (6, 177093),
    "Aust-Agder": (4, 122968),
    "Vest-Agder": (6, 196882),
    "Rogaland": (14, 499417),
    "Hordaland": (16, 541875),
    "Sogn og Fjordane": (4, 109424),
    "Møre og Romsdal": (8, 270624),
    "Sør-Trøndelag": (10, 347516),
    "Nord-Trøndelag": (5, 135440),
    "Nordland": (9, 243081),
    "Troms": (6, 169610),
    "Finnmark": (4, 75053),
}
assert sum(m for m, _ in DISTRICTS.values()) == STORTING_SEATS

LEVELLING = "Utjevning"  # the 19 mandates each district gives up, one each
LEVELLING_SEATS = len(DISTRICTS)
DISTRICT_SEATS = STORTING_SEATS - LEVELLING_SEATS  # 150
THRESHOLD = 0.04  # sperregrense: share of national votes needed for the levelling pool


def seats_in(district: str) -> int:
    """Mandates counted inside a district: the official number minus one, or 19 for Utjevning."""
    return LEVELLING_SEATS if district == LEVELLING else DISTRICTS[district][0] - 1


def split_voters(n_voters: int, sperregrense: bool = True) -> dict[str, int]:
    """Voters per district in proportion to population; rounding leftover to the largest district.

    With sperregrense off, Utjevning is a real district and gets voters too, at the same
    voters-per-mandate rate as the rest.
    """
    pops = {d: p for d, (_, p) in DISTRICTS.items()}
    if not sperregrense:
        pops[LEVELLING] = sum(pops.values()) * LEVELLING_SEATS / DISTRICT_SEATS
    total_pop = sum(pops.values())
    out = {d: round(n_voters * p / total_pop) for d, p in pops.items()}
    biggest = max(pops, key=pops.get)
    out[biggest] += n_voters - sum(out.values())
    return out


def simulate_districts(candidates: list[str], n_voters: int, rng: random.Random,
                       sperregrense: bool = True) -> dict[str, list[Ballot]]:
    """One ballot list per district (plus Utjevning when sperregrense is off)."""
    return {d: simulate(candidates, n, rng) for d, n in split_voters(n_voters, sperregrense).items()}


def parse_split(text: str) -> list[int]:
    """'1' -> [1], '2,1,1' -> [2, 1, 1]. Positive integers only."""
    try:
        weights = [int(x) for x in text.replace(" ", "").split(",") if x != ""]
    except ValueError:
        raise ValueError(f"split must be numbers separated by commas, got '{text}'")
    if not weights or any(w <= 0 for w in weights):
        raise ValueError("split weights must be positive, e.g. 1 or 2,1,1")
    return weights


def split_seats(seats: int, ranking: list[str], weights: list[int]) -> dict[str, int]:
    """Share a district's mandates among the top parties in ranking order.

    weights [1] = winner takes all. [2, 1, 1] = half / quarter / quarter.
    Largest-remainder rounding, ties in remainder go to the better-ranked party.
    Parties below the top len(weights) get nothing.
    """
    out = {p: 0 for p in ranking}
    top = ranking[: len(weights)]
    w = weights[: len(top)]
    total = sum(w)
    exact = [seats * wi / total for wi in w]
    floors = [int(x) for x in exact]
    leftover = seats - sum(floors)
    order = sorted(range(len(top)), key=lambda i: (-(exact[i] - floors[i]), i))
    for i in order[:leftover]:
        floors[i] += 1
    for p, n in zip(top, floors):
        out[p] = n
    return out


@dataclass
class SeatsResult:
    seats: dict[str, int]  # national total per party
    by_district: dict[str, dict[str, int]]
    largest: list[str]  # party or parties with the most seats nationally


def _finish(by_district: dict[str, dict[str, int]], candidates: list[str]) -> SeatsResult:
    seats = {c: sum(d[c] for d in by_district.values()) for c in candidates}
    top = max(seats.values())
    return SeatsResult(seats=seats, by_district=by_district,
                       largest=[c for c in candidates if seats[c] == top])


def mj_seats(district_ballots: dict[str, list[Ballot]], candidates: list[str],
             weights: list[int]) -> SeatsResult:
    """Majority Judgment per district; the district's mandates split by weights."""
    by_district = {}
    for d, ballots in district_ballots.items():
        ranking = [r.name for r in mj_rank(ballots, candidates)]
        by_district[d] = split_seats(seats_in(d), ranking, weights)
    return _finish(by_district, candidates)


def plurality_seats(district_votes: dict[str, dict[str, int]], candidates: list[str],
                    weights: list[int]) -> SeatsResult:
    """Most votes first per district; the district's mandates split by weights."""
    by_district = {}
    for d, votes in district_votes.items():
        ranking = sorted(candidates, key=lambda c: -votes[c])  # stable: ties keep input order
        by_district[d] = split_seats(seats_in(d), ranking, weights)
    return _finish(by_district, candidates)


def storting_seats(district_votes: dict[str, dict[str, int]], candidates: list[str]) -> SeatsResult:
    """Norway's real district count: modified Sainte-Lague with each district's seat number."""
    by_district = {d: sainte_lague(votes, seats_in(d)) for d, votes in district_votes.items()}
    return _finish(by_district, candidates)


@dataclass
class Election:
    district_ballots: dict[str, list[Ballot]]
    district_votes: dict[str, dict[str, int]]
    ambiguous: int
    mj_national: list[MJResult]  # grade table over all ballots, ranked
    votes: dict[str, int]  # national vote totals
    mj: SeatsResult
    plurality: SeatsResult
    storting: SeatsResult
    sperregrense: bool
    eligible: list[str]  # parties allowed into the levelling pool (all parties when off)

    @property
    def districts(self) -> list[str]:
        """Every place mandates are counted, Utjevning last."""
        return list(DISTRICTS) + [LEVELLING]

    @property
    def n_voters(self) -> int:
        return sum(len(b) for b in self.district_ballots.values())

    def picks(self) -> dict[str, list[str]]:
        return {"Majority Judgment": self.mj.largest, "Plurality": self.plurality.largest,
                "Storting": self.storting.largest}

    def agree(self) -> str | None:
        """The party every system makes largest, or None."""
        first = self.mj.largest
        return first[0] if len(first) == 1 and all(v == first for v in self.picks().values()) else None


def levelling_pool(e_votes: dict[str, int], mj_national: list[MJResult], candidates: list[str],
                   weights: list[int], eligible: list[str]) -> tuple[dict[str, int], dict[str, int], dict[str, int]]:
    """The 19 Utjevning mandates counted on national votes, only among eligible parties.

    Returns (mj, plurality, storting) seat dicts over all candidates.
    """
    mj_ranking = [r.name for r in mj_national if r.name in eligible]
    pl_ranking = sorted(eligible, key=lambda c: -e_votes[c])
    zero = {c: 0 for c in candidates}
    mj = {**zero, **split_seats(LEVELLING_SEATS, mj_ranking, weights)}
    pl = {**zero, **split_seats(LEVELLING_SEATS, pl_ranking, weights)}
    st = {**zero, **sainte_lague({c: e_votes[c] for c in eligible}, LEVELLING_SEATS)}
    return mj, pl, st


def count_all(district_ballots: dict[str, list[Ballot]], candidates: list[str],
              rng: random.Random, weights: list[int], sperregrense: bool = True) -> Election:
    """Count all three systems. district_ballots must include Utjevning iff sperregrense is off."""
    if sperregrense == (LEVELLING in district_ballots):
        raise ValueError("Utjevning ballots must be present exactly when sperregrense is off")
    district_votes, ambiguous = {}, 0
    for d, ballots in district_ballots.items():
        district_votes[d], amb = collect_votes(ballots, candidates, rng)
        ambiguous += amb
    all_ballots = [b for ballots in district_ballots.values() for b in ballots]
    votes = {c: sum(v[c] for v in district_votes.values()) for c in candidates}
    mj_national = mj_rank(all_ballots, candidates)

    mj = mj_seats(district_ballots, candidates, weights)
    pl = plurality_seats(district_votes, candidates, weights)
    st = storting_seats(district_votes, candidates)

    eligible = list(candidates)
    if sperregrense:
        total = sum(votes.values()) or 1
        eligible = [c for c in candidates if votes[c] / total >= THRESHOLD] or list(candidates)
        pool = levelling_pool(votes, mj_national, candidates, weights, eligible)
        for res, seats in zip((mj, pl, st), pool):
            res.by_district[LEVELLING] = seats
        mj, pl, st = (_finish(r.by_district, candidates) for r in (mj, pl, st))

    return Election(
        district_ballots=district_ballots,
        district_votes=district_votes,
        ambiguous=ambiguous,
        mj_national=mj_national,
        votes=votes,
        mj=mj,
        plurality=pl,
        storting=st,
        sperregrense=sperregrense,
        eligible=eligible,
    )


# ---------------------------------------------------------------- cli

def print_report(e: Election, candidates: list[str], weights: list[int]) -> None:
    n = e.n_voters
    width = max(len(c) for c in candidates)
    S = STORTING_SEATS

    mode = (f"sperregrense ON: {LEVELLING_SEATS} Utjevning mandates on national votes, parties >= {THRESHOLD:.0%}: "
            f"{', '.join(e.eligible)}" if e.sperregrense
            else f"sperregrense OFF: {LEVELLING} is a 20th district with its own {len(e.district_ballots[LEVELLING])} voters")
    print(mode)
    print()
    header = " ".join(f"{g.label:>10}" for g in GRADES_DESC)
    print(f"Majority Judgment  ({n} voters, {len(DISTRICTS)} districts + {LEVELLING}, split {','.join(map(str, weights))})")
    print(f"{'':{width}} {header}   majority      p      q   seats")
    for r in e.mj_national:
        row = " ".join(f"{r.counts[g]:>10}" for g in GRADES_DESC)
        print(f"{r.name:{width}} {row}   {r.majority.label:<10} {r.p:5.2f}  {r.q:5.2f}   {e.mj.seats[r.name]:>3}")
    print(f"Largest party: {' = '.join(e.mj.largest)}")
    print()

    print(f"Plurality  (split {','.join(map(str, weights))})")
    for c in sorted(candidates, key=lambda c: -e.plurality.seats[c]):
        print(f"{c:{width}} {e.votes[c]:>7} votes ({e.votes[c] / n:5.1%})   {e.plurality.seats[c]:>3} seats ({e.plurality.seats[c] / S:5.1%})")
    print(f"Largest party: {' = '.join(e.plurality.largest)}  (ambiguous ballots: {e.ambiguous})")
    print()

    print("Storting (Norway, modified Sainte-Lague per district, same votes)")
    for c in sorted(candidates, key=lambda c: -e.storting.seats[c]):
        print(f"{c:{width}} {e.votes[c]:>7} votes ({e.votes[c] / n:5.1%})   {e.storting.seats[c]:>3} seats ({e.storting.seats[c] / S:5.1%})")
    print(f"Largest party: {' = '.join(e.storting.largest)}")
    print()

    dw = max(len(d) for d in e.districts)
    print(f"{'District':{dw}}  seats  voters  {'MJ winner':{width}}  {'Plurality':{width}}  {'Storting':{width}}")
    for d in e.districts:
        def top(res):
            best = max(res.by_district[d].values())
            return " = ".join(c for c in candidates if res.by_district[d][c] == best)
        voters = str(len(e.district_ballots[d])) if d in e.district_ballots else "national"
        print(f"{d:{dw}}  {seats_in(d):>5}  {voters:>8}  "
              f"{top(e.mj):{width}}  {top(e.plurality):{width}}  {top(e.storting):{width}}")
    print()

    winner = e.agree()
    if winner:
        print(f"All three systems make {winner} the largest party")
    else:
        print("Systems DIFFER: " + ", ".join(f"{k} -> {' = '.join(v)}" for k, v in e.picks().items()))


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--candidates", nargs="+", default=["Far-left", "Center", "Far-right"])
    ap.add_argument("--voters", type=int, default=10000)
    ap.add_argument("--sperregrense", action=argparse.BooleanOptionalAction, default=True,
                    help="on: 19 Utjevning mandates by national votes for parties >= 4%%; "
                         "off: Utjevning is a 20th district with its own voters")
    ap.add_argument("--split", default="1",
                    help="mandate split among top parties per district: 1 (winner takes all), 2,1,1 or 3,2,1")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args(argv)

    weights = parse_split(args.split)
    rng = random.Random(args.seed)
    district_ballots = simulate_districts(args.candidates, args.voters, rng, args.sperregrense)
    e = count_all(district_ballots, args.candidates, rng, weights, args.sperregrense)
    print(f"seed {args.seed}")
    print_report(e, args.candidates, weights)


if __name__ == "__main__":
    main()
