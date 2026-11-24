"""Majority Judgment vs plurality voting on simulated ballots.

Every voter picks one favourite party (Excellent) and gives every other party a
random grade from Very Good down to Reject (random.choice, no hidden logic).

Voters are spread over Norway's 19 Storting districts by population. Each district
gives up one mandate (150 stay in the districts), and every system is counted per
district so that all three hand out the 169 mandates:
  Majority Judgment  district ranking by median grade, mandates by --split
  Storting (Norway)  modified Sainte-Lague (1.4, 3, 5, 7, ...) per district, real seat counts
  Plurality          most votes (the favourite) takes all of the district's mandates

The 19 remaining mandates ("Utjevning") are levelling seats: each party gets its fair
national share (Sainte-Lague on national votes) minus what it won in the districts.
  --sperregrense     (default) only parties with at least 4% of the national vote qualify
  --no-sperregrense  every party qualifies

--split applies to Majority Judgment only. Storting always uses Sainte-Lague and
plurality is always winner takes all.
--split 1       winner takes all of the district's mandates (default)
--split 2,1,1   half to the winner, a quarter each to second and third
--split 3,2,1   most / ok / least to the top three, nothing to the rest
--split norway  Norway's own split: modified Sainte-Lague (1.4, 3, 5, ...) per district on
                supporters = voters who graded the party Good or better, since grades give
                no vote count. Then the only difference between Majority Judgment and the
                Storting count is how voters express themselves, not how mandates are split.

Real voters: --cses YEAR loads Norwegian respondents from the CSES Integrated Module
Dataset (data/cses_imd_csv.zip, years 1997-2017). Their 0-10 like/dislike scores become
grades, their reported vote is the plurality/Storting vote.

Run:  python election.py --voters 10000 --split 2,1,1 --no-sperregrense
      python election.py --candidates Ap Hoyre Frp MDG
      python election.py --cses 2017 --split 2,1,1
"""
from __future__ import annotations

import argparse
import csv
import io
import math
import os
import random
import zipfile
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


GRADE_INDEX = {g: i for i, g in enumerate(GRADES_DESC)}  # 0 = Excellent ... 5 = Reject


def _bucket_majority(counts: list[int]) -> int:
    """Index (in GRADES_DESC) of the majority grade, from per-grade counts, best first.

    Same rule as majority_grade: middle grade; even count with different middles ->
    drop the worst grade and try again. Works on a copy of the counts.
    """
    counts = list(counts)
    n = sum(counts)
    if n == 0:
        raise ValueError("no grades")

    def at(pos: int) -> int:
        acc = 0
        for i, c in enumerate(counts):
            acc += c
            if pos < acc:
                return i
        raise IndexError(pos)

    while True:
        if n % 2 == 1:
            return at(n // 2)
        lo, hi = at(n // 2 - 1), at(n // 2)
        if lo == hi:
            return hi
        for i in range(len(counts) - 1, -1, -1):  # drop the worst grade
            if counts[i]:
                counts[i] -= 1
                break
        n -= 1


def grade_counts(grades: list[Grade]) -> list[int]:
    """Per-grade counts, best first (same order as GRADES_DESC)."""
    counts = [0] * len(GRADES_DESC)
    for g in grades:
        counts[GRADE_INDEX[g]] += 1
    return counts


def mj_key(grades: list[Grade], counts: list[int] | None = None) -> list[tuple[int, int]]:
    """Sort key giving the same order as compare(): one (median rank, -above) per round.

    Each round: majority grade, then how many grades are above it, then the worst grade
    is removed. Both sides of compare() do exactly this in lockstep, so comparing the
    sequences lexicographically is the same tie-break, computed once per party in O(n).
    Lower is better (rank 0 = Excellent, more above = more negative).
    """
    counts = list(counts) if counts is not None else grade_counts(grades)
    n = sum(counts)
    key = []
    while n > 0:
        m = _bucket_majority(counts)
        key.append((m, -sum(counts[:m])))
        for i in range(len(counts) - 1, -1, -1):
            if counts[i]:
                counts[i] -= 1
                break
        n -= 1
    return key


def mj_rank(ballots: list[Ballot], candidates: list[str]) -> list[MJResult]:
    """Candidates ranked best first by Majority Judgment."""
    results = []
    keys = {}
    for name in candidates:
        grades = grades_for(ballots, name)
        n = len(grades)
        counts = grade_counts(grades)
        m = _bucket_majority(counts)
        results.append(
            MJResult(
                name=name,
                counts={g: counts[i] for i, g in enumerate(GRADES_DESC)},
                majority=GRADES_DESC[m],
                p=sum(counts[:m]) / n,
                q=sum(counts[m + 1:]) / n,
                _grades=grades,
            )
        )
        keys[name] = mj_key(grades, counts)
    results.sort(key=lambda r: keys[r.name])
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


def divisor_method(votes: dict[str, float], seats: int, divisor) -> dict[str, int]:
    """Highest-averages allocation: seats go one at a time to the party with the highest
    votes / divisor(seats already won). Ties: more votes, then name."""
    won = {p: 0 for p in votes}
    for _ in range(seats):
        quotient = {p: votes[p] / divisor(won[p]) for p in votes}
        best = max(quotient.values())
        tied = [p for p in votes if quotient[p] == best]
        winner = min(tied, key=lambda p: (-votes[p], p))
        won[winner] += 1
    return won


def sainte_lague(votes: dict[str, float], seats: int, first_divisor: float = FIRST_DIVISOR) -> dict[str, int]:
    """Modified Sainte-Lague as used for the Norwegian Storting.

    Each party's votes are divided by first_divisor (1.4), then 3, 5, 7, ...
    The 1.4 makes the first seat harder to win for small parties.
    """
    return divisor_method(votes, seats, lambda k: first_divisor if k == 0 else 2 * k + 1)


def dhondt(votes: dict[str, float], seats: int) -> dict[str, int]:
    """D'Hondt: divide by 1, 2, 3, 4, ... Favours large parties a little more than Sainte-Lague."""
    return divisor_method(votes, seats, lambda k: k + 1)


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


def seats_in(district: str, levelling: bool = True) -> int:
    """Mandates counted inside a district.

    With levelling seats: the official number minus one (19 for Utjevning itself).
    Without: the full official number, and Utjevning has none.
    """
    if district == LEVELLING:
        return LEVELLING_SEATS if levelling else 0
    return DISTRICTS[district][0] - 1 if levelling else DISTRICTS[district][0]


def split_voters(n_voters: int) -> dict[str, int]:
    """Voters per district in proportion to population; rounding leftover to the largest district."""
    pops = {d: p for d, (_, p) in DISTRICTS.items()}
    total_pop = sum(pops.values())
    out = {d: round(n_voters * p / total_pop) for d, p in pops.items()}
    biggest = max(pops, key=pops.get)
    out[biggest] += n_voters - sum(out.values())
    return out


def simulate_districts(candidates: list[str], n_voters: int, rng: random.Random) -> dict[str, list[Ballot]]:
    """One ballot list per district."""
    return {d: simulate(candidates, n, rng) for d, n in split_voters(n_voters).items()}


NORWAY = "norway"  # split option: modified Sainte-Lague per district, like the Storting count
DHONDT = "dhondt"  # split option: D'Hondt divisors 1, 2, 3, ...
SL_CAP = "norway_cap"  # split option: Sainte-Lague; for Majority Judgment with the median cap (MJ_CAP)
MJ_CAP = 2  # Sainte-Lague cap: a slice may show at most this many grades above the party's median
PROPORTIONAL = {NORWAY: sainte_lague, DHONDT: dhondt, SL_CAP: sainte_lague}
MJ_SUPPORT_GRADE = Grade.GOOD  # "supporter" / approval: graded the party this or better
Split = "list[int] | str"


def is_proportional(weights) -> bool:
    return isinstance(weights, str) and weights in PROPORTIONAL


def split_label(weights) -> str:
    return weights if isinstance(weights, str) else ",".join(map(str, weights))


def parse_split(text: str):
    """'1' -> [1], '2,1,1' -> [2, 1, 1], 'norway' -> NORWAY. Positive integers only."""
    if isinstance(text, list):
        return text
    if text.strip().lower() in PROPORTIONAL:
        return text.strip().lower()
    try:
        weights = [int(x) for x in text.replace(" ", "").split(",") if x != ""]
    except ValueError:
        raise ValueError(f"split must be numbers separated by commas, got '{text}'")
    if not weights or any(w <= 0 for w in weights):
        raise ValueError("split weights must be positive, e.g. 1 or 2,1,1")
    return weights


def split_seats(seats: int, ranking: list[str], weights: Split) -> dict[str, int]:
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
    eligible: list[str] | None = None  # parties allowed levelling seats (None = no levelling)


def _finish(by_district: dict[str, dict[str, int]], candidates: list[str],
            eligible: list[str] | None = None) -> SeatsResult:
    seats = {c: sum(d[c] for d in by_district.values()) for c in candidates}
    top = max(seats.values())
    return SeatsResult(seats=seats, by_district=by_district,
                       largest=[c for c in candidates if seats[c] == top], eligible=eligible)


def mj_support(ballots: list[Ballot], candidates: list[str]) -> dict[str, int]:
    """Supporters per party: voters who graded it MJ_SUPPORT_GRADE or better.

    Also the approval count: Approval voting treats Good or better as a tick."""
    bar = GRADE_INDEX[MJ_SUPPORT_GRADE]
    return {c: sum(grade_counts(grades_for(ballots, c))[: bar + 1]) for c in candidates}


approvals = mj_support

# Score voting needs points. Excellent 5 ... Reject 0; the only place grades become numbers.
GRADE_SCORE = {g: len(GRADES_DESC) - 1 - i for i, g in enumerate(GRADES_DESC)}


def score_totals(ballots: list[Ballot], candidates: list[str]) -> dict[str, int]:
    """Sum of points per party (Score / Range voting)."""
    return {c: sum(GRADE_SCORE[g] for g in grades_for(ballots, c)) for c in candidates}


def score_rank(ballots: list[Ballot], candidates: list[str]) -> list[str]:
    """Highest average points first (same ballots as Majority Judgment, mean instead of median)."""
    totals = score_totals(ballots, candidates)
    return sorted(candidates, key=lambda c: -totals[c])  # stable: ties keep input order


def approval_rank(ballots: list[Ballot], candidates: list[str]) -> list[str]:
    """Most approvals (Good or better) first."""
    app = approvals(ballots, candidates)
    return sorted(candidates, key=lambda c: -app[c])


def quantile_mj(sorted_grades: dict[str, list[Grade]], seats: int, first_divisor: float = FIRST_DIVISOR,
                dhondt: bool = False, cap: int | None = None) -> dict[str, int]:
    """Majority Judgment's own proportional rule: Sainte-Lague or D'Hondt translated to grades.

    Q = voters per mandate. A party's voters are lined up best grade first and cut into
    slices of Q, one slice per mandate. A party holding k mandates is judged on the grade
    at one position in its (k+1)-th slice:
      Sainte-Lague  the middle of the slice, position (2k+1)/2 * Q; Norway's 1.4 first
                    divisor becomes 0.7 * Q for the first mandate
      D'Hondt       the end of the slice, position (k+1) * Q
    Mandates go one at a time to the party with the best grade there. Ties: the normal
    Majority Judgment tie-break on the slice. With one mandate and first divisor 1 the
    judged position is the median, so this is plain Majority Judgment.

    cap ("Sainte-Lague cap"): a slice may show at most `cap` grades above the party's own
    median in the district. A party most of whose voters reject it cannot win mandates on
    its fans alone. Grades are only ever lowered, never lifted.

    sorted_grades: every party's grades, best first (all parties same length).
    """
    candidates = list(sorted_grades)
    won = {c: 0 for c in candidates}
    if seats <= 0 or not candidates:
        return won
    n = len(next(iter(sorted_grades.values())))
    if n == 0:
        return won
    q = n / seats
    median_rank = {c: GRADE_INDEX[majority_grade(sorted_grades[c])] for c in candidates} if cap is not None else {}
    for _ in range(seats):
        best_key, best = None, None
        for c in candidates:
            k = won[c]
            pos = (k + 1) * q if dhondt else (first_divisor if k == 0 else 2 * k + 1) * q / 2
            idx = min(n - 1, max(0, math.ceil(pos) - 1))
            lo, hi = min(n, int(k * q)), min(n, math.ceil((k + 1) * q))
            slice_key = mj_key(sorted_grades[c][lo:hi]) if hi > lo else [(len(GRADES_DESC), 0)]
            g = GRADE_INDEX[sorted_grades[c][idx]]
            if cap is not None:
                g = max(g, median_rank[c] - cap)  # rank 0 is best, so "above" means a smaller rank
            key = (g, slice_key)
            if best_key is None or key < best_key:
                best_key, best = key, c
        won[best] += 1
    return won


def two_round(ballots: list[Ballot], votes: dict[str, int], candidates: list[str]) -> tuple[list[str], dict[str, int] | None]:
    """Two-round system. Round one: the single vote. If nobody has over 50%, a runoff between
    the top two: every voter backs whichever of the two they graded higher (equal grade = abstain).

    Returns (ranking, runoff counts or None). Ranking: runoff winner, runner-up, then the
    rest by first-round votes.
    """
    by_votes = sorted(candidates, key=lambda c: -votes[c])
    total = sum(votes.values())
    if len(candidates) < 2 or votes[by_votes[0]] * 2 > total:
        return by_votes, None
    a, b = by_votes[0], by_votes[1]
    runoff = {a: 0, b: 0}
    for ballot in ballots:
        ga, gb = ballot.get(a, Grade.REJECT), ballot.get(b, Grade.REJECT)
        if ga.beats(gb):
            runoff[a] += 1
        elif gb.beats(ga):
            runoff[b] += 1
    winner, loser = (a, b) if runoff[a] >= runoff[b] else (b, a)  # tie: first-round leader
    return [winner, loser] + by_votes[2:], runoff


def levelling_seats(votes: dict[str, int], district_seats: dict[str, int], eligible: list[str],
                    total: int = STORTING_SEATS, allocate=None) -> dict[str, int]:
    """Norway's levelling rule: fill each eligible party's deficit against its fair national share.

    1. Parties under the threshold keep their district seats; those seats are set aside.
    2. The remaining seats are shared among eligible parties by modified Sainte-Lague on
       national votes: that is each party's fair share.
    3. A party that already won more district seats than its fair share is taken out of
       the pool (it keeps what it has) and step 2 is redone without it.
    4. Each party left gets fair share minus district seats. These add up to exactly the
       seats the districts gave up (19). The split rule never applies here.
    """
    if allocate is None:
        allocate = lambda pool, seats: sainte_lague({c: votes[c] for c in pool}, seats)
    pool = [c for c in eligible]
    fair: dict[str, int] = {}
    while pool:
        remaining = total - sum(n for c, n in district_seats.items() if c not in pool)
        fair = allocate(pool, remaining)
        over = [c for c in pool if district_seats[c] > fair[c]]
        if not over:
            break
        pool = [c for c in pool if c not in over]
    out = {c: 0 for c in district_seats}
    if not pool:  # every eligible party is already over its share: plain proportional fallback
        out.update(allocate(list(eligible), total - sum(district_seats.values())))
        return out
    for c in pool:
        out[c] = fair[c] - district_seats[c]
    return out


# ---------------------------------------------------------------- systems

# grades = Majority Judgment (median), score = mean points, approval = Good or better,
# votes = one vote per voter, tworound = single vote with a runoff between the top two
METHODS = ("grades", "score", "approval", "votes", "tworound")
METHOD_LABELS = {"grades": "Majority Judgment", "score": "Score voting", "approval": "Approval voting",
                 "votes": "Single vote", "tworound": "Two-round"}
LEVELLING_MODES = ("threshold", "open", "none")  # 4% sperregrense / every party / no levelling seats


@dataclass(frozen=True)
class SystemSpec:
    """One electoral system: how voters are counted, how a district's mandates are split,
    and what happens with the 19 Utjevning levelling mandates."""
    name: str
    method: str = "grades"
    split: str = "1"
    levelling: str = "threshold"

    def validate(self) -> "SystemSpec":
        if self.method not in METHODS:
            raise ValueError(f"method must be one of {METHODS}, got '{self.method}'")
        if self.levelling not in LEVELLING_MODES:
            raise ValueError(f"levelling must be one of {LEVELLING_MODES}, got '{self.levelling}'")
        parse_split(self.split)
        if not self.name.strip():
            raise ValueError("system needs a name")
        return self

    def to_dict(self) -> dict:
        return {"name": self.name, "method": self.method, "split": self.split, "levelling": self.levelling}

    @staticmethod
    def from_dict(d: dict) -> "SystemSpec":
        return SystemSpec(str(d.get("name", "")), str(d.get("method", "grades")),
                          str(d.get("split", "1")), str(d.get("levelling", "threshold"))).validate()


PRESETS = {  # systems in use somewhere in the world; Majority Judgment is built via "New system"
    "Norwegian": SystemSpec("Norwegian", "votes", NORWAY, "threshold"),
    "American": SystemSpec("American", "votes", "1", "none"),
    # Two-round runoff between the top two, winner takes the district, no levelling: how
    # France elects its president (and Brazil). France's National Assembly also lets every
    # candidate with 12.5% of registered voters into round two, which is not modelled.
    "French": SystemSpec("French", "tworound", "1", "none"),
}


@dataclass
class Voters:
    """The shared electorate every system is counted on."""
    candidates: list[str]
    district_ballots: dict[str, list[Ballot]]
    district_votes: dict[str, dict[str, int]]
    ambiguous: int
    source: str

    @property
    def all_ballots(self) -> list[Ballot]:
        return [b for ballots in self.district_ballots.values() for b in ballots]

    @property
    def n_voters(self) -> int:
        return sum(len(b) for b in self.district_ballots.values())

    @property
    def votes(self) -> dict[str, int]:
        return {c: sum(v[c] for v in self.district_votes.values()) for c in self.candidates}

    def sorted_grades(self, district: str | None = None) -> dict[str, list[Grade]]:
        """Every party's grades best first, per district or national, computed once."""
        cache = self.__dict__.setdefault("_sorted_cache", {})
        if district not in cache:
            ballots = self.all_ballots if district is None else self.district_ballots[district]
            cache[district] = {c: sort_best_first(grades_for(ballots, c)) for c in self.candidates}
        return cache[district]

    def mj_ranking(self, district: str) -> list[str]:
        """Majority Judgment order in one district, computed once per electorate."""
        cache = self.__dict__.setdefault("_mj_rank_cache", {})
        if district not in cache:
            cache[district] = [r.name for r in mj_rank(self.district_ballots[district], self.candidates)]
        return cache[district]

    def mj_supporters(self, district: str | None = None) -> dict[str, int]:
        """Supporters per party (Good or better), per district or national, computed once."""
        return self.measure("approval", district)

    def measure(self, method: str, district: str | None = None) -> dict[str, int]:
        """What a method counts per party: votes, approvals, points or MJ supporters.
        Per district, or national when district is None. Computed once per electorate."""
        cache = self.__dict__.setdefault("_measure_cache", {})
        key = (method, district)
        if key not in cache:
            ballots = self.all_ballots if district is None else self.district_ballots[district]
            if method in ("votes", "tworound"):
                cache[key] = self.votes if district is None else dict(self.district_votes[district])
            elif method == "score":
                cache[key] = score_totals(ballots, self.candidates)
            else:  # grades and approval both measure support as Good or better
                cache[key] = mj_support(ballots, self.candidates)
        return cache[key]

    def ranking(self, method: str, district: str) -> list[str]:
        """Party order in one district under a method, computed once per electorate."""
        cache = self.__dict__.setdefault("_rank_cache", {})
        key = (method, district)
        if key not in cache:
            ballots = self.district_ballots[district]
            if method == "grades":
                cache[key] = self.mj_ranking(district)
            elif method == "score":
                cache[key] = score_rank(ballots, self.candidates)
            elif method == "approval":
                cache[key] = approval_rank(ballots, self.candidates)
            elif method == "tworound":
                cache[key] = two_round(ballots, self.district_votes[district], self.candidates)[0]
            else:
                cache[key] = sorted(self.candidates, key=lambda c: -self.district_votes[district][c])
        return cache[key]


def make_voters(candidates: list[str], district_ballots: dict[str, list[Ballot]], rng: random.Random,
                district_votes: dict[str, dict[str, int]] | None = None, source: str = "random") -> Voters:
    """Collect one vote per ballot (the top-graded party) unless real votes are given."""
    if LEVELLING in district_ballots:
        raise ValueError(f"{LEVELLING} is not a district; it is filled from national votes")
    ambiguous = 0
    if district_votes is None:
        district_votes = {}
        for d, ballots in district_ballots.items():
            district_votes[d], amb = collect_votes(ballots, candidates, rng)
            ambiguous += amb
    return Voters(candidates, district_ballots, district_votes, ambiguous, source)


def run_system(spec: SystemSpec, v: Voters) -> SeatsResult:
    """Count one system on the shared voters: district seats, then levelling if the spec has it."""
    spec.validate()
    weights = parse_split(spec.split)
    lev = spec.levelling != "none"
    grade_prop = spec.method == "grades" and is_proportional(weights)  # MJ's own proportional rule
    by_district: dict[str, dict[str, int]] = {}
    for d in DISTRICTS:
        seats = seats_in(d, lev)
        if grade_prop:
            by_district[d] = quantile_mj(v.sorted_grades(d), seats, dhondt=(weights == DHONDT),
                                         cap=MJ_CAP if weights == SL_CAP else None)
        elif is_proportional(weights):
            by_district[d] = PROPORTIONAL[weights](v.measure(spec.method, d), seats)
        else:
            by_district[d] = split_seats(seats, v.ranking(spec.method, d), weights)
    res = _finish(by_district, v.candidates)
    if not lev:
        return res
    # Levelling: fair national share minus district seats (Norway's deficit rule). The fair
    # share is computed with the method's own count (votes, approvals, points) and Sainte-Lague,
    # or, for Majority Judgment with a proportional split, with quantile_mj on all ballots.
    measure = v.measure(spec.method) if is_proportional(weights) else v.votes
    eligible = list(v.candidates)
    if spec.levelling == "threshold":
        total = sum(measure.values()) or 1
        eligible = [c for c in v.candidates if measure[c] / total >= THRESHOLD] or list(v.candidates)
    allocate = None
    if grade_prop:
        national = v.sorted_grades()
        allocate = lambda pool, seats: quantile_mj({c: national[c] for c in pool}, seats, dhondt=(weights == DHONDT),
                                                   cap=MJ_CAP if weights == SL_CAP else None)
    res.by_district[LEVELLING] = levelling_seats(measure, res.seats, eligible, allocate=allocate)
    return _finish(res.by_district, v.candidates, eligible)


@dataclass
class Election:
    """The three default systems on one electorate (used by the command line and tests)."""
    district_ballots: dict[str, list[Ballot]]
    district_votes: dict[str, dict[str, int]]
    ambiguous: int
    mj_national: list[MJResult]  # grade table over all ballots, ranked
    votes: dict[str, int]  # national vote totals
    mj: SeatsResult
    plurality: SeatsResult
    storting: SeatsResult
    sperregrense: bool
    eligible: list[str]  # parties allowed levelling seats (every party when sperregrense is off)

    @property
    def districts(self) -> list[str]:
        """Every place mandates are counted, Utjevning last."""
        return list(DISTRICTS) + [LEVELLING]

    @property
    def n_voters(self) -> int:
        return sum(len(b) for b in self.district_ballots.values())

    def picks(self) -> dict[str, list[str]]:
        return {"Majority Judgment": self.mj.largest, "Storting": self.storting.largest,
                "Plurality": self.plurality.largest}

    def agree(self) -> str | None:
        """The party every system makes largest, or None."""
        first = self.mj.largest
        return first[0] if len(first) == 1 and all(v == first for v in self.picks().values()) else None


def count_all(district_ballots: dict[str, list[Ballot]], candidates: list[str],
              rng: random.Random, weights: Split, sperregrense: bool = True,
              district_votes: dict[str, dict[str, int]] | None = None) -> Election:
    """The three default systems: MJ with the given split, Storting (votes + norway split),
    plurality (votes, winner takes all). All with levelling seats; sperregrense sets the 4% rule."""
    v = make_voters(candidates, district_ballots, rng, district_votes)
    lev = "threshold" if sperregrense else "open"
    mj = run_system(SystemSpec("Majority Judgment", "grades", split_label(weights), lev), v)
    st = run_system(SystemSpec("Storting", "votes", NORWAY, lev), v)
    pl = run_system(SystemSpec("Plurality", "votes", "1", lev), v)
    return Election(
        district_ballots=district_ballots,
        district_votes=v.district_votes,
        ambiguous=v.ambiguous,
        mj_national=mj_rank(v.all_ballots, candidates),
        votes=v.votes,
        mj=mj,
        plurality=pl,
        storting=st,
        sperregrense=sperregrense,
        eligible=st.eligible or list(candidates),
    )


# ---------------------------------------------------------------- cses data (real norwegian voters)

CSES_ZIP = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data", "cses_imd_csv.zip")
CSES_CACHE = os.path.join(os.path.dirname(CSES_ZIP), "cses_norway.csv")
CSES_YEARS = [1997, 2001, 2005, 2009, 2013, 2017]
CSES_SLOTS = "ABCDEFGHI"
# CSES party codes for Norway. Names inferred by matching each code's survey vote share
# to the official result year by year (e.g. 5780001 is the largest party in 1997-2013).
CSES_PARTIES = {
    "5780001": "Ap", "5780002": "Høyre", "5780003": "Frp", "5780004": "SV", "5780005": "MDG",
    "5780006": "Sp", "5780007": "KrF", "5780008": "Venstre", "5780009": "Rødt", "5780010": "Kystpartiet",
}
# 0-10 like/dislike score -> grade. Reject gets only 0, the other grades two scores each.
SCORE_TO_GRADE = {
    10: Grade.EXCELLENT, 9: Grade.EXCELLENT,
    8: Grade.VERY_GOOD, 7: Grade.VERY_GOOD,
    6: Grade.GOOD, 5: Grade.GOOD,
    4: Grade.ACCEPTABLE, 3: Grade.ACCEPTABLE,
    2: Grade.POOR, 1: Grade.POOR,
    0: Grade.REJECT,
}
CSES_COLUMNS = (["IMD1004", "IMD1008_YEAR", "IMD3002_LH_PL"]
                + [f"IMD3008_{s}" for s in CSES_SLOTS] + [f"IMD5000_{s}" for s in CSES_SLOTS])


def build_cses_cache(zip_path: str = CSES_ZIP, cache_path: str = CSES_CACHE) -> int:
    """Pull the Norwegian rows and the needed columns out of the 650 MB CSES file. Slow, once."""
    n = 0
    with zipfile.ZipFile(zip_path) as z:
        name = next(m for m in z.namelist() if m.endswith(".csv"))
        with z.open(name) as raw, open(cache_path, "w", newline="", encoding="utf-8") as out:
            reader = csv.DictReader(io.TextIOWrapper(raw, encoding="utf-8", errors="replace"))
            writer = csv.DictWriter(out, fieldnames=CSES_COLUMNS)
            writer.writeheader()
            for row in reader:
                if row["IMD1004"].startswith("NOR_"):
                    writer.writerow({c: row[c] for c in CSES_COLUMNS})
                    n += 1
    return n


def score_grade(text: str) -> Grade:
    """'0'..'10' -> grade; anything else (missing, refused, 96-99) counts as Reject."""
    return SCORE_TO_GRADE.get(int(text), Grade.REJECT) if text.isdigit() else Grade.REJECT


def load_cses(year: int, rng: random.Random, zip_path: str = CSES_ZIP,
              cache_path: str = CSES_CACHE) -> tuple[list[str], list[Ballot], list[str]]:
    """Real Norwegian respondents for one election year.

    Returns (parties, ballots, votes). Party slots that nobody rated that year are dropped.
    A respondent's vote is their reported vote if it is one of the rated parties, otherwise
    the party they rated highest (random among ties).
    """
    if year not in CSES_YEARS:
        raise ValueError(f"CSES has Norway for {CSES_YEARS}, not {year}")
    if not os.path.exists(cache_path):
        if not os.path.exists(zip_path):
            raise FileNotFoundError(f"{zip_path} not found; download cses_imd_csv.zip from cses.org into data/")
        print(f"first run: extracting Norway from {os.path.basename(zip_path)} ... (a minute or two)")
        build_cses_cache(zip_path, cache_path)
    with open(cache_path, newline="", encoding="utf-8") as f:
        rows = [r for r in csv.DictReader(f) if r["IMD1008_YEAR"] == str(year)]
    if not rows:
        raise ValueError(f"no Norwegian respondents for {year} in {cache_path}")

    # party in each slot: the code used that year; slots never rated are dropped
    slot_party: dict[str, str] = {}
    for s in CSES_SLOTS:
        codes = {r[f"IMD5000_{s}"] for r in rows} - {"9999999", ""}
        rated = any(r[f"IMD3008_{s}"].isdigit() and int(r[f"IMD3008_{s}"]) <= 10 for r in rows)
        if len(codes) == 1 and rated:
            code = codes.pop()
            slot_party[s] = CSES_PARTIES.get(code, f"party {code}")
    parties = list(slot_party.values())
    code_to_party = {code: CSES_PARTIES.get(code, f"party {code}") for code in CSES_PARTIES}

    ballots: list[Ballot] = []
    votes: list[str] = []
    for r in rows:
        ballot = {party: score_grade(r[f"IMD3008_{s}"]) for s, party in slot_party.items()}
        ballots.append(ballot)
        reported = code_to_party.get(r["IMD3002_LH_PL"])
        votes.append(reported if reported in parties else vote_from_ballot(ballot, parties, rng)[0])
    return parties, ballots, votes


def cses_districts(year: int, rng: random.Random, zip_path: str = CSES_ZIP,
                   cache_path: str = CSES_CACHE) -> Voters:
    """Spread the real respondents over the districts at random by population."""
    parties, ballots, votes = load_cses(year, rng, zip_path, cache_path)
    shares = split_voters(len(ballots))
    if len(ballots) < len(shares):
        raise ValueError(f"need at least {len(shares)} respondents, got {len(ballots)}")
    biggest = max(shares, key=shares.get)
    for d in shares:  # every district needs at least one respondent for the median to exist
        if shares[d] == 0:
            shares[d] += 1
            shares[biggest] -= 1
    order = list(range(len(ballots)))
    rng.shuffle(order)
    district_ballots: dict[str, list[Ballot]] = {}
    district_votes: dict[str, dict[str, int]] = {}
    pos = 0
    for d, n in shares.items():
        idx = order[pos:pos + n]
        pos += n
        district_ballots[d] = [ballots[i] for i in idx]
        counts = {p: 0 for p in parties}
        for i in idx:
            counts[votes[i]] += 1
        district_votes[d] = counts
    return make_voters(parties, district_ballots, rng, district_votes, source=f"CSES Norway {year}")


def cses_election(year: int, rng: random.Random, weights: Split, sperregrense: bool = True,
                  zip_path: str = CSES_ZIP, cache_path: str = CSES_CACHE) -> tuple[list[str], Election]:
    """Real respondents through the three default systems (command line)."""
    v = cses_districts(year, rng, zip_path, cache_path)
    return v.candidates, count_all(v.district_ballots, v.candidates, rng, weights, sperregrense, v.district_votes)


# ---------------------------------------------------------------- board (many systems at once)

def build_voters(settings: dict, zip_path: str = CSES_ZIP, cache_path: str = CSES_CACHE) -> Voters:
    """settings: {"source": "random" | "cses_2017", "parties": [...], "n": 10000, "seed": 42}."""
    seed = int(settings.get("seed", 42))
    rng = random.Random(seed)
    source = str(settings.get("source", "random"))
    if source.startswith("cses_"):
        return cses_districts(int(source[5:]), rng, zip_path, cache_path)
    parties = [str(p).strip() for p in settings.get("parties", []) if str(p).strip()]
    if len(parties) < 2:
        raise ValueError("enter at least two parties")
    if len(set(parties)) != len(parties):
        raise ValueError("party names must be unique")
    n = int(settings.get("n", 10000))
    if n < len(DISTRICTS):
        raise ValueError(f"need at least {len(DISTRICTS)} voters")
    return make_voters(parties, simulate_districts(parties, n, rng), rng, source="random")


def run_board(v: Voters, specs: list[SystemSpec]) -> dict:
    """Every system on the same voters, as plain JSON-ready data for the board page."""
    mj_national = mj_rank(v.all_ballots, v.candidates)
    mj_order = {r.name: i for i, r in enumerate(mj_national)}
    grade_rows = {r.name: r for r in mj_national}
    votes = v.votes
    points = v.measure("score")
    approved = v.measure("approval")
    n = v.n_voters
    blocks = []
    for spec in specs:
        res = run_system(spec, v)
        lev = spec.levelling != "none"
        rows = []
        for c in sorted(v.candidates, key=lambda c: (-res.seats[c], mj_order[c])):
            r = grade_rows[c]
            rows.append({
                "party": c,
                "counts": {g.label: r.counts[g] for g in GRADES_DESC},
                "median": r.majority.label,
                "p": round(r.p, 3),
                "q": round(r.q, 3),
                "score": round(points[c] / n, 2),
                "approvals": approved[c],
                "approval_share": round(approved[c] / n, 4),
                "votes": votes[c],
                "vote_share": round(votes[c] / n, 4),
                "seats": res.seats[c],
                "seat_share": round(res.seats[c] / STORTING_SEATS, 4),
            })
        districts = []
        for d in list(DISTRICTS) + ([LEVELLING] if lev else []):
            districts.append({
                "name": d,
                "seats": seats_in(d, lev),
                "voters": len(v.district_ballots[d]) if d in v.district_ballots else None,
                "by_party": {c: k for c, k in res.by_district[d].items() if k},
            })
        blocks.append({
            "spec": spec.to_dict(),
            "largest": res.largest,
            "eligible": res.eligible,
            "rows": rows,
            "levelling": {c: k for c, k in res.by_district[LEVELLING].items() if k} if lev else None,
            "districts": districts,
        })
    return {
        "source": v.source,
        "parties": list(v.candidates),
        "n_voters": n,
        "ambiguous": v.ambiguous,
        "blocks": blocks,
    }


# ---------------------------------------------------------------- cli

def print_report(e: Election, candidates: list[str], weights: Split) -> None:
    n = e.n_voters
    width = max(len(c) for c in candidates)
    S = STORTING_SEATS

    mode = (f"sperregrense ON: {LEVELLING_SEATS} {LEVELLING} levelling mandates only for parties >= {THRESHOLD:.0%}: "
            f"{', '.join(e.eligible)}" if e.sperregrense
            else f"sperregrense OFF: {LEVELLING_SEATS} {LEVELLING} levelling mandates open to every party")
    print(mode)
    print()
    header = " ".join(f"{g.label:>10}" for g in GRADES_DESC)
    print(f"Majority Judgment  ({n} voters, {len(DISTRICTS)} districts + {LEVELLING}, split {split_label(weights)})")
    print(f"{'':{width}} {header}   majority      p      q   seats")
    for r in e.mj_national:
        row = " ".join(f"{r.counts[g]:>10}" for g in GRADES_DESC)
        print(f"{r.name:{width}} {row}   {r.majority.label:<10} {r.p:5.2f}  {r.q:5.2f}   {e.mj.seats[r.name]:>3}")
    print(f"Largest party: {' = '.join(e.mj.largest)}")
    print()

    print("Storting (Norway, modified Sainte-Lague per district)")
    for c in sorted(candidates, key=lambda c: -e.storting.seats[c]):
        print(f"{c:{width}} {e.votes[c]:>7} votes ({e.votes[c] / n:5.1%})   {e.storting.seats[c]:>3} seats ({e.storting.seats[c] / S:5.1%})")
    print(f"Largest party: {' = '.join(e.storting.largest)}")
    print()

    print("Plurality  (winner takes all of the district's mandates, same votes)")
    for c in sorted(candidates, key=lambda c: -e.plurality.seats[c]):
        print(f"{c:{width}} {e.votes[c]:>7} votes ({e.votes[c] / n:5.1%})   {e.plurality.seats[c]:>3} seats ({e.plurality.seats[c] / S:5.1%})")
    print(f"Largest party: {' = '.join(e.plurality.largest)}  (ambiguous ballots: {e.ambiguous})")
    print()

    def seats_txt(res: SeatsResult, d: str) -> str:
        """Every party's mandates in the district, e.g. 'Ap 5, Høyre 3'."""
        return ", ".join(f"{c} {n}" for c, n in res.by_district[d].items() if n) or "-"

    dw = max(len(d) for d in e.districts)
    cols = {name: max(len(name), max(len(seats_txt(res, d)) for d in e.districts))
            for name, res in (("Majority Judgment", e.mj), ("Storting", e.storting), ("Plurality", e.plurality))}
    print(f"Mandates per district ({LEVELLING} = levelling seats, fair share minus district seats)")
    print(f"{'District':{dw}}  seats  voters  " + "  ".join(f"{name:{w}}" for name, w in cols.items()))
    for d in e.districts:
        voters = str(len(e.district_ballots[d])) if d in e.district_ballots else "national"
        print(f"{d:{dw}}  {seats_in(d):>5}  {voters:>8}  "
              f"{seats_txt(e.mj, d):{cols['Majority Judgment']}}  {seats_txt(e.storting, d):{cols['Storting']}}  "
              f"{seats_txt(e.plurality, d):{cols['Plurality']}}")
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
                    help="on: only parties >= 4%% get Utjevning levelling mandates; off: every party does")
    ap.add_argument("--split", default="1",
                    help="Majority Judgment mandate split per district: 1 (winner takes all), 2,1,1, 3,2,1, "
                         "norway (Sainte-Lague), norway_cap (Sainte-Lague cap) or dhondt")
    ap.add_argument("--cses", type=int, metavar="YEAR", choices=CSES_YEARS,
                    help="use real Norwegian voters from CSES for this election year instead of random ones")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args(argv)

    weights = parse_split(args.split)
    rng = random.Random(args.seed)
    if args.cses:
        candidates, e = cses_election(args.cses, rng, weights, args.sperregrense)
        print(f"CSES Norway {args.cses}: {e.n_voters} real respondents, parties {', '.join(candidates)}, seed {args.seed}")
    else:
        candidates = args.candidates
        district_ballots = simulate_districts(candidates, args.voters, rng)
        e = count_all(district_ballots, candidates, rng, weights, args.sperregrense)
        print(f"seed {args.seed}")
    print_report(e, candidates, weights)


if __name__ == "__main__":
    main()
