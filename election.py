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
MJ_SUPPORT_GRADE = Grade.GOOD  # under --split norway, an MJ "supporter" graded the party this or better
Split = "list[int] | str"


def split_label(weights) -> str:
    return weights if isinstance(weights, str) else ",".join(map(str, weights))


def parse_split(text: str):
    """'1' -> [1], '2,1,1' -> [2, 1, 1], 'norway' -> NORWAY. Positive integers only."""
    if text.strip().lower() == NORWAY:
        return NORWAY
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


def _finish(by_district: dict[str, dict[str, int]], candidates: list[str]) -> SeatsResult:
    seats = {c: sum(d[c] for d in by_district.values()) for c in candidates}
    top = max(seats.values())
    return SeatsResult(seats=seats, by_district=by_district,
                       largest=[c for c in candidates if seats[c] == top])


def mj_support(ballots: list[Ballot], candidates: list[str]) -> dict[str, int]:
    """Supporters per party: voters who graded it MJ_SUPPORT_GRADE or better (used by --split norway)."""
    return {c: sum(1 for g in grades_for(ballots, c) if g == MJ_SUPPORT_GRADE or g.beats(MJ_SUPPORT_GRADE))
            for c in candidates}


def mj_seats(district_ballots: dict[str, list[Ballot]], candidates: list[str],
             weights: Split) -> SeatsResult:
    """Majority Judgment per district; the district's mandates split by weights."""
    by_district = {}
    for d, ballots in district_ballots.items():
        if weights == NORWAY:
            by_district[d] = sainte_lague(mj_support(ballots, candidates), seats_in(d))
            continue
        ranking = [r.name for r in mj_rank(ballots, candidates)]
        by_district[d] = split_seats(seats_in(d), ranking, weights)
    return _finish(by_district, candidates)


def plurality_seats(district_votes: dict[str, dict[str, int]], candidates: list[str]) -> SeatsResult:
    """First past the post: the party with most votes takes all of the district's mandates."""
    by_district = {}
    for d, votes in district_votes.items():
        ranking = sorted(candidates, key=lambda c: -votes[c])  # stable: ties keep input order
        by_district[d] = split_seats(seats_in(d), ranking, [1])
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


def levelling_seats(votes: dict[str, int], district_seats: dict[str, int], eligible: list[str],
                    total: int = STORTING_SEATS) -> dict[str, int]:
    """Norway's levelling rule: fill each eligible party's deficit against its fair national share.

    1. Parties under the threshold keep their district seats; those seats are set aside.
    2. The remaining seats are shared among eligible parties by modified Sainte-Lague on
       national votes: that is each party's fair share.
    3. A party that already won more district seats than its fair share is taken out of
       the pool (it keeps what it has) and step 2 is redone without it.
    4. Each party left gets fair share minus district seats. These add up to exactly the
       seats the districts gave up (19). The split rule never applies here.
    """
    pool = [c for c in eligible]
    fair: dict[str, int] = {}
    while pool:
        remaining = total - sum(n for c, n in district_seats.items() if c not in pool)
        fair = sainte_lague({c: votes[c] for c in pool}, remaining)
        over = [c for c in pool if district_seats[c] > fair[c]]
        if not over:
            break
        pool = [c for c in pool if c not in over]
    out = {c: 0 for c in district_seats}
    if not pool:  # every eligible party is already over its share: plain proportional fallback
        out.update(sainte_lague({c: votes[c] for c in eligible}, total - sum(district_seats.values())))
        return out
    for c in pool:
        out[c] = fair[c] - district_seats[c]
    return out


def count_all(district_ballots: dict[str, list[Ballot]], candidates: list[str],
              rng: random.Random, weights: Split, sperregrense: bool = True,
              district_votes: dict[str, dict[str, int]] | None = None) -> Election:
    """Count all three systems: district seats, then the 19 Utjevning levelling seats.

    sperregrense: True = only parties with at least 4% of the national vote get levelling
    seats; False = every party does.
    district_votes: real votes per district (CSES). If None, each ballot votes for its
    top-graded party.
    """
    if LEVELLING in district_ballots:
        raise ValueError(f"{LEVELLING} is not a district; it is filled from national votes")
    ambiguous = 0
    if district_votes is None:
        district_votes = {}
        for d, ballots in district_ballots.items():
            district_votes[d], amb = collect_votes(ballots, candidates, rng)
            ambiguous += amb
    all_ballots = [b for ballots in district_ballots.values() for b in ballots]
    votes = {c: sum(v[c] for v in district_votes.values()) for c in candidates}
    mj_national = mj_rank(all_ballots, candidates)

    mj = mj_seats(district_ballots, candidates, weights)
    pl = plurality_seats(district_votes, candidates)
    st = storting_seats(district_votes, candidates)

    def eligible_for(support: dict[str, int]) -> list[str]:
        if not sperregrense:
            return list(candidates)
        total = sum(support.values()) or 1
        return [c for c in candidates if support[c] / total >= THRESHOLD] or list(candidates)

    eligible = eligible_for(votes)
    for res in (pl, st):
        res.by_district[LEVELLING] = levelling_seats(votes, res.seats, eligible)
    # Under the Norwegian split MJ measures everything in supporters (Good or better),
    # so its fair share and threshold use supporters too. Otherwise votes, like the rest.
    mj_measure = mj_support(all_ballots, candidates) if weights == NORWAY else votes
    mj.by_district[LEVELLING] = levelling_seats(mj_measure, mj.seats, eligible_for(mj_measure))
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


def cses_election(year: int, rng: random.Random, weights: Split, sperregrense: bool = True,
                  zip_path: str = CSES_ZIP, cache_path: str = CSES_CACHE) -> tuple[list[str], Election]:
    """Spread the real respondents over the districts by population and count all systems."""
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
    return parties, count_all(district_ballots, parties, rng, weights, sperregrense, district_votes)


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

    dw = max(len(d) for d in e.districts)
    print(f"{'District':{dw}}  seats  voters  {'MJ winner':{width}}  {'Storting':{width}}  {'Plurality':{width}}")
    for d in e.districts:
        def top(res):
            best = max(res.by_district[d].values())
            return " = ".join(c for c in candidates if res.by_district[d][c] == best)
        voters = str(len(e.district_ballots[d])) if d in e.district_ballots else "national"
        print(f"{d:{dw}}  {seats_in(d):>5}  {voters:>8}  "
              f"{top(e.mj):{width}}  {top(e.storting):{width}}  {top(e.plurality):{width}}")
    if True:
        print()
        print(f"{LEVELLING} (levelling, fair share minus district seats):")
        for name, res in (("Majority Judgment", e.mj), ("Storting", e.storting), ("Plurality", e.plurality)):
            parts = ", ".join(f"{c} {n}" for c, n in res.by_district[LEVELLING].items() if n)
            print(f"  {name:<18} {parts or '-'}")
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
                    help="Majority Judgment mandate split per district: 1 (winner takes all), 2,1,1, 3,2,1 "
                         "or norway (modified Sainte-Lague on supporters, like the Storting count)")
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
