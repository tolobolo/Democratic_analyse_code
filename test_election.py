import random

import pytest

from election import (
    GRADES_DESC,
    Grade,
    grades_for,
    compare,
    majority_grade,
    mj_rank,
    mj_winner,
    plurality_tally,
    sainte_lague,
    simulate,
    storting_from_votes,
    storting_tally,
    collect_votes,
    DISTRICTS,
    STORTING_SEATS,
    count_all,
    parse_split,
    simulate_districts,
    split_seats,
    split_voters,
    NORWAY,
    LEVELLING,
    LEVELLING_SEATS,
    levelling_seats,
    seats_in,
    SCORE_TO_GRADE,
    load_cses,
    cses_election,
    score_grade,
)

G = Grade


def test_grade_order_and_labels():
    assert GRADES_DESC == [G.EXCELLENT, G.VERY_GOOD, G.GOOD, G.ACCEPTABLE, G.POOR, G.REJECT]
    assert G.EXCELLENT.beats(G.VERY_GOOD) and G.POOR.beats(G.REJECT)
    assert G.REJECT.loses_to(G.POOR) and not G.GOOD.beats(G.GOOD)
    assert G.VERY_GOOD.label == "Very Good"
    assert G.REJECT.label == "Reject"


def test_majority_grade_odd_and_even():
    assert majority_grade([G.POOR, G.EXCELLENT, G.GOOD]) == G.GOOD
    # even, middles equal: that is the median
    assert majority_grade([G.EXCELLENT, G.GOOD, G.GOOD, G.POOR]) == G.GOOD
    # even, middles differ: drop the worst and take the middle of what is left
    assert majority_grade([G.EXCELLENT, G.GOOD, G.ACCEPTABLE, G.POOR]) == G.GOOD
    assert majority_grade([G.GOOD, G.REJECT]) == G.GOOD


def test_blank_counts_as_reject():
    ballots = [{"A": G.GOOD}, {"A": G.GOOD, "B": G.EXCELLENT}, {}]
    assert grades_for(ballots, "B") == [G.REJECT, G.EXCELLENT, G.REJECT]
    assert len(grades_for(ballots, "A")) == 3


def test_tie_break_more_above_median_wins():
    # Both have majority grade GOOD. A has one grade above it, B has none.
    ballots = [
        {"A": G.EXCELLENT, "B": G.GOOD},
        {"A": G.GOOD, "B": G.GOOD},
        {"A": G.GOOD, "B": G.GOOD},
        {"A": G.GOOD, "B": G.GOOD},
        {"A": G.POOR, "B": G.REJECT},
    ]
    ranked = mj_rank(ballots, ["A", "B"])
    assert ranked[0].majority == ranked[1].majority == G.GOOD
    assert ranked[0].name == "A"
    assert ranked[0].p == pytest.approx(0.2)
    assert ranked[1].q == pytest.approx(0.2)


def test_tie_break_remove_worst_and_retry():
    # Same median (Acceptable), same count above (2). Remove the worst from both
    # and start over: A's median becomes Excellent, B's becomes Good, so A wins.
    a = [G.EXCELLENT, G.EXCELLENT, G.ACCEPTABLE, G.POOR, G.REJECT]
    b = [G.GOOD, G.GOOD, G.ACCEPTABLE, G.POOR, G.REJECT]
    assert majority_grade(a) == majority_grade(b) == G.ACCEPTABLE
    assert compare(a, b) < 0
    assert compare(b, a) > 0
    assert compare(a, list(a)) == 0


def test_mj_winner_prefers_higher_median():
    ballots = [
        {"A": G.EXCELLENT, "B": G.ACCEPTABLE},
        {"A": G.REJECT, "B": G.ACCEPTABLE},
        {"A": G.REJECT, "B": G.ACCEPTABLE},
    ]
    assert mj_winner(ballots, ["A", "B"]) == "B"


def test_plurality_simple_and_tie():
    rng = random.Random(0)
    ballots = [
        {"A": G.EXCELLENT, "B": G.POOR},
        {"A": G.GOOD, "B": G.POOR},
        {"A": G.REJECT, "B": G.GOOD},
    ]
    res = plurality_tally(ballots, ["A", "B"], rng)
    assert res.counts == {"A": 2, "B": 1}
    assert res.winners == ["A"]
    assert res.ambiguous == 0

    tie = plurality_tally(ballots[1:], ["A", "B"], rng)
    assert tie.winners == ["A", "B"]


def test_plurality_ambiguous_ballot():
    rng = random.Random(0)
    res = plurality_tally([{"A": G.GOOD, "B": G.GOOD}], ["A", "B"], rng)
    assert res.ambiguous == 1
    assert sum(res.counts.values()) == 1


def test_simulate_one_favourite_random_rest():
    ballots = simulate(["A", "B", "C"], 300, random.Random(1))
    assert len(ballots) == 300
    for b in ballots:
        vals = list(b.values())
        assert vals.count(G.EXCELLENT) == 1  # exactly one favourite
    # the other grades all show up for every party
    for c in ["A", "B", "C"]:
        assert set(grades_for(ballots, c)) == set(G)


def test_simulate_reproducible():
    assert simulate(["A", "B"], 100, random.Random(7)) == simulate(["A", "B"], 100, random.Random(7))


def test_sainte_lague_worked_example():
    # 1.4, 3, 5, 7...: A 37.9, 17.7, 10.6, 7.57 | B 17.1, 8.0 | C 16.4, 7.67 -> 8 seats
    assert sainte_lague({"A": 53, "B": 24, "C": 23}, 8) == {"A": 4, "B": 2, "C": 2}


def test_first_divisor_penalises_small_party():
    votes = {"A": 92, "B": 8}
    assert sainte_lague(votes, 7, first_divisor=1.0) == {"A": 6, "B": 1}
    assert sainte_lague(votes, 7, first_divisor=1.4) == {"A": 7, "B": 0}


def test_sainte_lague_edge_cases():
    assert sainte_lague({"A": 100, "B": 0}, 5) == {"A": 5, "B": 0}
    assert sainte_lague({"Only": 1}, 169) == {"Only": 169}
    seats = sainte_lague({"A": 400, "B": 350, "C": 250}, 169)
    assert sum(seats.values()) == 169


def test_storting_uses_same_votes_as_plurality():
    ballots = simulate(["A", "B", "C"], 500, random.Random(2))
    counts, amb = collect_votes(ballots, ["A", "B", "C"], random.Random(0))
    st = storting_from_votes(counts, amb)
    assert st.votes == counts
    assert sum(st.seats.values()) == 169
    assert st.ambiguous == amb
    assert st.largest  # at least one largest party


def test_storting_tally_wrapper():
    st = storting_tally([{"A": G.EXCELLENT, "B": G.POOR}] * 10, ["A", "B"], random.Random(0), seats=3)
    assert st.seats == {"A": 3, "B": 0}
    assert st.largest == ["A"]


def test_split_seats_rules():
    parties = ["A", "B", "C", "D"]
    assert split_seats(8, parties, [1]) == {"A": 8, "B": 0, "C": 0, "D": 0}
    assert split_seats(8, parties, [2, 1, 1]) == {"A": 4, "B": 2, "C": 2, "D": 0}
    assert split_seats(6, parties, [3, 2, 1]) == {"A": 3, "B": 2, "C": 1, "D": 0}
    # 7 seats at 2,1,1: exact 3.5, 1.75, 1.75 -> floors 3,1,1, two leftovers to the
    # largest remainders (.75, .75) -> 3, 2, 2
    assert split_seats(7, parties, [2, 1, 1]) == {"A": 3, "B": 2, "C": 2, "D": 0}
    # 4 seats at 3,2,1: 2, 1.33, .67 -> floors 2,1,0, one leftover to .67 -> 2,1,1
    assert split_seats(4, parties, [3, 2, 1]) == {"A": 2, "B": 1, "C": 1, "D": 0}
    # fewer parties than weights: 3:2 only
    assert split_seats(5, ["A", "B"], [3, 2, 1]) == {"A": 3, "B": 2}


def test_parse_split():
    assert parse_split("norway") == NORWAY and parse_split(" Norway ") == NORWAY
    assert parse_split("1") == [1]
    assert parse_split("2, 1,1") == [2, 1, 1]
    with pytest.raises(ValueError):
        parse_split("2,0")
    with pytest.raises(ValueError):
        parse_split("a,b")


def test_districts_table():
    assert len(DISTRICTS) == 19
    assert sum(m for m, _ in DISTRICTS.values()) == STORTING_SEATS == 169
    assert DISTRICTS["Akershus"][0] == 20 and DISTRICTS["Finnmark"][0] == 4


def test_split_voters_sums_and_is_proportional():
    v = split_voters(10000)
    assert sum(v.values()) == 10000
    assert v["Oslo"] > 8 * v["Finnmark"]
    assert LEVELLING not in v


def test_seats_in():
    assert seats_in("Oslo") == 19 and seats_in("Finnmark") == 3
    assert seats_in(LEVELLING) == LEVELLING_SEATS == 19
    assert sum(seats_in(d) for d in DISTRICTS) == 150


def test_all_systems_hand_out_169_seats_both_modes():
    parties = ["A", "B", "C"]
    for sperregrense in (True, False):
        db = simulate_districts(parties, 5000, random.Random(4))
        assert sum(len(b) for b in db.values()) == 5000
        assert LEVELLING not in db
        for weights in ([1], [2, 1, 1], [3, 2, 1]):
            e = count_all(db, parties, random.Random(0), weights, sperregrense)
            for res in (e.mj, e.plurality, e.storting):
                assert sum(res.seats.values()) == 169
                assert set(res.by_district) == set(DISTRICTS) | {LEVELLING}
                assert sum(res.by_district[LEVELLING].values()) == 19
            assert sum(e.votes.values()) == 5000


def test_count_all_rejects_utjevning_ballots():
    db = simulate_districts(["A", "B"], 200, random.Random(0))
    db[LEVELLING] = db["Oslo"]
    with pytest.raises(ValueError):
        count_all(db, ["A", "B"], random.Random(0), [1])


def test_sperregrense_blocks_small_party_from_pool():
    # C is the favourite of 2% of voters: wins nothing from Utjevning when ON, can when OFF.
    parties = ["A", "B", "C"]

    def ballots(n):
        out = []
        for i in range(n):
            fav = "C" if i % 50 == 0 else ("A" if i % 2 else "B")
            out.append({c: G.EXCELLENT if c == fav else G.POOR for c in parties})
        return out

    db_on = {d: ballots(100) for d in DISTRICTS}
    e_on = count_all(db_on, parties, random.Random(0), [3, 2, 1], True)
    assert e_on.eligible == ["A", "B"]
    assert e_on.mj.by_district[LEVELLING]["C"] == 0
    assert e_on.plurality.by_district[LEVELLING]["C"] == 0
    assert e_on.storting.by_district[LEVELLING]["C"] == 0

    # OFF: C is eligible. Under Storting it has no district seats and 2% of the vote,
    # so levelling tops it up to its fair share. (Under the 3,2,1 split C already holds
    # third-place district seats above its fair share, so it gets none there.)
    e_off = count_all(db_on, parties, random.Random(0), [3, 2, 1], False)
    assert e_off.eligible == parties
    assert e_off.storting.by_district[LEVELLING]["C"] > 0
    for res in (e_off.mj, e_off.plurality, e_off.storting):
        assert sum(res.by_district[LEVELLING].values()) == 19


def test_levelling_fills_deficit():
    votes = {"A": 500, "B": 300, "C": 200}
    # fair share of 169 by Sainte-Lague: A 85, B 50, C 34 (approx). A already has too many
    # district seats, so it is taken out and B and C fill their deficits with the 19.
    district = {"A": 100, "B": 40, "C": 10}
    lev = levelling_seats(votes, district, ["A", "B", "C"])
    assert lev["A"] == 0
    assert sum(lev.values()) == 19
    final = {c: district[c] + lev[c] for c in votes}
    # B and C end up at their fair share of the 69 seats left after A
    assert final["B"] + final["C"] == 69
    assert final["B"] > final["C"]

    # ineligible party keeps its district seats, gets no levelling seats
    lev2 = levelling_seats(votes, {"A": 70, "B": 50, "C": 30}, ["A", "B"])
    assert lev2["C"] == 0 and sum(lev2.values()) == 19

    # nothing to correct: pure proportional top-up
    lev3 = levelling_seats(votes, {"A": 75, "B": 45, "C": 30}, ["A", "B", "C"])
    assert sum(lev3.values()) == 19 and all(n >= 0 for n in lev3.values())


def test_levelling_makes_storting_proportional():
    parties = ["A", "B", "C"]
    db = simulate_districts(parties, 20000, random.Random(11))
    e = count_all(db, parties, random.Random(0), [1], True)
    total = sum(e.votes.values())
    for c in parties:
        assert abs(e.storting.seats[c] / 169 - e.votes[c] / total) < 0.02
    # the pool is always exactly 19 and only goes to parties below their fair share
    for res in (e.mj, e.plurality, e.storting):
        pool = res.by_district[LEVELLING]
        assert sum(pool.values()) == 19
        fair = sainte_lague(e.votes, 169)
        for c in parties:
            district_only = res.seats[c] - pool[c]
            if pool[c] > 0:
                assert district_only + pool[c] <= fair[c] + 1  # filled up to (about) fair share


def test_winner_takes_all_when_unanimous():
    parties = ["A", "B"]
    db = {d: [{"A": G.EXCELLENT, "B": G.REJECT}] * 10 for d in DISTRICTS}
    e = count_all(db, parties, random.Random(0), [1], True)
    assert e.mj.seats == {"A": 169, "B": 0}
    assert e.plurality.seats == {"A": 169, "B": 0}
    assert e.storting.seats == {"A": 169, "B": 0}
    assert e.agree() == "A"


def test_bullet_voters_only_flip_median_past_half():
    # agent.md: bullet voting does not break the median unless the group is over 50%.
    honest = [{"A": G.GOOD} for _ in range(100)]
    attack = [{"A": G.REJECT} for _ in range(99)]
    assert majority_grade(grades_for(honest + attack, "A")) == G.GOOD
    attack = [{"A": G.REJECT} for _ in range(101)]
    assert majority_grade(grades_for(honest + attack, "A")) == G.REJECT


def test_score_to_grade_mapping():
    assert SCORE_TO_GRADE[10] == SCORE_TO_GRADE[9] == G.EXCELLENT
    assert SCORE_TO_GRADE[0] == G.REJECT and SCORE_TO_GRADE[1] == G.POOR
    assert score_grade("7") == G.VERY_GOOD
    for missing in ("", "96", "98", "99", "abc"):
        assert score_grade(missing) == G.REJECT


def _write_cache(path, repeat=1):
    import csv
    cols = (["IMD1004", "IMD1008_YEAR", "IMD3002_LH_PL"]
            + [f"IMD3008_{s}" for s in "ABCDEFGHI"] + [f"IMD5000_{s}" for s in "ABCDEFGHI"])
    codes = ["5780001", "5780002", "5780003"] + ["9999999"] * 6
    rows = [
        # vote Ap, scores Ap 10, Høyre 5, Frp 0
        ("5780001", ["10", "5", "0"]),
        # vote missing -> highest rated (Høyre 9)
        ("9999999", ["3", "9", "1"]),
        # vote for a party not rated this year -> highest rated (Frp 8)
        ("5780009", ["2", "99", "8"]),
    ]
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=cols)
        w.writeheader()
        for vote, scores in rows * repeat:
            row = {"IMD1004": "NOR_2017", "IMD1008_YEAR": "2017", "IMD3002_LH_PL": vote}
            for i, s in enumerate("ABCDEFGHI"):
                row[f"IMD3008_{s}"] = scores[i] if i < 3 else "99"
                row[f"IMD5000_{s}"] = codes[i]
            w.writerow(row)


def test_load_cses_from_cache(tmp_path):
    cache = tmp_path / "cache.csv"
    _write_cache(cache)
    parties, ballots, votes = load_cses(2017, random.Random(0), zip_path="missing.zip", cache_path=str(cache))
    assert parties == ["Ap", "Høyre", "Frp"]  # empty slots dropped
    assert ballots[0] == {"Ap": G.EXCELLENT, "Høyre": G.GOOD, "Frp": G.REJECT}
    assert ballots[2]["Høyre"] == G.REJECT  # missing score counts as Reject
    assert votes == ["Ap", "Høyre", "Frp"]
    with pytest.raises(ValueError):
        load_cses(2013, random.Random(0), zip_path="missing.zip", cache_path=str(cache))
    with pytest.raises(ValueError):
        load_cses(2000, random.Random(0), zip_path="missing.zip", cache_path=str(cache))


def test_cses_election_counts_real_votes(tmp_path):
    cache = tmp_path / "cache.csv"
    _write_cache(cache, repeat=14)  # 42 respondents, 14 per pattern
    parties, e = cses_election(2017, random.Random(0), [1], True, zip_path="missing.zip", cache_path=str(cache))
    assert e.n_voters == 42
    assert e.votes == {"Ap": 14, "Høyre": 14, "Frp": 14}
    assert all(len(b) >= 1 for b in e.district_ballots.values())
    for res in (e.mj, e.plurality, e.storting):
        assert sum(res.seats.values()) == 169
    # too few respondents for 19 districts
    small = tmp_path / "small.csv"
    _write_cache(small, repeat=2)
    with pytest.raises(ValueError):
        cses_election(2017, random.Random(0), [1], True, zip_path="missing.zip", cache_path=str(small))


def test_plurality_is_always_winner_takes_all():
    parties = ["A", "B", "C"]
    db = simulate_districts(parties, 5000, random.Random(9))
    results = [count_all(db, parties, random.Random(0), w, True) for w in ([1], [2, 1, 1], NORWAY)]
    for e in results:
        assert e.plurality.by_district == results[0].plurality.by_district
        assert e.storting.by_district == results[0].storting.by_district
        for d, seats in e.plurality.by_district.items():
            if d != LEVELLING:
                assert sorted(seats.values())[-2] == 0  # one party has everything in the district
        for res in (e.mj, e.plurality, e.storting):
            assert sum(res.seats.values()) == 169
    assert results[0].mj.by_district != results[2].mj.by_district  # split does change MJ


def test_norway_split_mj_uses_good_or_better():
    parties = ["A", "B"]
    # Everyone grades A Good and B Acceptable: only A has supporters, so A takes every district seat.
    db = {d: [{"A": G.GOOD, "B": G.ACCEPTABLE}] * 20 for d in DISTRICTS}
    e = count_all(db, parties, random.Random(0), NORWAY, True)
    assert all(v["A"] == seats_in(d) and v["B"] == 0 for d, v in e.mj.by_district.items() if d != LEVELLING)
    # Half grade A Excellent / B Poor, half grade B Very Good / A Acceptable: supporters 10 vs 10
    db = {d: [{"A": G.EXCELLENT, "B": G.POOR}] * 10 + [{"A": G.ACCEPTABLE, "B": G.VERY_GOOD}] * 10 for d in DISTRICTS}
    e = count_all(db, parties, random.Random(0), NORWAY, True)
    assert abs(e.mj.seats["A"] - e.mj.seats["B"]) <= 19  # proportional, not winner takes all


def test_norway_split_mj_levelling_uses_supporters():
    parties = ["A", "B"]
    # Everyone's favourite is A (all votes), but everyone also grades B Good: supporters 20 vs 20.
    db = {d: [{"A": G.EXCELLENT, "B": G.GOOD}] * 20 for d in DISTRICTS}
    e = count_all(db, parties, random.Random(0), NORWAY, True)
    assert e.storting.seats == {"A": 169, "B": 0}  # votes: A has them all
    assert abs(e.mj.seats["A"] - e.mj.seats["B"]) <= 1  # supporters: equal, after levelling too
