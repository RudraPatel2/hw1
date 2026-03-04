#!/usr/bin/env python3
"""
test_movie_recommender.py

Automatic tester for movie_recommender.py (CLASS-BASED version).

Usage:
    python test_movie_recommender.py

Assumes movie_recommender.py defines a class MovieRecommender with methods:
- load_movies(path) -> (loaded_count, skipped_count)
- load_ratings(path) -> (loaded_count, skipped_count)
- data_summary() -> str
- top_n_movies(n) -> list[(movie_name, avg, count)]
- top_n_movies_in_genre(genre, n) -> list[(movie_name, avg, count)]
- top_n_genres(n) -> list[(genre, avg)]
- user_genre_preference(user_id) -> (genre, score) or None
- recommend_movies_for_user(user_id, k) -> list[(movie_name, avg, count)] or list[(movie_name, avg)] depending on your implementation

This tester expects (movie_name, avg, count) for top_n_movies/top_n_movies_in_genre/recommend.
If your recommend() returns only (name, avg), adjust the check function accordingly.
"""

from __future__ import annotations
from typing import Any, List, Tuple, Optional

from movie_recommender import MovieRecommender  # <-- matches your current setup

MOVIES_FILE = "movies.txt"
RATINGS_FILE = "ratings.txt"


def approx(a: float, b: float, eps: float = 1e-12) -> bool:
    return abs(a - b) <= eps


def print_section(title: str) -> None:
    print("\n" + "=" * 10 + f" {title} " + "=" * 10)


def check_equal(label: str, got: Any, expected: Any) -> None:
    print(f"{label} (GOT):      {got}")
    print(f"{label} (EXPECTED): {expected}")
    ok = got == expected
    print(f"[{'PASS' if ok else 'FAIL'}] {label}")


def check_list_movie_rows(
    label: str,
    got: List[Tuple[str, float, int]],
    expected: List[Tuple[str, float, int]],
) -> None:
    print(f"{label} (GOT):")
    print(got)
    print(f"{label} (EXPECTED):")
    print(expected)

    ok = True
    if len(got) != len(expected):
        ok = False
    else:
        for (gn, ga, gc), (en, ea, ec) in zip(got, expected):
            if gn != en or gc != ec or not approx(ga, ea):
                ok = False
                break

    print(f"[{'PASS' if ok else 'FAIL'}] {label}")


def check_list_genres(
    label: str,
    got: List[Tuple[str, float]],
    expected: List[Tuple[str, float]],
) -> None:
    print(f"{label} (GOT):")
    print(got)
    print(f"{label} (EXPECTED):")
    print(expected)

    ok = True
    if len(got) != len(expected):
        ok = False
    else:
        for (gg, gs), (eg, es) in zip(got, expected):
            if gg != eg or not approx(gs, es):
                ok = False
                break

    print(f"[{'PASS' if ok else 'FAIL'}] {label}")


def check_tuple_str_float(
    label: str,
    got: Optional[Tuple[str, float]],
    expected: Optional[Tuple[str, float]],
) -> None:
    print(f"{label} (GOT):      {got}")
    print(f"{label} (EXPECTED): {expected}")

    if expected is None:
        ok = got is None
    else:
        ok = got is not None and got[0] == expected[0] and approx(got[1], expected[1])

    print(f"[{'PASS' if ok else 'FAIL'}] {label}")


def main() -> None:
    rec = MovieRecommender()

    print_section("LOAD FILES")
    try:
        m_loaded, m_skipped = rec.load_movies(MOVIES_FILE)
        r_loaded, r_skipped = rec.load_ratings(RATINGS_FILE)
    except FileNotFoundError as e:
        print(f"[FAIL] Missing file: {e}")
        return

    print(f"Movies loaded={m_loaded}, skipped={m_skipped}")
    print(f"Ratings loaded={r_loaded}, skipped={r_skipped}")

    print_section("DATA SUMMARY")
    summary = rec.data_summary()
    print(summary)

    print_section("TOP MOVIES")
    top3_movies = rec.top_n_movies(3)
    print(top3_movies)

    print_section("TOP MOVIES IN GENRE (case-insensitive input test)")
    top3_action = rec.top_n_movies_in_genre("aCtIoN", 3)
    print(top3_action)

    print_section("TOP GENRES")
    top4_genres = rec.top_n_genres(4)
    print(top4_genres)

    print_section("USER PREFERRED GENRE")
    user42_pref = rec.user_genre_preference("42")
    print(user42_pref)

    print_section("RECOMMENDATIONS")
    user42_recs = rec.recommend_movies_for_user("42", 3)
    print(user42_recs)

    # --------------------------
    # Expected checks for NEW dataset
    # --------------------------
    print_section("EXPECTED VALUE CHECKS (new dataset)")

    expected_summary_lines = [
        "Movies: 14",
        "Genres: 4",
        "Users with ratings: 16",
        "Total ratings: 42",
    ]
    check_equal("data_summary().splitlines()", summary.splitlines(), expected_summary_lines)

    expected_top3_movies = [
        ("Neon Future (2016)", 4.75, 2),
        ("Alpha Force (2010)", 4.625, 4),
        ("Tears of Steel (2014)", 4.375, 4),
    ]
    check_list_movie_rows("top_n_movies(3)", top3_movies, expected_top3_movies)

    expected_top3_action = [
        ("Alpha Force (2010)", 4.625, 4),
        ("Action Tie (2020)", 4.0, 2),
        ("Beta Strike (2011)", 4.0, 4),
    ]
    check_list_movie_rows("top_n_movies_in_genre('aCtIoN', 3)", top3_action, expected_top3_action)

    expected_top4_genres = [
        ("Sci-Fi", 4.25),
        ("Action", 3.90625),
        ("Drama", 3.6875),
        ("Comedy", 3.6111111111111107),
    ]
    check_list_genres("top_n_genres(4)", top4_genres, expected_top4_genres)

    expected_user42_pref = ("Sci-Fi", 4.75)
    check_tuple_str_float("user_genre_preference('42')", user42_pref, expected_user42_pref)

    expected_user42_recs = [
        ("Quantum Drift (2021)", 4.25, 4),
        ("Space Oddity (2018)", 4.166666666666667, 3),
        ("Orbit Zero (2017)", 3.8333333333333335, 3),
    ]
    check_list_movie_rows("recommend_movies_for_user('42', 3)", user42_recs, expected_user42_recs)


if __name__ == "__main__":
    main()
