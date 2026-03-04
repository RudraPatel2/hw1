# test_movie_recommender.py
"""
Automatic tester for movie_recommender.py

Usage:
    python test_movie_recommender.py

What it does:
- Loads input files
- Calls each required function
- Prints results
- Compares printed results against expected values (for the sample dataset)
"""

from __future__ import annotations

from typing import Any, List, Tuple, Optional
from movie_recommender import MovieRecommender


MOVIES_FILE = "movies.txt"
RATINGS_FILE = "ratings.txt"


def approx(a: float, b: float, eps: float = 1e-12) -> bool:
    """Float compare helper (no rounding; just tolerance for float representation)."""
    return abs(a - b) <= eps


def print_section(title: str) -> None:
    print("\n" + "=" * 10 + f" {title} " + "=" * 10)


def check_equal(label: str, got: Any, expected: Any) -> None:
    """Print got vs expected and PASS/FAIL for exact-equality checks."""
    print(f"{label} (GOT):      {got}")
    print(f"{label} (EXPECTED): {expected}")
    ok = got == expected
    print(f"[{'PASS' if ok else 'FAIL'}] {label}")


def check_list_movie_rows(
    label: str,
    got: List[Tuple[str, float, int]],
    expected: List[Tuple[str, float, int]],
) -> None:
    """Print got vs expected and PASS/FAIL for lists of (name, avg, count)."""
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


def check_tuple_str_float(
    label: str,
    got: Optional[Tuple[str, float]],
    expected: Tuple[str, float],
) -> None:
    """Print got vs expected and PASS/FAIL for (string, float)."""
    print(f"{label} (GOT):      {got}")
    print(f"{label} (EXPECTED): {expected}")
    ok = got is not None and got[0] == expected[0] and approx(got[1], expected[1])
    print(f"[{'PASS' if ok else 'FAIL'}] {label}")


def check_list_genres(
    label: str,
    got: List[Tuple[str, float]],
    expected: List[Tuple[str, float]],
) -> None:
    """Print got vs expected and PASS/FAIL for lists of (genre, score)."""
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


def main() -> None:
    rec = MovieRecommender()

    print_section("LOAD FILES")
    try:
        m_loaded, m_skipped = rec.load_movies(MOVIES_FILE)
        r_loaded, r_skipped = rec.load_ratings(RATINGS_FILE)
    except FileNotFoundError as e:
        print(f"[FAIL] Missing file: {e}")
        print("Make sure movies.txt and ratings.txt are in the same folder as this tester.")
        return

    print(f"Movies loaded={m_loaded}, skipped={m_skipped}")
    print(f"Ratings loaded={r_loaded}, skipped={r_skipped}")

    # --------------------------
    # Call each required function
    # --------------------------

    print_section("DATA SUMMARY")
    summary = rec.data_summary()
    print(summary)

    print_section("MOVIE POPULARITY (Top N Movies)")
    top3_movies = rec.top_n_movies(3)
    print(top3_movies)

    print_section("MOVIE POPULARITY IN GENRE (Top N Movies in Genre)")
    top3_action = rec.top_n_movies_in_genre("Action", 3)
    print(top3_action)

    print_section("GENRE POPULARITY (Top N Genres)")
    top3_genres = rec.top_n_genres(3)
    print(top3_genres)

    print_section("USER PREFERENCE FOR GENRE")
    user6_pref = rec.user_genre_preference("6")
    print(user6_pref)

    print_section("RECOMMEND MOVIES (3 unseen from user's top genre)")
    user6_recs = rec.recommend_movies_for_user("6", 3)
    print(user6_recs)

    # --------------------------
    # Compare against expected values (sample dataset)
    # --------------------------

    print_section("EXPECTED VALUE CHECKS (sample dataset)")

    expected_summary_lines = [
        "Movies: 9",
        "Genres: 3",
        "Users with ratings: 36",
        "Total ratings: 54",
    ]
    check_equal("data_summary().splitlines()", summary.splitlines(), expected_summary_lines)

    expected_top3_movies = [
        ("Heat (1995)", 4.25, 6),
        ("Father of the Bride Part II (1995)", 4.0, 6),
        ("Grumpier Old Men (1995)", 4.0, 6),
    ]
    check_list_movie_rows("top_n_movies(3)", top3_movies, expected_top3_movies)

    expected_top3_action = [
        ("Heat (1995)", 4.25, 6),
        ("Sudden Death (1995)", 3.3333333333333335, 6),
        ("GoldenEye (1995)", 3.0, 6),
    ]
    check_list_movie_rows("top_n_movies_in_genre('Action', 3)", top3_action, expected_top3_action)

    expected_top3_genres = [
        ("Action", 3.527777777777778),
        ("Comedy", 3.5),
        ("Adventure", 3.361111111111111),
    ]
    check_list_genres("top_n_genres(3)", top3_genres, expected_top3_genres)

    expected_user6_pref = ("Comedy", 4.333333333333333)
    check_tuple_str_float("user_genre_preference('6')", user6_pref, expected_user6_pref)

    expected_user6_recs: List[Tuple[str, float, int]] = []
    check_equal("recommend_movies_for_user('6', 3)", user6_recs, expected_user6_recs)


if __name__ == "__main__":
    main()
