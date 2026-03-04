#!/usr/bin/env python3
"""
movie_recommender.py

HW1 Movie Recommendation System (Prototype)

Data files:
- Movies:  movie_genre|movie_id|movie_name
- Ratings: movie_name|rating|user_id

Edge-case handling:
- Skips malformed lines (wrong field count, empty required fields)
- Movies:
  - Movie names are treated case-sensitive identifiers.
  - Leading/trailing whitespace is stripped.
  - Duplicate movie names are skipped (first one wins) to avoid double-adding to genre lists.
- Ratings:
  - Rating must be numeric float in [0, 5]
  - Skips ratings for unknown movies (not present in movies file)
  - Enforces user can rate a given movie only once (duplicate skipped)
- Genre input to queries is case-insensitive.

Compatible with Python 3.12.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Tuple, Optional
import os


@dataclass(frozen=True)
class Movie:
    """Represents a movie from the movies file."""
    genre: str
    movie_id: str
    name: str  # includes year, e.g. "Toy Story (1995)"


@dataclass(frozen=True)
class Rating:
    """Represents a single user rating for a movie."""
    movie_name: str  # includes year; matches movies file name field
    rating: float    # 0..5 inclusive
    user_id: str


def _canon_movie_name(name: str) -> str:
    """Movie names are case-sensitive; only strip outer whitespace."""
    return name.strip()


def _genre_key(genre: str) -> str:
    """Normalize genre for case-insensitive matching."""
    return genre.strip().lower()


class MovieRecommender:
    """
    Stores loaded movies/ratings and provides methods to compute popularity,
    genre stats, user preferences, and recommendations.
    """

    def __init__(self) -> None:
        # Movies
        self.movies_by_name: Dict[str, Movie] = {}           # movie_name -> Movie
        self.genre_display_by_key: Dict[str, str] = {}       # normalized genre -> display casing
        self.movies_by_genre_key: Dict[str, List[str]] = {}  # normalized genre -> [movie_name]

        # Ratings
        self.ratings_by_movie: Dict[str, List[float]] = {}   # movie_name -> [ratings]
        self.ratings_by_user: Dict[str, Dict[str, float]] = {}  # user_id -> {movie_name: rating}

    def reset(self) -> None:
        """Clear all loaded data."""
        self.__init__()

    # -----------------------
    # Loading / Validation
    # -----------------------

    def load_movies(self, path: str) -> Tuple[int, int]:
        """
        Load the movies file.

        Args:
            path: File path to movies text file.

        Returns:
            (loaded_count, skipped_count)
        """
        if not os.path.exists(path):
            raise FileNotFoundError(path)

        loaded = 0
        skipped = 0

        with open(path, "r", encoding="utf-8") as f:
            for _line_num, raw in enumerate(f, start=1):
                line = raw.rstrip("\n")
                if not line.strip():
                    continue

                parts = line.split("|")
                if len(parts) != 3:
                    skipped += 1
                    continue

                genre_raw, movie_id_raw, movie_name_raw = parts
                genre = genre_raw.strip()
                movie_id = movie_id_raw.strip()
                movie_name = _canon_movie_name(movie_name_raw)

                if not genre or not movie_id or not movie_name:
                    skipped += 1
                    continue

                # Duplicate movie name: skip to avoid corrupting movies_by_genre lists.
                if movie_name in self.movies_by_name:
                    skipped += 1
                    continue

                gk = _genre_key(genre)
                self.genre_display_by_key.setdefault(gk, genre)

                self.movies_by_name[movie_name] = Movie(genre=genre, movie_id=movie_id, name=movie_name)
                self.movies_by_genre_key.setdefault(gk, []).append(movie_name)
                loaded += 1

        return loaded, skipped

    def load_ratings(self, path: str) -> Tuple[int, int]:
        """
        Load the ratings file.

        Args:
            path: File path to ratings text file.

        Returns:
            (loaded_count, skipped_count)
        """
        if not os.path.exists(path):
            raise FileNotFoundError(path)

        loaded = 0
        skipped = 0

        with open(path, "r", encoding="utf-8") as f:
            for _line_num, raw in enumerate(f, start=1):
                line = raw.rstrip("\n")
                if not line.strip():
                    continue

                parts = line.split("|")
                if len(parts) != 3:
                    skipped += 1
                    continue

                movie_name_raw, rating_str_raw, user_id_raw = parts
                movie_name = _canon_movie_name(movie_name_raw)
                rating_str = rating_str_raw.strip()
                user_id = user_id_raw.strip()

                if not movie_name or not rating_str or not user_id:
                    skipped += 1
                    continue

                try:
                    rating = float(rating_str)
                except ValueError:
                    skipped += 1
                    continue

                if rating < 0 or rating > 5:
                    skipped += 1
                    continue

                # IMPORTANT: skip ratings for movies not in movies file.
                if movie_name not in self.movies_by_name:
                    skipped += 1
                    continue

                # Enforce: user can rate a particular movie only once
                user_map = self.ratings_by_user.setdefault(user_id, {})
                if movie_name in user_map:
                    skipped += 1
                    continue

                user_map[movie_name] = rating
                self.ratings_by_movie.setdefault(movie_name, []).append(rating)
                loaded += 1

        return loaded, skipped

    def data_summary(self) -> str:
        """Return a short summary of currently loaded data."""
        num_movies = len(self.movies_by_name)
        num_genres = len(self.genre_display_by_key)
        num_users = len(self.ratings_by_user)
        num_ratings = sum(len(v) for v in self.ratings_by_movie.values())
        return (
            f"Movies: {num_movies}\n"
            f"Genres: {num_genres}\n"
            f"Users with ratings: {num_users}\n"
            f"Total ratings: {num_ratings}"
        )

    # -----------------------
    # Movie popularity
    # -----------------------

    def movie_stats(self) -> Dict[str, Tuple[float, int]]:
        """
        Compute (average_rating, rating_count) for each movie that has ratings.

        Returns:
            Dict mapping movie_name -> (avg_rating, count)

        Notes:
            - Does not round averages.
            - Only includes movies that have at least 1 rating.
        """
        stats: Dict[str, Tuple[float, int]] = {}
        for movie_name, vals in self.ratings_by_movie.items():
            if not vals:
                continue
            avg = sum(vals) / len(vals)
            stats[movie_name] = (avg, len(vals))
        return stats

    def top_n_movies(self, n: int) -> List[Tuple[str, float, int]]:
        """
        Return the top n movies ranked by average rating.

        Tie-breakers:
            1) higher avg_rating
            2) alphabetical movie_name
        """
        if n <= 0:
            return []

        stats = self.movie_stats()
        ranked = sorted(
            ((name, avg, cnt) for name, (avg, cnt) in stats.items()),
            key=lambda t: (-t[1], t[0]),
        )
        return ranked[:n]

    # -----------------------
    # Movie popularity in genre
    # -----------------------

    def top_n_movies_in_genre(self, genre: str, n: int) -> List[Tuple[str, float, int]]:
        """
        Return the top n movies within a genre ranked by average rating.

        Args:
            genre: genre name (matched case-insensitively)
            n: number of movies to return

        Returns:
            List of (movie_name, avg_rating, rating_count)

        Notes:
            - Only includes movies in that genre that ALSO have ratings.
        """
        if n <= 0:
            return []

        gk = _genre_key(genre)
        candidates = self.movies_by_genre_key.get(gk, [])
        stats = self.movie_stats()

        ranked: List[Tuple[str, float, int]] = []
        for name in candidates:
            if name in stats:
                avg, cnt = stats[name]
                ranked.append((name, avg, cnt))

        ranked.sort(key=lambda t: (-t[1], t[0]))
        return ranked[:n]

    # -----------------------
    # Genre popularity
    # -----------------------

    def genre_popularity(self) -> Dict[str, float]:
        """
        Compute genre popularity as:
            average of (average movie ratings) within that genre.

        Returns:
            Dict mapping genre_display -> popularity_score (float, not rounded)

        Notes:
            - Uses each movie's average rating (across all users).
            - Only counts movies that have at least 1 rating.
        """
        stats = self.movie_stats()
        gk_to_avgs: Dict[str, List[float]] = {}

        for movie_name, (avg, _cnt) in stats.items():
            mv = self.movies_by_name.get(movie_name)
            if mv is None:
                continue
            gk = _genre_key(mv.genre)
            gk_to_avgs.setdefault(gk, []).append(avg)

        out: Dict[str, float] = {}
        for gk, avgs in gk_to_avgs.items():
            if avgs:
                display = self.genre_display_by_key.get(gk, gk)
                out[display] = sum(avgs) / len(avgs)
        return out

    def top_n_genres(self, n: int) -> List[Tuple[str, float]]:
        """
        Return top n genres ranked by genre popularity.

        Tie-breakers:
            1) higher popularity_score
            2) alphabetical genre
        """
        if n <= 0:
            return []

        pop = self.genre_popularity()
        ranked = sorted(pop.items(), key=lambda t: (-t[1], t[0]))
        return ranked[:n]

    # -----------------------
    # User preference for genre
    # -----------------------

    def user_genre_preference(self, user_id: str) -> Optional[Tuple[str, float]]:
        """
        Compute the user's most preferred genre.

        ASSIGNMENT definition:
            For each genre, compute the average of the GLOBAL movie-average-ratings
            for movies in that genre that the user rated.
            Then choose the genre with the highest value.

        Returns:
            (best_genre_display, preference_score) or None if user has no usable ratings.

        Tie-breakers:
            1) higher preference_score
            2) alphabetical genre display
        """
        user_ratings = self.ratings_by_user.get(user_id)
        if not user_ratings:
            return None

        stats = self.movie_stats()  # global movie averages

        gk_to_sum: Dict[str, float] = {}
        gk_to_cnt: Dict[str, int] = {}

        for movie_name in user_ratings.keys():
            mv = self.movies_by_name.get(movie_name)
            if mv is None:
                continue
            if movie_name not in stats:
                continue
            avg, _cnt = stats[movie_name]
            gk = _genre_key(mv.genre)
            gk_to_sum[gk] = gk_to_sum.get(gk, 0.0) + avg
            gk_to_cnt[gk] = gk_to_cnt.get(gk, 0) + 1

        if not gk_to_cnt:
            return None

        best_display: Optional[str] = None
        best_score: Optional[float] = None

        for gk, cnt in gk_to_cnt.items():
            score = gk_to_sum[gk] / cnt
            display = self.genre_display_by_key.get(gk, gk)
            if (
                best_score is None
                or score > best_score
                or (score == best_score and display < (best_display or ""))
            ):
                best_score = score
                best_display = display

        assert best_display is not None and best_score is not None
        return best_display, best_score

    # -----------------------
    # Recommend movies
    # -----------------------

    def recommend_movies_for_user(self, user_id: str, k: int = 3) -> List[Tuple[str, float, int]]:
        """
        Recommend up to k movies for the user:

        - Find user's top genre (per user_genre_preference)
        - From that genre, pick the most popular movies (by global avg rating)
          that the user has NOT rated yet.
        - Return up to k results as (movie_name, avg_rating, rating_count)
        """
        if k <= 0:
            return []

        pref = self.user_genre_preference(user_id)
        if pref is None:
            return []

        top_genre_display, _score = pref
        gk = _genre_key(top_genre_display)

        already_rated = set(self.ratings_by_user.get(user_id, {}).keys())
        stats = self.movie_stats()

        candidates: List[Tuple[str, float, int]] = []
        for movie_name in self.movies_by_genre_key.get(gk, []):
            if movie_name in already_rated:
                continue
            if movie_name not in stats:
                continue
            avg, cnt = stats[movie_name]
            candidates.append((movie_name, avg, cnt))

        candidates.sort(key=lambda t: (-t[1], t[0]))
        return candidates[:k]


def run_cli() -> None:
    """Run the command-line interface loop."""
    rec = MovieRecommender()

    while True:
        print("\n=== Movie Recommender CLI ===")
        print("1) Load movies file")
        print("2) Load ratings file")
        print("3) Show data summary")
        print("4) Reset loaded data")
        print("5) Top N movies (overall)")
        print("6) Top N movies in a genre")
        print("7) Top N genres")
        print("8) User top genre preference")
        print("9) Recommend 3 movies for a user")
        print("0) Exit")

        choice = input("Select: ").strip()

        if choice == "1":
            path = input("Movies file path: ").strip()
            try:
                loaded, skipped = rec.load_movies(path)
                print(f"Loaded {loaded} movies; skipped {skipped} lines.")
            except FileNotFoundError:
                print("File not found.")
            except OSError as e:
                print(f"File error: {e}")

        elif choice == "2":
            path = input("Ratings file path: ").strip()
            try:
                loaded, skipped = rec.load_ratings(path)
                print(f"Loaded {loaded} ratings; skipped {skipped} lines.")
            except FileNotFoundError:
                print("File not found.")
            except OSError as e:
                print(f"File error: {e}")

        elif choice == "3":
            print("\n--- Summary ---")
            print(rec.data_summary())

        elif choice == "4":
            rec.reset()
            print("Data cleared.")

        elif choice == "5":
            try:
                n = int(input("N: ").strip())
            except ValueError:
                print("Invalid N.")
                continue
            for name, avg, cnt in rec.top_n_movies(n):
                print(f"{name} | avg={avg} | count={cnt}")

        elif choice == "6":
            genre = input("Genre: ").strip()
            try:
                n = int(input("N: ").strip())
            except ValueError:
                print("Invalid N.")
                continue
            for name, avg, cnt in rec.top_n_movies_in_genre(genre, n):
                print(f"{name} | avg={avg} | count={cnt}")

        elif choice == "7":
            try:
                n = int(input("N: ").strip())
            except ValueError:
                print("Invalid N.")
                continue
            for genre, score in rec.top_n_genres(n):
                print(f"{genre} | score={score}")

        elif choice == "8":
            user_id = input("User id: ").strip()
            pref = rec.user_genre_preference(user_id)
            if pref is None:
                print("No preference found (user missing or no usable ratings).")
            else:
                g, avg = pref
                print(f"Top genre for user {user_id}: {g} (score={avg})")

        elif choice == "9":
            user_id = input("User id: ").strip()
            recs = rec.recommend_movies_for_user(user_id, k=3)
            if not recs:
                print("No recommendations available.")
            else:
                print(f"Recommendations for user {user_id}:")
                for name, avg, cnt in recs:
                    print(f"{name} | avg={avg} | count={cnt}")

        elif choice == "0":
            print("Bye.")
            return

        else:
            print("Invalid choice.")


if __name__ == "__main__":
    run_cli()
