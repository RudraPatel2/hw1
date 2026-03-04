# movie_recommender.py
"""
HW1 Movie Recommendation System (Prototype)

Data files:
- Movies:  movie_genre|movie_id|movie_name
- Ratings: movie_name|rating|user_id

This program provides a CLI to load data and run recommendation-related features.
Compatible with Python 3.12.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Tuple, Optional


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


class MovieRecommender:
    """
    Stores loaded movies/ratings and provides methods to compute popularity,
    genre stats, user preferences, and recommendations.
    """

    def __init__(self) -> None:
        # Movies
        self.movies_by_name: Dict[str, Movie] = {}
        self.movies_by_genre: Dict[str, List[str]] = {}  # genre -> [movie_name]

        # Ratings
        self.ratings_by_movie: Dict[str, List[float]] = {}  # movie_name -> [ratings]
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
        loaded = 0
        skipped = 0

        with open(path, "r", encoding="utf-8") as f:
            for _line_num, raw in enumerate(f, start=1):
                line = raw.strip()
                if not line:
                    continue

                parts = line.split("|")
                if len(parts) != 3:
                    skipped += 1
                    continue

                genre, movie_id, movie_name = (p.strip() for p in parts)
                if not genre or not movie_id or not movie_name:
                    skipped += 1
                    continue

                # If duplicates exist, last one wins (simple policy).
                self.movies_by_name[movie_name] = Movie(genre=genre, movie_id=movie_id, name=movie_name)
                self.movies_by_genre.setdefault(genre, []).append(movie_name)
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
        loaded = 0
        skipped = 0

        with open(path, "r", encoding="utf-8") as f:
            for _line_num, raw in enumerate(f, start=1):
                line = raw.strip()
                if not line:
                    continue

                parts = line.split("|")
                if len(parts) != 3:
                    skipped += 1
                    continue

                movie_name, rating_str, user_id = (p.strip() for p in parts)
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
        num_genres = len(self.movies_by_genre)
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
            - Only includes movies that appear in the ratings file.
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

        Args:
            n: number of movies to return (if n <= 0 returns [])

        Returns:
            List of (movie_name, avg_rating, rating_count)

        Tie-breakers (deterministic):
            1) higher avg_rating
            2) higher rating_count
            3) alphabetical movie_name
        """
        if n <= 0:
            return []

        stats = self.movie_stats()
        ranked = sorted(
            ((name, avg, cnt) for name, (avg, cnt) in stats.items()),
            key=lambda t: (-t[1], -t[2], t[0])
        )
        return ranked[:n]

    # -----------------------
    # Movie popularity in genre
    # -----------------------

    def top_n_movies_in_genre(self, genre: str, n: int) -> List[Tuple[str, float, int]]:
        """
        Return the top n movies within a genre ranked by average rating.

        Args:
            genre: genre name exactly as in movies file (case-sensitive)
            n: number of movies to return

        Returns:
            List of (movie_name, avg_rating, rating_count)

        Notes:
            - Only movies in that genre that ALSO have ratings will appear.
        """
        if n <= 0:
            return []

        stats = self.movie_stats()
        candidates = self.movies_by_genre.get(genre, [])
        ranked: List[Tuple[str, float, int]] = []
        for name in candidates:
            if name in stats:
                avg, cnt = stats[name]
                ranked.append((name, avg, cnt))

        ranked.sort(key=lambda t: (-t[1], -t[2], t[0]))
        return ranked[:n]

    # -----------------------
    # Genre popularity
    # -----------------------

    def genre_popularity(self) -> Dict[str, float]:
        """
        Compute genre popularity as:
            average of (average movie ratings) within that genre.

        Returns:
            Dict mapping genre -> popularity_score (float, not rounded)

        Notes:
            - Uses each movie's average rating (across all users).
            - Only counts movies that have at least 1 rating.
            - Ignores ratings for movies not present in the movies file.
        """
        stats = self.movie_stats()

        genre_to_movie_avgs: Dict[str, List[float]] = {}
        for movie_name, (avg, _cnt) in stats.items():
            movie = self.movies_by_name.get(movie_name)
            if movie is None:
                continue
            genre_to_movie_avgs.setdefault(movie.genre, []).append(avg)

        out: Dict[str, float] = {}
        for g, avgs in genre_to_movie_avgs.items():
            if avgs:
                out[g] = sum(avgs) / len(avgs)
        return out

    def top_n_genres(self, n: int) -> List[Tuple[str, float]]:
        """
        Return top n genres ranked by genre popularity.

        Args:
            n: number of genres

        Returns:
            List of (genre, popularity_score)

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

        For each genre, compute:
            average of the user's ratings for movies in that genre
        and return the genre with the highest average.

        Args:
            user_id: user id as string (since we store ids as strings)

        Returns:
            (best_genre, avg_user_rating_in_genre) or None if user has no usable ratings.

        Tie-breakers:
            1) higher avg rating in genre
            2) genre name alphabetical
        """
        user_ratings = self.ratings_by_user.get(user_id)
        if not user_ratings:
            return None

        genre_to_vals: Dict[str, List[float]] = {}
        for movie_name, r in user_ratings.items():
            movie = self.movies_by_name.get(movie_name)
            if movie is None:
                continue
            genre_to_vals.setdefault(movie.genre, []).append(r)

        if not genre_to_vals:
            return None

        genre_to_avg: List[Tuple[str, float]] = []
        for g, vals in genre_to_vals.items():
            if vals:
                genre_to_avg.append((g, sum(vals) / len(vals)))

        if not genre_to_avg:
            return None

        genre_to_avg.sort(key=lambda t: (-t[1], t[0]))
        return genre_to_avg[0]

    # -----------------------
    # Recommend movies
    # -----------------------

    def recommend_movies_for_user(self, user_id: str, k: int = 3) -> List[Tuple[str, float, int]]:
        """
        Recommend up to k movies for the user:

        - Find user's top genre (by average of user's ratings in that genre)
        - From that genre, pick the most popular movies (by average rating overall)
          that the user has NOT rated yet.
        - Return up to k results.

        Args:
            user_id: user id as string
            k: number of recommendations (default 3)

        Returns:
            List of (movie_name, avg_rating, rating_count)

        Notes:
            - If user has no ratings, returns [].
            - If user's top genre has fewer than k unseen rated movies, returns fewer.
        """
        if k <= 0:
            return []

        pref = self.user_genre_preference(user_id)
        if pref is None:
            return []

        top_genre, _score = pref
        already_rated = set(self.ratings_by_user.get(user_id, {}).keys())

        # Start with ranked movies in that genre (by avg rating overall)
        ranked_in_genre = self.top_n_movies_in_genre(top_genre, n=10**9)  # effectively "all"
        unseen = [t for t in ranked_in_genre if t[0] not in already_rated]
        return unseen[:k]


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
            genre = input("Genre (case-sensitive): ").strip()
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
                print(f"Top genre for user {user_id}: {g} (avg={avg})")

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