# test_movie_recommender.py

from movie_recommender import MovieRecommender


def main():
    rec = MovieRecommender()
    rec.load_movies("movies.txt")
    rec.load_ratings("ratings.txt")

    print("=== BASIC STRUCTURE TESTS ===")

    # 1. Every top movie must actually exist in stats
    top_movies = rec.top_n_movies(5)
    stats = rec.movie_stats()
    assert all(name in stats for name, _, _ in top_movies)
    print("[PASS] top movies exist in stats")

    # 2. Sorted correctly by average
    avgs = [avg for _, avg, _ in top_movies]
    assert avgs == sorted(avgs, reverse=True)
    print("[PASS] movies sorted by descending average")

    # 3. Genre movies belong to that genre
    for genre in rec.movies_by_genre:
        top_genre_movies = rec.top_n_movies_in_genre(genre, 5)
        for name, _, _ in top_genre_movies:
            assert rec.movies_by_name[name].genre == genre
    print("[PASS] genre filtering correct")

    # 4. Recommended movies are unseen by user
    for user in rec.ratings_by_user:
        recs = rec.recommend_movies_for_user(user, 3)
        rated = rec.ratings_by_user[user]
        for name, _, _ in recs:
            assert name not in rated
    print("[PASS] recommendations exclude already rated movies")

    # 5. Recommended movies are from user's top genre
    for user in rec.ratings_by_user:
        pref = rec.user_genre_preference(user)
        if pref is None:
            continue
        top_genre = pref[0]
        recs = rec.recommend_movies_for_user(user, 3)
        for name, _, _ in recs:
            assert rec.movies_by_name[name].genre == top_genre
    print("[PASS] recommendations match top genre")

    print("\nAll logical tests passed.")


if __name__ == "__main__":
    main()