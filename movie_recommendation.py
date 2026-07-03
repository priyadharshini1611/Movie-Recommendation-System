import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

movies = pd.DataFrame({
    "title": [
        "Harry Potter",
        "Frozen",
        "Interstellar",
        "KGF",
        "The Conjuring",
        "Avatar",
        "Inception",
        "Jumanji"
    ],
    "genres": [
        "fantasy magic adventure",
        "animation fantasy music",
        "science space drama",
        "action crime drama",
        "horror thriller mystery",
        "science fantasy adventure",
        "science thriller dream",
        "fantasy adventure comedy"
    ]
})

vectorizer = TfidfVectorizer()
genre_matrix = vectorizer.fit_transform(movies["genres"])

similarity = cosine_similarity(genre_matrix)

movie_name = input("Enter a movie name: ")

if movie_name in movies["title"].values:
    index = movies[movies["title"] == movie_name].index[0]
    scores = list(enumerate(similarity[index]))
    scores = sorted(scores, key=lambda x: x[1], reverse=True)

    print("\nRecommended Movies:")
    count = 0

    for i, score in scores:
        if movies.iloc[i]["title"] != movie_name:
            print(movies.iloc[i]["title"])
            count += 1

        if count == 3:
            break
else:
    print("Movie not found in database")
