from pathlib import Path
import json
import time

import requests
from dotenv import load_dotenv
import os



load_dotenv()

API_KEY = os.getenv("TMDB_API_KEY")

BASE_URL = "https://api.themoviedb.org/3"

OUTPUT_DIR = Path(__file__).resolve().parents[1] / "data" / "raw"
OUTPUT_FILE = OUTPUT_DIR / "movies.json"

NB_PAGES = 150
# LANGUAGE = "fr-FR"
LANGUAGE = "en-US"


if not API_KEY:
    raise ValueError("TMDB_API_KEY est absente du fichier .env")



def get_movies_page(page):
    url = f"{BASE_URL}/discover/movie"

    params = {
        "api_key": API_KEY,
        "language": LANGUAGE,
        "page": page
    }

    try:
        response = requests.get(
            url,
            params=params,
            timeout=10
        )

        response.raise_for_status()

        data = response.json()

        if not data:
            print(f"Page {page} : réponse vide")
            return []

        return data.get("results", [])

    except requests.exceptions.RequestException as error:
        print(f"Erreur HTTP page {page} : {error}")
        return []


def get_movie_details(movie_id):

    url = f"{BASE_URL}/movie/{movie_id}"

    params = {
        "api_key": API_KEY,
        "language": LANGUAGE,
        "append_to_response": "keywords"
    }

    try:
        response = requests.get(
            url,
            params=params,
            timeout=10
        )

        response.raise_for_status()

        return response.json()

    except requests.exceptions.RequestException as error:
        print(f"Erreur film {movie_id} : {error}")
        return None


def extract_movie(movie):

    movie_id = movie.get("id")

    details = get_movie_details(movie_id)

    if details is None:
        return None

    genres = [
        genre["name"]
        for genre in details.get("genres", [])
    ]

    keywords_data = details.get("keywords", {})

    keywords = [keyword["name"]for keyword in keywords_data.get("keywords", [])]

    return {
        "movie_id": movie_id,
        "title": details.get("title"),
        "overview": details.get("overview"),
        "release_date": details.get("release_date"),
        "runtime": details.get("runtime"),
        "original_language": details.get("original_language"),
        "genres": genres,
        "keywords": keywords,
        "budget": details.get("budget"),
        "revenue": details.get("revenue"),
        "popularity": details.get("popularity"),
        "vote_average": details.get("vote_average"),
        "vote_count": details.get("vote_count")
    }

def main():

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    movies = []

    print("=" * 60)
    print("EXTRACTION TMDB")
    print("=" * 60)

    for page in range(1, NB_PAGES + 1):

        print(f"Page {page}/{NB_PAGES}")

        page_movies = get_movies_page(page)

        if not page_movies:
            print("Aucun film récupéré. Arrêt.")
            break

        for movie in page_movies:

            movie_data = extract_movie(movie)

            if movie_data:
                movies.append(movie_data)

            time.sleep(0.05)

        print(f"Films récupérés : {len(movies)}")

        time.sleep(0.2)


    with open(
        OUTPUT_FILE,
        "w",
        encoding="utf-8"
    ) as file:

        json.dump(
            movies,
            file,
            ensure_ascii=False,
            indent=2
        )

    print("=" * 60)
    print(f"Extraction terminée : {len(movies)} films")
    print(f"Fichier : {OUTPUT_FILE}")
    print("=" * 60)


if __name__ == "__main__":
    main()