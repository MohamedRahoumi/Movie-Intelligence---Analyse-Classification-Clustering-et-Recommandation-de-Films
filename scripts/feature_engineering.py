import pandas as pd
import ast
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]

INPUT_FILE = ROOT / "data" / "processed" / "movies_clean.csv"
OUTPUT_FILE = ROOT / "data" / "processed" / "movies_engineered.csv"


df = pd.read_csv(INPUT_FILE)

print("=" * 60)
print("ÉTAPE 4 - FEATURE ENGINEERING")
print("=" * 60)

print("\nNombre de films :", len(df))


df["release_date"] = pd.to_datetime(
    df["release_date"],
    errors="coerce"
)

df["release_year"] = df["release_date"].dt.year

df["release_month"] = df["release_date"].dt.month

df["release_decade"] = (
    df["release_year"] // 10
) * 10


df["genres"] = df["genres"].apply(ast.literal_eval)

df["keywords"] = df["keywords"].apply(ast.literal_eval)

df["genre_count"] = df["genres"].apply(len)

df["keyword_count"] = df["keywords"].apply(len)



df["runtime_category"] = pd.cut(
    df["runtime"],
    bins=[0, 90, 120, 150, float("inf")],
    labels=[
        "Court",
        "Moyen",
        "Long",
        "Très long"
    ]
)



df["is_long_movie"] = (
    df["runtime"] > 120
).astype(int)

df["is_multi_genre"] = (
    df["genre_count"] > 1
).astype(int)

df["roi"] = df["revenue"] / df["budget"]

annee_reference = df["release_year"].max()

df["movie_age"] = (
    annee_reference - df["release_year"]
)



OUTPUT_FILE.parent.mkdir(
    parents=True,
    exist_ok=True
)

df.to_csv(
    OUTPUT_FILE,
    index=False,
    encoding="utf-8"
)


nouvelles_variables = [
    "release_year",
    "release_month",
    "release_decade",
    "genre_count",
    "keyword_count",
    "runtime_category",
    "is_long_movie",
    "is_multi_genre",
    "roi",
    "movie_age"
]

print("\nNouvelles variables créées :")

print(nouvelles_variables)

print("\nAperçu :")

print(
    df[
        [
            "title",
            "release_year",
            "release_month",
            "release_decade",
            "genre_count",
            "keyword_count",
            "runtime_category",
            "is_long_movie",
            "is_multi_genre",
            "roi",
            "movie_age"
        ]
    ].head()
)


print("\nFichier sauvegardé :")

print(OUTPUT_FILE)

print("\n" + "=" * 60)

print("ÉTAPE 4 TERMINÉE")

print("=" * 60)