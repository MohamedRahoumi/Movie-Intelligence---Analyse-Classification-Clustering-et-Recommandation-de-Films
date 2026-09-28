import pandas as pd
from pathlib import Path
from pymongo import MongoClient


ROOT = Path(__file__).resolve().parents[1]

INPUT_FILE = ROOT / "data" / "raw" / "movies.json"
OUTPUT_FILE = ROOT / "data" / "processed" / "movies_clean.csv"


df = pd.read_json(INPUT_FILE)

print("Nombre de lignes :", len(df))
print("Nombre de colonnes :", len(df.columns))


print("\nDoublons movie_id :", df["movie_id"].duplicated().sum())

df = df.drop_duplicates(subset="movie_id")


df["release_date"] = pd.to_datetime(
    df["release_date"],
    errors="coerce"
)


colonnes_numeriques = [
    "runtime",
    "budget",
    "revenue",
    "popularity",
    "vote_average",
    "vote_count"
]

for colonne in colonnes_numeriques:
    df[colonne] = pd.to_numeric(
        df[colonne],
        errors="coerce"
    )


print("\n= VALEURS INCOHERENTES =")

print("vote_average > 10 :",
      (df["vote_average"] > 10).sum())

print("vote_average < 0 :",
      (df["vote_average"] < 0).sum())

print("runtime < 0 :",
      (df["runtime"] < 0).sum())

print("budget < 0 :",
      (df["budget"] < 0).sum())

print("revenue < 0 :",
      (df["revenue"] < 0).sum())


df.loc[df["vote_average"] > 10, "vote_average"] = pd.NA
df.loc[df["vote_average"] < 0, "vote_average"] = pd.NA

df.loc[df["runtime"] < 0, "runtime"] = pd.NA
df.loc[df["budget"] < 0, "budget"] = pd.NA
df.loc[df["revenue"] < 0, "revenue"] = pd.NA


df.loc[df["runtime"] == 0, "runtime"] = pd.NA
df.loc[df["budget"] == 0, "budget"] = pd.NA
df.loc[df["revenue"] == 0, "revenue"] = pd.NA
df.loc[df["vote_average"] == 0, "vote_average"] = pd.NA


df["overview"] = df["overview"].fillna("Unknown")

df["overview"] = df["overview"].apply(
    lambda x: "Unknown"
    if not isinstance(x, str) or x.strip() == ""
    else x.strip()
)


df["genres"] = df["genres"].apply(
    lambda x: x
    if isinstance(x, list) and len(x) > 0
    else ["Unknown"]
)


df["keywords"] = df["keywords"].apply(
    lambda x: x
    if isinstance(x, list) and len(x) > 0
    else ["Unknown"]
)


colonnes_a_remplir_par_mediane = [
    "runtime",
    "budget",
    "revenue",
    "popularity",
    "vote_average"
]

for colonne in colonnes_a_remplir_par_mediane:

    mediane = df[colonne].median()

    df[colonne] = df[colonne].fillna(mediane)

    print(
        f"{colonne} → valeurs manquantes remplacées par : {mediane}"
    )


df["vote_count"] = df["vote_count"].fillna(0)


print(
    "\nDates manquantes :",
    df["release_date"].isna().sum()
)

df = df.dropna(subset=["release_date"])


print("\n========== VALEURS MANQUANTES APRES NETTOYAGE ==========")

print(df.isnull().sum())


print("\n========== ZEROS RESTANTS ==========")

print("runtime = 0 :",
      (df["runtime"] == 0).sum())

print("budget = 0 :",
      (df["budget"] == 0).sum())

print("revenue = 0 :",
      (df["revenue"] == 0).sum())

print("vote_average = 0 :",
      (df["vote_average"] == 0).sum())

print("vote_count = 0 :",
      (df["vote_count"] == 0).sum())


variables_numeriques = df.select_dtypes(
    include="number"
).columns.tolist()

print("\nVariables numériques :")
print(variables_numeriques)


variables_categorielles = df.select_dtypes(
    exclude="number"
).columns.tolist()

print("\nVariables catégorielles / non numériques :")
print(variables_categorielles)


OUTPUT_FILE.parent.mkdir(
    parents=True,
    exist_ok=True
)

df.to_csv(
    OUTPUT_FILE,
    index=False,
    encoding="utf-8"
)

print("\nCSV sauvegardé :")
print(OUTPUT_FILE)


client = MongoClient(
    "mongodb://admin:admin123@localhost:27017/"
)

db = client["movie_intelligence"]

collection = db["movies"]

collection.delete_many({})


films = df.to_dict(orient="records")


for film in films:

    for key, value in film.items():

        if isinstance(value, list):
            continue

        if pd.isna(value):
            film[key] = None


if films:
    collection.insert_many(films)


print("\nMongoDB :")
print("Films insérés :", len(films))


client.close()


print("\nÉtape 2 terminée.")