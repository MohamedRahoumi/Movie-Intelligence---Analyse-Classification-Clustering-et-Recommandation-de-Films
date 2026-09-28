import pandas as pd
import re
from pathlib import Path
from sklearn.feature_extraction.text import TfidfVectorizer
import joblib
from scipy import sparse


ROOT = Path(__file__).resolve().parents[1]

INPUT_FILE = ROOT / "data" / "processed" / "movies_engineered.csv"

OUTPUT_DIR = ROOT / "outputs" / "tfidf"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


df = pd.read_csv(INPUT_FILE)


print("=" * 60)
print("ÉTAPE 5 - NLP AVEC TF-IDF")
print("=" * 60)

print("\nNombre de films :", len(df))


df["overview"] = df["overview"].fillna("")

print(
    "\nNombre de descriptions vides :",
    (df["overview"].str.strip() == "").sum()
)


def nettoyer_texte(texte):
    texte = texte.lower()
    texte = re.sub(
        r"[^a-zàâçéèêëîïôûùüÿñæœ\s]",
        " ",
        texte
    )
    texte = re.sub(r"\s+", " ", texte)
    return texte.strip()


df["overview_clean"] = df["overview"].apply(nettoyer_texte)


experiences = [
    {
        "max_features": 500,
        "ngram_range": (1, 1)
    },
    {
        "max_features": 1000,
        "ngram_range": (1, 1)
    },
    {
        "max_features": 2000,
        "ngram_range": (1, 1)
    },
    {
        "max_features": 1000,
        "ngram_range": (1, 2)
    }
]


resultats = []


for experience in experiences:

    vectorizer = TfidfVectorizer(
        max_features=experience["max_features"],
        ngram_range=experience["ngram_range"],
        stop_words="english"
    )

    X_tfidf = vectorizer.fit_transform(
        df["overview_clean"]
    )

    resultats.append({
        "max_features": experience["max_features"],
        "ngram_range": experience["ngram_range"],
        "documents": X_tfidf.shape[0],
        "termes": X_tfidf.shape[1]
    })

    print("\n" + "-" * 60)
    print(
        "max_features =",
        experience["max_features"]
    )
    print(
        "ngram_range =",
        experience["ngram_range"]
    )
    print(
        "Dimensions :",
        X_tfidf.shape
    )


experiences_df = pd.DataFrame(resultats)

print("\n" + "=" * 60)
print("COMPARAISON DES EXPÉRIENCES")
print("=" * 60)

print(experiences_df)


experiences_df.to_csv(
    OUTPUT_DIR / "tfidf_experiments.csv",
    index=False,
    encoding="utf-8"
)


vectorizer_final = TfidfVectorizer(
    max_features=1000,
    ngram_range=(1, 2),
    stop_words="english"
)

X_tfidf_final = vectorizer_final.fit_transform(
    df["overview_clean"]
)


terms = vectorizer_final.get_feature_names_out()

scores = X_tfidf_final.sum(axis=0).A1

terms_df = pd.DataFrame({
    "term": terms,
    "tfidf_score": scores
})

terms_df = terms_df.sort_values(
    "tfidf_score",
    ascending=False
)


print("\n" + "=" * 60)
print("TERMES LES PLUS REPRÉSENTATIFS")
print("=" * 60)

print(
    terms_df.head(20)
)


terms_df.to_csv(
    OUTPUT_DIR / "tfidf_terms.csv",
    index=False,
    encoding="utf-8"
)


sparse.save_npz(
    OUTPUT_DIR / "tfidf_matrix.npz",
    X_tfidf_final
)


joblib.dump(
    vectorizer_final,
    OUTPUT_DIR / "tfidf_vectorizer.joblib"
)


print("\n" + "=" * 60)
print("MATRICE TF-IDF FINALE")
print("=" * 60)

print(
    "Nombre de documents :",
    X_tfidf_final.shape[0]
)

print(
    "Nombre de termes :",
    X_tfidf_final.shape[1]
)

print(
    "Dimensions de la matrice :",
    X_tfidf_final.shape
)

print(
    "\nFichiers sauvegardés dans :",
    OUTPUT_DIR
)

print("\nFichiers créés :")
print("- tfidf_experiments.csv")
print("- tfidf_terms.csv")
print("- tfidf_matrix.npz")
print("- tfidf_vectorizer.joblib")


print("\n" + "=" * 60)
print("ÉTAPE 5 TERMINÉE")
print("=" * 60)