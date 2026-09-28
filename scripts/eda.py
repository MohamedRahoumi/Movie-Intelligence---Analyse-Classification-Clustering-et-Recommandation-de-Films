import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import ast
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]

INPUT_FILE = ROOT / "data" / "processed" / "movies_clean.csv"

OUTPUT_DIR = ROOT / "outputs" / "graphiques_etape3"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


df = pd.read_csv(INPUT_FILE)

print("=" * 60)
print("ÉTAPE 3 - EDA ET VISUALISATION")
print("=" * 60)

print("\nNombre de films :", len(df))
print("Nombre de colonnes :", len(df.columns))

print("\nColonnes :")
print(df.columns.tolist())


df["release_date"] = pd.to_datetime(
    df["release_date"],
    errors="coerce"
)

# Création de l'année
df["year"] = df["release_date"].dt.year


# Conversion des colonnes numériques
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


# ============================================================
# FONCTION POUR LIRE LES LISTES DE GENRES
# ============================================================

def convertir_liste(valeur):

    if pd.isna(valeur):
        return []

    if isinstance(valeur, list):
        return valeur

    try:
        resultat = ast.literal_eval(valeur)

        if isinstance(resultat, list):
            return resultat

        return []

    except (ValueError, SyntaxError):
        return []


df["genres"] = df["genres"].apply(convertir_liste)


# ============================================================
# 4. DISTRIBUTION DES NOTES
# ============================================================

plt.figure(figsize=(10, 6))

sns.histplot(
    df["vote_average"].dropna(),
    bins=20,
    kde=True
)

plt.title("Distribution des notes des films")
plt.xlabel("Note moyenne")
plt.ylabel("Nombre de films")

plt.tight_layout()

plt.savefig(
    OUTPUT_DIR / "01_distribution_notes.png"
)

plt.show()


print("\n--- Interprétation : distribution des notes ---")

moyenne_notes = df["vote_average"].mean()

print(
    f"La note moyenne des films est de "
    f"{moyenne_notes:.2f}/10."
)

print(
    "Le graphique permet d'observer la répartition "
    "des notes et de voir autour de quelles valeurs "
    "les films sont principalement concentrés."
)


# ============================================================
# 5. DISTRIBUTION DE LA POPULARITÉ
# ============================================================

plt.figure(figsize=(10, 6))

sns.histplot(
    df["popularity"].dropna(),
    bins=30,
    kde=True
)

plt.title("Distribution de la popularité des films")
plt.xlabel("Popularité")
plt.ylabel("Nombre de films")

plt.tight_layout()

plt.savefig(
    OUTPUT_DIR / "02_distribution_popularite.png"
)

plt.show()


print("\n--- Interprétation : popularité ---")

print(
    "La distribution permet d'identifier si la majorité "
    "des films ont une popularité faible, moyenne ou élevée."
)

print(
    "Une forte concentration à gauche indique généralement "
    "que peu de films ont une popularité très élevée."
)


# ============================================================
# 6. FILMS PAR GENRE
# ============================================================

genres = []

for liste_genres in df["genres"]:

    for genre in liste_genres:

        if genre:
            genres.append(genre)


genres_df = pd.Series(genres)

nombre_genres = genres_df.value_counts()


plt.figure(figsize=(12, 7))

sns.barplot(
    x=nombre_genres.values,
    y=nombre_genres.index
)

plt.title("Nombre de films par genre")
plt.xlabel("Nombre de films")
plt.ylabel("Genre")

plt.tight_layout()

plt.savefig(
    OUTPUT_DIR / "03_films_par_genre.png"
)

plt.show()


print("\n--- Interprétation : films par genre ---")

print(
    "Les genres les plus représentés sont :"
)

print(nombre_genres.head(5))

print(
    "Cette analyse permet d'identifier les genres "
    "les plus présents dans le catalogue."
)


# ============================================================
# 7. SORTIES PAR ANNÉE
# ============================================================

films_par_annee = (
    df["year"]
    .dropna()
    .value_counts()
    .sort_index()
)


plt.figure(figsize=(14, 6))

sns.lineplot(
    x=films_par_annee.index,
    y=films_par_annee.values
)

plt.title("Nombre de films sortis par année")
plt.xlabel("Année")
plt.ylabel("Nombre de films")

plt.tight_layout()

plt.savefig(
    OUTPUT_DIR / "04_sorties_par_annee.png"
)

plt.show()


print("\n--- Interprétation : sorties par année ---")

annee_max = films_par_annee.idxmax()
nombre_max = films_par_annee.max()

print(
    f"L'année avec le plus grand nombre de films "
    f"dans le dataset est {int(annee_max)}, "
    f"avec {nombre_max} films."
)

print(
    "Le graphique permet d'observer l'évolution "
    "du nombre de films au fil des années."
)


# ============================================================
# 8. DISTRIBUTION DE LA DURÉE
# ============================================================

plt.figure(figsize=(10, 6))

sns.histplot(
    df["runtime"].dropna(),
    bins=30,
    kde=True
)

plt.title("Distribution de la durée des films")
plt.xlabel("Durée (minutes)")
plt.ylabel("Nombre de films")

plt.tight_layout()

plt.savefig(
    OUTPUT_DIR / "05_distribution_duree.png"
)

plt.show()


print("\n--- Interprétation : durée ---")

duree_moyenne = df["runtime"].mean()

print(
    f"La durée moyenne des films est de "
    f"{duree_moyenne:.2f} minutes."
)

print(
    "La distribution permet de voir quelles durées "
    "sont les plus fréquentes dans le catalogue."
)


# ============================================================
# 9. BUDGET ET REVENUS
# ============================================================

budget_revenus = df[
    ["budget", "revenue"]
].dropna()


plt.figure(figsize=(10, 6))

sns.scatterplot(
    data=budget_revenus,
    x="budget",
    y="revenue"
)

plt.title("Relation entre budget et revenus")
plt.xlabel("Budget")
plt.ylabel("Revenus")

plt.tight_layout()

plt.savefig(
    OUTPUT_DIR / "06_budget_revenus.png"
)

plt.show()


print("\n--- Interprétation : budget / revenus ---")

print(
    "Le nuage de points permet d'observer la relation "
    "entre le budget investi et les revenus générés."
)

correlation_budget_revenue = (
    df["budget"]
    .corr(df["revenue"])
)

print(
    f"Corrélation budget/revenus : "
    f"{correlation_budget_revenue:.2f}"
)


# ============================================================
# 10. VOTES ET POPULARITÉ
# ============================================================

votes_popularite = df[
    ["vote_count", "popularity"]
].dropna()


plt.figure(figsize=(10, 6))

sns.scatterplot(
    data=votes_popularite,
    x="vote_count",
    y="popularity"
)

plt.title("Relation entre nombre de votes et popularité")
plt.xlabel("Nombre de votes")
plt.ylabel("Popularité")

plt.tight_layout()

plt.savefig(
    OUTPUT_DIR / "07_votes_popularite.png"
)

plt.show()


print("\n--- Interprétation : votes / popularité ---")

correlation_votes_popularite = (
    df["vote_count"]
    .corr(df["popularity"])
)

print(
    f"Corrélation votes/popularité : "
    f"{correlation_votes_popularite:.2f}"
)

print(
    "Le nuage de points permet d'étudier si les films "
    "ayant beaucoup de votes ont également une popularité élevée."
)


# ============================================================
# 11. BOXPLOT DES NOTES
# ============================================================

plt.figure(figsize=(10, 5))

sns.boxplot(
    x=df["vote_average"].dropna()
)

plt.title("Boxplot des notes")
plt.xlabel("Note moyenne")

plt.tight_layout()

plt.savefig(
    OUTPUT_DIR / "08_boxplot_notes.png"
)

plt.show()


print("\n--- Interprétation : boxplot des notes ---")

print(
    "Le boxplot permet d'observer la médiane, "
    "la dispersion des notes et les éventuelles valeurs atypiques."
)


# ============================================================
# 12. BOXPLOT DE LA POPULARITÉ
# ============================================================

plt.figure(figsize=(10, 5))

sns.boxplot(
    x=df["popularity"].dropna()
)

plt.title("Boxplot de la popularité")
plt.xlabel("Popularité")

plt.tight_layout()

plt.savefig(
    OUTPUT_DIR / "09_boxplot_popularite.png"
)

plt.show()


print("\n--- Interprétation : boxplot popularité ---")

print(
    "Le boxplot permet d'identifier la dispersion "
    "de la popularité et les valeurs extrêmes."
)


# ============================================================
# 13. MATRICE DE CORRÉLATION
# ============================================================

colonnes_correlation = [
    "runtime",
    "budget",
    "revenue",
    "popularity",
    "vote_average",
    "vote_count"
]

correlation = df[colonnes_correlation].corr()


plt.figure(figsize=(10, 8))

sns.heatmap(
    correlation,
    annot=True,
    cmap="coolwarm",
    fmt=".2f"
)

plt.title("Matrice de corrélation des variables numériques")

plt.tight_layout()

plt.savefig(
    OUTPUT_DIR / "10_matrice_correlation.png"
)

plt.show()


print("\n--- Interprétation : corrélations ---")

print(
    "La matrice de corrélation permet d'identifier "
    "les relations entre les variables numériques."
)

print(
    "Une corrélation proche de 1 indique une relation "
    "linéaire positive forte."
)

print(
    "Une corrélation proche de -1 indique une relation "
    "linéaire négative forte."
)

print(
    "Une corrélation proche de 0 indique une faible "
    "relation linéaire."
)


# ============================================================
# 14. RÉSUMÉ STATISTIQUE
# ============================================================

print("\n")
print("=" * 60)
print("RÉSUMÉ STATISTIQUE")
print("=" * 60)

print(
    df[
        [
            "runtime",
            "budget",
            "revenue",
            "popularity",
            "vote_average",
            "vote_count"
        ]
    ].describe()
)



print("\n")
print("=" * 60)
print("ÉTAPE 3 TERMINÉE")
print("=" * 60)

print(
    f"\nLes graphiques sont sauvegardés dans :\n"
    f"{OUTPUT_DIR}"
)