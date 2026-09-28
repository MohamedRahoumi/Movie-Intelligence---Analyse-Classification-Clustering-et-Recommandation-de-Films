import pandas as pd
import numpy as np
from pathlib import Path

from sklearn.model_selection import (
    StratifiedKFold,
    cross_validate,
    GridSearchCV
)
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import LinearSVC
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
    classification_report
)

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns


ROOT = Path(__file__).resolve().parents[1]

INPUT_FILE = ROOT / "data" / "processed" / "movies_engineered.csv"
OUTPUT_DIR = ROOT / "outputs" / "validation"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


df = pd.read_csv(INPUT_FILE)

print("=" * 60)
print("ÉTAPE 7 - VALIDATION ET OPTIMISATION")
print("=" * 60)

print("\nNombre de films :", len(df))
seuil = df["vote_count"].median()

df["high_engagement"] = (
    df["vote_count"] >= seuil
).astype(int)

colonnes_numeriques = [
    "runtime",
    "budget",
    "revenue",
    "popularity",
    "vote_average",
    "release_year",
    "release_month",
    "genre_count",
    "keyword_count",
    "is_long_movie",
    "is_multi_genre",
    "roi",
    "movie_age"
]

colonnes_categoriques = [
    "release_decade"
]

features = colonnes_numeriques + colonnes_categoriques

X = df[features]
y = df["high_engagement"]

preprocessor = ColumnTransformer(
    transformers=[
        (
            "num",
            StandardScaler(),
            colonnes_numeriques
        ),
        (
            "cat",
            OneHotEncoder(handle_unknown="ignore"),
            colonnes_categoriques
        )
    ]
)


print("\n" + "=" * 60)
print("CROSS-VALIDATION (StratifiedKFold, k=5)")
print("=" * 60)

cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

scoring = [
    "accuracy",
    "precision",
    "recall",
    "f1",
    "roc_auc"
]

modeles = {
    "Logistic Regression": Pipeline([
        ("preprocessor", preprocessor),
        ("classifier", LogisticRegression(
            max_iter=1000,
            random_state=42
        ))
    ]),

    "Random Forest": Pipeline([
        ("preprocessor", preprocessor),
        ("classifier", RandomForestClassifier(
            n_estimators=100,
            random_state=42
        ))
    ]),

    "Linear SVM": Pipeline([
        ("preprocessor", preprocessor),
        ("classifier", CalibratedClassifierCV(
            LinearSVC(
                max_iter=2000,
                random_state=42
            ),
            cv=3
        ))
    ])
}

resultats_cv = []

for nom, pipeline in modeles.items():

    print(f"\n--- {nom} ---")

    scores = cross_validate(
        pipeline, X, y,
        cv=cv,
        scoring=scoring,
        return_train_score=False
    )

    ligne = {"Modèle": nom}

    for metrique in scoring:
        cle = f"test_{metrique}"
        moyenne = scores[cle].mean()
        ecart = scores[cle].std()
        ligne[metrique] = round(moyenne, 4)
        ligne[f"{metrique}_std"] = round(ecart, 4)
        print(f"  {metrique:12s} : {moyenne:.4f} ± {ecart:.4f}")

    resultats_cv.append(ligne)


resultats_cv_df = pd.DataFrame(resultats_cv)

print("\n" + "=" * 60)
print("RÉSUMÉ CROSS-VALIDATION")
print("=" * 60)

colonnes_affichage = ["Modèle"] + scoring
print("\n", resultats_cv_df[colonnes_affichage].to_string(index=False))

resultats_cv_df.to_csv(
    OUTPUT_DIR / "cross_validation.csv",
    index=False,
    encoding="utf-8"
)


# ============================================================
# 3. GRIDSEARCHCV SUR RANDOM FOREST
# ============================================================

print("\n" + "=" * 60)
print("GRIDSEARCHCV - RANDOM FOREST")
print("=" * 60)

# Pipeline pour GridSearch
pipeline_rf = Pipeline([
    ("preprocessor", preprocessor),
    ("classifier", RandomForestClassifier(random_state=42))
])

# Grille d'hyperparamètres
param_grid = {
    "classifier__n_estimators": [100, 200, 300],
    "classifier__max_depth": [10, 20, None],
    "classifier__min_samples_split": [2, 5]
}

print("\nGrille testée :")
for param, valeurs in param_grid.items():
    print(f"  {param} : {valeurs}")

grid_search = GridSearchCV(
    pipeline_rf,
    param_grid,
    cv=cv,
    scoring="f1",
    n_jobs=-1,
    verbose=1
)

grid_search.fit(X, y)

print(f"\nMeilleurs hyperparamètres :")
for param, valeur in grid_search.best_params_.items():
    print(f"  {param} : {valeur}")

print(f"\nMeilleur F1-score (CV) : {grid_search.best_score_:.4f}")



print("\n" + "=" * 60)
print("COMPARAISON AVANT / APRÈS OPTIMISATION")
print("=" * 60)

rf_avant = resultats_cv_df[
    resultats_cv_df["Modèle"] == "Random Forest"
].iloc[0]

scores_apres = cross_validate(
    grid_search.best_estimator_,
    X, y,
    cv=cv,
    scoring=scoring,
    return_train_score=False
)

comparaison = []

for metrique in scoring:
    avant = rf_avant[metrique]
    apres = round(scores_apres[f"test_{metrique}"].mean(), 4)
    diff = round(apres - avant, 4)
    comparaison.append({
        "Métrique": metrique,
        "Avant": avant,
        "Après": apres,
        "Différence": diff
    })

comparaison_df = pd.DataFrame(comparaison)

print("\n", comparaison_df.to_string(index=False))

comparaison_df.to_csv(
    OUTPUT_DIR / "comparaison_avant_apres.csv",
    index=False,
    encoding="utf-8"
)


x_pos = np.arange(len(scoring))
width = 0.35

avant_vals = [comparaison_df.loc[i, "Avant"] for i in range(len(scoring))]
apres_vals = [comparaison_df.loc[i, "Après"] for i in range(len(scoring))]

plt.figure(figsize=(10, 6))

plt.bar(x_pos - width/2, avant_vals, width, label="Avant (défaut)")
plt.bar(x_pos + width/2, apres_vals, width, label="Après (GridSearch)")

plt.xticks(x_pos, scoring)
plt.ylabel("Score")
plt.title("Random Forest : avant vs après optimisation")
plt.ylim(0, 1)
plt.legend()
plt.tight_layout()

plt.savefig(OUTPUT_DIR / "avant_apres_optimisation.png")
plt.close()


print("\n" + "=" * 60)
print("ÉTAPE 7 TERMINÉE")
print("=" * 60)

print(f"\nFichiers sauvegardés dans : {OUTPUT_DIR}")
