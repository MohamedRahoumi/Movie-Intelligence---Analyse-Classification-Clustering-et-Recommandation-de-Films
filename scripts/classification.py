import pandas as pd
import numpy as np
from pathlib import Path

from sklearn.model_selection import train_test_split
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
OUTPUT_DIR = ROOT / "outputs" / "classification"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


df = pd.read_csv(INPUT_FILE)

print("=" * 60)
print("ÉTAPE 6 - CLASSIFICATION")
print("=" * 60)

print("\nNombre de films :", len(df))


seuil = df["vote_count"].median()

df["high_engagement"] = (
    df["vote_count"] >= seuil
).astype(int)

print(f"\nSeuil choisi : {seuil} (médiane de vote_count)")
print("\nDistribution de la cible :")
print(df["high_engagement"].value_counts())
print(f"\nProportion : {df['high_engagement'].mean():.2%}")



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

print("\nFeatures utilisées :", features)
print("Dimensions X :", X.shape)


X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

print(f"\nTrain : {len(X_train)} films")
print(f"Test  : {len(X_test)} films")


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


resultats = []

for nom, pipeline in modeles.items():

    print("\n" + "=" * 60)
    print(f"MODÈLE : {nom}")
    print("=" * 60)

    
    pipeline.fit(X_train, y_train)

    
    y_pred = pipeline.predict(X_test)
    y_proba = pipeline.predict_proba(X_test)[:, 1]

    # Métriques
    acc = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred)
    rec = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    roc = roc_auc_score(y_test, y_proba)

    resultats.append({
        "Modèle": nom,
        "Accuracy": round(acc, 4),
        "Precision": round(prec, 4),
        "Recall": round(rec, 4),
        "F1-score": round(f1, 4),
        "ROC-AUC": round(roc, 4)
    })

    print(f"\nAccuracy  : {acc:.4f}")
    print(f"Precision : {prec:.4f}")
    print(f"Recall    : {rec:.4f}")
    print(f"F1-score  : {f1:.4f}")
    print(f"ROC-AUC   : {roc:.4f}")

    print("\nRapport de classification :")
    print(classification_report(y_test, y_pred))

    # Matrice de confusion
    cm = confusion_matrix(y_test, y_pred)

    plt.figure(figsize=(6, 5))

    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=["Faible", "Fort"],
        yticklabels=["Faible", "Fort"]
    )

    plt.title(f"Matrice de confusion - {nom}")
    plt.xlabel("Prédiction")
    plt.ylabel("Réalité")
    plt.tight_layout()

    filename = nom.lower().replace(" ", "_")

    plt.savefig(
        OUTPUT_DIR / f"confusion_{filename}.png"
    )

    plt.close()

    print(f"Matrice sauvegardée : confusion_{filename}.png")


print("\n" + "=" * 60)
print("COMPARAISON DES MODÈLES")
print("=" * 60)

resultats_df = pd.DataFrame(resultats)

print("\n", resultats_df.to_string(index=False))

resultats_df.to_csv(
    OUTPUT_DIR / "comparaison_modeles.csv",
    index=False,
    encoding="utf-8"
)


# Graphique comparatif
metriques = ["Accuracy", "Precision", "Recall", "F1-score", "ROC-AUC"]

resultats_melted = resultats_df.melt(
    id_vars="Modèle",
    value_vars=metriques,
    var_name="Métrique",
    value_name="Score"
)

plt.figure(figsize=(12, 6))

sns.barplot(
    data=resultats_melted,
    x="Métrique",
    y="Score",
    hue="Modèle"
)

plt.title("Comparaison des modèles de classification")
plt.ylim(0, 1)
plt.tight_layout()

plt.savefig(
    OUTPUT_DIR / "comparaison_modeles.png"
)

plt.close()


print("\n" + "=" * 60)
print("ÉTAPE 6 TERMINÉE")
print("=" * 60)

print(f"\nFichiers sauvegardés dans : {OUTPUT_DIR}")
