import streamlit as st
import pandas as pd
import numpy as np
import ast
import re
from pathlib import Path

from sklearn.model_selection import train_test_split, StratifiedKFold, cross_validate
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import LinearSVC
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, roc_auc_score, confusion_matrix
)
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction.text import TfidfVectorizer

import matplotlib.pyplot as plt
import seaborn as sns



st.set_page_config(
    page_title="Movie Intelligence",
    page_icon="🎬",
    layout="wide"
)

ROOT = Path(__file__).resolve().parents[0]
DATA_FILE = ROOT / "data" / "processed" / "movies_engineered.csv"
GRAPHS_DIR = ROOT / "outputs" / "graphiques_etape3"



@st.cache_data
def charger_donnees():
    df = pd.read_csv(DATA_FILE)
    df["release_date"] = pd.to_datetime(df["release_date"], errors="coerce")
    df["genres"] = df["genres"].apply(ast.literal_eval)
    df["keywords"] = df["keywords"].apply(ast.literal_eval)
    return df


@st.cache_data
def preparer_tfidf(overviews):
    """Nettoie les textes et calcule la matrice TF-IDF."""
    def nettoyer(texte):
        texte = str(texte).lower()
        texte = re.sub(r"[^a-zàâçéèêëîïôûùüÿñæœ\s]", " ", texte)
        texte = re.sub(r"\s+", " ", texte)
        return texte.strip()

    textes = overviews.fillna("").apply(nettoyer)

    vectorizer = TfidfVectorizer(
        max_features=1000,
        ngram_range=(1, 2),
        stop_words="english"
    )
    matrice = vectorizer.fit_transform(textes)
    return matrice, vectorizer, textes


df = charger_donnees()


# ============================================================
# SIDEBAR - NAVIGATION
# ============================================================

st.sidebar.title("🎬 Movie Intelligence")

page = st.sidebar.radio(
    "Navigation",
    [
        "📊 Dashboard",
        "🤖 Classification",
        "🔬 Clustering"
    ]
)

st.sidebar.markdown("---")
st.sidebar.info(f"**{len(df)}** films chargés")


# ============================================================
# PAGE 1 : DASHBOARD
# ============================================================

if page == "📊 Dashboard":

    st.title("📊 Dashboard — Analyse exploratoire")

    # --- KPIs ---
    col1, col2, col3, col4 = st.columns(4)

    col1.metric("Films", f"{len(df):,}")
    col2.metric("Note moyenne", f"{df['vote_average'].mean():.2f}")
    col3.metric("Budget moyen", f"{df['budget'].mean()/1e6:.0f} M$")
    col4.metric("Revenu moyen", f"{df['revenue'].mean()/1e6:.0f} M$")

    st.markdown("---")

    # --- Graphiques EDA (images pré-générées) ---
    st.subheader("Visualisations")

    graphiques = {
        "Distribution des notes": "01_distribution_notes.png",
        "Distribution de la popularité": "02_distribution_popularite.png",
        "Films par genre": "03_films_par_genre.png",
        "Sorties par année": "04_sorties_par_annee.png",
        "Distribution de la durée": "05_distribution_duree.png",
        "Budget vs Revenus": "06_budget_revenus.png",
        "Votes vs Popularité": "07_votes_popularite.png",
        "Boxplot des notes": "08_boxplot_notes.png",
        "Boxplot de la popularité": "09_boxplot_popularite.png",
        "Matrice de corrélation": "10_matrice_correlation.png"
    }

    # Affichage 2 par 2
    noms = list(graphiques.keys())

    for i in range(0, len(noms), 2):
        cols = st.columns(2)

        for j, col in enumerate(cols):
            idx = i + j
            if idx < len(noms):
                nom = noms[idx]
                fichier = GRAPHS_DIR / graphiques[nom]
                if fichier.exists():
                    col.image(str(fichier), caption=nom, use_container_width=True)

    # --- Statistiques descriptives ---
    st.markdown("---")
    st.subheader("Statistiques descriptives")

    colonnes_stats = [
        "runtime", "budget", "revenue",
        "popularity", "vote_average", "vote_count"
    ]

    st.dataframe(
        df[colonnes_stats].describe().round(2),
        use_container_width=True
    )


elif page == "🤖 Classification":

    st.title("🤖 Classification — Prédiction du high engagement")

    # --- Préparation ---
    seuil = df["vote_count"].median()

    st.info(
        f"**Variable cible** : `high_engagement` "
        f"(vote_count ≥ {seuil:.0f}, médiane du dataset)"
    )

    y = (df["vote_count"] >= seuil).astype(int)

    col1, col2 = st.columns(2)
    col1.metric("Classe 0 (faible)", f"{(y == 0).sum()}")
    col2.metric("Classe 1 (fort)", f"{(y == 1).sum()}")

    colonnes_numeriques = [
        "runtime", "budget", "revenue", "popularity",
        "vote_average", "release_year", "release_month",
        "genre_count", "keyword_count", "is_long_movie",
        "is_multi_genre", "roi", "movie_age"
    ]
    colonnes_categoriques = ["release_decade"]

    X = df[colonnes_numeriques + colonnes_categoriques]

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", StandardScaler(), colonnes_numeriques),
            ("cat", OneHotEncoder(handle_unknown="ignore"), colonnes_categoriques)
        ]
    )

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    modeles = {
        "Logistic Regression": Pipeline([
            ("preprocessor", preprocessor),
            ("classifier", LogisticRegression(max_iter=1000, random_state=42))
        ]),
        "Random Forest": Pipeline([
            ("preprocessor", preprocessor),
            ("classifier", RandomForestClassifier(n_estimators=100, random_state=42))
        ]),
        "Linear SVM": Pipeline([
            ("preprocessor", preprocessor),
            ("classifier", CalibratedClassifierCV(
                LinearSVC(max_iter=2000, random_state=42), cv=3
            ))
        ])
    }

    # --- Entraînement et évaluation ---
    st.markdown("---")
    st.subheader("Comparaison des modèles (Train/Test split)")

    resultats = []

    for nom, pipeline in modeles.items():
        pipeline.fit(X_train, y_train)
        y_pred = pipeline.predict(X_test)
        y_proba = pipeline.predict_proba(X_test)[:, 1]

        resultats.append({
            "Modèle": nom,
            "Accuracy": round(accuracy_score(y_test, y_pred), 4),
            "Precision": round(precision_score(y_test, y_pred), 4),
            "Recall": round(recall_score(y_test, y_pred), 4),
            "F1-score": round(f1_score(y_test, y_pred), 4),
            "ROC-AUC": round(roc_auc_score(y_test, y_proba), 4)
        })

    resultats_df = pd.DataFrame(resultats)
    st.dataframe(resultats_df, use_container_width=True, hide_index=True)

    # --- Matrices de confusion ---
    st.subheader("Matrices de confusion")

    cols = st.columns(3)

    for i, (nom, pipeline) in enumerate(modeles.items()):
        y_pred = pipeline.predict(X_test)
        cm = confusion_matrix(y_test, y_pred)

        fig, ax = plt.subplots(figsize=(5, 4))
        sns.heatmap(
            cm, annot=True, fmt="d", cmap="Blues",
            xticklabels=["Faible", "Fort"],
            yticklabels=["Faible", "Fort"],
            ax=ax
        )
        ax.set_title(nom)
        ax.set_xlabel("Prédiction")
        ax.set_ylabel("Réalité")

        cols[i].pyplot(fig)
        plt.close(fig)

    # --- Cross-validation ---
    st.markdown("---")
    st.subheader("Cross-Validation (StratifiedKFold, k=5)")

    cv_file = ROOT / "outputs" / "validation" / "cross_validation.csv"

    if cv_file.exists():
        cv_df = pd.read_csv(cv_file)
        metriques_cv = ["accuracy", "precision", "recall", "f1", "roc_auc"]

        affichage = cv_df[["Modèle"] + metriques_cv].copy()
        affichage.columns = ["Modèle"] + [m.replace("_", " ").title() for m in metriques_cv]
        st.dataframe(affichage, use_container_width=True, hide_index=True)
    else:
        st.warning("Fichier cross_validation.csv non trouvé. Lancez `python scripts/validation.py`.")

    # --- Avant / Après optimisation ---
    st.subheader("Optimisation GridSearchCV (Random Forest)")

    avant_file = ROOT / "outputs" / "validation" / "comparaison_avant_apres.csv"

    if avant_file.exists():
        avant_df = pd.read_csv(avant_file)
        st.dataframe(avant_df, use_container_width=True, hide_index=True)

        fig, ax = plt.subplots(figsize=(10, 5))
        x_pos = np.arange(len(avant_df))
        width = 0.35

        ax.bar(x_pos - width/2, avant_df.iloc[:, 1], width, label="Avant (défaut)")
        ax.bar(x_pos + width/2, avant_df.iloc[:, 2], width, label="Après (GridSearch)")

        ax.set_xticks(x_pos)
        ax.set_xticklabels(avant_df.iloc[:, 0])
        ax.set_ylabel("Score")
        ax.set_ylim(0, 1)
        ax.set_title("Random Forest : avant vs après optimisation")
        ax.legend()

        st.pyplot(fig)
        plt.close(fig)
    else:
        st.warning("Fichier comparaison_avant_apres.csv non trouvé. Lancez `python scripts/validation.py`.")


# ============================================================
# PAGE 3 : CLUSTERING
# ============================================================

elif page == "🔬 Clustering":

    st.title("🔬 Clustering — Groupes de films")

    # --- Données structurées ---
    colonnes_struct = [
        "runtime", "budget", "revenue", "popularity",
        "vote_average", "vote_count", "genre_count",
        "keyword_count", "roi", "movie_age"
    ]

    X_struct = df[colonnes_struct].values
    scaler = StandardScaler()
    X_struct_scaled = scaler.fit_transform(X_struct)

    # --- TF-IDF ---
    X_tfidf, vectorizer, _ = preparer_tfidf(df["overview"])

    # --- Silhouette Scores ---
    st.subheader("Recherche du meilleur K (Silhouette Score)")

    sil_file = ROOT / "outputs" / "clustering" / "silhouette_scores.csv"

    if sil_file.exists():
        sil_df = pd.read_csv(sil_file)
    else:
        # Calcul à la volée
        k_values = range(2, 11)
        sil_struct = []
        sil_tfidf = []

        for k in k_values:
            km = KMeans(n_clusters=k, random_state=42, n_init=10)

            labels_s = km.fit_predict(X_struct_scaled)
            sil_struct.append(silhouette_score(X_struct_scaled, labels_s))

            labels_t = km.fit_predict(X_tfidf)
            sil_tfidf.append(silhouette_score(X_tfidf, labels_t))

        sil_df = pd.DataFrame({
            "K": list(k_values),
            "silhouette_structurees": sil_struct,
            "silhouette_tfidf": sil_tfidf
        })

    col1, col2 = st.columns(2)

    with col1:
        fig, ax = plt.subplots(figsize=(8, 4))
        ax.plot(sil_df["K"], sil_df["silhouette_structurees"], marker="o")
        ax.set_title("Silhouette — Variables structurées")
        ax.set_xlabel("K")
        ax.set_ylabel("Score")
        ax.set_xticks(sil_df["K"])
        ax.grid(True, alpha=0.3)
        st.pyplot(fig)
        plt.close(fig)

    with col2:
        fig, ax = plt.subplots(figsize=(8, 4))
        ax.plot(sil_df["K"], sil_df["silhouette_tfidf"], marker="o", color="orange")
        ax.set_title("Silhouette — TF-IDF")
        ax.set_xlabel("K")
        ax.set_ylabel("Score")
        ax.set_xticks(sil_df["K"])
        ax.grid(True, alpha=0.3)
        st.pyplot(fig)
        plt.close(fig)

    best_k_struct = int(sil_df.loc[sil_df["silhouette_structurees"].idxmax(), "K"])
    best_k_tfidf = int(sil_df.loc[sil_df["silhouette_tfidf"].idxmax(), "K"])

    st.success(
        f"**Meilleur K structurées** : {best_k_struct} — "
        f"**Meilleur K TF-IDF** : {best_k_tfidf}"
    )

    # --- Clustering structuré ---
    st.markdown("---")
    st.subheader(f"Clusters — Variables structurées (K={best_k_struct})")

    km_struct = KMeans(n_clusters=best_k_struct, random_state=42, n_init=10)
    df["cluster_struct"] = km_struct.fit_predict(X_struct_scaled)

    stats = df.groupby("cluster_struct")[colonnes_struct].mean().round(2)
    st.dataframe(stats, use_container_width=True)

    # Distribution par cluster
    col1, col2 = st.columns(2)

    with col1:
        fig, ax = plt.subplots(figsize=(6, 4))
        df["cluster_struct"].value_counts().sort_index().plot(
            kind="bar", ax=ax, color=sns.color_palette("viridis", best_k_struct)
        )
        ax.set_title("Nombre de films par cluster")
        ax.set_xlabel("Cluster")
        ax.set_ylabel("Nombre de films")
        st.pyplot(fig)
        plt.close(fig)

    with col2:
        fig, ax = plt.subplots(figsize=(6, 4))
        for c in range(best_k_struct):
            subset = df[df["cluster_struct"] == c]
            ax.scatter(
                subset["budget"] / 1e6,
                subset["revenue"] / 1e6,
                label=f"Cluster {c}",
                alpha=0.5, s=15
            )
        ax.set_xlabel("Budget (M$)")
        ax.set_ylabel("Revenu (M$)")
        ax.set_title("Budget vs Revenu par cluster")
        ax.legend()
        st.pyplot(fig)
        plt.close(fig)

    # Interprétation
    st.markdown("**Interprétation :**")
    for c in range(best_k_struct):
        sub = df[df["cluster_struct"] == c]
        st.write(
            f"- **Cluster {c}** ({len(sub)} films) : "
            f"popularité moy. = {sub['popularity'].mean():.1f}, "
            f"note moy. = {sub['vote_average'].mean():.2f}, "
            f"budget moy. = {sub['budget'].mean()/1e6:.0f}M$, "
            f"revenu moy. = {sub['revenue'].mean()/1e6:.0f}M$"
        )

    # --- Clustering TF-IDF + SVD 2D ---
    st.markdown("---")
    st.subheader(f"Clusters — TF-IDF (K={best_k_tfidf}) + TruncatedSVD 2D")

    km_tfidf = KMeans(n_clusters=best_k_tfidf, random_state=42, n_init=10)
    labels_tfidf = km_tfidf.fit_predict(X_tfidf)

    svd = TruncatedSVD(n_components=2, random_state=42)
    X_2d = svd.fit_transform(X_tfidf)

    st.caption(
        f"Variance expliquée par TruncatedSVD : "
        f"{svd.explained_variance_ratio_.sum():.2%}"
    )

    fig, ax = plt.subplots(figsize=(12, 7))
    scatter = ax.scatter(
        X_2d[:, 0], X_2d[:, 1],
        c=labels_tfidf, cmap="viridis",
        alpha=0.5, s=20
    )
    plt.colorbar(scatter, ax=ax, label="Cluster")
    ax.set_title(f"Clusters TF-IDF en 2D (K={best_k_tfidf})")
    ax.set_xlabel("Composante 1")
    ax.set_ylabel("Composante 2")
    st.pyplot(fig)
    plt.close(fig)

    # Termes dominants par cluster TF-IDF
    st.markdown("**Termes dominants par cluster :**")
    terms = vectorizer.get_feature_names_out()

    for c in range(best_k_tfidf):
        indices = np.where(labels_tfidf == c)[0]
        cluster_mean = X_tfidf[indices].mean(axis=0).A1
        top_idx = cluster_mean.argsort()[-10:][::-1]
        top_terms = [terms[i] for i in top_idx]
        st.write(f"- **Cluster {c}** ({len(indices)} films) : {', '.join(top_terms)}")
