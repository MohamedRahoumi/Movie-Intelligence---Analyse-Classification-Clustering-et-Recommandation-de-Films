import pandas as pd
import numpy as np
from pathlib import Path
from scipy import sparse

from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.decomposition import TruncatedSVD

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns


ROOT = Path(__file__).resolve().parents[1]

INPUT_FILE = ROOT / "data" / "processed" / "movies_engineered.csv"
TFIDF_FILE = ROOT / "outputs" / "tfidf" / "tfidf_matrix.npz"

OUTPUT_DIR = ROOT / "outputs" / "clustering"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


df = pd.read_csv(INPUT_FILE)
X_tfidf = sparse.load_npz(TFIDF_FILE)

print("=" * 60)
print("ÉTAPE 8 - CLUSTERING")
print("=" * 60)

print(f"\nNombre de films : {len(df)}")
print(f"Dimensions TF-IDF : {X_tfidf.shape}")

print("\n" + "=" * 60)
print("CLUSTERING - VARIABLES STRUCTURÉES")
print("=" * 60)

colonnes_structurees = [
    "runtime",
    "budget",
    "revenue",
    "popularity",
    "vote_average",
    "vote_count",
    "genre_count",
    "keyword_count",
    "roi",
    "movie_age"
]

X_struct = df[colonnes_structurees].values

scaler = StandardScaler()
X_struct_scaled = scaler.fit_transform(X_struct)

k_values = range(2, 11)
silhouette_scores_struct = []

for k in k_values:
    kmeans = KMeans(
        n_clusters=k,
        random_state=42,
        n_init=10
    )
    labels = kmeans.fit_predict(X_struct_scaled)
    score = silhouette_score(X_struct_scaled, labels)
    silhouette_scores_struct.append(score)
    print(f"  K={k}  →  Silhouette = {score:.4f}")

# Graphique Silhouette
plt.figure(figsize=(10, 5))

plt.plot(
    list(k_values),
    silhouette_scores_struct,
    marker="o"
)

plt.title("Silhouette Score - Variables structurées")
plt.xlabel("Nombre de clusters (K)")
plt.ylabel("Silhouette Score")
plt.xticks(list(k_values))
plt.grid(True, alpha=0.3)
plt.tight_layout()

plt.savefig(OUTPUT_DIR / "silhouette_structurees.png")
plt.close()

# Meilleur K
best_k_struct = list(k_values)[
    np.argmax(silhouette_scores_struct)
]

print(f"\nMeilleur K (structurées) : {best_k_struct}")
print(f"Silhouette : {max(silhouette_scores_struct):.4f}")

print("\n" + "=" * 60)
print("CLUSTERING - TF-IDF")
print("=" * 60)

silhouette_scores_tfidf = []

for k in k_values:
    kmeans = KMeans(
        n_clusters=k,
        random_state=42,
        n_init=10
    )
    labels = kmeans.fit_predict(X_tfidf)
    score = silhouette_score(X_tfidf, labels)
    silhouette_scores_tfidf.append(score)
    print(f"  K={k}  →  Silhouette = {score:.4f}")

# Graphique Silhouette
plt.figure(figsize=(10, 5))

plt.plot(
    list(k_values),
    silhouette_scores_tfidf,
    marker="o",
    color="orange"
)

plt.title("Silhouette Score - TF-IDF")
plt.xlabel("Nombre de clusters (K)")
plt.ylabel("Silhouette Score")
plt.xticks(list(k_values))
plt.grid(True, alpha=0.3)
plt.tight_layout()

plt.savefig(OUTPUT_DIR / "silhouette_tfidf.png")
plt.close()

best_k_tfidf = list(k_values)[
    np.argmax(silhouette_scores_tfidf)
]

print(f"\nMeilleur K (TF-IDF) : {best_k_tfidf}")
print(f"Silhouette : {max(silhouette_scores_tfidf):.4f}")


print("\n" + "=" * 60)
print(f"ANALYSE DES CLUSTERS (K={best_k_struct}, structurées)")
print("=" * 60)

kmeans_struct = KMeans(
    n_clusters=best_k_struct,
    random_state=42,
    n_init=10
)

df["cluster_struct"] = kmeans_struct.fit_predict(X_struct_scaled)

print("\nNombre de films par cluster :")
print(df["cluster_struct"].value_counts().sort_index())

print("\nCaractéristiques moyennes par cluster :")

stats_cluster = df.groupby("cluster_struct")[
    colonnes_structurees
].mean().round(2)

print(stats_cluster.to_string())

# Interprétation
print("\n--- Interprétation des clusters (structurées) ---")

for cluster_id in range(best_k_struct):
    cluster_data = df[df["cluster_struct"] == cluster_id]
    n = len(cluster_data)
    pop_moy = cluster_data["popularity"].mean()
    vote_moy = cluster_data["vote_average"].mean()
    budget_moy = cluster_data["budget"].mean()
    revenue_moy = cluster_data["revenue"].mean()

    print(
        f"\nCluster {cluster_id} ({n} films) : "
        f"popularité moy={pop_moy:.1f}, "
        f"note moy={vote_moy:.2f}, "
        f"budget moy={budget_moy/1e6:.0f}M, "
        f"revenu moy={revenue_moy/1e6:.0f}M"
    )


print("\n" + "=" * 60)
print(f"ANALYSE DES CLUSTERS (K={best_k_tfidf}, TF-IDF)")
print("=" * 60)

kmeans_tfidf = KMeans(
    n_clusters=best_k_tfidf,
    random_state=42,
    n_init=10
)

df["cluster_tfidf"] = kmeans_tfidf.fit_predict(X_tfidf)

print("\nNombre de films par cluster :")
print(df["cluster_tfidf"].value_counts().sort_index())

svd = TruncatedSVD(n_components=2, random_state=42)
X_tfidf_2d = svd.fit_transform(X_tfidf)

print(
    f"\nVariance expliquée par SVD 2D : "
    f"{svd.explained_variance_ratio_.sum():.2%}"
)

# Visualisation 2D
plt.figure(figsize=(12, 8))

scatter = plt.scatter(
    X_tfidf_2d[:, 0],
    X_tfidf_2d[:, 1],
    c=df["cluster_tfidf"],
    cmap="viridis",
    alpha=0.5,
    s=20
)

plt.colorbar(scatter, label="Cluster")
plt.title(
    f"Clusters TF-IDF visualisés en 2D (TruncatedSVD, K={best_k_tfidf})"
)
plt.xlabel("Composante 1")
plt.ylabel("Composante 2")
plt.tight_layout()

plt.savefig(OUTPUT_DIR / "clusters_tfidf_2d.png")
plt.close()

# Interprétation : termes fréquents par cluster
print("\n--- Interprétation des clusters TF-IDF ---")

import joblib

vectorizer = joblib.load(
    ROOT / "outputs" / "tfidf" / "tfidf_vectorizer.joblib"
)
terms = vectorizer.get_feature_names_out()

for cluster_id in range(best_k_tfidf):
    indices = df[df["cluster_tfidf"] == cluster_id].index
    cluster_tfidf = X_tfidf[indices]

    # Moyenne des scores TF-IDF par terme dans ce cluster
    mean_scores = cluster_tfidf.mean(axis=0).A1
    top_indices = mean_scores.argsort()[-10:][::-1]
    top_terms = [terms[i] for i in top_indices]

    n = len(indices)
    print(f"\nCluster {cluster_id} ({n} films) :")
    print(f"  Termes dominants : {', '.join(top_terms)}")


resultats_silhouette = pd.DataFrame({
    "K": list(k_values),
    "silhouette_structurees": silhouette_scores_struct,
    "silhouette_tfidf": silhouette_scores_tfidf
})

resultats_silhouette.to_csv(
    OUTPUT_DIR / "silhouette_scores.csv",
    index=False,
    encoding="utf-8"
)

stats_cluster.to_csv(
    OUTPUT_DIR / "clusters_structurees_stats.csv",
    encoding="utf-8"
)


print("\n" + "=" * 60)
print("ÉTAPE 8 TERMINÉE")
print("=" * 60)

print(f"\nFichiers sauvegardés dans : {OUTPUT_DIR}")
