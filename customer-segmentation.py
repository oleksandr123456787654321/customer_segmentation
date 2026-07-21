"""
customer_segmentation.py
--------------------------
Clusters mall customers by age, annual income, and spending score.
Uses K-means (with K chosen via elbow method + silhouette score) and
compares it against hierarchical clustering on the same data.
"""

import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans, AgglomerativeClustering
from sklearn.metrics import silhouette_score, adjusted_rand_score
from sklearn.decomposition import PCA
from scipy.cluster.hierarchy import dendrogram, linkage

DATA_PATH = 'Mall_Customers.csv'
OUTPUT_DIR = 'outputs'
FEATURE_COLUMNS = ['Age', 'Annual Income (k$)', 'Spending Score (1-100)']
K = 6  # chosen via elbow method + silhouette score


# ----------------------------------------------------------------------
# Step 1 — Load and explore
# ----------------------------------------------------------------------
def load_data(path):
    df = pd.read_csv(path)
    print(f"Loaded {df.shape[0]} rows, {df.shape[1]} columns")
    assert df.isna().sum().sum() == 0, "Unexpected missing values"
    return df


def plot_distributions(df):
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    sns.histplot(df['Age'], bins=20, ax=axes[0]).set_title('Age Distribution')
    sns.histplot(df['Annual Income (k$)'], bins=20, ax=axes[1]).set_title('Annual Income Distribution')
    sns.histplot(df['Spending Score (1-100)'], bins=20, ax=axes[2]).set_title('Spending Score Distribution')
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, 'distributions.png'), dpi=150, bbox_inches='tight')


def plot_income_vs_spending(df):
    plt.figure(figsize=(8, 6))
    sns.scatterplot(data=df, x='Annual Income (k$)', y='Spending Score (1-100)')
    plt.title('Annual Income vs Spending Score')
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, 'income_vs_spending.png'), dpi=150, bbox_inches='tight')


# ----------------------------------------------------------------------
# Step 2 — Scale features and choose K
# ----------------------------------------------------------------------
def scale_features(df, columns):
    scaler = StandardScaler()
    scaled = scaler.fit_transform(df[columns])
    return scaled, scaler


def plot_elbow(scaled_features, k_range=range(1, 11)):
    inertias = []
    for k in k_range:
        km = KMeans(n_clusters=k, random_state=42, n_init=10)
        km.fit(scaled_features)
        inertias.append(km.inertia_)

    plt.figure(figsize=(8, 5))
    plt.plot(list(k_range), inertias, marker='o')
    plt.xlabel('Number of Clusters (K)')
    plt.ylabel('Inertia')
    plt.title('Elbow Method for Choosing K')
    plt.xticks(list(k_range))
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, 'elbow_method.png'), dpi=150, bbox_inches='tight')


def plot_silhouette(scaled_features, k_range=range(2, 11)):
    scores = []
    for k in k_range:
        km = KMeans(n_clusters=k, random_state=42, n_init=10)
        labels = km.fit_predict(scaled_features)
        scores.append(silhouette_score(scaled_features, labels))

    plt.figure(figsize=(8, 5))
    plt.plot(list(k_range), scores, marker='o', color='green')
    plt.xlabel('Number of Clusters (K)')
    plt.ylabel('Silhouette Score')
    plt.title('Silhouette Score for Choosing K')
    plt.xticks(list(k_range))
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, 'silhouette_scores.png'), dpi=150, bbox_inches='tight')

    best_k = list(k_range)[scores.index(max(scores))]
    print(f"Best K by silhouette score: {best_k} (score={max(scores):.3f})")
    return best_k


# ----------------------------------------------------------------------
# Step 3 — K-means clustering + PCA visualization
# ----------------------------------------------------------------------
def run_kmeans(scaled_features, df, k):
    km = KMeans(n_clusters=k, random_state=42, n_init=10)
    cluster_labels = km.fit_predict(scaled_features)
    df = df.copy()
    df['Cluster'] = cluster_labels
    return df, km


def plot_clusters_pca(scaled_features, cluster_labels, k):
    pca = PCA(n_components=2, random_state=42)
    components = pca.fit_transform(scaled_features)

    plt.figure(figsize=(9, 7))
    scatter = plt.scatter(components[:, 0], components[:, 1],
                           c=cluster_labels, cmap='tab10', s=60, alpha=0.8)
    plt.xlabel(f'PC1 ({pca.explained_variance_ratio_[0]:.1%} variance)')
    plt.ylabel(f'PC2 ({pca.explained_variance_ratio_[1]:.1%} variance)')
    plt.title(f'Customer Clusters (K={k}) — PCA Projection')
    plt.colorbar(scatter, label='Cluster')
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, 'clusters_pca.png'), dpi=150, bbox_inches='tight')


def summarize_clusters(df):
    summary = df.groupby('Cluster')[FEATURE_COLUMNS].mean().round(1)
    summary['Count'] = df.groupby('Cluster').size()
    return summary


# ----------------------------------------------------------------------
# Step 4 — Hierarchical clustering comparison
# ----------------------------------------------------------------------
def plot_dendrogram(scaled_features):
    linkage_matrix = linkage(scaled_features, method='ward')

    plt.figure(figsize=(12, 6))
    dendrogram(linkage_matrix, truncate_mode='lastp', p=30)
    plt.xlabel('Customers (or customer clusters)')
    plt.ylabel('Distance')
    plt.title('Hierarchical Clustering Dendrogram')
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, 'dendrogram.png'), dpi=150, bbox_inches='tight')


def run_hierarchical(scaled_features, k):
    hc = AgglomerativeClustering(n_clusters=k, linkage='ward')
    return hc.fit_predict(scaled_features)


def compare_clusterings(kmeans_labels, hierarchical_labels):
    score = adjusted_rand_score(kmeans_labels, hierarchical_labels)
    print(f"\nAdjusted Rand Index (K-means vs Hierarchical): {score:.3f}")

    crosstab = pd.crosstab(
        pd.Series(kmeans_labels, name='K-Means Cluster'),
        pd.Series(hierarchical_labels, name='Hierarchical Cluster')
    )
    print("\nCross-tabulation:")
    print(crosstab)


# ----------------------------------------------------------------------
# Main
# ----------------------------------------------------------------------
def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    df = load_data(DATA_PATH)
    plot_distributions(df)
    plot_income_vs_spending(df)

    scaled_features, scaler = scale_features(df, FEATURE_COLUMNS)
    plot_elbow(scaled_features)
    plot_silhouette(scaled_features)

    df_clustered, kmeans_model = run_kmeans(scaled_features, df, k=K)
    plot_clusters_pca(scaled_features, df_clustered['Cluster'], k=K)

    print("\nCluster summary:")
    print(summarize_clusters(df_clustered))

    plot_dendrogram(scaled_features)
    hierarchical_labels = run_hierarchical(scaled_features, k=K)
    compare_clusterings(df_clustered['Cluster'].values, hierarchical_labels)

    plt.show()


if __name__ == '__main__':
    main()