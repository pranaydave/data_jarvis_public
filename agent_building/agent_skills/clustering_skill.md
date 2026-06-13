# Clustering Analysis Skill

## When to Use This Skill
Apply this skill for cluster analysis on tabular customer or behavioral data, especially for:
- Customer segmentation
- Market segmentation
- Pattern discovery in multi-dimensional behavioral data

---

## Step 1: Data Preprocessing
Always standardize features before clustering — K-Means is sensitive to feature scale.

```python
from sklearn.preprocessing import StandardScaler
features = ['Annual Income (k$)', 'Spending Score (1-100)', 'Age']
scaler = StandardScaler()
X = scaler.fit_transform(df[features])
```

Never use ID columns as features.

---

## Step 2: Find Optimal Number of Clusters
Always test K from 2 to 8 using BOTH methods before fitting a final model.

**Elbow Method:** Plot inertia (within-cluster sum of squares) vs K. The "elbow" is where
the rate of decrease sharply slows — this is your candidate K.

**Silhouette Score:** Measures how similar each point is to its own cluster vs. other clusters.
Range: -1 to 1. Score > 0.5 = well-separated clusters. Choose the K with the highest silhouette score.

**Decision rule:** Choose K where the elbow is visible AND silhouette score is at or near its peak.

```python
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

inertias, silhouettes = [], []
K_range = range(2, 9)

for k in K_range:
    km = KMeans(n_clusters=k, random_state=42, n_init=10)
    labels = km.fit_predict(X)
    inertias.append(km.inertia_)
    silhouettes.append(silhouette_score(X, labels))

best_k = list(K_range)[silhouettes.index(max(silhouettes))]
```

---

## Step 3: Fit the Final Model

```python
km_final = KMeans(n_clusters=best_k, random_state=42, n_init=10)
df['Cluster'] = km_final.fit_predict(X)
```

---

## Step 4: Visualization — Always Use a 4-Panel Layout

Create a `fig, axes = plt.subplots(2, 2, figsize=(14, 10))` figure:

| Position | Plot | Details |
|----------|------|---------|
| `axes[0,0]` | Elbow Curve | Inertia vs K; mark chosen K with a red dashed vertical line |
| `axes[0,1]` | Silhouette Scores | Score vs K; mark chosen K with a red dashed vertical line |
| `axes[1,0]` | Scatter Plot | Annual Income vs Spending Score, colored by cluster; mark centroids as gold stars |
| `axes[1,1]` | Cluster Profiles | Horizontal bar chart of mean feature values per cluster (normalized) |

Add a main `fig.suptitle` and call `plt.tight_layout()` before `plt.show()`.

---

## Step 5: Business Segment Naming
After clustering, compute per-cluster means and assign business labels:

| Income Level | Spending Level | Segment Name | Strategy |
|-------------|---------------|--------------|----------|
| High (>70k) | High (>60) | **Premium Customers** | Loyalty programs, exclusive offers |
| High (>70k) | Low (<40) | **Cautious High-Earners** | Targeted promotions, trust-building |
| Low (<45k) | High (>60) | **Young Spenders** | Instalment plans, trendy products |
| Low (<45k) | Low (<40) | **Budget Conscious** | Discounts, value bundles |
| Medium | Medium | **Balanced Customers** | General marketing, cross-sell |

---

## Step 6: Required Printed Output
Always print the following after clustering:

1. `"Optimal K = X  |  Silhouette Score = Y.YY"`
2. A cluster summary table: Cluster ID, Size, Mean Age, Mean Income, Mean Spending, Segment Name
3. One actionable recommendation per segment

---

## Step 7: Complete Code Template

```python
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score

# 1. Preprocess
features = ['Annual Income (k$)', 'Spending Score (1-100)', 'Age']
scaler = StandardScaler()
X = scaler.fit_transform(df[features])

# 2. Find optimal K
inertias, silhouettes = [], []
K_range = range(2, 9)
for k in K_range:
    km = KMeans(n_clusters=k, random_state=42, n_init=10)
    labels = km.fit_predict(X)
    inertias.append(km.inertia_)
    silhouettes.append(silhouette_score(X, labels))

best_k = list(K_range)[silhouettes.index(max(silhouettes))]
print(f"Optimal K = {best_k}  |  Silhouette Score = {max(silhouettes):.2f}")

# 3. Fit final model
km_final = KMeans(n_clusters=best_k, random_state=42, n_init=10)
df['Cluster'] = km_final.fit_predict(X)

# 4. Four-panel visualization
colors = plt.cm.tab10.colors
fig, axes = plt.subplots(2, 2, figsize=(14, 10))
fig.suptitle(f'Customer Segmentation — K-Means (K={best_k})', fontsize=16, fontweight='bold')

axes[0, 0].plot(list(K_range), inertias, 'bo-')
axes[0, 0].axvline(best_k, color='red', linestyle='--', label=f'K={best_k}')
axes[0, 0].set(title='Elbow Curve', xlabel='K', ylabel='Inertia')
axes[0, 0].legend()

axes[0, 1].plot(list(K_range), silhouettes, 'go-')
axes[0, 1].axvline(best_k, color='red', linestyle='--', label=f'K={best_k}')
axes[0, 1].set(title='Silhouette Scores', xlabel='K', ylabel='Silhouette Score')
axes[0, 1].legend()

for cluster_id in range(best_k):
    mask = df['Cluster'] == cluster_id
    axes[1, 0].scatter(df.loc[mask, 'Annual Income (k$)'],
                       df.loc[mask, 'Spending Score (1-100)'],
                       c=[colors[cluster_id]], label=f'Cluster {cluster_id}', alpha=0.7)
centroids_orig = scaler.inverse_transform(km_final.cluster_centers_)
axes[1, 0].scatter(centroids_orig[:, 0], centroids_orig[:, 1],
                   c='gold', marker='*', s=300, zorder=5, label='Centroids')
axes[1, 0].set(title='Income vs Spending by Cluster',
               xlabel='Annual Income (k$)', ylabel='Spending Score')
axes[1, 0].legend()

cluster_means = df.groupby('Cluster')[features].mean()
cluster_means_norm = (cluster_means - cluster_means.min()) / (cluster_means.max() - cluster_means.min())
cluster_means_norm.T.plot(kind='barh', ax=axes[1, 1], colormap='tab10')
axes[1, 1].set(title='Cluster Profiles (Normalized)', xlabel='Normalized Mean Value')

plt.tight_layout()
plt.show()

# 5. Print cluster summary with business labels
def assign_label(row):
    inc, spend = row['Annual Income (k$)'], row['Spending Score (1-100)']
    if inc > 70 and spend > 60: return 'Premium Customers'
    if inc > 70 and spend < 40: return 'Cautious High-Earners'
    if inc < 45 and spend > 60: return 'Young Spenders'
    if inc < 45 and spend < 40: return 'Budget Conscious'
    return 'Balanced Customers'

summary = cluster_means.copy()
summary['Size'] = df['Cluster'].value_counts().sort_index()
summary['Segment'] = summary.apply(assign_label, axis=1)
print("\nCluster Summary:")
print(summary.to_string())
```
