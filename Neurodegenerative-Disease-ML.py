import os
import kagglehub
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.decomposition import PCA

# 1. DOWNLOAD AND LOAD DATASET

path = kagglehub.dataset_download(
    "gowtha69/neurodegenerative-disorders-dataset"
)

file_path = os.path.join(path, "NDCTD-45K.csv")
df = pd.read_csv(file_path)

print("Dataset shape:", df.shape)

# ------------------------------------------
# PREPROCESSING

y = df["Diagnosis"]

# Remove ID, target and possible leakage variables
remove_columns = [
    "Patient_ID",
    "Diagnosis",
    "Disease_Stage",
    "Disease_Severity"
]

X = df.drop(columns=remove_columns)

# Use numerical features
X = X.select_dtypes(include=["number"])

# Split before imputation/scaling to avoid data leakage
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)

# Imputation
imputer = SimpleImputer(strategy="median")

X_train = imputer.fit_transform(X_train)
X_test = imputer.transform(X_test)

# Scaling
scaler = StandardScaler()

X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Prepare all data for clustering
X_all = imputer.transform(X)
X_all_scaled = scaler.transform(X_all)


# ------------------------------------------
# CLASSIFICATION

models = {
    "Logistic Regression": LogisticRegression(
        max_iter=1000,
        random_state=42
    ),

    "Random Forest": RandomForestClassifier(
        n_estimators=100,
        random_state=42,
        n_jobs=-1
    ),

    "SVM": SVC(
        kernel="rbf",
        random_state=42
    )
}

results = {}

for name, model in models.items():

    if name == "Random Forest":
        model.fit(X_train, y_train)
        predictions = model.predict(X_test)
    else:
        model.fit(X_train_scaled, y_train)
        predictions = model.predict(X_test_scaled)

    accuracy = accuracy_score(y_test, predictions)
    results[name] = accuracy

    print("\n==============================")
    print(name)
    print("==============================")
    print("Accuracy:", accuracy * 100, "%")
    print(classification_report(y_test, predictions))

    # Confusion matrix
    cm = confusion_matrix(y_test, predictions)

    plt.figure(figsize=(6, 5))
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=sorted(y.unique()),
        yticklabels=sorted(y.unique())
    )

    plt.xlabel("Predicted Diagnosis")
    plt.ylabel("Actual Diagnosis")
    plt.title(name + " - Confusion Matrix")
    plt.tight_layout()

    os.makedirs("results", exist_ok=True)

    filename = name.lower().replace(" ", "_") + "_confusion_matrix.png"
    plt.savefig("results/" + filename, dpi=300)
    plt.show()


# ------------------------------------------
# MODEL COMPARISON

print("\n===== MODEL COMPARISON =====")

for name, accuracy in results.items():
    print(f"{name}: {accuracy * 100:.2f}%")


# ------------------------------------------
# RANDOM FOREST FEATURE IMPORTANCE

rf = models["Random Forest"]

importance = pd.Series(
    rf.feature_importances_,
    index=X.columns
)

top_features = importance.nlargest(15)

print("\n===== TOP 15 FEATURES =====")
print(top_features)

plt.figure(figsize=(9, 6))

top_features.sort_values().plot(
    kind="barh"
)

plt.xlabel("Feature Importance")
plt.ylabel("Features")
plt.title("Top 15 Features - Random Forest")
plt.tight_layout()

plt.savefig(
    "results/random_forest_feature_importance.png",
    dpi=300
)

plt.show()

# ------------------------------------------
# K-MEANS CLUSTERING

kmeans = KMeans(
    n_clusters=6,
    random_state=42,
    n_init=10
)

clusters = kmeans.fit_predict(X_all_scaled)

silhouette = silhouette_score(
    X_all_scaled,
    clusters
)

print("\n===== K-MEANS CLUSTERING =====")
print("Number of clusters:", 6)
print("Silhouette Score:", silhouette)


# ------------------------------------------
# PCA VISUALIZATION

pca = PCA(n_components=2)

X_pca = pca.fit_transform(X_all_scaled)

plt.figure(figsize=(8, 6))

plt.scatter(
    X_pca[:, 0],
    X_pca[:, 1],
    c=clusters,
    cmap="tab10",
    s=10
)

plt.xlabel("PC1")
plt.ylabel("PC2")
plt.title("K-Means Clustering Visualized Using PCA")
plt.colorbar(label="Cluster")
plt.tight_layout()

plt.savefig(
    "results/kmeans_pca.png",
    dpi=300
)

plt.show()


# ------------------------------------------
# CLUSTER VS DIAGNOSIS

comparison = pd.crosstab(
    clusters,
    y,
    normalize="index"
) * 100

print("\n===== CLUSTER VS DIAGNOSIS (%) =====")
print(comparison.round(2))

print("\n===== ANALYSIS COMPLETE =====")