# ============================================================
# CLASSIFICATION DU CANCER DU SEIN
# par Machine Learning: comparaison de la
#  Regression Logistique, Random Forest, KNN et SVM
# ============================================================

# Installation éventuelle des bibliothèques :
# pip install pandas numpy matplotlib seaborn scikit-learn

import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    classification_report,
    confusion_matrix,
    roc_curve
)


# ============================================================
# 1. CHARGEMENT DES DONNÉES
# ============================================================

FILE_PATH = "Copie de data.csv"

try:
    df = pd.read_csv(FILE_PATH)
    print("Fichier chargé avec succès.")
except FileNotFoundError:
    raise FileNotFoundError(
        f"Le fichier '{FILE_PATH}' est introuvable. "
        "Placez-le dans le même dossier que le script."
    )

print("\nDimensions initiales du dataset :", df.shape)

print("\nCinq premières lignes :")
print(df.head())


# ============================================================
# 2. ANALYSE EXPLORATOIRE DES DONNÉES — EDA
# ============================================================

print("\n" + "=" * 70)
print("INFORMATIONS GÉNÉRALES")
print("=" * 70)

df.info()

print("\n" + "=" * 70)
print("STATISTIQUES DESCRIPTIVES")
print("=" * 70)

print(df.describe(include="all").T)

print("\n" + "=" * 70)
print("VALEURS MANQUANTES")
print("=" * 70)

missing_values = df.isnull().sum()
print(missing_values[missing_values > 0])

print("\nNombre total de doublons :", df.duplicated().sum())

print("\nRépartition initiale de la variable diagnosis :")
print(df["diagnosis"].value_counts())

print("\nRépartition en pourcentage :")
print(df["diagnosis"].value_counts(normalize=True).mul(100).round(2))


# ============================================================
# 3. NETTOYAGE DES DONNÉES
# ============================================================

# Suppression de la colonne vide générée lors de l'export CSV
if "Unnamed: 32" in df.columns:
    df = df.drop(columns=["Unnamed: 32"])

# Suppression de l'identifiant, car il ne constitue pas
# une caractéristique médicale utile pour la prédiction
if "id" in df.columns:
    df = df.drop(columns=["id"])

# Suppression des doublons éventuels
df = df.drop_duplicates().reset_index(drop=True)

# Encodage manuel de la variable cible :
# B = Bénin = 0
# M = Malin = 1
df["diagnosis"] = df["diagnosis"].map({
    "B": 0,
    "M": 1
})

# Vérification de l'encodage
if df["diagnosis"].isnull().any():
    raise ValueError(
        "La colonne diagnosis contient des valeurs différentes de B et M."
    )

print("\nDimensions après nettoyage :", df.shape)

print("\nRépartition de la cible après encodage :")
print(df["diagnosis"].value_counts())


# ============================================================
# 4. VISUALISATIONS EDA
# ============================================================

# 4.1 Distribution de la variable cible
plt.figure(figsize=(7, 5))

sns.countplot(
    data=df,
    x="diagnosis"
)

plt.title("Répartition des diagnostics")
plt.xlabel("Diagnostic")
plt.ylabel("Nombre de patientes")
plt.xticks(
    ticks=[0, 1],
    labels=["Bénin", "Malin"]
)
plt.tight_layout()
plt.show()


# 4.2 Répartition en pourcentage
target_counts = df["diagnosis"].value_counts().sort_index()

plt.figure(figsize=(7, 5))

plt.pie(
    target_counts,
    labels=["Bénin", "Malin"],
    autopct="%1.1f%%",
    startangle=90
)

plt.title("Proportion des diagnostics")
plt.tight_layout()
plt.show()


# 4.3 Histogrammes des variables
df.drop(columns=["diagnosis"]).hist(
    figsize=(20, 18),
    bins=20
)

plt.suptitle(
    "Distribution des variables médicales",
    fontsize=16
)
plt.tight_layout()
plt.show()


# 4.4 Matrice de corrélation complète
correlation_matrix = df.corr(numeric_only=True)

plt.figure(figsize=(20, 16))

sns.heatmap(
    correlation_matrix,
    cmap="coolwarm",
    center=0
)

plt.title("Matrice de corrélation des variables")
plt.tight_layout()
plt.show()


# 4.5 Variables les plus corrélées avec le diagnostic
target_correlations = (
    correlation_matrix["diagnosis"]
    .drop("diagnosis")
    .abs()
    .sort_values(ascending=False)
)

print("\nVariables les plus corrélées avec le diagnostic :")
print(target_correlations.head(10))

plt.figure(figsize=(10, 6))

target_correlations.head(10).sort_values().plot(
    kind="barh"
)

plt.title("Top 10 des variables corrélées au diagnostic")
plt.xlabel("Valeur absolue de la corrélation")
plt.ylabel("Variable")
plt.tight_layout()
plt.show()


# 4.6 Boxplots de quelques variables importantes
important_features = [
    "radius_worst",
    "perimeter_worst",
    "area_worst",
    "concave points_worst"
]

for feature in important_features:
    if feature in df.columns:

        plt.figure(figsize=(7, 5))

        sns.boxplot(
            data=df,
            x="diagnosis",
            y=feature
        )

        plt.title(
            f"Distribution de {feature} selon le diagnostic"
        )

        plt.xlabel("Diagnostic")
        plt.ylabel(feature)

        plt.xticks(
            ticks=[0, 1],
            labels=["Bénin", "Malin"]
        )

        plt.tight_layout()
        plt.show()


# ============================================================
# 5. SÉPARATION DES VARIABLES EXPLICATIVES ET DE LA CIBLE
# ============================================================

X = df.drop(columns=["diagnosis"])
y = df["diagnosis"]

print("\nDimensions de X :", X.shape)
print("Dimensions de y :", y.shape)


# ============================================================
# 6. SÉPARATION TRAIN / TEST
# ============================================================

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)

print("\nTaille de l'ensemble d'entraînement :", X_train.shape)
print("Taille de l'ensemble de test :", X_test.shape)

print("\nRépartition de la cible dans l'entraînement :")
print(y_train.value_counts(normalize=True).round(3))

print("\nRépartition de la cible dans le test :")
print(y_test.value_counts(normalize=True).round(3))


# ============================================================
# 7. DÉFINITION DES MODÈLES
# ============================================================

models = {
    "Logistic Regression": Pipeline([
        (
            "scaler",
            StandardScaler()
        ),
        (
            "classifier",
            LogisticRegression(
                max_iter=2000,
                random_state=42
            )
        )
    ]),

    "Random Forest": RandomForestClassifier(
        n_estimators=300,
        random_state=42,
        class_weight="balanced"
    ),

    "KNN": Pipeline([
        (
            "scaler",
            StandardScaler()
        ),
        (
            "classifier",
            KNeighborsClassifier(
                n_neighbors=5
            )
        )
    ]),

    "SVM": Pipeline([
        (
            "scaler",
            StandardScaler()
        ),
        (
            "classifier",
            SVC(
                kernel="rbf",
                probability=True,
                random_state=42
            )
        )
    ])
}


# ============================================================
# 8. ENTRAÎNEMENT ET ÉVALUATION
# ============================================================

results = []
roc_results = {}
trained_models = {}
confusion_matrices = {}

for model_name, model in models.items():

    print("\n" + "=" * 70)
    print(f"ENTRAÎNEMENT DU MODÈLE : {model_name}")
    print("=" * 70)

    # Entraînement
    model.fit(X_train, y_train)

    # Sauvegarde du modèle entraîné
    trained_models[model_name] = model

    # Prédictions
    y_pred = model.predict(X_test)

    # Probabilités de la classe maligne
    y_proba = model.predict_proba(X_test)[:, 1]

    # Calcul des métriques
    accuracy = accuracy_score(y_test, y_pred)

    precision = precision_score(
        y_test,
        y_pred,
        zero_division=0
    )

    recall = recall_score(
        y_test,
        y_pred,
        zero_division=0
    )

    f1 = f1_score(
        y_test,
        y_pred,
        zero_division=0
    )

    roc_auc = roc_auc_score(
        y_test,
        y_proba
    )

    results.append({
        "Modèle": model_name,
        "Accuracy": accuracy,
        "Précision": precision,
        "Rappel": recall,
        "F1-score": f1,
        "ROC-AUC": roc_auc
    })

    # Rapport de classification
    print("\nRapport de classification :")

    print(
        classification_report(
            y_test,
            y_pred,
            target_names=["Bénin", "Malin"],
            zero_division=0
        )
    )

    # Matrice de confusion
    cm = confusion_matrix(y_test, y_pred)

    confusion_matrices[model_name] = cm

    plt.figure(figsize=(6, 5))

    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cbar=False,
        xticklabels=["Bénin", "Malin"],
        yticklabels=["Bénin", "Malin"]
    )

    plt.title(
        f"Matrice de confusion — {model_name}"
    )

    plt.xlabel("Classe prédite")
    plt.ylabel("Classe réelle")
    plt.tight_layout()
    plt.show()

    # Calcul de la courbe ROC
    fpr, tpr, thresholds = roc_curve(
        y_test,
        y_proba
    )

    roc_results[model_name] = {
        "fpr": fpr,
        "tpr": tpr,
        "thresholds": thresholds,
        "auc": roc_auc
    }


# ============================================================
# 9. TABLEAU COMPARATIF DES MODÈLES
# ============================================================

results_df = pd.DataFrame(results)

results_df = results_df.sort_values(
    by="ROC-AUC",
    ascending=False
).reset_index(drop=True)

print("\n" + "=" * 70)
print("COMPARAISON DES MODÈLES")
print("=" * 70)

print(
    results_df.round(4).to_string(index=False)
)


# ============================================================
# 10. COURBES ROC DES QUATRE MODÈLES
# ============================================================

plt.figure(figsize=(9, 7))

for model_name, roc_data in roc_results.items():

    plt.plot(
        roc_data["fpr"],
        roc_data["tpr"],
        linewidth=2,
        label=(
            f"{model_name} "
            f"(AUC = {roc_data['auc']:.3f})"
        )
    )

plt.plot(
    [0, 1],
    [0, 1],
    linestyle="--",
    label="Modèle aléatoire"
)

plt.title("Comparaison des courbes ROC")
plt.xlabel("Taux de faux positifs")
plt.ylabel("Taux de vrais positifs")
plt.legend()
plt.grid(alpha=0.3)
plt.tight_layout()
plt.show()


# ============================================================
# 11. COMPARAISON GRAPHIQUE DES MÉTRIQUES
# ============================================================

metrics_to_plot = [
    "Accuracy",
    "Précision",
    "Rappel",
    "F1-score",
    "ROC-AUC"
]

results_long = results_df.melt(
    id_vars="Modèle",
    value_vars=metrics_to_plot,
    var_name="Métrique",
    value_name="Score"
)

plt.figure(figsize=(13, 7))

sns.barplot(
    data=results_long,
    x="Modèle",
    y="Score",
    hue="Métrique"
)

plt.title("Comparaison des performances des modèles")
plt.xlabel("Modèle")
plt.ylabel("Score")
plt.ylim(0, 1.05)
plt.xticks(rotation=15)
plt.legend(
    title="Métrique",
    bbox_to_anchor=(1.02, 1),
    loc="upper left"
)
plt.tight_layout()
plt.show()


# ============================================================
# 12. SÉLECTION AUTOMATIQUE DU MEILLEUR MODÈLE
# ============================================================

best_model_name = results_df.iloc[0]["Modèle"]
best_auc = results_df.iloc[0]["ROC-AUC"]

best_model = trained_models[best_model_name]

print("\n" + "=" * 70)
print("MEILLEUR MODÈLE")
print("=" * 70)

print(f"Meilleur modèle selon le ROC-AUC : {best_model_name}")
print(f"ROC-AUC obtenu : {best_auc:.4f}")


# ============================================================
# 13. IMPORTANCE DES VARIABLES — RANDOM FOREST
# ============================================================

random_forest_model = trained_models["Random Forest"]

feature_importances = pd.DataFrame({
    "Variable": X.columns,
    "Importance": random_forest_model.feature_importances_
})

feature_importances = feature_importances.sort_values(
    by="Importance",
    ascending=False
)

print("\nVariables les plus importantes selon Random Forest :")
print(feature_importances.head(15).to_string(index=False))

plt.figure(figsize=(10, 7))

sns.barplot(
    data=feature_importances.head(15),
    x="Importance",
    y="Variable"
)

plt.title(
    "Les 15 variables les plus importantes — Random Forest"
)
plt.xlabel("Importance")
plt.ylabel("Variable")
plt.tight_layout()
plt.show()


# ============================================================
# 14. TEST SUR UNE OBSERVATION DU JEU DE TEST
# ============================================================

sample = X_test.iloc[[0]]
real_class = y_test.iloc[0]

predicted_class = best_model.predict(sample)[0]
predicted_probability = best_model.predict_proba(sample)[0, 1]

print("\n" + "=" * 70)
print("EXEMPLE DE PRÉDICTION")
print("=" * 70)

print(
    "Classe réelle :",
    "Malin" if real_class == 1 else "Bénin"
)

print(
    "Classe prédite :",
    "Malin" if predicted_class == 1 else "Bénin"
)

print(
    f"Probabilité de malignité : "
    f"{predicted_probability * 100:.2f} %"
)


# ============================================================
# 15. EXPORT DES RÉSULTATS
# ============================================================

results_df.to_csv(
    "comparaison_modeles.csv",
    index=False,
    encoding="utf-8-sig"
)

feature_importances.to_csv(
    "importance_variables_random_forest.csv",
    index=False,
    encoding="utf-8-sig"
)

print("\nLes fichiers suivants ont été générés :")
print("- comparaison_modeles.csv")
print("- importance_variables_random_forest.csv")

print("\nAnalyse terminée avec succès.")
