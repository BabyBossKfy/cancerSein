# ============================================================
# APPLICATION STREAMLIT
# CLASSIFICATION DU CANCER DU SEIN
# Machine Learning : Logistic Regression, Random Forest, KNN, SVM
# Développé par Franck Kianguebeni
# ============================================================

from __future__ import annotations

import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import streamlit as st
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
    roc_curve,
)


# ============================================================
# CONFIGURATION STREAMLIT
# ============================================================

st.set_page_config(
    page_title="Classification Cancer du Sein",
    page_icon="🧬",
    layout="wide",
)

APP_NAME = "Classification du Cancer du Sein"
SIGNATURE = "Développé par Franck Kianguebeni - Data Analyst / BI Engineer"


# ============================================================
# STYLE
# ============================================================

st.markdown(
    """
<style>
.block-container {
    padding-top: 1rem;
    padding-bottom: 2rem;
    max-width: 1500px;
}

.hero {
    padding: 1.5rem 1.8rem;
    border-radius: 22px;
    background: linear-gradient(120deg, #4b0033, #8a005d 60%, #e75480);
    color: white;
    margin-bottom: 1.2rem;
    box-shadow: 0 14px 34px rgba(75,0,51,.22);
}

.hero h1 {
    margin: 0;
    font-size: 2.2rem;
}

.hero p {
    margin: .35rem 0 0;
    opacity: .93;
}

.kpi-card {
    padding: 1rem;
    border: 1px solid #ead5df;
    border-radius: 18px;
    background: white;
    min-height: 115px;
    box-shadow: 0 6px 18px rgba(138,0,93,.06);
}

.kpi-label {
    font-size: .85rem;
    color: #64748b;
}

.kpi-value {
    font-size: 1.65rem;
    font-weight: 780;
    color: #4b0033;
    margin: .25rem 0;
}

.kpi-note {
    font-size: .78rem;
    color: #64748b;
}

.section-note {
    padding: .85rem 1rem;
    border-radius: 13px;
    background: #fff0f6;
    color: #4b0033;
    border-left: 4px solid #8a005d;
    margin-bottom: 1rem;
}

.footer {
    text-align: center;
    color: #64748b;
    font-size: .85rem;
    margin-top: 2rem;
    padding-top: 1rem;
    border-top: 1px solid #e5e7eb;
}

[data-testid="stMetric"] {
    background: white;
    border: 1px solid #ead5df;
    padding: .85rem;
    border-radius: 14px;
}
</style>
""",
    unsafe_allow_html=True,
)


# ============================================================
# FONCTIONS D'AFFICHAGE
# ============================================================

def render_hero() -> None:
    st.markdown(
        """
<div class="hero">
    <h1>🧬 Classification du Cancer du Sein</h1>
    <p>Analyse exploratoire, comparaison de modèles Machine Learning et prédiction du diagnostic</p>
</div>
""",
        unsafe_allow_html=True,
    )


def render_kpi(label: str, value: str, note: str = "") -> None:
    st.markdown(
        f"""
<div class="kpi-card">
    <div class="kpi-label">{label}</div>
    <div class="kpi-value">{value}</div>
    <div class="kpi-note">{note}</div>
</div>
""",
        unsafe_allow_html=True,
    )


def format_number(value) -> str:
    return f"{value:,.0f}".replace(",", " ")


# ============================================================
# CHARGEMENT ET NETTOYAGE DES DONNÉES
# ============================================================

@st.cache_data
def load_default_data() -> pd.DataFrame:
    file_path = "Copie de data.csv"
    return pd.read_csv(file_path)


def clean_dataset(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    if "Unnamed: 32" in df.columns:
        df = df.drop(columns=["Unnamed: 32"])

    if "id" in df.columns:
        df = df.drop(columns=["id"])

    df = df.drop_duplicates().reset_index(drop=True)

    if "diagnosis" not in df.columns:
        raise ValueError("La colonne 'diagnosis' est introuvable dans le dataset.")

    df["diagnosis"] = df["diagnosis"].map(
        {
            "B": 0,
            "M": 1,
            "0": 0,
            "1": 1,
            0: 0,
            1: 1,
            0.0: 0,
            1.0: 1,
        }
    )

    if df["diagnosis"].isnull().any():
        raise ValueError(
            "La colonne diagnosis doit contenir uniquement B, M, 0 ou 1."
        )

    df["diagnosis"] = df["diagnosis"].astype(int)

    for col in df.columns:
        if col != "diagnosis":
            df[col] = pd.to_numeric(df[col], errors="coerce")

    numeric_cols = [col for col in df.columns if col != "diagnosis"]

    for col in numeric_cols:
        if df[col].isnull().any():
            df[col] = df[col].fillna(df[col].median())

    return df


# ============================================================
# MACHINE LEARNING
# ============================================================

def get_models() -> dict:
    return {
        "Logistic Regression": Pipeline(
            [
                ("scaler", StandardScaler()),
                (
                    "classifier",
                    LogisticRegression(
                        max_iter=2000,
                        random_state=42,
                    ),
                ),
            ]
        ),

        "Random Forest": RandomForestClassifier(
            n_estimators=300,
            random_state=42,
            class_weight="balanced",
        ),

        "KNN": Pipeline(
            [
                ("scaler", StandardScaler()),
                (
                    "classifier",
                    KNeighborsClassifier(
                        n_neighbors=5,
                    ),
                ),
            ]
        ),

        "SVM": Pipeline(
            [
                ("scaler", StandardScaler()),
                (
                    "classifier",
                    SVC(
                        kernel="rbf",
                        probability=True,
                        random_state=42,
                    ),
                ),
            ]
        ),
    }


@st.cache_resource
def train_all_models(df: pd.DataFrame) -> dict:
    X = df.drop(columns=["diagnosis"])
    y = df["diagnosis"]

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.20,
        random_state=42,
        stratify=y,
    )

    models = get_models()

    results = []
    roc_results = {}
    trained_models = {}
    confusion_matrices = {}
    reports = {}

    for model_name, model in models.items():
        model.fit(X_train, y_train)

        trained_models[model_name] = model

        y_pred = model.predict(X_test)
        y_proba = model.predict_proba(X_test)[:, 1]

        accuracy = accuracy_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred, zero_division=0)
        recall = recall_score(y_test, y_pred, zero_division=0)
        f1 = f1_score(y_test, y_pred, zero_division=0)
        roc_auc = roc_auc_score(y_test, y_proba)

        results.append(
            {
                "Modèle": model_name,
                "Accuracy": accuracy,
                "Précision": precision,
                "Rappel": recall,
                "F1-score": f1,
                "ROC-AUC": roc_auc,
            }
        )

        cm = confusion_matrix(y_test, y_pred)
        confusion_matrices[model_name] = cm

        report = classification_report(
            y_test,
            y_pred,
            target_names=["Bénin", "Malin"],
            output_dict=True,
            zero_division=0,
        )

        reports[model_name] = pd.DataFrame(report).transpose()

        fpr, tpr, thresholds = roc_curve(y_test, y_proba)

        roc_results[model_name] = {
            "fpr": fpr,
            "tpr": tpr,
            "thresholds": thresholds,
            "auc": roc_auc,
        }

    results_df = pd.DataFrame(results).sort_values(
        by="ROC-AUC",
        ascending=False,
    ).reset_index(drop=True)

    best_model_name = results_df.iloc[0]["Modèle"]
    best_model = trained_models[best_model_name]

    return {
        "X": X,
        "y": y,
        "X_train": X_train,
        "X_test": X_test,
        "y_train": y_train,
        "y_test": y_test,
        "trained_models": trained_models,
        "results_df": results_df,
        "roc_results": roc_results,
        "confusion_matrices": confusion_matrices,
        "reports": reports,
        "best_model_name": best_model_name,
        "best_model": best_model,
    }


def get_feature_importance(model, feature_names) -> pd.DataFrame:
    if hasattr(model, "feature_importances_"):
        importance_df = pd.DataFrame(
            {
                "Variable": feature_names,
                "Importance": model.feature_importances_,
            }
        ).sort_values(
            by="Importance",
            ascending=False,
        )

        return importance_df

    return pd.DataFrame(columns=["Variable", "Importance"])


# ============================================================
# SIDEBAR
# ============================================================

st.sidebar.title("Navigation")
st.sidebar.caption("Cancer du sein - Machine Learning")

uploaded_file = st.sidebar.file_uploader(
    "Importer un fichier CSV",
    type=["csv"],
)

page = st.sidebar.radio(
    "Menu",
    [
        "Résumé exécutif",
        "Données",
        "Analyse exploratoire",
        "Comparaison des modèles",
        "Courbes ROC",
        "Importance des variables",
        "Prédiction individuelle",
        "Exporter les résultats",
    ],
)


# ============================================================
# CHARGEMENT GLOBAL
# ============================================================

render_hero()

st.warning(
    "Cette application est un outil pédagogique et analytique. "
    "Elle ne doit pas être utilisée comme diagnostic médical réel. "
    "Toute décision médicale doit être validée par un professionnel de santé."
)

try:
    with st.spinner("Chargement des données et entraînement des modèles..."):
        if uploaded_file is not None:
            raw_df = pd.read_csv(uploaded_file)
            data_source = "Fichier importé"
        else:
            raw_df = load_default_data()
            data_source = "Fichier local : Copie de data.csv"

        df = clean_dataset(raw_df)

        ml = train_all_models(df)

        X = ml["X"]
        y = ml["y"]
        X_train = ml["X_train"]
        X_test = ml["X_test"]
        y_train = ml["y_train"]
        y_test = ml["y_test"]

        trained_models = ml["trained_models"]
        results_df = ml["results_df"]
        roc_results = ml["roc_results"]
        confusion_matrices = ml["confusion_matrices"]
        reports = ml["reports"]
        best_model_name = ml["best_model_name"]
        best_model = ml["best_model"]

except Exception as exc:
    st.error("Erreur lors du chargement ou de l'entraînement du modèle.")
    st.exception(exc)
    st.stop()


# ============================================================
# PAGE 1 : RÉSUMÉ EXÉCUTIF
# ============================================================

if page == "Résumé exécutif":
    st.subheader("Résumé exécutif")

    total = len(df)
    benign_count = int((df["diagnosis"] == 0).sum())
    malignant_count = int((df["diagnosis"] == 1).sum())

    benign_pct = benign_count / total * 100
    malignant_pct = malignant_count / total * 100

    c1, c2, c3, c4 = st.columns(4)

    with c1:
        render_kpi("Observations", format_number(total), "Nombre total de lignes")

    with c2:
        render_kpi(
            "Variables médicales",
            format_number(X.shape[1]),
            "Variables explicatives",
        )

    with c3:
        render_kpi("Bénins", f"{benign_count}", f"{benign_pct:.2f} %")

    with c4:
        render_kpi("Malins", f"{malignant_count}", f"{malignant_pct:.2f} %")

    st.markdown(
        f"""
<div class="section-note">
Le meilleur modèle selon le ROC-AUC est : <b>{best_model_name}</b>.
</div>
""",
        unsafe_allow_html=True,
    )

    col1, col2 = st.columns(2)

    with col1:
        fig, ax = plt.subplots(figsize=(6, 4))
        sns.countplot(data=df, x="diagnosis", ax=ax)
        ax.set_title("Répartition des diagnostics")
        ax.set_xlabel("Diagnostic")
        ax.set_ylabel("Nombre")
        ax.set_xticklabels(["Bénin", "Malin"])
        st.pyplot(fig)

    with col2:
        fig, ax = plt.subplots(figsize=(6, 4))
        values = df["diagnosis"].value_counts().sort_index()
        ax.pie(
            values,
            labels=["Bénin", "Malin"],
            autopct="%1.1f%%",
            startangle=90,
        )
        ax.set_title("Proportion des diagnostics")
        st.pyplot(fig)

    st.subheader("Classement des modèles")

    st.dataframe(
        results_df.style.format(
            {
                "Accuracy": "{:.4f}",
                "Précision": "{:.4f}",
                "Rappel": "{:.4f}",
                "F1-score": "{:.4f}",
                "ROC-AUC": "{:.4f}",
            }
        ).highlight_max(axis=0),
        use_container_width=True,
        hide_index=True,
    )


# ============================================================
# PAGE 2 : DONNÉES
# ============================================================

elif page == "Données":
    st.subheader("Données du projet")

    st.info(f"Source des données : {data_source}")

    c1, c2, c3 = st.columns(3)

    c1.metric("Lignes", df.shape[0])
    c2.metric("Colonnes", df.shape[1])
    c3.metric("Doublons", int(df.duplicated().sum()))

    st.subheader("Aperçu du dataset")
    st.dataframe(df.head(500), use_container_width=True)

    st.subheader("Structure des colonnes")

    structure_df = pd.DataFrame(
        {
            "Colonne": df.columns,
            "Type": [str(df[col].dtype) for col in df.columns],
            "Valeurs manquantes": [int(df[col].isna().sum()) for col in df.columns],
            "Valeurs uniques": [int(df[col].nunique()) for col in df.columns],
        }
    )

    st.dataframe(
        structure_df,
        use_container_width=True,
        hide_index=True,
    )

    st.subheader("Statistiques descriptives")
    st.dataframe(df.describe().T, use_container_width=True)


# ============================================================
# PAGE 3 : ANALYSE EXPLORATOIRE
# ============================================================

elif page == "Analyse exploratoire":
    st.subheader("Analyse exploratoire des données")

    numeric_cols = X.columns.tolist()

    selected_feature = st.selectbox(
        "Choisir une variable à analyser",
        numeric_cols,
    )

    col1, col2 = st.columns(2)

    with col1:
        fig, ax = plt.subplots(figsize=(7, 5))
        sns.histplot(df[selected_feature], kde=True, ax=ax)
        ax.set_title(f"Distribution de {selected_feature}")
        st.pyplot(fig)

    with col2:
        fig, ax = plt.subplots(figsize=(7, 5))
        sns.boxplot(
            data=df,
            x="diagnosis",
            y=selected_feature,
            ax=ax,
        )
        ax.set_title(f"{selected_feature} selon le diagnostic")
        ax.set_xlabel("Diagnostic")
        ax.set_xticklabels(["Bénin", "Malin"])
        st.pyplot(fig)

    st.subheader("Matrice de corrélation")

    corr = df.corr(numeric_only=True)

    fig, ax = plt.subplots(figsize=(18, 12))
    sns.heatmap(
        corr,
        cmap="coolwarm",
        center=0,
        ax=ax,
    )
    ax.set_title("Matrice de corrélation")
    st.pyplot(fig)

    st.subheader("Top 10 des variables les plus corrélées au diagnostic")

    target_corr = (
        corr["diagnosis"]
        .drop("diagnosis")
        .abs()
        .sort_values(ascending=False)
    )

    top_corr = target_corr.head(10).reset_index()
    top_corr.columns = ["Variable", "Corrélation absolue"]

    st.dataframe(
        top_corr,
        use_container_width=True,
        hide_index=True,
    )

    fig, ax = plt.subplots(figsize=(9, 6))
    sns.barplot(
        data=top_corr.sort_values("Corrélation absolue"),
        x="Corrélation absolue",
        y="Variable",
        ax=ax,
    )
    ax.set_title("Top 10 des variables corrélées avec le diagnostic")
    st.pyplot(fig)


# ============================================================
# PAGE 4 : COMPARAISON DES MODÈLES
# ============================================================

elif page == "Comparaison des modèles":
    st.subheader("Comparaison des modèles Machine Learning")

    st.markdown(
        """
<div class="section-note">
Les modèles comparés sont : Logistic Regression, Random Forest, KNN et SVM.
Le meilleur modèle est sélectionné automatiquement selon le ROC-AUC.
</div>
""",
        unsafe_allow_html=True,
    )

    st.dataframe(
        results_df.style.format(
            {
                "Accuracy": "{:.4f}",
                "Précision": "{:.4f}",
                "Rappel": "{:.4f}",
                "F1-score": "{:.4f}",
                "ROC-AUC": "{:.4f}",
            }
        ).highlight_max(axis=0),
        use_container_width=True,
        hide_index=True,
    )

    model_names = list(trained_models.keys())

    selected_model = st.selectbox(
        "Choisir un modèle à inspecter",
        model_names,
        index=model_names.index(best_model_name),
    )

    st.subheader(f"Matrice de confusion — {selected_model}")

    cm = confusion_matrices[selected_model]

    fig, ax = plt.subplots(figsize=(6, 5))

    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cbar=False,
        xticklabels=["Bénin", "Malin"],
        yticklabels=["Bénin", "Malin"],
        ax=ax,
    )

    ax.set_xlabel("Classe prédite")
    ax.set_ylabel("Classe réelle")
    ax.set_title(f"Matrice de confusion — {selected_model}")

    st.pyplot(fig)

    st.subheader("Rapport de classification")

    st.dataframe(
        reports[selected_model],
        use_container_width=True,
    )

    st.subheader("Comparaison graphique des métriques")

    metrics_to_plot = [
        "Accuracy",
        "Précision",
        "Rappel",
        "F1-score",
        "ROC-AUC",
    ]

    results_long = results_df.melt(
        id_vars="Modèle",
        value_vars=metrics_to_plot,
        var_name="Métrique",
        value_name="Score",
    )

    fig, ax = plt.subplots(figsize=(12, 6))

    sns.barplot(
        data=results_long,
        x="Modèle",
        y="Score",
        hue="Métrique",
        ax=ax,
    )

    ax.set_title("Comparaison des performances des modèles")
    ax.set_ylim(0, 1.05)
    ax.tick_params(axis="x", rotation=15)

    ax.legend(
        title="Métrique",
        bbox_to_anchor=(1.02, 1),
        loc="upper left",
    )

    st.pyplot(fig)


# ============================================================
# PAGE 5 : COURBES ROC
# ============================================================

elif page == "Courbes ROC":
    st.subheader("Courbes ROC des modèles")

    st.markdown(
        """
<div class="section-note">
La courbe ROC permet de comparer la capacité des modèles à distinguer les diagnostics bénins et malins.
Plus l'AUC est proche de 1, meilleure est la performance du modèle.
</div>
""",
        unsafe_allow_html=True,
    )

    fig, ax = plt.subplots(figsize=(9, 7))

    for model_name, roc_data in roc_results.items():
        ax.plot(
            roc_data["fpr"],
            roc_data["tpr"],
            linewidth=2,
            label=f"{model_name} - AUC = {roc_data['auc']:.3f}",
        )

    ax.plot(
        [0, 1],
        [0, 1],
        linestyle="--",
        color="gray",
        label="Modèle aléatoire",
    )

    ax.set_title("Comparaison des courbes ROC")
    ax.set_xlabel("Taux de faux positifs")
    ax.set_ylabel("Taux de vrais positifs")
    ax.legend()
    ax.grid(alpha=0.3)

    st.pyplot(fig)

    st.subheader("Valeurs ROC-AUC par modèle")

    auc_df = pd.DataFrame(
        [
            {
                "Modèle": model_name,
                "ROC-AUC": roc_data["auc"],
            }
            for model_name, roc_data in roc_results.items()
        ]
    ).sort_values(
        by="ROC-AUC",
        ascending=False,
    )

    st.dataframe(
        auc_df.style.format(
            {
                "ROC-AUC": "{:.4f}",
            }
        ),
        use_container_width=True,
        hide_index=True,
    )


# ============================================================
# PAGE 6 : IMPORTANCE DES VARIABLES
# ============================================================

elif page == "Importance des variables":
    st.subheader("Importance des variables — Random Forest")

    rf_model = trained_models["Random Forest"]

    importance_df = get_feature_importance(
        rf_model,
        X.columns,
    )

    if importance_df.empty:
        st.info("L'importance des variables n'est pas disponible pour ce modèle.")
    else:
        top_n = st.slider(
            "Nombre de variables à afficher",
            min_value=5,
            max_value=min(30, len(importance_df)),
            value=15,
        )

        st.dataframe(
            importance_df.head(top_n),
            use_container_width=True,
            hide_index=True,
        )

        fig, ax = plt.subplots(figsize=(10, 7))
        sns.barplot(
            data=importance_df.head(top_n).sort_values("Importance"),
            x="Importance",
            y="Variable",
            ax=ax,
        )
        ax.set_title(
            f"Top {top_n} des variables les plus importantes — Random Forest"
        )
        st.pyplot(fig)


# ============================================================
# PAGE 7 : PRÉDICTION INDIVIDUELLE
# ============================================================

elif page == "Prédiction individuelle":
    st.subheader("Prédiction individuelle")

    st.markdown(
        """
<div class="section-note">
Cette section permet de saisir les caractéristiques médicales d'une observation
et d'obtenir une prédiction : Bénin ou Malin.
</div>
""",
        unsafe_allow_html=True,
    )

    st.info(f"Modèle utilisé : {best_model_name}")

    manual_values = {}

    with st.form("prediction_form"):
        cols = st.columns(3)

        for idx, col in enumerate(X.columns):
            with cols[idx % 3]:
                default_value = float(X[col].median())

                manual_values[col] = st.number_input(
                    col,
                    value=default_value,
                    step=0.01,
                    format="%.4f",
                    key=f"input_{col}",
                )

        submit = st.form_submit_button(
            "Prédire le diagnostic",
            use_container_width=True,
        )

    if submit:
        input_df = pd.DataFrame([manual_values])

        prediction = best_model.predict(input_df)[0]
        probability_malignant = best_model.predict_proba(input_df)[0, 1]

        label = "Malin" if prediction == 1 else "Bénin"

        c1, c2, c3 = st.columns(3)

        c1.metric("Diagnostic prédit", label)
        c2.metric(
            "Probabilité de malignité",
            f"{probability_malignant * 100:.2f} %",
        )
        c3.metric("Modèle", best_model_name)

        if prediction == 1:
            st.error(
                "Résultat du modèle : la tumeur est classée comme potentiellement maligne. "
                "Ce résultat doit impérativement être validé par un professionnel de santé."
            )
        else:
            st.success(
                "Résultat du modèle : la tumeur est classée comme potentiellement bénigne. "
                "Ce résultat ne remplace pas une validation médicale."
            )

        st.subheader("Données saisies")

        st.dataframe(
            input_df,
            use_container_width=True,
            hide_index=True,
        )


# ============================================================
# PAGE 8 : EXPORT
# ============================================================

elif page == "Exporter les résultats":
    st.subheader("Exporter les résultats")

    st.write("Téléchargement du tableau comparatif des modèles.")

    csv_results = results_df.to_csv(
        index=False,
        encoding="utf-8-sig",
    ).encode("utf-8-sig")

    st.download_button(
        label="Télécharger comparaison_modeles.csv",
        data=csv_results,
        file_name="comparaison_modeles.csv",
        mime="text/csv",
        use_container_width=True,
    )

    rf_model = trained_models["Random Forest"]

    importance_df = get_feature_importance(
        rf_model,
        X.columns,
    )

    if not importance_df.empty:
        csv_importance = importance_df.to_csv(
            index=False,
            encoding="utf-8-sig",
        ).encode("utf-8-sig")

        st.download_button(
            label="Télécharger importance_variables_random_forest.csv",
            data=csv_importance,
            file_name="importance_variables_random_forest.csv",
            mime="text/csv",
            use_container_width=True,
        )

    st.success("Les fichiers sont prêts au téléchargement.")


# ============================================================
# FOOTER
# ============================================================

st.markdown(
    f"""
<div class="footer">
    {SIGNATURE}<br>
    Application Streamlit de classification du cancer du sein par Machine Learning.
</div>
""",
    unsafe_allow_html=True,
)