import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
import os
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

RND = 42
sns.set(style="whitegrid")

st.set_page_config(page_title="Digits Classification with PCA", layout="wide")
st.title("🧮 Digits Classification (0-9) with PCA & ML Models")

# Load dataset
digits = load_digits()
X = pd.DataFrame(digits.data)
y = pd.Series(digits.target, name="target")

# Sidebar controls
st.sidebar.header("Controls")
show_images = st.sidebar.checkbox("Show sample images", True)
show_pca = st.sidebar.checkbox("Show PCA variance plot", True)

# Dataset info
st.subheader("Dataset Overview")
st.write("Shape:", X.shape)
st.write("Unique classes:", sorted(y.unique()))

# Class distribution
fig, ax = plt.subplots(figsize=(7,4))
sns.countplot(x=y, ax=ax)
ax.set_title("Class Distribution")
st.pyplot(fig)

# Sample images
if show_images:
    fig, axes = plt.subplots(2, 5, figsize=(10,4))
    for ax, img, label in zip(axes.ravel(), digits.images, digits.target):
        ax.imshow(img, cmap="gray")
        ax.axis("off")
        ax.set_title(label)
    st.pyplot(fig)

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=RND
)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# PCA analysis
pca_full = PCA().fit(X_train_scaled)
explained_cumsum = np.cumsum(pca_full.explained_variance_ratio_)

if show_pca:
    fig, ax = plt.subplots(figsize=(8,5))
    ax.plot(range(1, len(explained_cumsum)+1), explained_cumsum, marker="o")
    ax.axhline(0.90, color="gray", linestyle="--")
    ax.axhline(0.95, color="gray", linestyle="--")
    ax.set_xlabel("Number of components")
    ax.set_ylabel("Cumulative explained variance")
    ax.set_title("PCA cumulative explained variance")
    st.pyplot(fig)

# Quick PCA(0.95)
pca_95 = PCA(n_components=0.95, svd_solver="full")
X_train_pca95 = pca_95.fit_transform(X_train_scaled)
X_test_pca95 = pca_95.transform(X_test_scaled)

st.sidebar.subheader("Select model")
model_choice = st.sidebar.selectbox("Classifier", ["LogisticRegression", "SVC", "RandomForest", "KNN"])

models = {
    'LogisticRegression': LogisticRegression(max_iter=2000, random_state=RND),
    'SVC': SVC(random_state=RND),
    'RandomForest': RandomForestClassifier(random_state=RND),
    'KNN': KNeighborsClassifier()
}

# --- keep trained model in session state ---
if "trained_model" not in st.session_state:
    st.session_state.trained_model = None

# Training block
if st.sidebar.button("Train Model"):
    model = models[model_choice]
    model.fit(X_train_pca95, y_train)
    preds = model.predict(X_test_pca95)
    acc = accuracy_score(y_test, preds)

    # save trained model in session
    st.session_state.trained_model = model  

    st.subheader(f"Results for {model_choice}")
    st.write(f"Accuracy: **{acc:.4f}**")

    st.text("Classification Report:")
    st.text(classification_report(y_test, preds))

    cm = confusion_matrix(y_test, preds)
    fig, ax = plt.subplots(figsize=(8,6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    ax.set_title("Confusion Matrix")
    st.pyplot(fig)

# Save model (only if trained)
if st.sidebar.button("Save Model"):
    if st.session_state.trained_model is not None:
        os.makedirs("exported_model", exist_ok=True)
        joblib.dump(st.session_state.trained_model, "exported_model/best_model.pkl")
        joblib.dump(pca_95, "exported_model/pca.pkl")
        joblib.dump(scaler, "exported_model/scaler.pkl")
        st.success("✅ Model & artifacts saved to exported_model/")
    else:
        st.warning("⚠️ Please train a model first before saving.")