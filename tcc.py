import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import joblib
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score, confusion_matrix
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
import xgboost as xgb
from sklearn.decomposition import PCA
import warnings
warnings.filterwarnings("ignore")

# Use this script as an integrated version if multipage app isn't used
# Otherwise, split this into eda_dashboard.py, model_training.py, prediction_app.py, and Home.py

st.set_page_config(page_title="Customer Churn App", layout="wide")
page = st.sidebar.selectbox("Navigate", ["🏠 Home", "📊 EDA", "⚙️ Model Training", "🔮 Predict"])

if page == "🏠 Home":
    st.title("📦 Customer Churn Prediction App")
    st.markdown("""
    Welcome to the **Customer Churn Prediction** App! 👋

    This multi-page Streamlit application allows you to:
    - 📊 Explore your data (EDA)
    - ⚙️ Train classification models with hyperparameter tuning
    - 🔮 Predict churn from new data with any trained model

    Use the sidebar to navigate between different modules.

    ---
    **Developed by Ayush Singh**
    """)

elif page == "📊 EDA":
    st.title("📊 Customer Churn EDA Dashboard")
    uploaded_file = st.file_uploader("Upload your Churn CSV file", type=["csv"], key="eda")
    if uploaded_file is not None:
        df = pd.read_csv(uploaded_file)
        df['TotalCharges'] = pd.to_numeric(df['TotalCharges'], errors='coerce')
        df['TotalCharges'].fillna(df['TotalCharges'].median(), inplace=True)
        if 'customerID' in df.columns:
            df.drop('customerID', axis=1, inplace=True)

        st.subheader("📌 Dataset Overview")
        st.write(df.head())

        with st.expander("Show data summary"):
            st.write(df.describe(include='all'))
            st.write("Missing Values:")
            st.write(df.isnull().sum())

        chart_type = st.selectbox("Select Chart Type", [
            "Churn Distribution", "Numerical Feature Distribution", "Boxplot by Churn",
            "Categorical Countplot", "Correlation Heatmap"])

        if chart_type == "Churn Distribution":
            fig, ax = plt.subplots()
            sns.countplot(data=df, x='Churn', palette='Set2', ax=ax)
            st.pyplot(fig)

        elif chart_type == "Numerical Feature Distribution":
            num_cols = ['tenure', 'MonthlyCharges', 'TotalCharges']
            num_feature = st.selectbox("Select numerical feature", num_cols)
            fig, ax = plt.subplots()
            sns.histplot(data=df, x=num_feature, kde=True, hue='Churn', palette='Set2', multiple='stack', ax=ax)
            st.pyplot(fig)

        elif chart_type == "Boxplot by Churn":
            num_cols = ['tenure', 'MonthlyCharges', 'TotalCharges']
            num_feature = st.selectbox("Select numerical feature for boxplot", num_cols)
            fig, ax = plt.subplots()
            sns.boxplot(data=df, x='Churn', y=num_feature, palette='Set1', ax=ax)
            st.pyplot(fig)

        elif chart_type == "Categorical Countplot":
            cat_cols = [
                'gender', 'SeniorCitizen', 'Partner', 'Dependents', 'PhoneService', 'MultipleLines',
                'InternetService', 'OnlineSecurity', 'OnlineBackup', 'DeviceProtection', 'TechSupport',
                'StreamingTV', 'StreamingMovies', 'Contract', 'PaperlessBilling', 'PaymentMethod']
            cat_feature = st.selectbox("Select categorical feature", cat_cols)
            fig, ax = plt.subplots()
            sns.countplot(data=df, x=cat_feature, hue='Churn', palette='Set2', ax=ax)
            plt.xticks(rotation=45)
            st.pyplot(fig)

        elif chart_type == "Correlation Heatmap":
            corr_cols = ['tenure', 'MonthlyCharges', 'TotalCharges', 'SeniorCitizen']
            fig, ax = plt.subplots(figsize=(8, 5))
            sns.heatmap(df[corr_cols].corr(), annot=True, cmap='coolwarm', ax=ax)
            st.pyplot(fig)

elif page == "⚙️ Model Training":
    st.title("🤖 Customer Churn - Model Training & Evaluation")
    uploaded_file = st.file_uploader("Upload preprocessed churn CSV", type=["csv"], key="model")
    if uploaded_file is not None:
        df = pd.read_csv(uploaded_file)
        if 'customerID' in df.columns:
            df.drop('customerID', axis=1, inplace=True)
        df['TotalCharges'] = pd.to_numeric(df['TotalCharges'], errors='coerce')
        df.fillna(0, inplace=True)
        if 'Churn' not in df.columns:
            st.error("Target column 'Churn' not found.")
            st.stop()

        X = df.drop('Churn', axis=1)
        y = df['Churn'].map({'Yes': 1, 'No': 0}) if df['Churn'].dtype == object else df['Churn']
        X = pd.get_dummies(X, drop_first=True)
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        model_choice = st.selectbox("Select Classification Algorithm", [
            "Logistic Regression", "KNN", "Decision Tree", "Random Forest", "SVM", "Naive Bayes", "XGBoost"])

        scale_models = ["Logistic Regression", "KNN", "SVM"]
        model_params = {
            "Logistic Regression": (LogisticRegression(max_iter=1000), {
                'model__C': [0.01, 0.1, 1, 10], 'model__solver': ['liblinear'] }),
            "KNN": (KNeighborsClassifier(), {'model__n_neighbors': [3, 5, 7], 'model__weights': ['uniform', 'distance']}),
            "Decision Tree": (DecisionTreeClassifier(), {'model__max_depth': [3, 5, 10, None], 'model__criterion': ['gini', 'entropy']}),
            "Random Forest": (RandomForestClassifier(), {'model__n_estimators': [50, 100], 'model__max_depth': [None, 5, 10]}),
            "SVM": (SVC(probability=True), {'model__C': [0.1, 1, 10], 'model__kernel': ['linear', 'rbf']}),
            "Naive Bayes": (GaussianNB(), {}),
            "XGBoost": (xgb.XGBClassifier(use_label_encoder=False, eval_metric='logloss'), {
                'model__n_estimators': [100], 'model__max_depth': [3, 6], 'model__learning_rate': [0.01, 0.1]})
        }

        clf, params = model_params[model_choice]
        pipe = Pipeline([('scaler', StandardScaler()), ('model', clf)]) if model_choice in scale_models else Pipeline([('model', clf)])
        with st.spinner("Training model with hyperparameter tuning..."):
            search = GridSearchCV(pipe, params, cv=3, scoring='accuracy', n_jobs=-1) if params else None
            best_model = search.fit(X_train, y_train).best_estimator_ if search else pipe.fit(X_train, y_train)
            y_pred = best_model.predict(X_test)
            y_proba = best_model.predict_proba(X_test)[:, 1] if hasattr(best_model, "predict_proba") else y_pred

            st.success("Training complete!")
            st.write(f"**Accuracy:** {accuracy_score(y_test, y_pred):.4f}")
            st.write(f"**F1 Score:** {f1_score(y_test, y_pred):.4f}")
            st.write(f"**ROC AUC Score:** {roc_auc_score(y_test, y_proba):.4f}")
            st.write(f"**Best Parameters:** {search.best_params_ if search else 'Default'}")

            fig, ax = plt.subplots()
            sns.heatmap(confusion_matrix(y_test, y_pred), annot=True, fmt='d', cmap='Blues', ax=ax)
            ax.set_title('Confusion Matrix')
            ax.set_xlabel('Predicted')
            ax.set_ylabel('Actual')
            st.pyplot(fig)

elif page == "🔮 Predict":
    st.title("🔮 Customer Churn - Prediction App")
    uploaded_file = st.file_uploader("Upload CSV File for Prediction", type=["csv"], key="predict")
    model_choice = st.selectbox("Select Model", ["Logistic Regression", "KNN", "Decision Tree", "Random Forest", "SVM", "Naive Bayes", "XGBoost"])
    
    apply_scaling = st.checkbox("Apply Standard Scaling", value=True)
    apply_pca = st.checkbox("Apply PCA (for high-dimensional data)")

    if uploaded_file is not None:
        df_pred = pd.read_csv(uploaded_file)

        # Preprocessing
        if 'customerID' in df_pred.columns:
            df_pred.drop('customerID', axis=1, inplace=True)
        if 'Churn' in df_pred.columns:
            df_pred.drop('Churn', axis=1, inplace=True)

        df_pred['TotalCharges'] = pd.to_numeric(df_pred['TotalCharges'], errors='coerce')
        df_pred.fillna(0, inplace=True)

        bin_map = {'Yes': 1, 'No': 0, 'Male': 1, 'Female': 0}
        for col in df_pred.columns:
            if df_pred[col].nunique() == 2 and df_pred[col].dtype == object:
                df_pred[col] = df_pred[col].map(bin_map)
        
        df_pred = pd.get_dummies(df_pred, drop_first=True)

        # Feature alignment (simulate trained model structure)
        model_features = df_pred.columns.tolist()  # Normally you'd load with joblib
        df_pred = df_pred.reindex(columns=model_features, fill_value=0)

        # Apply Scaling and PCA
        if apply_scaling:
            scaler = StandardScaler()
            df_pred = scaler.fit_transform(df_pred)

        if apply_pca:
            pca = PCA(n_components=min(10, df_pred.shape[1]))
            df_pred = pca.fit_transform(df_pred)

        st.info("⚠️ Using simulated churn probabilities. Replace with real model.predict_proba in production.")
        
        # 👉 Set decision threshold
        threshold = st.slider("Set churn threshold", 0.0, 1.0, 0.5, 0.01)

        # 🔄 Simulated probabilities (replace with model.predict_proba(X)[:,1] for real)
        np.random.seed(42)
        probs = np.random.rand(df_pred.shape[0])

        # ✅ Predict churn based on threshold
        preds = (probs >= threshold).astype(int)

        # 💡 Confidence = probability of predicted class
        confidence = np.where(preds == 1, probs * 100, (1 - probs) * 100)
        status = np.where(preds == 1, "Will Churn", "Will Not Churn")

        # 📋 Results
        result_df = pd.DataFrame({
            "Churn Probability": probs.round(4),
            "Predicted Label": preds,
            "Status": status,
            "Confidence (%)": confidence.round(2)
        })

        st.subheader("📊 Prediction Results")
        st.write(result_df.head())

        # 📥 Download
        csv = result_df.to_csv(index=False).encode()
        st.download_button("📥 Download Predictions", csv, "predictions.csv", "text/csv")

