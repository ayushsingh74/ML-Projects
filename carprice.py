import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
import requests
import re
import base64
import plotly.express as px
from sklearn.model_selection import train_test_split, cross_val_predict
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.linear_model import LinearRegression, Ridge, Lasso, ElasticNet
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor, AdaBoostRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.svm import SVR
import xgboost as xgb
import warnings
warnings.filterwarnings("ignore")

st.set_page_config(page_title="Car Price App", layout="wide")

def set_bg_image(image_url):
    response = requests.get(image_url)
    encoded_string = base64.b64encode(response.content).decode()
    page_bg_img = f"""
    <style>
    .stApp {{
        background-image: url("data:image/jpg;base64,{encoded_string}");
        background-size: cover;
        background-position: center;
        background-repeat: no-repeat;
        background-attachment: fixed;
    }}
    </style>
    """
    st.markdown(page_bg_img, unsafe_allow_html=True)

# ✅ Set image (replace this with your actual path)
set_bg_image("https://raw.githubusercontent.com/ayushsingh74/ML-Projects/main/shrinivas_pawar-porschen.jpg")
  # ✅ Make sure this file exists in the same folder


page = st.sidebar.selectbox("Navigate", ["🏠 Home", "📊 EDA", "⚙️ Model Training", "🔮 Predict"])

# Regressors
regressors = {
    "Linear Regression": make_pipeline(StandardScaler(), LinearRegression()),
    "Ridge": make_pipeline(StandardScaler(), Ridge()),
    "Lasso": make_pipeline(StandardScaler(), Lasso()),
    "ElasticNet": make_pipeline(StandardScaler(), ElasticNet()),
    "Decision Tree": DecisionTreeRegressor(max_depth=5, random_state=42),
    "Random Forest": RandomForestRegressor(n_estimators=100, max_depth=6, random_state=42),
    "KNN": make_pipeline(StandardScaler(), KNeighborsRegressor(n_neighbors=5)),
    "SVR": make_pipeline(StandardScaler(), SVR(kernel='rbf', C=100, epsilon=10)),
    "GBoost": GradientBoostingRegressor(n_estimators=100, learning_rate=0.1, max_depth=3, random_state=42),
    "AdaBoost": AdaBoostRegressor(n_estimators=100, learning_rate=0.1, random_state=42),
    "XGBoost": xgb.XGBRegressor(verbosity=0)
}

if page == "🏠 Home":
    st.title("🚗 Car Price Prediction App")
    st.markdown("""
    Welcome to the **Car Price Prediction** App! 👋

    This multi-page Streamlit app allows you to:
    - 📊 Explore car sales data
    - ⚙️ Train regression models with hyperparameter tuning
    - 🔮 Predict Selling Price using a trained model

    Use the sidebar to navigate.

    ---
    **Developed by Ayush Singh**
    """)

elif page == "📊 EDA":
    st.title("📊 Car Dataset - EDA")
    file = st.file_uploader("Upload your Car CSV file", type=["csv"], key="eda")

    if file is not None:
        df = pd.read_csv(file)

        st.subheader("Dataset Preview")
        st.dataframe(df.head())

        st.write("Missing Values:")
        st.write(df.isnull().sum())

        st.write("Summary Statistics")
        st.write(df.describe())

        # Dropdown for advanced visualizations
        vis_choice = st.selectbox("Select Visualization", ["None", "Heatmap", "Distribution", "Pairplot", "Model Metrics Comparison"])

        if vis_choice == "Heatmap":
            st.subheader("Correlation Heatmap")
            numeric_df = df.select_dtypes(include=[np.number])
            fig = px.imshow(numeric_df.corr(), text_auto=True, aspect="auto", color_continuous_scale="RdBu_r")
            st.plotly_chart(fig, use_container_width=True)

        elif vis_choice == "Distribution":
            num_cols = df.select_dtypes(include=np.number).columns.tolist()
            column = st.selectbox("Choose column", num_cols)
            fig = px.histogram(df, x=column, marginal="box", title=f"Distribution of {column}")
            st.plotly_chart(fig, use_container_width=True)

        elif vis_choice == "Pairplot":
            numeric_df = df.select_dtypes(include=[np.number])
            if len(numeric_df.columns) > 5:
                st.warning("Pairplot is resource-heavy. Showing only top 5 numerical columns.")
                numeric_df = numeric_df.iloc[:, :5]
            st.subheader("Pairplot")
            with st.spinner("Generating pairplot..."):
                fig = sns.pairplot(numeric_df)
                st.pyplot(fig)

        elif vis_choice == "Model Metrics Comparison":
            st.subheader("Model Metrics Visualization")
            model_names = ["Linear Regression", "Ridge", "Lasso"]
            r2_scores = [0.8, 0.81, 0.79]
            mae_scores = [1.2, 1.1, 1.3]
            mse_scores = [2.5, 2.4, 2.6]
            rmse_scores = [1.58, 1.55, 1.61]

            metrics_df = pd.DataFrame({
                "Model": model_names,
                "R2": r2_scores,
                "MAE": mae_scores,
                "MSE": mse_scores,
                "RMSE": rmse_scores
            })

            metric_to_plot = st.selectbox("Select Metric", ["R2", "MAE", "MSE", "RMSE"])
            fig = px.bar(metrics_df, x="Model", y=metric_to_plot, title=f"{metric_to_plot} Comparison")
            st.plotly_chart(fig)

elif page == "⚙️ Model Training":
    st.title("⚙️ Model Training")
    file = st.file_uploader("Upload Cleaned CSV", type=["csv"], key="train")

    if file is not None:
        df = pd.read_csv(file)

        # Group rare Car Names
        top_names = df['Car_Name'].value_counts().nlargest(20).index
        df['Car_Name'] = df['Car_Name'].apply(lambda x: x if x in top_names else 'Other')

        X = df.drop(columns=['Selling_Price'])
        y = df['Selling_Price']
        X = pd.get_dummies(X, drop_first=True)

        # Remove outliers
        from scipy.stats import zscore
        z = np.abs(zscore(X.select_dtypes(include=[np.number])))
        df_no_outliers = df[(z < 3).all(axis=1)]
        X = df_no_outliers.drop(columns=['Selling_Price'])
        y = df_no_outliers['Selling_Price']
        X = pd.get_dummies(X, drop_first=True)

        model_rf = RandomForestRegressor()
        model_rf.fit(X, y)
        importances = pd.Series(model_rf.feature_importances_, index=X.columns)
        important_features = importances[importances > 0.01].index
        X = X[important_features]

        st.subheader("Select Model")
        selected_model_name = st.selectbox("Choose Model", list(regressors.keys()))

        if st.button("Train Model"):
            model = regressors[selected_model_name]
            y_pred = cross_val_predict(model, X, y, cv=5)
            mse = mean_squared_error(y, y_pred)
            mae = mean_absolute_error(y, y_pred)
            r2 = r2_score(y, y_pred)

            st.write(f"R2 Score: {r2:.4f}")
            st.write(f"Mean Squared Error: {mse:.4f}")
            st.write(f"Mean Absolute Error: {mae:.4f}")

            model.fit(X, y)
            joblib.dump((model, X.columns), f"model_{selected_model_name}.pkl")
            st.success(f"{selected_model_name} saved successfully!")

elif page == "🔮 Predict":
    st.title("🔮 Make Prediction")

    model_choice = st.selectbox("Select a Trained Model", list(regressors.keys()))

    try:
        model, features = joblib.load(f"model_{model_choice}.pkl")
    except:
        st.warning("Please train the selected model first in the ⚙️ Model Training tab.")
        st.stop()

    file = st.file_uploader("Upload CSV for Car Options", type=["csv"], key="predict")

    car_name_list = []
    df_pred = None

    if file is not None:
        df_pred = pd.read_csv(file)

        if 'Car_Name' in df_pred.columns:
            top_names = df_pred['Car_Name'].value_counts().nlargest(20).index.tolist()
            car_name_list = sorted(top_names)

    if car_name_list:
        car_name = st.selectbox("Select Car Name", car_name_list)
    else:
        st.warning("No car names available. Please upload a CSV with a 'Car_Name' column.")
        car_name = st.text_input("Enter Car Name", "Swift")

    st.subheader("Enter Car Details")

    year = st.slider("Year", 2000, 2023, 2015)
    present_price = st.number_input("Present Price (in lakhs)", 0.0, 50.0, 5.0)
    kms_driven = st.number_input("Kilometers Driven", 0, 200000, 10000)
    fuel_type = st.selectbox("Fuel Type", ['Petrol', 'Diesel', 'CNG'])
    seller_type = st.selectbox("Seller Type", ['Dealer', 'Individual'])
    transmission = st.selectbox("Transmission", ['Manual', 'Automatic'])
    owner = st.selectbox("Owner", [0, 1, 2, 3])

    input_df = pd.DataFrame({
        'Year': [year],
        'Present_Price': [present_price],
        'Kms_Driven': [kms_driven],
        'Owner': [owner],
        'Car_Name': [car_name],
        'Fuel_Type_Diesel': [1 if fuel_type == 'Diesel' else 0],
        'Fuel_Type_Petrol': [1 if fuel_type == 'Petrol' else 0],
        'Seller_Type_Individual': [1 if seller_type == 'Individual' else 0],
        'Transmission_Manual': [1 if transmission == 'Manual' else 0],
    })

    for col in features:
        if col not in input_df.columns:
            input_df[col] = 0
    input_df = input_df[features]

    if st.button("Predict Selling Price"):
        prediction = model.predict(input_df)[0]
        st.success(f"Predicted Selling Price: ₹ {prediction:.2f} Lakhs")


