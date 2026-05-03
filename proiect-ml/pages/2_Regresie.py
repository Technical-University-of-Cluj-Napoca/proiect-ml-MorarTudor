import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import shap
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_squared_error
from sklearn.linear_model import LinearRegression
from catboost import CatBoostRegressor
from xgboost import XGBRegressor
from sklearn.svm import SVR
from interpret.glassbox import ExplainableBoostingRegressor

st.set_page_config(page_title="Regresie", layout="wide")
st.title("Regresie: Predicția Prețului la Motorină")

@st.cache_data
def load_data():
    return pd.read_csv('data/regresie.csv')

df = load_data()
X = df.drop('Pret_Motorina', axis=1)
y = df['Pret_Motorina']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)

with st.expander("Vezi Analiza Exploratorie a Datelor (EDA)"):
    st.dataframe(df.head())
    fig, ax = plt.subplots(figsize=(8, 4))
    sns.histplot(df['Pret_Motorina'], bins=30, kde=True, ax=ax, color='purple')
    st.pyplot(fig)

st.subheader("Testează un Model (Top 5)")
model_ales = st.selectbox("Alege modelul:", ["Linear Regression", "CatBoost Regressor", "EBM Regressor", "XGBoost Regressor", "SVR"])

@st.cache_resource
def get_model(name):
    if name == "Linear Regression":
        m = LinearRegression()
    elif name == "CatBoost Regressor":
        m = CatBoostRegressor(verbose=0, random_state=42)
    elif name == "EBM Regressor":
        m = ExplainableBoostingRegressor(random_state=42)
    elif name == "XGBoost Regressor":
        m = XGBRegressor(random_state=42)
    else:
        m = SVR(kernel='linear')
    m.fit(X_train, y_train)
    return m

model = get_model(model_ales)
y_pred = model.predict(X_test)
st.info(f"**Performanță:** Scor R2: {r2_score(y_test, y_pred):.4f} | RMSE: {np.sqrt(mean_squared_error(y_test, y_pred)):.4f}")

st.subheader("Calculează Prețul Estimativ")
inputs = []
cols = st.columns(4)
for i in range(1, 9):
    with cols[(i-1) % 4]:
        val = st.number_input(f"Feature_{i}", value=float(X[f'Feature_{i}'].mean()))
        inputs.append(val)

if st.button("Estimează Preț!"):
    input_df = pd.DataFrame([inputs], columns=X.columns)
    pred = model.predict(input_df)[0]
    st.success(f"⛽ Prețul estimat pentru motorină este: **{pred:.2f} RON**")

    if model_ales in ["CatBoost Regressor", "XGBoost Regressor"]:
        st.subheader("🧠 Explicabilitate SHAP (Impactul factorilor)")
        explainer = shap.TreeExplainer(model)
        shap_values = explainer.shap_values(input_df)
        
        fig, ax = plt.subplots()
        shap.summary_plot(shap_values, input_df, plot_type="bar", show=False)
        st.pyplot(fig)