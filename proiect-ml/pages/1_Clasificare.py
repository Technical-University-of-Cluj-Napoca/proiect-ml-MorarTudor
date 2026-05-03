import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import shap
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score
from sklearn.svm import SVC
from catboost import CatBoostClassifier
from xgboost import XGBClassifier
from sklearn.neighbors import KNeighborsClassifier
from interpret.glassbox import ExplainableBoostingClassifier

st.set_page_config(page_title="Clasificare", layout="wide")
st.title("Clasificare: Aprobare Credit Bancar")

@st.cache_data
def load_data():
    df = pd.read_csv('data/clasificare.csv')
    return df

df = load_data()
X = df.drop('Aprobare_Credit', axis=1)
y = df['Aprobare_Credit']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)

with st.expander("Vezi Analiza Exploratorie a Datelor (EDA)"):
    st.write("Primele rânduri din setul de date:")
    st.dataframe(df.head())
    
    col1, col2 = st.columns(2)
    with col1:
        fig, ax = plt.subplots()
        sns.countplot(data=df, x='Aprobare_Credit', ax=ax, palette='Set2')
        ax.set_title("Distribuția Claselor")
        st.pyplot(fig)
    with col2:
        fig2, ax2 = plt.subplots(figsize=(6, 5))
        sns.heatmap(df.corr(), annot=False, cmap='coolwarm', ax=ax2)
        ax2.set_title("Matricea de Corelație")
        st.pyplot(fig2)

st.subheader("Testează un Model (Top 5)")
model_ales = st.selectbox("Alege modelul:", ["SVM", "CatBoost", "XGBoost", "KNN", "EBM"])

@st.cache_resource
def get_model(name):
    if name == "SVM":
        m = SVC(probability=True, random_state=42)
    elif name == "CatBoost":
        m = CatBoostClassifier(verbose=0, random_state=42)
    elif name == "XGBoost":
        m = XGBClassifier(eval_metric='logloss', random_state=42)
    elif name == "KNN":
        m = KNeighborsClassifier()
    else:
        m = ExplainableBoostingClassifier(random_state=42)
    m.fit(X_train, y_train)
    return m

model = get_model(model_ales)

y_pred = model.predict(X_test)
st.info(f"**Performanța pe setul de test:** Acuratețe: {accuracy_score(y_test, y_pred):.2f} | Scor F1: {f1_score(y_test, y_pred):.2f}")

st.subheader("Fă o predicție nouă")
inputs = []
cols = st.columns(5)
for i in range(1, 11):
    with cols[(i-1) % 5]:
        val = st.number_input(f"Feature_{i}", value=float(X[f'Feature_{i}'].mean()))
        inputs.append(val)

if st.button("Află Rezultatul!"):
    input_df = pd.DataFrame([inputs], columns=X.columns)
    pred = model.predict(input_df)[0]
    if pred == 1:
        st.success("Credit APROBAT!")
    else:
        st.error("Credit RESPINS!")

    if model_ales in ["CatBoost", "XGBoost"]:
        st.subheader("Explicabilitate SHAP (De ce s-a luat această decizie?)")
        explainer = shap.TreeExplainer(model)
        shap_values = explainer.shap_values(input_df)
        
        fig, ax = plt.subplots()
        shap.summary_plot(shap_values, input_df, plot_type="bar", show=False)
        st.pyplot(fig)
        st.write("*Graficul de mai sus arată care trăsături (Features) au contat cel mai mult pentru acest client.*")