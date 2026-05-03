import streamlit as st

# Configurarea paginii principale
st.set_page_config(page_title="Proiect ML", page_icon="🤖", layout="wide")

st.title("Proiect Machine Learning - Sisteme Inteligente")
st.subheader("Analiza Comparată a Modelelor de Machine Learning")

st.markdown("""
**Bine ați venit!** Această aplicație interactivă prezintă rezultatele proiectului de Machine Learning.
În meniul din stânga, puteți naviga între cele două probleme abordate:
1. **Problema de Clasificare**: Predicția aprobării unui credit bancar.
2. **Problema de Regresie**: Predicția prețului la motorină.

Fiecare pagină include explorarea datelor, selecția celor mai bune modele, predicții în timp real și explicabilitate prin intermediul bibliotecii SHAP.
""")