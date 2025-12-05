# ProyectoFinal_IA
# ================================
# Proyecto Final IA 
# Dashboard Streamlit: Predicción de Fallas en Motores Ingenieria en Robotica y Sistemas Inteligentes
# ================================

# ---- Importación de librerías ----
import streamlit as st            # Framework para crear la aplicación web
import pandas as pd               # Para manejar el dataset
import numpy as np                # Para cálculos numéricos
from sklearn.ensemble import RandomForestClassifier  # Modelo de Machine Learning
from sklearn.model_selection import train_test_split # Para dividir datos en train/test
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix, classification_report
import matplotlib.pyplot as plt   # Para gráficas
import seaborn as sns             # Para gráficas avanzadas

# Configuración de la página para que se vea más limpio
st.set_page_config(page_title="PF IA - Predicción Fallas", layout="wide")

# -------------------------
# Ruta del dataset
DATA_PATH = "ai4i2020.csv"  
# -------------------------

# Esta función carga los datos 
@st.cache_data
def cargar_datos(path):
    df = pd.read_csv(path)     # Lee el CSV
    return df                  # Regresa el DataFrame

# Intento cargar el dataset
try:
    df = cargar_datos(DATA_PATH)      # Carga el CSV
except Exception as e:                
    st.error(f"Error cargando el dataset: {e}")
    st.stop()                         

# ---------------------------------------
# Preprocesamiento de datos
# ---------------------------------------
def preprocess(df):
    df_clean = df.copy()   # Se hace copia del dataset original

    # Normalizamos nombres de columnas (sin espacios, corchetes, etc.)
    df_clean.columns = [
        c.strip().replace(" ", "_").replace("[", "").replace("]", "")
         .replace("/", "_").replace(".", "")
        for c in df_clean.columns
    ]

    # Buscamos la columna "machine failure" o similar
    label_candidates = [
        c for c in df_clean.columns
        if 'failure' in c.lower() or c.lower() == 'machine_failure'
    ]

    # Si no encuentra la columna de etiqueta, marcamos error
    if len(label_candidates) == 0:
        raise ValueError("No se encontró columna de falla en el dataset.")

    label_col = label_candidates[0]   # Tomamos la primera coincidencia

    # X = todas las columnas numéricas excepto la etiqueta
    X = df_clean.select_dtypes(include=[np.number]).drop(columns=[label_col])

    # y = la etiqueta (0 = no falla, 1 = falla)
    y = df_clean[label_col].astype(int)

    return X, y, label_col, df_clean   # Regresamos datos procesados

# Aplicamos preprocesamiento
try:
    X, y, label_col, df_clean = preprocess(df)
except Exception as e:
    st.error(f"Error en preprocesamiento: {e}")
    st.stop()

# ---------------------------------------
# División en Train y Test
# ---------------------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# ---------------------------------------
# Entrenamiento del Modelo
# ---------------------------------------
model = RandomForestClassifier(      # Creamos el modelo Random Forest
    n_estimators=150,                # Árboles en el bosque
    random_state=42,                 # Reproducibilidad
    n_jobs=-1                        # Usa todos los núcleos del CPU
)

model.fit(X_train, y_train)          # Entrenamos el modelo

y_pred = model.predict(X_test)       # Hacemos predicciones en test

# Cálculo de métricas
acc = accuracy_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred, zero_division=0)
cm = confusion_matrix(y_test, y_pred)

# ---------------------------------------
# MENÚ LATERAL
# ---------------------------------------
st.sidebar.title("Navegación")     # Título del menú lateral

op = st.sidebar.radio(             # Opciones del menú
    "Ir a:", 
    ["1. Contexto y Datos", 
     "2. Análisis Exploratorio (EDA)", 
     "3. Evaluación del Modelo", 
     "4. Simulador en Vivo"]
)

# ---------------------------------------
# OPCIÓN 1 — CONTEXTO
# ---------------------------------------
if op == "1. Contexto y Datos":

    st.title("🔧 Predicción de Fallas en Motores ")
    st.markdown("""
    **Problema:** Los robots utilizan motores y actuadores que pueden fallar y parar todo el sistema.  
    **Objetivo:** Predecir fallas usando sensores (temperatura, torque, rpm, desgaste).
    """)

    st.subheader("Vista del Dataset")
    st.dataframe(df_clean.head(10))   # Muestra primeras 10 filas

    st.write("Columnas disponibles:", list(df_clean.columns))  # Lista las columnas

# ---------------------------------------
# OPCIÓN 2 — EDA
# ---------------------------------------
elif op == "2. Análisis Exploratorio (EDA)":

    st.title("📊 Exploración de Datos")

    # Dos columnas lado a lado
    col1, col2 = st.columns(2)

    # Gráfica de distribución de fallas
    with col1:
        st.subheader("Distribución de fallas")
        counts = y.value_counts().rename(index={0: "No falla", 1: "Falla"})
        st.bar_chart(counts)
        st.write(counts)

    # Heatmap de correlaciones
    with col2:
        st.subheader("Correlación entre variables")
        corr = X.corr()                         # Matriz de correlación
        fig, ax = plt.subplots(figsize=(8,6))   # Creamos figura
        sns.heatmap(corr, annot=True, cmap="vlag", ax=ax)  # Heatmap
        st.pyplot(fig)

    # Selección de variable para análisis
    st.subheader("Explorar variable:")
    var = st.selectbox("Variable:", X.columns)

    # Gráficas descriptivas
    fig2, ax2 = plt.subplots(1,2, figsize=(12,4))
    sns.histplot(X[var], kde=True, ax=ax2[0])        # Histograma
    ax2[0].set_title(f"Histograma de {var}")
    sns.boxplot(x=y, y=X[var], ax=ax2[1])            # Boxplot por falla
    ax2[1].set_title(f"{var} vs falla")
    st.pyplot(fig2)

# ---------------------------------------
# OPCIÓN 3 — Evaluación
# ---------------------------------------
elif op == "3. Evaluación del Modelo":

    st.title("⚙️ Evaluación del Modelo")

    # Métricas principales
    st.metric("Accuracy", f"{acc:.4f}")
    st.metric("F1 Score", f"{f1:.4f}")

    st.subheader("Reporte de Clasificación")
    st.text(classification_report(y_test, y_pred))

    st.subheader("Matriz de Confusión")
    fig_cm, ax_cm = plt.subplots()
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", ax=ax_cm)
    st.pyplot(fig_cm)

    st.subheader("Importancia de Características")
    feat_imp = pd.Series(model.feature_importances_, index=X.columns)
    st.bar_chart(feat_imp)

# ---------------------------------------
# OPCIÓN 4 — Simulador
# ---------------------------------------
elif op == "4. Simulador en Vivo":

    st.title("🚀 Simulador en Vivo")

    st.markdown("Modifica los valores de sensores para predecir si el motor fallará.")

    # Dos columnas para sliders
    colA, colB = st.columns(2)

    input_data = {}  # Diccionario para inputs del usuario

    # Generar sliders para cada característica
    for i, col_name in enumerate(X.columns):
        col = X[col_name]

        lo = float(col.min())         # Valor mínimo
        hi = float(col.max())         # Valor máximo
        default = float(col.median()) # Valor por defecto

        slider_col = colA if i % 2 == 0 else colB

        # Creamos slider
        val = slider_col.slider(
            col_name,
            min_value=lo,
            max_value=hi,
            value=default
        )

        input_data[col_name] = val   # Guardamos valor

    # Botón para predecir
    if st.button("Predecir"):
        input_df = pd.DataFrame([input_data])      # Convertimos inputs a DataFrame
        pred = model.predict(input_df)[0]          # Predicción 0/1
        prob = model.predict_proba(input_df)[0]    # Probabilidades

        if pred == 1:
            st.error("Error: ¡El motor fallará pronto! ⚠️")
        else:
            st.success("✅ Motor en estado normal")

        st.write("Probabilidades [No falla / Falla]:", np.round(prob,3))


st.sidebar.markdown("---")
st.sidebar.write("Autor: Abraham Gamez Gonzalez - Proyecto Final IA")
