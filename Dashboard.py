import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# ======================================
# 1️⃣ CONFIGURACIÓN INICIAL
# ======================================
st.set_page_config(page_title="Dashboard Diabetes", layout="wide")
st.title("📊 Dashboard Interactivo de Pacientes con Diabetes")
st.markdown("Visualiza, analiza y explora información médica de pacientes con diabetes.")

# ======================================
# 2️⃣ CARGA Y LIMPIEZA DE DATOS
# ======================================
@st.cache_data
def cargar_datos():
    df = pd.read_excel(r"c:\Users\Usuario\Downloads\DIABETES DATOS HD.xlsx")
    df.columns = df.columns.str.lower().str.strip()
    return df

df = cargar_datos()

# ======================================
# 3️⃣ FILTROS LATERALES
# ======================================
st.sidebar.header("🔍 Filtros de Análisis")

# Edad
min_edad = int(df["edad"].min())
max_edad = int(df["edad"].max())
edad_rango = st.sidebar.slider("Edad", min_edad, max_edad, (min_edad, max_edad))

# Género
generos = df["genero"].dropna().unique().tolist()
genero_sel = st.sidebar.multiselect("Género", generos, default=generos)

# IMC
min_imc = float(df["imc"].min())
max_imc = float(df["imc"].max())
imc_rango = st.sidebar.slider("IMC", min_imc, max_imc, (min_imc, max_imc))

# Tipo de diabetes
if "tipo_diabetes" in df.columns:
    tipos = df["tipo_diabetes"].dropna().unique().tolist()
    tipo_sel = st.sidebar.multiselect("Tipo de diabetes", tipos, default=tipos)
else:
    tipo_sel = []

# Fumador
if "fumador" in df.columns:
    fumadores = df["fumador"].dropna().unique().tolist()
    fumador_sel = st.sidebar.multiselect("Fumador", fumadores, default=fumadores)
else:
    fumador_sel = []

# Aplicar filtros
df_filtrado = df[
    (df["edad"].between(*edad_rango)) &
    (df["genero"].isin(genero_sel)) &
    (df["imc"].between(*imc_rango))
]

if tipo_sel:
    df_filtrado = df_filtrado[df_filtrado["tipo_diabetes"].isin(tipo_sel)]
if fumador_sel:
    df_filtrado = df_filtrado[df_filtrado["fumador"].isin(fumador_sel)]

# ======================================
# 4️⃣ MÉTRICAS RESUMEN
# ======================================
st.subheader("📈 Resumen General de los Pacientes Filtrados")

col1, col2, col3, col4 = st.columns(4)
col1.metric("👥 Total Pacientes", df_filtrado.shape[0])
col2.metric("📆 Promedio Edad", f"{df_filtrado['edad'].mean():.1f} años")
col3.metric("⚖️ IMC Promedio", f"{df_filtrado['imc'].mean():.1f}")
if "colesterol_total" in df_filtrado.columns:
    col4.metric("🩸 Colesterol Promedio", f"{df_filtrado['colesterol_total'].mean():.1f} mg/dL")

# ======================================
# 5️⃣ GRÁFICOS PRINCIPALES
# ======================================
st.markdown("---")
st.subheader("📊 Visualizaciones")

col1, col2 = st.columns(2)

with col1:
    if "tipo_diabetes" in df_filtrado.columns:
        fig1 = px.pie(df_filtrado, names="tipo_diabetes", title="Distribución por Tipo de Diabetes")
        st.plotly_chart(fig1, use_container_width=True)
    else:
        st.info("No se encontró la columna 'tipo_diabetes'.")

with col2:
    if {"pre_sisto", "pre_diasto"}.issubset(df_filtrado.columns):
        fig2 = px.bar(
            x=["Sistólica", "Diastólica"],
            y=[df_filtrado["pre_sisto"].mean(), df_filtrado["pre_diasto"].mean()],
            title="Presión Arterial Promedio (mmHg)",
            color=["Sistólica", "Diastólica"]
        )
        st.plotly_chart(fig2, use_container_width=True)

# ======================================
# 6️⃣ GRÁFICO: Relación Edad vs Presión
# ======================================
st.markdown("---")
st.subheader("📉 Relación entre Edad y Presión Arterial")

if {"edad", "pre_sisto", "pre_diasto"}.issubset(df_filtrado.columns):
    fig3 = px.scatter(
        df_filtrado,
        x="edad",
        y="pre_sisto",
        color="genero",
        trendline="ols",
        title="Relación entre Edad y Presión Sistólica"
    )
    st.plotly_chart(fig3, use_container_width=True)

# ======================================
# 7️⃣ MATRIZ DE CORRELACIÓN
# ======================================
st.markdown("---")
st.subheader("📊 Matriz de Correlación entre Variables Numéricas")

corr = df_filtrado.select_dtypes(include=np.number).corr()
fig_corr = px.imshow(corr, text_auto=".2f", aspect="auto", title="Correlación de Variables")
st.plotly_chart(fig_corr, use_container_width=True)

# ======================================
# 8️⃣ MODELO PREDICTIVO SIMPLE
# ======================================
st.markdown("---")
st.subheader("🤖 Predicción de Complicaciones (modelo demostrativo)")

if {"imc", "colesterol_total", "edad", "fumador_cod"}.issubset(df.columns) and "complicaciones" in df.columns:
    X = df[["imc", "colesterol_total", "edad", "fumador_cod"]]
    y = df["complicaciones"]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    modelo = LogisticRegression(max_iter=1000, multi_class='multinomial')
    modelo.fit(X_train_scaled, y_train)

    # Seleccionar valores manualmente
    st.write("### Ingresa valores para predecir complicaciones:")
    edad_pred = st.number_input("Edad", min_value=10, max_value=100, value=40)
    imc_pred = st.number_input("IMC", min_value=10.0, max_value=50.0, value=25.0)
    col_pred = st.number_input("Colesterol total", min_value=100.0, max_value=400.0, value=200.0)
    fumador_pred = st.selectbox("¿Fumador?", ["No", "Sí"])
    fumador_cod = 1 if fumador_pred == "Sí" else 0

    X_new = scaler.transform([[imc_pred, col_pred, edad_pred, fumador_cod]])
    pred = modelo.predict(X_new)[0]
    st.success(f"🔮 Predicción de complicación: **{pred}**")
else:
    st.info("No se encontraron las columnas necesarias para la predicción.")

# ======================================
# 9️⃣ EXPORTAR DATOS FILTRADOS
# ======================================
st.markdown("---")
st.subheader("💾 Exportar datos filtrados")

@st.cache_data
def convertir_excel(df):
    return df.to_excel(index=False, engine='openpyxl')

if st.button("Descargar datos filtrados"):
    df_filtrado.to_excel("pacientes_filtrados.xlsx", index=False)
    st.success("Archivo 'pacientes_filtrados.xlsx' generado con éxito.")
