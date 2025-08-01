import streamlit as st
import pandas as pd
import plotly.express as px
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
import plotly.graph_objects as go

df = pd.read_csv("garments_worker_productivity_limpio.csv")

st.set_page_config(layout="wide")
st.title("Dashboard de Productividad de Trabajadores de Confección")

if df.isnull().values.any():
    st.warning("Datos faltantes detectados. Aplicando interpolación...")
    df.interpolate(method="linear", inplace=True)

st.sidebar.header("Filtros")
departamento = st.sidebar.multiselect(
    "Departamento",
    sorted(df["Departamento"].unique()),
    default=df["Departamento"].unique(),
)
dia = st.sidebar.multiselect(
    "Día", sorted(df["Día"].unique()), default=df["Día"].unique()
)
equipo = st.sidebar.multiselect(
    "Equipo", sorted(df["Equipo"].unique()), default=df["Equipo"].unique()
)
trimestre = st.sidebar.multiselect(
    "Trimestre", sorted(df["Trimestre"].unique()), default=df["Trimestre"].unique()
)

df_filtrado = df[
    (df["Departamento"].isin(departamento))
    & (df["Día"].isin(dia))
    & (df["Equipo"].isin(equipo))
    & (df["Trimestre"].isin(trimestre))
]

st.markdown("## Indicadores Clave")
col1, col2, col3 = st.columns(3)
col1.metric(
    "Promedio de Productividad Real", f"{df_filtrado['Productividad_Real'].mean():.2f}"
)
col2.metric(
    "Productividad Objetivo Promedio", f"{df_filtrado['Productividad_Meta'].mean():.2f}"
)
col3.metric("Incentivo Promedio", f"{df_filtrado['Incentivo'].mean():.2f}")

st.subheader("Datos filtrados")
st.dataframe(df_filtrado)

col1, col2 = st.columns(2)
with col1:
    st.subheader("Productividad Real por Departamento")
    grafico1 = px.histogram(
        df_filtrado,
        x="Departamento",
        y="Productividad_Real",
        color="Departamento",
        barmode="group",
    )
    st.plotly_chart(grafico1, use_container_width=True)

with col2:
    st.subheader("Distribución de Días Trabajados")
    grafico2 = px.pie(df_filtrado, names="Día", title="Distribución de Días", hole=0.3)
    st.plotly_chart(grafico2, use_container_width=True)

st.subheader("Distribución de Productividad por Día de la Semana")
grafico_stacked = px.histogram(
    df_filtrado,
    x="Día",
    y="Productividad_Real",
    color="Departamento",
    barmode="stack",
    title="Productividad acumulada por día y departamento",
)
st.plotly_chart(grafico_stacked, use_container_width=True)

st.subheader("Comparaciones adicionales")
col3, col4 = st.columns(2)
with col3:
    st.markdown("### Departamento vs Productividad Real")
    grafico3 = px.histogram(
        df_filtrado,
        x="Departamento",
        y="Productividad_Real",
        color="Día",
        barmode="group",
    )
    st.plotly_chart(grafico3, use_container_width=True)

with col4:
    st.markdown("### Equipo vs Productividad Real")
    grafico4 = px.histogram(
        df_filtrado,
        x="Equipo",
        y="Productividad_Real",
        color="Departamento",
        barmode="group",
    )
    st.plotly_chart(grafico4, use_container_width=True)

st.subheader("Relación entre Horas Extra, Incentivo y Productividad")
grafico5 = px.scatter(
    df_filtrado,
    x="Horas_Extra",
    y="Incentivo",
    size="Productividad_Real",
    color="Departamento",
    hover_data=["Equipo", "Día"],
)
st.plotly_chart(grafico5, use_container_width=True)

st.subheader("Mapa de Calor de Correlaciones")
df_numerico = df_filtrado.select_dtypes(include=["int64", "float64"])
if df_numerico.shape[1] > 1:
    fig, ax = plt.subplots(figsize=(10, 6))
    sns.heatmap(
        df_numerico.corr(), annot=True, cmap="YlGnBu", fmt=".2f", linewidths=0.5, ax=ax
    )
    st.pyplot(fig)
else:
    st.info("No hay suficientes variables numéricas para generar el mapa de calor.")

with st.expander("Explicación de variables", expanded=False):
    st.markdown(
        """
    - *Fecha*: Día exacto del registro (formato YYYY-MM-DD).
    - *Trimestre*: Periodo del año en el que se registró (Quarter1, Quarter2...).
    - *Departamento*: Área de trabajo del empleado (costura, acabado).
    - *Día*: Día de la semana en que se registró el dato.
    - *Equipo*: Número de equipo asignado al trabajador.
    - *Productividad_Meta*: Meta de productividad asignada al equipo (valor entre 0 y 1).
    - *SMV*: Tiempo estándar en minutos que debería tomar completar la unidad de trabajo.
    - *WIP*: Número de unidades en progreso durante la jornada.
    - *Horas_Extra*: Cantidad de tiempo adicional trabajado (en minutos).
    - *Incentivo*: Valor del incentivo monetario recibido por desempeño.
    - *Tiempo_Inactivo*: Tiempo no productivo durante la jornada (en minutos).
    - *Trabajadores_Inactivos*: Cantidad de trabajadores sin actividad registrada.
    - *Cambios_Estilo*: Número de veces que se cambió el estilo o modelo en la jornada.
    - *Numero_Trabajadores*: Total de trabajadores presentes en el equipo.
    - *Productividad_Real*: Productividad lograda realmente por el equipo (valor entre 0 y 1).
    """
    )
    st.caption(
        "Esta sección describe las variables incluidas en el análisis de productividad del sector textil."
    )

csv = df_filtrado.to_csv(index=False).encode("utf-8-sig")
st.download_button(
    label="Descargar datos filtrados en CSV",
    data=csv,
    file_name="productividad_filtrada.csv",
    mime="text/csv",
)
