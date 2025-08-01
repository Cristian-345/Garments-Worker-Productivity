import streamlit as st
import pandas as pd
import plotly.express as px
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
import plotly.graph_objects as go
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder


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

with col1:
    st.metric(
        "Promedio de Productividad Real",
        f"{df_filtrado['Productividad_Real'].mean():.2f}",
    )
    st.caption("Promedio de productividad real entre 0 y 1")

with col2:
    st.metric(
        "Productividad Objetivo Promedio",
        f"{df_filtrado['Productividad_Meta'].mean():.2f}",
    )
    st.caption("Promedio de meta establecida")

with col3:
    st.metric("Incentivo Promedio", f"{df_filtrado['Incentivo'].mean():.2f}")
    st.caption("Promedio de incentivo aplicado")


st.subheader("Datos filtrados")
st.dataframe(df_filtrado)

col1, col2 = st.columns(2)

with col1:
    st.subheader("Productividad Real por Departamento")
    st.caption(
        "Histograma que muestra la distribución de la productividad real según el departamento."
    )
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
    st.caption(
        "Gráfico de pastel que muestra la proporción de los días en que se registró la productividad."
    )
    grafico2 = px.pie(df_filtrado, names="Día", title="Distribución de Días", hole=0.3)
    st.plotly_chart(grafico2, use_container_width=True)


st.subheader("Distribución de Productividad por Día de la Semana")
st.caption(
    "Gráfico de barras apiladas que muestra cómo se distribuye la productividad real según el día y el departamento."
)
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
    st.caption("Histograma agrupado por departamento y día de la semana.")
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
    st.caption("Histograma agrupado por número de equipo y departamento.")
    grafico4 = px.histogram(
        df_filtrado,
        x="Equipo",
        y="Productividad_Real",
        color="Departamento",
        barmode="group",
    )
    st.plotly_chart(grafico4, use_container_width=True)

st.subheader("Relación entre Horas Extra, Incentivo y Productividad")
st.caption(
    "Gráfico de dispersión que relaciona las horas extra trabajadas con el incentivo aplicado y la productividad real."
)
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
st.caption(
    "Correlaciones entre variables cuantitativas del rendimiento de los trabajadores."
)

# Filtramos solo columnas numéricas
df_numerico = df_filtrado.select_dtypes(include=["int64", "float64"])

# Verificamos que haya al menos 2 variables numéricas para evitar errores
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
    - **Fecha**: Día exacto del registro (formato YYYY-MM-DD).
    - **Trimestre**: Periodo del año en el que se registró (Quarter1, Quarter2...).
    - **Departamento**: Área de trabajo del empleado (costura, acabado).
    - **Día**: Día de la semana en que se registró el dato.
    - **Equipo**: Número de equipo asignado al trabajador.
    - **Productividad_Meta**: Meta de productividad asignada al equipo (valor entre 0 y 1).
    - **SMV (Standard Minute Value)**: Tiempo estándar en minutos que debería tomar completar la unidad de trabajo.
    - **WIP (Work In Progress)**: Número de unidades en progreso durante la jornada.
    - **Horas_Extra**: Cantidad de tiempo adicional trabajado (en minutos).
    - **Incentivo**: Valor del incentivo monetario recibido por desempeño.
    - **Tiempo_Inactivo**: Tiempo no productivo durante la jornada (en minutos).
    - **Trabajadores_Inactivos**: Cantidad de trabajadores sin actividad registrada.
    - **Cambios_Estilo**: Número de veces que se cambió el estilo o modelo en la jornada.
    - **Numero_Trabajadores**: Total de trabajadores presentes en el equipo.
    - **Productividad_Real**: Productividad lograda realmente por el equipo (valor entre 0 y 1).
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
    help="Descarga los datos actualmente filtrados en formato CSV.",
)

st.markdown("## Comparativa de Equipo Seleccionado vs Promedio")
st.caption(
    "Selecciona un registro de productividad para compararlo visualmente con el promedio del dataset filtrado."
)

if not df_filtrado.empty:
    idx_registro = st.selectbox(
        "Selecciona el índice del registro a comparar:",
        df_filtrado.index,
        format_func=lambda x: f"Registro #{x}",
    )
    registro_sel = df_filtrado.loc[idx_registro]

    radar_vars = [
        "Productividad_Real",
        "Productividad_Meta",
        "Horas_Extra",
        "Incentivo",
        "SMV",
        "Tiempo_Inactivo",
        "Trabajadores_Inactivos",
    ]

    reg_vals = [registro_sel[var] for var in radar_vars]
    prom_vals = [df_filtrado[var].mean() for var in radar_vars]

    fig_radar = go.Figure()
    fig_radar.add_trace(
        go.Scatterpolar(
            r=reg_vals, theta=radar_vars, fill="toself", name="Registro Seleccionado"
        )
    )
    fig_radar.add_trace(
        go.Scatterpolar(
            r=prom_vals, theta=radar_vars, fill="toself", name="Promedio General"
        )
    )

    fig_radar.update_layout(
        polar=dict(
            radialaxis=dict(
                visible=True, range=[0, max(max(reg_vals), max(prom_vals)) + 1]
            )
        ),
        showlegend=True,
    )

    st.plotly_chart(fig_radar, use_container_width=True)
else:
    st.info("No hay registros filtrados para comparar.")

st.markdown("## Registros Destacados de Productividad")
st.caption(
    "Tabla con registros destacados según alta productividad, alto incentivo y muchas horas extra."
)

if not df_filtrado.empty:
    destacados = pd.DataFrame()

    # Mejores registros por productividad real
    top_productivos = df_filtrado.sort_values(
        "Productividad_Real", ascending=False
    ).head(3)
    destacados = pd.concat([destacados, top_productivos])

    # Registros con mayor incentivo
    top_incentivo = df_filtrado.sort_values("Incentivo", ascending=False).head(3)
    destacados = pd.concat([destacados, top_incentivo])

    # Registros con más horas extra
    top_horas_extra = df_filtrado.sort_values("Horas_Extra", ascending=False).head(3)
    destacados = pd.concat([destacados, top_horas_extra])

    # Eliminar duplicados si un registro destaca en más de una categoría
    destacados = destacados.drop_duplicates()

    st.dataframe(destacados)
else:
    st.info("No hay registros destacados para mostrar.")

st.markdown("## Análisis de Correlación Interactivo")
st.caption(
    "Selecciona dos variables numéricas para analizar su relación mediante un gráfico de dispersión."
)

# Seleccionamos solo columnas numéricas
cols_corr = df_filtrado.select_dtypes(include=["int64", "float64"]).columns.tolist()

if len(cols_corr) >= 2:
    var_x = st.selectbox("Variable X", cols_corr, key="corr_x")
    var_y = st.selectbox("Variable Y", cols_corr, key="corr_y")

    if var_x and var_y:
        fig_corr = px.scatter(
            df_filtrado,
            x=var_x,
            y=var_y,
            color="Departamento",
            hover_data=["Equipo", "Día"],
        )
        st.plotly_chart(fig_corr, use_container_width=True)
else:
    st.info("No hay suficientes variables numéricas para el análisis de correlación.")


st.markdown("## Predicción de Aceptación")
st.caption(
    "Selecciona características para predecir la aceptación del vehículo usando un modelo básico de machine learning."
)

if not df.empty:
    df_model = df.copy()
    columnas_codificar = [
        "Precio_Compra",
        "Costo_Mantenimiento",
        "Num_Puertas",
        "Capacidad_Personas",
        "Tamano_Maletero",
        "Seguridad",
        "Aceptacion",
    ]

    # Crear LabelEncoders para cada columna categórica
    encoders = {col: LabelEncoder().fit(df_model[col]) for col in columnas_codificar}
    for col in columnas_codificar:
        df_model[col] = encoders[col].transform(df_model[col])

    # Variables predictoras y objetivo
    X = df_model.drop("Aceptacion", axis=1)
    y = df_model["Aceptacion"]

    # Entrenar modelo RandomForest
    modelo_rf = RandomForestClassifier(n_estimators=50, random_state=42)
    modelo_rf.fit(X, y)

    st.markdown("### Selecciona las características:")
    input_usuario = {}
    for col in columnas_codificar[:-1]:  # Excepto "Aceptacion"
        opciones = df[col].unique()
        input_usuario[col] = st.selectbox(f"{col}", opciones, key=f"pred_{col}")

    if st.button("Predecir aceptación"):
        df_input = pd.DataFrame([input_usuario])
        # Codificar entradas
        for col in columnas_codificar[:-1]:
            df_input[col] = encoders[col].transform(df_input[col])
        df_input = df_input[X.columns]

        # Predicción
        prediccion = modelo_rf.predict(df_input)[0]
        etiqueta_prediccion = encoders["Aceptacion"].inverse_transform([prediccion])[0]

        st.success(f"Predicción de aceptación: **{etiqueta_prediccion}**")
else:
    st.info("No hay datos suficientes para entrenar el modelo de predicción.")
