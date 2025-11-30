import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import io
from datetime import datetime, timedelta
import json
from collections import Counter
import string
import random
import hashlib
import math
import time
import warnings
import unicodedata
import re
import pickle
from io import BytesIO
warnings.filterwarnings('ignore')
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor, IsolationForest, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression, LinearRegression, Ridge, Lasso, ElasticNet
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    mean_squared_error, r2_score, mean_absolute_error,
    confusion_matrix, classification_report,
    silhouette_score, davies_bouldin_score, calinski_harabasz_score,
    roc_auc_score, roc_curve, precision_recall_curve
)
from sklearn.preprocessing import LabelEncoder, StandardScaler, MinMaxScaler, RobustScaler
from sklearn.cluster import KMeans, DBSCAN, AgglomerativeClustering
from sklearn.impute import SimpleImputer, KNNImputer
from sklearn.decomposition import PCA
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier, LocalOutlierFactor
from scipy.stats import gaussian_kde, stats, chi2, zscore
from scipy.spatial.distance import cdist
try:
    from fuzzywuzzy import fuzz, process
    FUZZY_AVAILABLE = True
except ImportError:
    FUZZY_AVAILABLE = False
try:
    import networkx as nx
    from networkx.algorithms import community
    NETWORKX_AVAILABLE = True
except ImportError:
    NETWORKX_AVAILABLE = False
try:
    import openpyxl
    from openpyxl.styles import Font, PatternFill, Alignment
    OPENPYXL_AVAILABLE = True
except ImportError:
    OPENPYXL_AVAILABLE = False
try:
    from wordcloud import WordCloud
    WORDCLOUD_AVAILABLE = True
except ImportError:
    WORDCLOUD_AVAILABLE = False

# PARÁMETRO CONFIGURABLE: Umbral de alerta de incremento patrimonial inusual (%)
UMBRAL_ALERTA_INCREMENTO = 100.0

st.set_page_config(
    page_title="AURORA-ETHICS M.A.S.C. & S.A.H.",
    layout="wide",
    page_icon="🔍",
    initial_sidebar_state="expanded"
)

st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Poppins:wght@300;400;600;700&display=swap');
    * {
        font-family: 'Poppins', sans-serif;
    }
    .stApp {
        background: white !important;
    }
    [data-testid="stSidebar"] {
        background: white !important;
        border-right: 2px solid rgba(102, 126, 234, 0.2);
    }
    [data-testid="stSidebar"] [data-testid="stMarkdownContainer"] {
        color: #667eea;
    }
    @keyframes colorPulse {
        0% { background: linear-gradient(135deg, rgba(102, 126, 234, 0.95) 0%, rgba(118, 75, 162, 0.95) 100%); }
        25% { background: linear-gradient(135deg, rgba(0, 71, 187, 0.95) 0%, rgba(102, 126, 234, 0.95) 100%); }
        50% { background: linear-gradient(135deg, rgba(102, 126, 234, 0.95) 0%, rgba(118, 75, 162, 0.95) 100%); }
        75% { background: linear-gradient(135deg, rgba(0, 71, 187, 0.95) 0%, rgba(102, 126, 234, 0.95) 100%); }
        100% { background: linear-gradient(135deg, rgba(102, 126, 234, 0.95) 0%, rgba(118, 75, 162, 0.95) 100%); }
    }
    .hero-header {
        animation: colorPulse 4s ease-in-out infinite;
        padding: 3rem;
        border-radius: 25px;
        text-align: center;
        margin: 2rem 0;
        box-shadow: 0 20px 60px rgba(102, 126, 234, 0.4);
        backdrop-filter: blur(10px);
        border: 2px solid rgba(255, 255, 255, 0.2);
    }
    .brand-title {
        font-size: 3.5rem;
        font-weight: 700;
        background: linear-gradient(45deg, #ffffff, #f0f0f0);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        text-shadow: 2px 2px 4px rgba(0,0,0,0.2);
        letter-spacing: 3px;
        margin: 1rem 0;
        animation: pulse 2s ease-in-out infinite;
    }
    .quote-text {
        color: rgba(255, 255, 255, 0.95);
        font-size: 1.3rem;
        font-style: italic;
        margin-bottom: 0.5rem;
        text-shadow: 1px 1px 3px rgba(0,0,0,0.3);
    }
    .author-text {
        color: rgba(255, 255, 255, 0.85);
        font-size: 1rem;
        margin-bottom: 0.3rem;
    }
    .signature-text {
        color: rgba(255, 255, 255, 0.75);
        font-size: 0.9rem;
        font-weight: 300;
    }
    @keyframes fadeIn {
        from { opacity: 0; transform: translateY(-30px); }
        to { opacity: 1; transform: translateY(0); }
    }
    @keyframes pulse {
        0%, 100% { transform: scale(1); }
        50% { transform: scale(1.02); }
    }
    .stMetric {
        background: linear-gradient(135deg, rgba(255, 255, 255, 0.95) 0%, rgba(255, 255, 255, 0.85) 100%);
        padding: 1.5rem;
        border-radius: 15px;
        box-shadow: 0 8px 25px rgba(102, 126, 234, 0.3);
        transition: all 0.3s ease;
        border: 1px solid rgba(102, 126, 234, 0.2);
    }
    .stMetric:hover {
        transform: translateY(-5px);
        box-shadow: 0 12px 35px rgba(102, 126, 234, 0.4);
    }
    .info-box {
        background: linear-gradient(135deg, rgba(102, 126, 234, 0.1) 0%, rgba(118, 75, 162, 0.1) 100%);
        padding: 1.5rem;
        border-radius: 15px;
        border-left: 5px solid #667eea;
        margin: 1rem 0;
        box-shadow: 0 4px 15px rgba(102, 126, 234, 0.2);
    }
    .stButton > button {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        border: none;
        border-radius: 12px;
        padding: 0.75rem 2rem;
        font-weight: 600;
        font-size: 1rem;
        box-shadow: 0 8px 20px rgba(102, 126, 234, 0.4);
        transition: all 0.3s ease;
        cursor: pointer;
    }
    .stButton > button:hover {
        transform: translateY(-3px);
        box-shadow: 0 12px 30px rgba(102, 126, 234, 0.6);
        background: linear-gradient(135deg, #764ba2 0%, #667eea 100%);
    }
    .stTabs [data-baseweb="tab-list"] {
        gap: 8px;
        background: rgba(255, 255, 255, 0.9);
        border-radius: 15px;
        padding: 0.5rem;
    }
    .stTabs [data-baseweb="tab"] {
        border-radius: 10px;
        color: #667eea;
        font-weight: 600;
        padding: 0.75rem 1.5rem;
        transition: all 0.3s ease;
    }
    .stTabs [aria-selected="true"] {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        box-shadow: 0 4px 15px rgba(102, 126, 234, 0.4);
    }
    .stProgress > div > div > div > div {
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
    }
    div[data-testid="stFileUploadDropzone"] {
        background: linear-gradient(135deg, rgba(102, 126, 234, 0.1) 0%, rgba(118, 75, 162, 0.1) 100%);
        border: 2px dashed rgba(102, 126, 234, 0.5);
        border-radius: 15px;
        transition: all 0.3s ease;
    }
    div[data-testid="stFileUploadDropzone"]:hover {
        border-color: rgba(102, 126, 234, 0.8);
        background: linear-gradient(135deg, rgba(102, 126, 234, 0.2) 0%, rgba(118, 75, 162, 0.2) 100%);
    }
</style>
""", unsafe_allow_html=True)

if 'datasets' not in st.session_state:
    st.session_state.datasets = {}
if 'dataset_seleccionado' not in st.session_state:
    st.session_state.dataset_seleccionado = None
if 'encoding_dicts' not in st.session_state:
    st.session_state.encoding_dicts = {}
if 'df_encoded' not in st.session_state:
    st.session_state.df_encoded = None
if 'ml_models' not in st.session_state:
    st.session_state.ml_models = {}
if 'df_procesado' not in st.session_state:
    st.session_state.df_procesado = None
if 'df_metricas_persona' not in st.session_state:
    st.session_state.df_metricas_persona = None
if 'df_servidores_unicos' not in st.session_state:
    st.session_state.df_servidores_unicos = None
if 'df_anomalias' not in st.session_state:
    st.session_state.df_anomalias = None
if 'df_alertas' not in st.session_state:
    st.session_state.df_alertas = None

def calcular_hash_md5(data):
    """Calcula hash MD5 de datos"""
    if isinstance(data, pd.DataFrame):
        data = data.to_csv(index=False).encode()
    elif isinstance(data, bytes):
        pass
    else:
        data = str(data).encode()
    return hashlib.md5(data).hexdigest()

def normalizar_texto(texto):
    """Normaliza texto: mayúsculas, sin acentos, sin espacios múltiples"""
    if pd.isna(texto):
        return ""
    texto = str(texto).upper()
    texto = unicodedata.normalize('NFKD', texto).encode('ASCII', 'ignore').decode('ASCII')
    texto = re.sub(r'\s+', ' ', texto).strip()
    return texto

def limpiar_valor_monetario(valor):
    """Limpia y convierte valores monetarios a float"""
    if pd.isna(valor):
        return 0.0
    if isinstance(valor, (int, float)):
        return float(valor)
    # Eliminar símbolos de moneda, comas, paréntesis
    valor_str = str(valor).replace('$', '').replace(',', '').replace('(', '').replace(')', '').strip()
    try:
        return float(valor_str)
    except:
        return 0.0

def eliminar_fila_encabezados_duplicada(df):
    """Elimina segunda fila si es idéntica a encabezados"""
    if len(df) <= 1:
        return df
    headers_original = df.columns.astype(str).str.strip().str.upper()
    primera_fila = df.iloc[0].astype(str).str.strip().str.upper().values
    if np.array_equal(primera_fila, headers_original):
        df = df.drop(index=0).reset_index(drop=True)
        st.info("🔄 Fila duplicada de encabezados eliminada automáticamente")
    return df

def procesar_archivo_cargado(uploaded_file):
    """Procesa archivo cargado y aplica limpieza inicial"""
    try:
        progress_bar = st.progress(0)
        status_text = st.empty()
        file_size = uploaded_file.size / (1024 * 1024)
        status_text.text(f"📥 Cargando {uploaded_file.name}... ({file_size:.2f} MB)")
        progress_bar.progress(10)
        start_time = datetime.now()
        if uploaded_file.name.endswith('.csv'):
            df = pd.read_csv(uploaded_file, encoding='utf-8', low_memory=False)
        elif uploaded_file.name.endswith(('.xls', '.xlsx')):
            df = pd.read_excel(uploaded_file, engine='openpyxl')
        else:
            st.error("❌ Formato no soportado. Use CSV, XLS o XLSX.")
            return None
        progress_bar.progress(30)
        status_text.text(f"🔄 Procesando datos... ({len(df)} registros)")

        # Eliminar fila duplicada de encabezados si existe
        df = eliminar_fila_encabezados_duplicada(df)

        progress_bar.progress(50)
        # Convertir columnas de fecha con manejo robusto de timezone
        fecha_cols = [col for col in df.columns if 'fecha' in col.lower() or 'actualizacion' in col.lower() or 'Fecha' in col]
        for col in fecha_cols:
            try:
                df[col] = pd.to_datetime(df[col], utc=True, errors='coerce')
            except:
                df[col] = pd.to_datetime(df[col], errors='coerce')
        progress_bar.progress(70)
        # Convertir columnas monetarias con limpieza robusta
        columnas_monetarias = [
            col for col in df.columns 
            if any(keyword in col.lower() for keyword in ['valor', 'monto', 'saldo', 'ingreso', 'adquisicion'])
        ]
        for col in columnas_monetarias:
            if col in df.columns:
                df[col] = df[col].apply(limpiar_valor_monetario)
        progress_bar.progress(85)
        # Crear NombreCompleto normalizado si existen las columnas
        if all(col in df.columns for col in ['declaracion_situacionPatrimonial_datosGenerales_nombre', 
                                               'declaracion_situacionPatrimonial_datosGenerales_primerApellido']):
            df['NombreCompleto'] = (
                df['declaracion_situacionPatrimonial_datosGenerales_nombre'].fillna('').astype(str) + ' ' +
                df['declaracion_situacionPatrimonial_datosGenerales_primerApellido'].fillna('').astype(str) + ' ' +
                df.get('declaracion_situacionPatrimonial_datosGenerales_segundoApellido', pd.Series([''] * len(df))).fillna('').astype(str)
            )
            df['NombreCompleto'] = df['NombreCompleto'].apply(normalizar_texto)
        progress_bar.progress(100)
        elapsed_time = (datetime.now() - start_time).total_seconds()
        status_text.text(f"✅ Archivo cargado exitosamente en {elapsed_time:.2f} segundos")
        return df
    except Exception as e:
        st.error(f"❌ Error al procesar archivo: {str(e)}")
        return None

def preparar_para_excel(df):
    """Convierte columnas datetime con timezone a timezone-unaware para Excel"""
    df = df.copy()
    for col in df.select_dtypes(include=['datetimetz', 'datetime']).columns:
        try:
            df[col] = df[col].dt.tz_localize(None)
        except:
            pass
    return df

def calcular_metricas_por_persona(df):
    """
    Calcula métricas patrimoniales por persona según la lógica correcta:
    - Ingresos Totales = Ingreso SP + Otros Ingresos
    - Detecta incrementos inusuales entre declaraciones
    - Retorna DataFrame con métricas agregadas por persona
    """
    # Definir nombres de columnas según dataset real
    col_ingreso_sp = 'declaracion_situacionPatrimonial_ingresos_remuneracionAnualCargoPublico_valor'
    col_otros_ingresos = 'declaracion_situacionPatrimonial_ingresos_otrosIngresosAnualesTotal_valor'
    col_ingresos_totales_original = 'declaracion_situacionPatrimonial_ingresos_totalIngresosAnualesNetos_valor'
    col_fecha_actualizacion = 'metadata_actualizacion'
    col_nombre = 'NombreCompleto'
    # Verificar que existan las columnas necesarias
    if col_nombre not in df.columns:
        st.error(f"❌ Columna '{col_nombre}' no encontrada")
        return pd.DataFrame()
    # Crear copia de trabajo
    df_trabajo = df.copy()
    # Asegurar que columnas monetarias sean numéricas
    for col in [col_ingreso_sp, col_otros_ingresos, col_ingresos_totales_original]:
        if col in df_trabajo.columns:
            df_trabajo[col] = pd.to_numeric(df_trabajo[col], errors='coerce').fillna(0.0)
    # Calcular Ingresos Totales correctamente (SP + Otros)
    ingreso_sp_vals = df_trabajo[col_ingreso_sp] if col_ingreso_sp in df_trabajo.columns else 0
    otros_ingresos_vals = df_trabajo[col_otros_ingresos] if col_otros_ingresos in df_trabajo.columns else 0
    df_trabajo['ingresos_totales_calc'] = ingreso_sp_vals + otros_ingresos_vals
    # Parsear fecha de actualización
    if col_fecha_actualizacion in df_trabajo.columns:
        df_trabajo[col_fecha_actualizacion] = pd.to_datetime(df_trabajo[col_fecha_actualizacion], utc=True, errors='coerce')
    else:
        df_trabajo[col_fecha_actualizacion] = pd.NaT
    # Ordenar por persona y fecha
    df_trabajo = df_trabajo.sort_values([col_nombre, col_fecha_actualizacion])
    # Calcular incrementos entre declaraciones sucesivas por persona
    df_trabajo['ingreso_prev'] = df_trabajo.groupby(col_nombre)['ingresos_totales_calc'].shift(1)
    df_trabajo['delta_abs'] = df_trabajo['ingresos_totales_calc'] - df_trabajo['ingreso_prev']
    df_trabajo['delta_pct'] = ((df_trabajo['ingresos_totales_calc'] - df_trabajo['ingreso_prev']) / 
                                (df_trabajo['ingreso_prev'] + 1)) * 100
    # Marcar alertas de incremento inusual
    df_trabajo['alerta_incremento'] = df_trabajo['delta_pct'] > UMBRAL_ALERTA_INCREMENTO
    # Agregar métricas por persona
    metricas_persona = df_trabajo.groupby(col_nombre).agg({
        col_nombre: 'count',  # n_declaraciones
        col_ingreso_sp: 'mean' if col_ingreso_sp in df_trabajo.columns else lambda x: 0,
        col_otros_ingresos: 'mean' if col_otros_ingresos in df_trabajo.columns else lambda x: 0,
        'ingresos_totales_calc': ['mean', 'max', lambda x: x[x > 0].min() if (x > 0).any() else 0, 'last'],
        'delta_abs': 'max',
        'delta_pct': 'max',
        'alerta_incremento': 'any',
        col_fecha_actualizacion: 'max'
    })
    # Renombrar columnas
    metricas_persona.columns = [
    'n_declaraciones',
    'ingreso_sp_promedio',
    'otros_ingresos_promedio',
    'ingresos_totales_promedio',
    'max_ingresos_totales',
    'min_ingresos_totales',
    'ultimo_ingresos_totales',
    'delta_max_abs',
    'delta_max_pct',
    'alerta_incremento_inusual',
    'fecha_ultima_actualizacion'
]
    metricas_persona = metricas_persona.reset_index()
    return metricas_persona
def seccion_inicio():
    if not st.session_state.datasets:
        st.info("ℹ️ Por favor, cargue archivos desde el panel lateral para comenzar.")
        return
    df = st.session_state.datasets[st.session_state.dataset_seleccionado]
    st.markdown(f"### 📁 Resumen: {st.session_state.dataset_seleccionado}")
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("📊 Filas", f"{len(df):,}")
    with col2:
        st.metric("📋 Columnas", f"{len(df.columns):,}")
    with col3:
        memory_usage = df.memory_usage(deep=True).sum() / (1024**2)
        st.metric("💾 Memoria", f"{memory_usage:.2f} MB")
    with col4:
        hash_md5 = calcular_hash_md5(df)
        st.metric("🔒 Hash MD5", hash_md5[:8])
    st.markdown("---")
    st.markdown("#### 📊 Tipos de Datos")
    tipos_datos = df.dtypes.value_counts()
    col1, col2 = st.columns(2)
    with col1:
        for tipo, cantidad in tipos_datos.items():
            st.write(f"**{tipo}**: {cantidad} columnas")
    with col2:
        fig = px.pie(
            values=tipos_datos.values,
            names=[str(x) for x in tipos_datos.index],
            title="Distribución de Tipos de Datos",
            color_discrete_sequence=px.colors.sequential.Purp
        )
        st.plotly_chart(fig, use_container_width=True)
    st.markdown("---")
    st.markdown("#### 🔍 Vista Previa de Datos")
    col1, col2 = st.columns(2)
    with col1:
        st.markdown("**Primeras 5 filas:**")
        st.dataframe(df.head(), use_container_width=True)
    with col2:
        st.markdown("**Últimas 5 filas:**")
        st.dataframe(df.tail(), use_container_width=True)
    st.markdown("---")
    st.markdown("#### 📋 Información Detallada")
    info_df = pd.DataFrame({
        'Columna': df.columns,
        'Tipo': df.dtypes.values,
        'No Nulos': df.count().values,
        '% Nulos': ((df.isnull().sum() / len(df)) * 100).values,
        'Valores Únicos': [df[col].nunique() for col in df.columns]
    })
    st.dataframe(info_df, use_container_width=True)
    st.markdown("---")
    st.markdown("### 💾 Descargar Dataset con Cálculos")
    col1, col2 = st.columns(2)
    with col1:
        if st.button("📊 Generar Dataset con Métricas Calculadas"):
            with st.spinner("Calculando métricas patrimoniales..."):
                try:
                    df_exportar = df.copy()
                    # Aplicar limpieza de columnas monetarias
                    col_ingreso_sp = 'declaracion_situacionPatrimonial_ingresos_remuneracionAnualCargoPublico_valor'
                    col_otros_ingresos = 'declaracion_situacionPatrimonial_ingresos_otrosIngresosAnualesTotal_valor'
                    if col_ingreso_sp in df_exportar.columns:
                        df_exportar[col_ingreso_sp] = pd.to_numeric(df_exportar[col_ingreso_sp], errors='coerce').fillna(0)
                    if col_otros_ingresos in df_exportar.columns:
                        df_exportar[col_otros_ingresos] = pd.to_numeric(df_exportar[col_otros_ingresos], errors='coerce').fillna(0)
                    # Calcular Total de Ingresos correctamente
                    df_exportar['CALC_Total_Ingresos'] = (
                        df_exportar.get(col_ingreso_sp, 0) + 
                        df_exportar.get(col_otros_ingresos, 0)
                    )
                    st.session_state.df_procesado = df_exportar
                    st.success("✅ Dataset con métricas calculadas generado exitosamente")
                    st.markdown("#### 📊 Nuevas Columnas Agregadas:")
                    nuevas_cols = [col for col in df_exportar.columns if col.startswith('CALC_')]
                    for col in nuevas_cols:
                        st.write(f"✓ **{col}**")
                except Exception as e:
                    st.error(f"Error generando dataset: {e}")
    with col2:
        if st.session_state.df_procesado is not None:
            try:
                output = BytesIO()
                df_para_excel = preparar_para_excel(st.session_state.df_procesado)
                with pd.ExcelWriter(output, engine='openpyxl') as writer:
                    df_para_excel.to_excel(writer, sheet_name='Datos con Calculos', index=False)
                output.seek(0)
                st.download_button(
                    label="⬇️ Descargar Dataset Procesado (Excel)",
                    data=output,
                    file_name=f"dataset_procesado_{datetime.now().strftime('%Y%m%d_%H%M%S')}.xlsx",
                    mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
                )
            except Exception as e:
                st.error(f"Error exportando: {e}")
        else:
            st.info("Primero genere el dataset con métricas usando el botón de la izquierda")

def seccion_eda():
    if not st.session_state.datasets:
        st.warning("⚠️ No hay datasets cargados. Por favor, cargue archivos en la sección Inicio.")
        return
    df = st.session_state.datasets[st.session_state.dataset_seleccionado]
    st.markdown("# 🔍 EDA - Análisis Exploratorio de Datos")
    tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
        "📊 Información General",
        "🔢 Valores Únicos",
        "❌ Análisis de Nulos",
        "📈 Distribuciones",
        "🔗 Correlaciones",
        "📊 Cargos e Ingresos"
    ])
    with tab1:
        st.markdown("### 📊 Información General del Dataset")
        col1, col2, col3, col4, col5 = st.columns(5)
        with col1:
            st.metric("📊 Total Registros", f"{len(df):,}")
        with col2:
            st.metric("📋 Total Columnas", f"{len(df.columns):,}")
        with col3:
            duplicados = df.duplicated().sum()
            st.metric("🔄 Duplicados", f"{duplicados:,}")
        with col4:
            memoria = df.memory_usage(deep=True).sum() / (1024**2)
            st.metric("💾 Memoria Total", f"{memoria:.2f} MB")
        with col5:
            nulos_totales = df.isnull().sum().sum()
            st.metric("❌ Nulos Totales", f"{nulos_totales:,}")
        st.markdown("---")
        st.markdown("### 📋 Resumen Estadístico")
        col1, col2 = st.columns(2)
        with col1:
            st.markdown("#### Columnas Numéricas")
            numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
            st.write(f"**Total:** {len(numeric_cols)} columnas")
            if numeric_cols:
                st.dataframe(df[numeric_cols].describe(), use_container_width=True)
        with col2:
            st.markdown("#### Columnas Categóricas")
            cat_cols = df.select_dtypes(include=['object']).columns.tolist()
            st.write(f"**Total:** {len(cat_cols)} columnas")
            if cat_cols:
                cat_summary = pd.DataFrame({
                    'Columna': cat_cols[:10],
                    'Valores Únicos': [df[col].nunique() for col in cat_cols[:10]],
                    'Valor Más Frecuente': [df[col].mode()[0] if len(df[col].mode()) > 0 else None for col in cat_cols[:10]]
                })
                st.dataframe(cat_summary, use_container_width=True)
        st.markdown("---")
        st.markdown("### 🎯 Información Detallada por Columna")
        try:
            memoria_por_col = df.memory_usage(deep=True)
            info_detallada = pd.DataFrame({
                'Columna': df.columns,
                'Tipo de Dato': df.dtypes.astype(str).values,
                'Valores No Nulos': df.count().values,
                'Valores Nulos': df.isnull().sum().values,
                '% Nulos': ((df.isnull().sum() / len(df)) * 100).round(2).values,
                'Valores Únicos': [df[col].nunique() for col in df.columns],
                'Memoria (KB)': (memoria_por_col[1:] / 1024).round(2).values
            })
            st.dataframe(info_detallada, use_container_width=True)
        except Exception as e:
            info_simple = pd.DataFrame({
                'Columna': df.columns,
                'Tipo de Dato': df.dtypes.astype(str).values,
                'Valores No Nulos': df.count().values,
                'Valores Únicos': [df[col].nunique() for col in df.columns]
            })
            st.dataframe(info_simple, use_container_width=True)
    with tab2:
        st.markdown("### 🔢 Análisis de Valores Únicos")
        columna_seleccionada = st.selectbox(
            "Seleccione una columna para analizar:",
            df.columns.tolist(),
            key="valores_unicos_col"
        )
        valores_unicos = df[columna_seleccionada].value_counts()
        total_unicos = df[columna_seleccionada].nunique()
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("🎯 Valores Únicos", f"{total_unicos:,}")
        with col2:
            st.metric("📊 Total Registros", f"{len(df):,}")
        with col3:
            porcentaje_unicos = (total_unicos / len(df)) * 100
            st.metric("📈 % Unicidad", f"{porcentaje_unicos:.2f}%")
        st.markdown("---")
        col1, col2 = st.columns(2)
        with col1:
            st.markdown("#### 📊 Tabla de Frecuencias (Top 50)")
            frecuencias_df = pd.DataFrame({
                'Valor': valores_unicos.index,
                'Frecuencia': valores_unicos.values,
                '% del Total': ((valores_unicos.values / len(df)) * 100).round(2)
            })
            st.dataframe(frecuencias_df.head(50), use_container_width=True)
        with col2:
            st.markdown("#### 📈 Gráfico de Distribución")
            if total_unicos <= 50:
                fig = px.bar(
                    x=valores_unicos.index[:20],
                    y=valores_unicos.values[:20],
                    title=f"Top 20 valores más frecuentes",
                    labels={'x': columna_seleccionada, 'y': 'Frecuencia'},
                    color=valores_unicos.values[:20],
                    color_continuous_scale='Purples'
                )
                st.plotly_chart(fig, use_container_width=True)
            else:
                fig = px.histogram(
                    df,
                    x=columna_seleccionada,
                    title=f"Distribución de {columna_seleccionada}",
                    color_discrete_sequence=['#667eea']
                )
                st.plotly_chart(fig, use_container_width=True)
        st.markdown("---")
        if df[columna_seleccionada].dtype == 'object' and WORDCLOUD_AVAILABLE:
            st.markdown("#### ☁️ Nube de Palabras")
            try:
                texto = ' '.join(df[columna_seleccionada].dropna().astype(str).tolist())
                if len(texto) > 0:
                    wordcloud = WordCloud(
                        width=800,
                        height=400,
                        background_color='white',
                        colormap='twilight',
                        max_words=100
                    ).generate(texto)
                    fig, ax = plt.subplots(figsize=(12, 6))
                    ax.imshow(wordcloud, interpolation='bilinear')
                    ax.axis('off')
                    ax.set_title(f'Nube de Palabras: {columna_seleccionada}', fontsize=16, pad=20)
                    st.pyplot(fig)
            except Exception as e:
                st.warning(f"No se pudo generar la nube de palabras: {e}")
    with tab3:
        st.markdown("### ❌ Análisis de Valores Nulos")
        nulos_por_columna = df.isnull().sum()
        porcentaje_nulos = (nulos_por_columna / len(df)) * 100
        nulos_df = pd.DataFrame({
            'Columna': df.columns,
            'Valores Nulos': nulos_por_columna.values,
            '% Nulos': porcentaje_nulos.values
        }).sort_values('Valores Nulos', ascending=False)
        col1, col2 = st.columns(2)
        with col1:
            st.markdown("#### 📊 Estadísticas de Nulos")
            columnas_con_nulos = (nulos_df['Valores Nulos'] > 0).sum()
            st.metric("📋 Columnas con Nulos", columnas_con_nulos)
            st.metric("❌ Total de Nulos", f"{nulos_df['Valores Nulos'].sum():,}")
            st.metric("📈 % Promedio de Nulos", f"{nulos_df['% Nulos'].mean():.2f}%")
        with col2:
            st.markdown("#### 🎯 Top 10 Columnas con Más Nulos")
            fig = px.bar(
                nulos_df.head(10),
                x='% Nulos',
                y='Columna',
                orientation='h',
                title="Porcentaje de Nulos por Columna",
                color='% Nulos',
                color_continuous_scale='Reds'
            )
            st.plotly_chart(fig, use_container_width=True)
        st.markdown("---")
        st.markdown("#### 📋 Tabla Completa de Nulos")
        st.dataframe(nulos_df, use_container_width=True)
        st.markdown("---")
        st.markdown("#### 🔥 Heatmap de Nulos")
        if len(df.columns) <= 50:
            nulos_matrix = df.isnull().astype(int)
            fig, ax = plt.subplots(figsize=(15, 8))
            sns.heatmap(
                nulos_matrix,
                cmap='RdPu',
                cbar_kws={'label': 'Nulo (1) / No Nulo (0)'},
                ax=ax,
                yticklabels=False
            )
            ax.set_title('Mapa de Calor de Valores Nulos', fontsize=16, pad=20)
            st.pyplot(fig)
        else:
            st.info("Demasiadas columnas para mostrar heatmap completo. Mostrando muestra aleatoria de 50 columnas.")
            sample_cols = np.random.choice(df.columns, min(50, len(df.columns)), replace=False)
            nulos_matrix = df[sample_cols].isnull().astype(int)
            fig, ax = plt.subplots(figsize=(15, 8))
            sns.heatmap(
                nulos_matrix,
                cmap='RdPu',
                cbar_kws={'label': 'Nulo (1) / No Nulo (0)'},
                ax=ax,
                yticklabels=False
            )
            ax.set_title('Mapa de Calor de Valores Nulos (Muestra)', fontsize=16, pad=20)
            st.pyplot(fig)
    with tab4:
        st.markdown("### 📈 Análisis de Distribuciones")
        all_cols = df.columns.tolist()
        columna_distribucion = st.selectbox(
            "Seleccione una columna:",
            all_cols,
            key="dist_col"
        )
        if df[columna_distribucion].dtype in [np.number, 'int64', 'float64']:
            datos_validos = pd.to_numeric(df[columna_distribucion], errors='coerce').dropna()
            if len(datos_validos) == 0:
                st.warning("⚠️ No hay datos numéricos válidos en esta columna.")
            else:
                col1, col2, col3, col4 = st.columns(4)
                with col1:
                    st.metric("📊 Media", f"{datos_validos.mean():.2f}")
                with col2:
                    st.metric("📍 Mediana", f"{datos_validos.median():.2f}")
                with col3:
                    st.metric("📉 Desv. Est.", f"{datos_validos.std():.2f}")
                with col4:
                    moda_val = datos_validos.mode()
                    st.metric("🎯 Moda", f"{moda_val[0] if len(moda_val) > 0 else 'N/A'}")
                st.markdown("---")
                col1, col2 = st.columns(2)
                with col1:
                    st.markdown("#### 📊 Histograma")
                    fig = px.histogram(
                        datos_validos,
                        nbins=50,
                        title=f"Distribución de {columna_distribucion}",
                        color_discrete_sequence=['#667eea'],
                        marginal='box'
                    )
                    st.plotly_chart(fig, use_container_width=True)
                with col2:
                    st.markdown("#### 📦 Box Plot")
                    fig = px.box(
                        y=datos_validos,
                        title=f"Box Plot de {columna_distribucion}",
                        color_discrete_sequence=['#764ba2']
                    )
                    st.plotly_chart(fig, use_container_width=True)
                st.markdown("---")
                col1, col2 = st.columns(2)
                with col1:
                    st.markdown("#### 📈 Gráfico Q-Q")
                    try:
                        fig, ax = plt.subplots(figsize=(8, 6))
                        stats.probplot(datos_validos, dist="norm", plot=ax)
                        ax.set_title(f"Q-Q Plot: {columna_distribucion}")
                        ax.grid(True, alpha=0.3)
                        st.pyplot(fig)
                    except Exception as e:
                        st.warning(f"No se pudo generar Q-Q plot: {e}")
                with col2:
                    st.markdown("#### 📊 Estadísticas Detalladas")
                    try:
                        stats_df = pd.DataFrame({
                            'Métrica': ['Mínimo', 'Q1 (25%)', 'Q2 (50%)', 'Q3 (75%)', 'Máximo', 'Rango', 'IQR', 'Asimetría', 'Curtosis'],
                            'Valor': [
                                datos_validos.min(),
                                datos_validos.quantile(0.25),
                                datos_validos.quantile(0.50),
                                datos_validos.quantile(0.75),
                                datos_validos.max(),
                                datos_validos.max() - datos_validos.min(),
                                datos_validos.quantile(0.75) - datos_validos.quantile(0.25),
                                datos_validos.skew(),
                                datos_validos.kurtosis()
                            ]
                        })
                        st.dataframe(stats_df, use_container_width=True)
                    except Exception as e:
                        st.warning(f"Error calculando estadísticas: {e}")
        else:
            st.info("La columna seleccionada no es numérica. Mostrando distribución categórica.")
            valores_unicos = df[columna_distribucion].value_counts().head(20)
            fig = px.bar(
                x=valores_unicos.values,
                y=valores_unicos.index,
                orientation='h',
                title=f"Top 20 valores de {columna_distribucion}",
                labels={'x': 'Frecuencia', 'y': 'Valor'},
                color=valores_unicos.values,
                color_continuous_scale='Purples'
            )
            st.plotly_chart(fig, use_container_width=True)
    with tab5:
        st.markdown("### 🔗 Análisis de Correlaciones")
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        if len(numeric_cols) < 2:
            st.warning("⚠️ Se necesitan al menos 2 columnas numéricas para calcular correlaciones.")
            return
        # Limitar a top 20 por varianza si hay muchas
        if len(numeric_cols) > 20:
            varianzas = df[numeric_cols].var().sort_values(ascending=False)
            numeric_cols = varianzas.head(20).index.tolist()
        metodo_correlacion = st.selectbox(
            "Seleccione el método de correlación:",
            ["Pearson", "Spearman", "Kendall"],
            key="metodo_corr"
        )
        metodo_map = {
            "Pearson": "pearson",
            "Spearman": "spearman",
            "Kendall": "kendall"
        }
        try:
            corr_matrix = df[numeric_cols].corr(method=metodo_map[metodo_correlacion])
            st.markdown("#### 🔥 Matriz de Correlación")
            fig, ax = plt.subplots(figsize=(16, 12))
            sns.heatmap(
                corr_matrix,
                annot=True if len(numeric_cols) <= 15 else False,
                fmt='.2f',
                cmap='twilight',
                center=0,
                square=True,
                linewidths=0.5,
                cbar_kws={"shrink": 0.8},
                ax=ax
            )
            ax.set_title(f'Matriz de Correlación ({metodo_correlacion})', fontsize=16, pad=20)
            plt.xticks(rotation=45, ha='right')
            plt.yticks(rotation=0)
            st.pyplot(fig)
            st.markdown("---")
            st.markdown("#### 🎯 Correlaciones Más Fuertes")
            correlaciones_pares = []
            for i in range(len(corr_matrix.columns)):
                for j in range(i+1, len(corr_matrix.columns)):
                    correlaciones_pares.append({
                        'Variable 1': corr_matrix.columns[i],
                        'Variable 2': corr_matrix.columns[j],
                        'Correlación': corr_matrix.iloc[i, j]
                    })
            corr_df = pd.DataFrame(correlaciones_pares)
            corr_df['Correlación Abs'] = corr_df['Correlación'].abs()
            corr_df = corr_df.sort_values('Correlación Abs', ascending=False)
            col1, col2 = st.columns(2)
            with col1:
                st.markdown("##### 🔴 Top 10 Correlaciones Positivas")
                top_positivas = corr_df[corr_df['Correlación'] > 0].head(10)
                st.dataframe(top_positivas[['Variable 1', 'Variable 2', 'Correlación']], use_container_width=True)
            with col2:
                st.markdown("##### 🔵 Top 10 Correlaciones Negativas")
                top_negativas = corr_df[corr_df['Correlación'] < 0].head(10)
                st.dataframe(top_negativas[['Variable 1', 'Variable 2', 'Correlación']], use_container_width=True)
            st.markdown("---")
            st.markdown("#### 📊 Gráfico de Dispersión Interactivo")
            col1, col2 = st.columns(2)
            with col1:
                var_x = st.selectbox("Variable X:", numeric_cols, key="scatter_x")
            with col2:
                var_y = st.selectbox("Variable Y:", numeric_cols, key="scatter_y", index=min(1, len(numeric_cols)-1))
            fig = px.scatter(
                df,
                x=var_x,
                y=var_y,
                title=f"Dispersión: {var_x} vs {var_y}",
                trendline="ols",
                color_discrete_sequence=['#667eea'],
                opacity=0.6
            )
            st.plotly_chart(fig, use_container_width=True)
        except Exception as e:
            st.error(f"Error calculando correlaciones: {e}")
    with tab6:
        st.markdown("### 📊 Análisis de Cargos e Ingresos")
        st.markdown("""
        <div class='info-box'>
            <p style='margin: 0; color: #667eea; font-weight: 600;'>
            📊 Análisis de relación entre cargos públicos e ingresos declarados
            </p>
        </div>
        """, unsafe_allow_html=True)
        cargo_col = 'declaracion_situacionPatrimonial_datosEmpleoCargoComision_empleoCargoComision'
        col_ingreso_sp = 'declaracion_situacionPatrimonial_ingresos_remuneracionAnualCargoPublico_valor'
        col_otros_ingresos = 'declaracion_situacionPatrimonial_ingresos_otrosIngresosAnualesTotal_valor'
        if cargo_col in df.columns and (col_ingreso_sp in df.columns or col_otros_ingresos in df.columns):
            df_temp = df.copy()
            # Calcular ingresos totales correctamente
            ingreso_sp_vals = pd.to_numeric(df_temp[col_ingreso_sp], errors='coerce').fillna(0) if col_ingreso_sp in df_temp.columns else 0
            otros_ingresos_vals = pd.to_numeric(df_temp[col_otros_ingresos], errors='coerce').fillna(0) if col_otros_ingresos in df_temp.columns else 0
            df_temp['Total_Ingresos'] = ingreso_sp_vals + otros_ingresos_vals
            st.markdown("#### 📊 Top 20 Cargos con Mayores Ingresos Promedio")
            cargos_ingresos = df_temp.groupby(cargo_col)['Total_Ingresos'].agg(['mean', 'count', 'sum']).reset_index()
            cargos_ingresos.columns = ['Cargo', 'Ingreso_Promedio', 'Cantidad', 'Total_Ingresos']
            cargos_ingresos = cargos_ingresos.sort_values('Ingreso_Promedio', ascending=False).head(20)
            fig = px.bar(
                cargos_ingresos,
                x='Ingreso_Promedio',
                y='Cargo',
                orientation='h',
                title='Top 20 Cargos con Mayor Ingreso Promedio',
                color='Ingreso_Promedio',
                color_continuous_scale='Purples',
                hover_data=['Cantidad', 'Total_Ingresos']
            )
            st.plotly_chart(fig, use_container_width=True)
            st.markdown("---")
            st.markdown("#### 📈 Dispersión: Cantidad de Servidores vs Ingreso Promedio por Cargo")
            fig = px.scatter(
                cargos_ingresos,
                x='Cantidad',
                y='Ingreso_Promedio',
                size='Total_Ingresos',
                hover_data=['Cargo'],
                title='Relación entre Cantidad de Servidores e Ingresos por Cargo',
                color='Ingreso_Promedio',
                color_continuous_scale='Viridis',
                labels={'Cantidad': 'Número de Servidores', 'Ingreso_Promedio': 'Ingreso Promedio (MXN)'}
            )
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.warning("⚠️ No se encontraron las columnas necesarias para el análisis de cargos e ingresos.")
def seccion_visualizacion_global():
    if not st.session_state.datasets:
        st.warning("⚠️ No hay datasets cargados. Por favor, cargue archivos en la sección Inicio.")
        return
    df = st.session_state.datasets[st.session_state.dataset_seleccionado]
    st.markdown("# 📈 Visualización Global")
    tipo_grafico = st.selectbox(
        "Seleccione el tipo de gráfico:",
        ["Dispersión", "Líneas", "Barras", "Histograma", "Box Plot", "Pastel", "Violin", "Densidad", "Heatmap"],
        key="tipo_graf_global"
    )
    st.markdown("---")
    if tipo_grafico == "Dispersión":
        col1, col2, col3 = st.columns(3)
        with col1:
            x_col = st.selectbox("Eje X:", df.columns.tolist(), key="scatter_x_global")
        with col2:
            y_col = st.selectbox("Eje Y:", df.columns.tolist(), key="scatter_y_global")
        with col3:
            color_col = st.selectbox("Color (opcional):", [None] + df.columns.tolist(), key="scatter_color_global")
        size_col = st.selectbox("Tamaño (opcional):", [None] + df.columns.tolist(), key="scatter_size_global")
        try:
            fig = px.scatter(
                df,
                x=x_col,
                y=y_col,
                color=color_col,
                size=size_col,
                title=f"Gráfico de Dispersión: {x_col} vs {y_col}",
                color_continuous_scale='Purples' if color_col and pd.api.types.is_numeric_dtype(df[color_col]) else None,
                opacity=0.7
            )
            st.plotly_chart(fig, use_container_width=True)
        except Exception as e:
            st.error(f"Error generando gráfico: {e}")
    elif tipo_grafico == "Líneas":
        col1, col2, col3 = st.columns(3)
        with col1:
            x_col = st.selectbox("Eje X:", df.columns.tolist(), key="line_x_global")
        with col2:
            y_col = st.selectbox("Eje Y:", df.columns.tolist(), key="line_y_global")
        with col3:
            color_col = st.selectbox("Agrupar por (opcional):", [None] + df.columns.tolist(), key="line_color_global")
        try:
            fig = px.line(
                df,
                x=x_col,
                y=y_col,
                color=color_col,
                title=f"Gráfico de Líneas: {y_col} por {x_col}",
                markers=True
            )
            st.plotly_chart(fig, use_container_width=True)
        except Exception as e:
            st.error(f"Error generando gráfico: {e}")
    elif tipo_grafico == "Barras":
        col1, col2, col3 = st.columns(3)
        with col1:
            x_col = st.selectbox("Eje X:", df.columns.tolist(), key="bar_x_global")
        with col2:
            y_col = st.selectbox("Eje Y:", df.columns.tolist(), key="bar_y_global")
        with col3:
            orientacion = st.radio("Orientación:", ["Vertical", "Horizontal"], key="bar_orient_global")
        color_col = st.selectbox("Color por (opcional):", [None] + df.columns.tolist(), key="bar_color_global")
        try:
            fig = px.bar(
                df,
                x=x_col if orientacion == "Vertical" else y_col,
                y=y_col if orientacion == "Vertical" else x_col,
                color=color_col,
                orientation='v' if orientacion == "Vertical" else 'h',
                title=f"Gráfico de Barras: {y_col} por {x_col}",
                color_discrete_sequence=px.colors.sequential.Purp
            )
            st.plotly_chart(fig, use_container_width=True)
        except Exception as e:
            st.error(f"Error generando gráfico: {e}")
    elif tipo_grafico == "Histograma":
        col1, col2 = st.columns(2)
        with col1:
            x_col = st.selectbox("Variable:", df.columns.tolist(), key="hist_x_global")
        with col2:
            bins = st.slider("Número de bins:", 10, 100, 30, key="hist_bins_global")
        color_col = st.selectbox("Color por (opcional):", [None] + df.columns.tolist(), key="hist_color_global")
        marginal = st.selectbox("Marginal:", [None, "rug", "box", "violin"], key="hist_marginal_global")
        try:
            fig = px.histogram(
                df,
                x=x_col,
                nbins=bins,
                color=color_col,
                marginal=marginal,
                title=f"Histograma: {x_col}",
                color_discrete_sequence=px.colors.sequential.Purp
            )
            st.plotly_chart(fig, use_container_width=True)
        except Exception as e:
            st.error(f"Error generando gráfico: {e}")
    elif tipo_grafico == "Box Plot":
        col1, col2 = st.columns(2)
        with col1:
            y_col = st.selectbox("Variable numérica:", df.columns.tolist(), key="box_y_global")
        with col2:
            x_col = st.selectbox("Agrupar por (opcional):", [None] + df.columns.tolist(), key="box_x_global")
        color_col = st.selectbox("Color por (opcional):", [None] + df.columns.tolist(), key="box_color_global")
        try:
            fig = px.box(
                df,
                y=y_col,
                x=x_col,
                color=color_col,
                title=f"Box Plot: {y_col}",
                color_discrete_sequence=px.colors.sequential.Purp
            )
            st.plotly_chart(fig, use_container_width=True)
        except Exception as e:
            st.error(f"Error generando gráfico: {e}")
    elif tipo_grafico == "Pastel":
        col1, col2 = st.columns(2)
        with col1:
            values_col = st.selectbox("Valores:", df.columns.tolist(), key="pie_values_global")
        with col2:
            names_col = st.selectbox("Etiquetas:", df.columns.tolist(), key="pie_names_global")
        try:
            df_grouped = df.groupby(names_col)[values_col].sum().reset_index()
            fig = px.pie(
                df_grouped,
                values=values_col,
                names=names_col,
                title=f"Gráfico de Pastel: {values_col} por {names_col}",
                color_discrete_sequence=px.colors.sequential.Purp
            )
            st.plotly_chart(fig, use_container_width=True)
        except Exception as e:
            st.error(f"Error generando gráfico: {e}")
    elif tipo_grafico == "Violin":
        col1, col2 = st.columns(2)
        with col1:
            y_col = st.selectbox("Variable numérica:", df.columns.tolist(), key="violin_y_global")
        with col2:
            x_col = st.selectbox("Agrupar por (opcional):", [None] + df.columns.tolist(), key="violin_x_global")
        try:
            fig = px.violin(
                df,
                y=y_col,
                x=x_col,
                title=f"Violin Plot: {y_col}",
                color_discrete_sequence=['#667eea'],
                box=True,
                points="all"
            )
            st.plotly_chart(fig, use_container_width=True)
        except Exception as e:
            st.error(f"Error generando gráfico: {e}")
    elif tipo_grafico == "Densidad":
        col1, col2 = st.columns(2)
        with col1:
            x_col = st.selectbox("Eje X:", df.columns.tolist(), key="density_x_global")
        with col2:
            y_col = st.selectbox("Eje Y:", df.columns.tolist(), key="density_y_global")
        try:
            fig = px.density_contour(
                df,
                x=x_col,
                y=y_col,
                title=f"Gráfico de Densidad: {x_col} vs {y_col}",
                color_discrete_sequence=['#667eea']
            )
            st.plotly_chart(fig, use_container_width=True)
        except Exception as e:
            st.error(f"Error generando gráfico: {e}")
    elif tipo_grafico == "Heatmap":
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        if len(numeric_cols) < 2:
            st.warning("⚠️ Se necesitan al menos 2 columnas numéricas para un heatmap.")
        else:
            cols_seleccionadas = st.multiselect(
                "Seleccione las columnas para el heatmap:",
                numeric_cols,
                default=numeric_cols[:min(10, len(numeric_cols))],
                key="heatmap_cols_global"
            )
            if cols_seleccionadas:
                try:
                    corr_matrix = df[cols_seleccionadas].corr()
                    fig = px.imshow(
                        corr_matrix,
                        title="Mapa de Calor de Correlaciones",
                        color_continuous_scale='RdBu_r',
                        aspect="auto",
                        text_auto=True
                    )
                    st.plotly_chart(fig, use_container_width=True)
                except Exception as e:
                    st.error(f"Error generando heatmap: {e}")
def seccion_visualizacion_personalizada():
    if not st.session_state.datasets:
        st.warning("⚠️ No hay datasets cargados.")
        return
    df = st.session_state.datasets[st.session_state.dataset_seleccionado]
    st.markdown("# 🎨 Visualización Personalizada")
    st.markdown("### ⚙️ Configuración de Gráficos")
    col1, col2, col3 = st.columns(3)
    with col1:
        num_graficos = st.slider("Número de gráficos:", 1, 6, 2, key="num_graphs")
    with col2:
        layout_tipo = st.selectbox(
            "Tipo de layout:",
            ["Cuadrícula", "Vertical", "Horizontal"],
            key="layout_type"
        )
    with col3:
        tema = st.selectbox(
            "Tema del gráfico:",
            ["plotly", "plotly_white", "plotly_dark", "ggplot2", "seaborn", "simple_white"],
            key="theme_select"
        )
    st.markdown("---")
    if layout_tipo == "Cuadrícula":
        if num_graficos <= 2:
            rows, cols = 1, num_graficos
        elif num_graficos <= 4:
            rows, cols = 2, 2
        else:
            rows, cols = 2, 3
    elif layout_tipo == "Vertical":
        rows, cols = num_graficos, 1
    else:
        rows, cols = 1, num_graficos
    fig = make_subplots(
        rows=rows,
        cols=cols,
        subplot_titles=[f"Gráfico {i+1}" for i in range(num_graficos)],
        vertical_spacing=0.12,
        horizontal_spacing=0.1
    )
    for i in range(num_graficos):
        with st.expander(f"⚙️ Configurar Gráfico {i+1}", expanded=i==0):
            col1, col2, col3 = st.columns(3)
            with col1:
                tipo_graf = st.selectbox(
                    "Tipo de gráfico:",
                    ["Scatter", "Bar", "Line", "Box", "Histogram", "Violin"],
                    key=f"tipo_{i}"
                )
            with col2:
                x_col = st.selectbox(
                    "Eje X:",
                    df.columns.tolist(),
                    key=f"x_{i}"
                )
            with col3:
                y_col = st.selectbox(
                    "Eje Y:",
                    df.columns.tolist(),
                    key=f"y_{i}",
                    index=min(1, len(df.columns)-1)
                )
            color_col = st.selectbox(
                "Color por (opcional):",
                [None] + df.columns.tolist(),
                key=f"color_{i}"
            )
            row_pos = (i // cols) + 1
            col_pos = (i % cols) + 1
            try:
                if tipo_graf == "Scatter":
                    scatter_data = df[[x_col, y_col]].dropna()
                    if color_col:
                        colors = df.loc[scatter_data.index, color_col]
                        fig.add_trace(
                            go.Scatter(
                                x=scatter_data[x_col],
                                y=scatter_data[y_col],
                                mode='markers',
                                name=f"{y_col} vs {x_col}",
                                marker=dict(
                                    color=colors if pd.api.types.is_numeric_dtype(colors) else None,
                                    size=8,
                                    opacity=0.6,
                                    colorscale='Purples'
                                )
                            ),
                            row=row_pos,
                            col=col_pos
                        )
                    else:
                        fig.add_trace(
                            go.Scatter(
                                x=scatter_data[x_col],
                                y=scatter_data[y_col],
                                mode='markers',
                                name=f"{y_col} vs {x_col}",
                                marker=dict(color='#667eea', size=8, opacity=0.6)
                            ),
                            row=row_pos,
                            col=col_pos
                        )
                elif tipo_graf == "Bar":
                    if df[x_col].dtype == 'object' or df[x_col].nunique() < 50:
                        df_grouped = df.groupby(x_col)[y_col].sum().reset_index()
                        fig.add_trace(
                            go.Bar(
                                x=df_grouped[x_col],
                                y=df_grouped[y_col],
                                name=f"{y_col} por {x_col}",
                                marker=dict(color='#764ba2')
                            ),
                            row=row_pos,
                            col=col_pos
                        )
                    else:
                        st.warning(f"Gráfico {i+1}: Demasiados valores únicos en X para gráfico de barras")
                elif tipo_graf == "Line":
                    line_data = df[[x_col, y_col]].dropna().sort_values(x_col)
                    fig.add_trace(
                        go.Scatter(
                            x=line_data[x_col],
                            y=line_data[y_col],
                            mode='lines+markers',
                            name=f"{y_col} vs {x_col}",
                            line=dict(color='#667eea', width=2)
                        ),
                        row=row_pos,
                        col=col_pos
                    )
                elif tipo_graf == "Box":
                    box_data = pd.to_numeric(df[y_col], errors='coerce').dropna()
                    fig.add_trace(
                        go.Box(
                            y=box_data,
                            name=y_col,
                            marker=dict(color='#764ba2')
                        ),
                        row=row_pos,
                        col=col_pos
                    )
                elif tipo_graf == "Histogram":
                    hist_data = pd.to_numeric(df[x_col], errors='coerce').dropna()
                    fig.add_trace(
                        go.Histogram(
                            x=hist_data,
                            name=x_col,
                            marker=dict(color='#667eea')
                        ),
                        row=row_pos,
                        col=col_pos
                    )
                elif tipo_graf == "Violin":
                    violin_data = pd.to_numeric(df[y_col], errors='coerce').dropna()
                    fig.add_trace(
                        go.Violin(
                            y=violin_data,
                            name=y_col,
                            marker=dict(color='#764ba2'),
                            box_visible=True,
                            meanline_visible=True
                        ),
                        row=row_pos,
                        col=col_pos
                    )
            except Exception as e:
                st.warning(f"Error en gráfico {i+1}: {e}")
    fig.update_layout(
        height=400 * rows,
        showlegend=True,
        title_text="Dashboard Personalizado",
        title_font_size=24,
        template=tema
    )
    st.plotly_chart(fig, use_container_width=True)
def seccion_machine_learning():
    if not st.session_state.datasets:
        st.warning("⚠️ No hay datasets cargados.")
        return
    df = st.session_state.datasets[st.session_state.dataset_seleccionado]
    st.markdown("# 🤖 Machine Learning")
    tab1, tab2, tab3 = st.tabs([
        "🔧 Preparación de Datos",
        "🎯 Entrenamiento de Modelos",
        "📊 Resultados y Evaluación"
    ])
    with tab1:
        st.markdown("### 🔧 Preparación y Codificación de Datos")
        st.markdown("""
        <div class='info-box'>
            <h4 style='color: #667eea; margin-top: 0;'>📌 Codificación de Variables</h4>
            <p>Los algoritmos de Machine Learning requieren datos numéricos. Esta sección convierte 
            las variables categóricas (texto) en valores numéricos usando LabelEncoder.</p>
        </div>
        """, unsafe_allow_html=True)
        if st.button("🔄 Codificar Variables Categóricas", key="encode_button"):
            with st.spinner("Codificando variables..."):
                try:
                    df_encoded = df.copy()
                    encoding_dicts = {}
                    categorical_cols = df.select_dtypes(include=['object']).columns.tolist()
                    progress_bar = st.progress(0)
                    total_cols = len(categorical_cols)
                    columnas_codificadas = 0
                    columnas_omitidas = 0
                    for idx, col in enumerate(categorical_cols):
                        progress_bar.progress((idx + 1) / total_cols)
                        unique_values = df[col].nunique()
                        if unique_values < 100 and unique_values > 1:
                            le = LabelEncoder()
                            valid_data = df[col].dropna()
                            if len(valid_data) > 0:
                                le.fit(valid_data)
                                df_encoded[col] = df[col].map(
                                    lambda x: le.transform([x])[0] if pd.notna(x) else np.nan
                                )
                                encoding_dicts[col] = dict(zip(le.classes_, le.transform(le.classes_)))
                                columnas_codificadas += 1
                        else:
                            columnas_omitidas += 1
                    st.session_state.df_encoded = df_encoded
                    st.session_state.encoding_dicts = encoding_dicts
                    progress_bar.progress(1.0)
                    st.success(f"✅ Codificación completada: {columnas_codificadas} columnas codificadas, {columnas_omitidas} omitidas (>100 valores únicos)")
                except Exception as e:
                    st.error(f"Error en la codificación: {e}")
        if st.session_state.df_encoded is not None:
            st.markdown("---")
            st.markdown("### 📊 Vista Previa del Dataset Codificado")
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("📋 Columnas Codificadas", len(st.session_state.encoding_dicts))
            with col2:
                st.metric("📊 Total Columnas", len(st.session_state.df_encoded.columns))
            with col3:
                numeric_cols = st.session_state.df_encoded.select_dtypes(include=[np.number]).columns
                st.metric("🔢 Columnas Numéricas", len(numeric_cols))
            st.dataframe(st.session_state.df_encoded.head(10), use_container_width=True)
            st.markdown("---")
            st.markdown("### 📖 Diccionario de Codificación")
            if st.session_state.encoding_dicts:
                columna_dict = st.selectbox(
                    "Seleccione una columna para ver su diccionario:",
                    list(st.session_state.encoding_dicts.keys()),
                    key="dict_select_ml"
                )
                if columna_dict:
                    st.markdown(f"#### Codificación de: {columna_dict}")
                    dict_df = pd.DataFrame([
                        {"Valor Original": k, "Código": v}
                        for k, v in st.session_state.encoding_dicts[columna_dict].items()
                    ])
                    st.dataframe(dict_df, use_container_width=True)
            st.markdown("---")
            if st.button("💾 Exportar Dataset Codificado", key="export_encoded"):
                try:
                    output = BytesIO()
                    df_para_excel = preparar_para_excel(st.session_state.df_encoded)
                    with pd.ExcelWriter(output, engine='openpyxl') as writer:
                        df_para_excel.to_excel(writer, sheet_name='Datos Codificados', index=False)
                        dict_rows = []
                        for col, mappings in st.session_state.encoding_dicts.items():
                            for original, codigo in mappings.items():
                                dict_rows.append({
                                    'Columna': col,
                                    'Valor Original': original,
                                    'Código': codigo
                                })
                        if dict_rows:
                            dict_df = pd.DataFrame(dict_rows)
                            dict_df.to_excel(writer, sheet_name='Diccionario', index=False)
                    output.seek(0)
                    st.download_button(
                        label="⬇️ Descargar Excel con Datos Codificados",
                        data=output,
                        file_name=f"dataset_codificado_{datetime.now().strftime('%Y%m%d_%H%M%S')}.xlsx",
                        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
                    )
                except Exception as e:
                    st.error(f"Error exportando: {e}")
    with tab2:
        st.markdown("### 🎯 Configuración y Entrenamiento de Modelos")
        datos_disponibles = st.radio(
            "Seleccione el dataset a utilizar:",
            ["Datos Originales", "Datos Codificados", "Servidores Públicos Únicos"],
            key="data_source_ml"
        )
        if datos_disponibles == "Datos Codificados":
            if st.session_state.df_encoded is None:
                st.warning("⚠️ Primero debe codificar las variables en la pestaña 'Preparación de Datos'")
                return
            df_ml = st.session_state.df_encoded
        elif datos_disponibles == "Servidores Públicos Únicos":
            if st.session_state.df_servidores_unicos is None:
                st.warning("⚠️ Primero debe generar la tabla de Servidores Públicos Únicos en la sección correspondiente")
                return
            df_ml = st.session_state.df_servidores_unicos
        else:
            df_ml = df
        st.markdown("---")
        tipo_modelo = st.selectbox(
            "Seleccione el tipo de modelo:",
            ["Clasificación", "Regresión", "Clustering"],
            key="model_type"
        )
        st.markdown("---")
        numeric_cols = df_ml.select_dtypes(include=[np.number]).columns.tolist()
        if len(numeric_cols) < 2:
            st.error("❌ Se necesitan al menos 2 columnas numéricas para entrenar modelos.")
            return
        st.markdown("### 📊 Selección de Variables")
        features = st.multiselect(
            "Seleccione las variables independientes (Features):",
            numeric_cols,
            default=numeric_cols[:min(5, len(numeric_cols))],
            key="features_select"
        )
        if tipo_modelo != "Clustering":
            target_options = [col for col in numeric_cols if col not in features]
            if not target_options:
                st.error("❌ Debe seleccionar al menos una variable como feature para tener opciones de target.")
                return
            target = st.selectbox(
                "Seleccione la variable objetivo (Target):",
                target_options,
                key="target_select"
            )
        if not features:
            st.warning("⚠️ Debe seleccionar al menos una variable independiente.")
            return
        st.markdown("---")
        if tipo_modelo == "Clasificación":
            if st.button("🚀 Entrenar TODOS los Modelos de Clasificación", key="train_all_clf"):
                with st.spinner("Entrenando todos los modelos..."):
                    try:
                        X = df_ml[features].dropna()
                        y = df_ml.loc[X.index, target].dropna()
                        common_index = X.index.intersection(y.index)
                        X = X.loc[common_index]
                        y = y.loc[common_index]
                        if len(np.unique(y)) < 2:
                            st.error("❌ La variable objetivo debe tener al menos 2 clases diferentes.")
                            return
                        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
                        scaler = StandardScaler()
                        X_train_scaled = scaler.fit_transform(X_train)
                        X_test_scaled = scaler.transform(X_test)
                        modelos = {
                            "Logistic Regression": LogisticRegression(max_iter=1000, random_state=42),
                            "Decision Tree": DecisionTreeClassifier(random_state=42),
                            "Random Forest": RandomForestClassifier(n_estimators=100, random_state=42),
                            "SVM": SVC(kernel='rbf', random_state=42, probability=True),
                            "K-Nearest Neighbors": KNeighborsClassifier(n_neighbors=5)
                        }
                        resultados = []
                        modelos_entrenados = {}
                        for nombre, modelo in modelos.items():
                            modelo.fit(X_train_scaled, y_train)
                            y_pred = modelo.predict(X_test_scaled)
                            accuracy = accuracy_score(y_test, y_pred)
                            precision = precision_score(y_test, y_pred, average='weighted', zero_division=0)
                            recall = recall_score(y_test, y_pred, average='weighted', zero_division=0)
                            f1 = f1_score(y_test, y_pred, average='weighted', zero_division=0)
                            resultados.append({
                                'Modelo': nombre,
                                'Accuracy': accuracy,
                                'Precision': precision,
                                'Recall': recall,
                                'F1-Score': f1
                            })
                            modelos_entrenados[nombre] = {
                                'model': modelo,
                                'scaler': scaler,
                                'X_test': X_test_scaled,
                                'y_test': y_test,
                                'y_pred': y_pred,
                                'features': features,
                                'target': target,
                                'type': 'classification',
                                'algorithm': nombre
                            }
                        st.session_state.ml_models['all_models'] = modelos_entrenados
                        resultados_df = pd.DataFrame(resultados)
                        st.session_state.ml_models['resultados_clf'] = resultados_df
                        st.success("✅ Todos los modelos de clasificación entrenados exitosamente")
                        st.dataframe(resultados_df, use_container_width=True)
                        # Botón para descargar resultados
                        output = BytesIO()
                        df_para_excel = preparar_para_excel(resultados_df)
                        with pd.ExcelWriter(output, engine='openpyxl') as writer:
                            df_para_excel.to_excel(writer, sheet_name='Resultados Clasificación', index=False)
                        output.seek(0)
                        st.download_button(
                            label="⬇️ Descargar Resultados Clasificación (Excel)",
                            data=output,
                            file_name=f"resultados_clasificacion_{datetime.now().strftime('%Y%m%d_%H%M%S')}.xlsx",
                            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
                        )
                    except Exception as e:
                        st.error(f"Error entrenando modelos: {e}")
        elif tipo_modelo == "Regresión":
            st.markdown("### 📈 Algoritmos de Regresión")
            algoritmo = st.selectbox(
                "Seleccione el algoritmo:",
                ["Linear Regression", "Ridge", "Lasso", "ElasticNet", "Random Forest Regressor"],
                key="reg_algo"
            )
            if st.button("🚀 Entrenar Modelo de Regresión", key="train_reg"):
                with st.spinner("Entrenando modelo..."):
                    try:
                        X = df_ml[features].dropna()
                        y = df_ml.loc[X.index, target].dropna()
                        common_index = X.index.intersection(y.index)
                        X = X.loc[common_index]
                        y = y.loc[common_index]
                        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
                        scaler = StandardScaler()
                        X_train_scaled = scaler.fit_transform(X_train)
                        X_test_scaled = scaler.transform(X_test)
                        if algoritmo == "Linear Regression":
                            model = LinearRegression()
                        elif algoritmo == "Ridge":
                            model = Ridge(alpha=1.0, random_state=42)
                        elif algoritmo == "Lasso":
                            model = Lasso(alpha=1.0, random_state=42)
                        elif algoritmo == "ElasticNet":
                            model = ElasticNet(alpha=1.0, random_state=42)
                        elif algoritmo == "Random Forest Regressor":
                            model = RandomForestRegressor(n_estimators=100, random_state=42)
                        model.fit(X_train_scaled, y_train)
                        y_pred = model.predict(X_test_scaled)
                        st.session_state.ml_models['last_model'] = {
                            'model': model,
                            'scaler': scaler,
                            'X_test': X_test_scaled,
                            'y_test': y_test,
                            'y_pred': y_pred,
                            'features': features,
                            'target': target,
                            'type': 'regression',
                            'algorithm': algoritmo
                        }
                        st.success(f"✅ Modelo {algoritmo} entrenado exitosamente")
                    except Exception as e:
                        st.error(f"Error entrenando modelo: {e}")
        elif tipo_modelo == "Clustering":
            st.markdown("### 🔍 Algoritmos de Clustering")
            algoritmo = st.selectbox(
                "Seleccione el algoritmo:",
                ["K-Means", "DBSCAN", "Agglomerative Clustering"],
                key="cluster_algo"
            )
            if algoritmo == "K-Means":
                # Método del codo
                st.markdown("#### 📏 Método del Codo para seleccionar k")
                if st.button("📊 Calcular Método del Codo"):
                    try:
                        X = df_ml[features].dropna()
                        scaler = StandardScaler()
                        X_scaled = scaler.fit_transform(X)
                        k_range = range(1, 11)
                        inertias = []
                        for k in k_range:
                            kmeans = KMeans(n_clusters=k, random_state=42)
                            kmeans.fit(X_scaled)
                            inertias.append(kmeans.inertia_)
                        fig = px.line(
                            x=list(k_range),
                            y=inertias,
                            title='Método del Codo',
                            labels={'x': 'Número de clusters (k)', 'y': 'Inercia'},
                            markers=True
                        )
                        st.plotly_chart(fig, use_container_width=True)
                        st.session_state.ml_models['codo_data'] = {'k': list(k_range), 'inercia': inertias}
                    except Exception as e:
                        st.error(f"Error calculando método del codo: {e}")
                n_clusters = st.slider("Número de clusters:", 2, 10, 3, key="kmeans_clusters")
            elif algoritmo == "DBSCAN":
                eps = st.slider("Epsilon:", 0.1, 5.0, 0.5, key="dbscan_eps")
                min_samples = st.slider("Min samples:", 2, 20, 5, key="dbscan_min")
            elif algoritmo == "Agglomerative Clustering":
                n_clusters = st.slider("Número de clusters:", 2, 10, 3, key="agg_clusters")
            if st.button("🚀 Entrenar Modelo de Clustering", key="train_cluster"):
                with st.spinner("Entrenando modelo..."):
                    try:
                        X = df_ml[features].dropna()
                        scaler = StandardScaler()
                        X_scaled = scaler.fit_transform(X)
                        if algoritmo == "K-Means":
                            model = KMeans(n_clusters=n_clusters, random_state=42)
                        elif algoritmo == "DBSCAN":
                            model = DBSCAN(eps=eps, min_samples=min_samples)
                        elif algoritmo == "Agglomerative Clustering":
                            model = AgglomerativeClustering(n_clusters=n_clusters)
                        clusters = model.fit_predict(X_scaled)
                        # Crear DataFrame con clusters
                        df_result = X.copy()
                        df_result['Cluster'] = clusters
                        st.session_state.ml_models['clustering_result'] = df_result
                        st.session_state.ml_models['last_model'] = {
                            'model': model,
                            'scaler': scaler,
                            'X': X_scaled,
                            'clusters': clusters,
                            'features': features,
                            'type': 'clustering',
                            'algorithm': algoritmo
                        }
                        st.success(f"✅ Modelo {algoritmo} entrenado exitosamente")
                        # Mostrar resultados
                        st.markdown("#### 📊 Resultados del Clustering")
                        cluster_counts = pd.Series(clusters).value_counts().sort_index()
                        st.dataframe(cluster_counts.rename('Cantidad').reset_index().rename(columns={'index': 'Cluster'}), use_container_width=True)
                        # Botón para descargar
                        output = BytesIO()
                        df_para_excel = preparar_para_excel(df_result)
                        with pd.ExcelWriter(output, engine='openpyxl') as writer:
                            df_para_excel.to_excel(writer, sheet_name='Clusters Asignados', index=False)
                        output.seek(0)
                        st.download_button(
                            label="⬇️ Descargar Resultados Clustering (Excel)",
                            data=output,
                            file_name=f"resultados_clustering_{datetime.now().strftime('%Y%m%d_%H%M%S')}.xlsx",
                            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
                        )
                    except Exception as e:
                        st.error(f"Error entrenando modelo: {e}")
    with tab3:
        st.markdown("### 📊 Resultados y Evaluación del Modelo")
        if 'all_models' in st.session_state.ml_models:
            st.markdown("#### 📋 Resultados Comparativos de Clasificación")
            resultados_df = st.session_state.ml_models['resultados_clf']
            st.dataframe(resultados_df, use_container_width=True)
            selected_model = st.selectbox("Seleccione un modelo para ver detalles:", list(st.session_state.ml_models['all_models'].keys()))
            model_data = st.session_state.ml_models['all_models'][selected_model]
        elif 'last_model' in st.session_state.ml_models:
            model_data = st.session_state.ml_models['last_model']
        else:
            st.info("ℹ️ Primero entrene un modelo en la pestaña 'Entrenamiento de Modelos'")
            return
        st.markdown(f"""
        <div class='info-box'>
            <h4 style='color: #667eea; margin-top: 0;'>📌 Modelo Entrenado</h4>
            <p><strong>Tipo:</strong> {model_data['type'].title()}</p>
            <p><strong>Algoritmo:</strong> {model_data['algorithm']}</p>
            <p><strong>Features:</strong> {', '.join(model_data['features'][:5])}{'...' if len(model_data['features']) > 5 else ''}</p>
        </div>
        """, unsafe_allow_html=True)
        st.markdown("---")
        if model_data['type'] == 'classification':
            st.markdown("### 🎯 Métricas de Clasificación")
            y_test = model_data['y_test']
            y_pred = model_data['y_pred']
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                accuracy = accuracy_score(y_test, y_pred)
                st.metric("🎯 Accuracy", f"{accuracy:.4f}")
            with col2:
                precision = precision_score(y_test, y_pred, average='weighted', zero_division=0)
                st.metric("🎪 Precision", f"{precision:.4f}")
            with col3:
                recall = recall_score(y_test, y_pred, average='weighted', zero_division=0)
                st.metric("📡 Recall", f"{recall:.4f}")
            with col4:
                f1 = f1_score(y_test, y_pred, average='weighted', zero_division=0)
                st.metric("⚖️ F1-Score", f"{f1:.4f}")
            st.markdown("---")
            st.markdown("### 📊 Matriz de Confusión")
            cm = confusion_matrix(y_test, y_pred)
            fig, ax = plt.subplots(figsize=(10, 8))
            sns.heatmap(cm, annot=True, fmt='d', cmap='Purples', ax=ax)
            ax.set_title('Matriz de Confusión')
            ax.set_ylabel('Valor Real')
            ax.set_xlabel('Valor Predicho')
            st.pyplot(fig)
            st.markdown("---")
            if hasattr(model_data['model'], 'feature_importances_'):
                st.markdown("### 🎯 Importancia de Features")
                importances = model_data['model'].feature_importances_
                feature_importance_df = pd.DataFrame({
                    'Feature': model_data['features'],
                    'Importancia': importances
                }).sort_values('Importancia', ascending=False)
                fig = px.bar(
                    feature_importance_df,
                    x='Importancia',
                    y='Feature',
                    orientation='h',
                    title='Importancia de Variables',
                    color='Importancia',
                    color_continuous_scale='Purples'
                )
                st.plotly_chart(fig, use_container_width=True)
        elif model_data['type'] == 'regression':
            st.markdown("### 📈 Métricas de Regresión")
            y_test = model_data['y_test']
            y_pred = model_data['y_pred']
            col1, col2, col3 = st.columns(3)
            with col1:
                mse = mean_squared_error(y_test, y_pred)
                st.metric("📊 MSE", f"{mse:.4f}")
            with col2:
                rmse = np.sqrt(mse)
                st.metric("📉 RMSE", f"{rmse:.4f}")
            with col3:
                r2 = r2_score(y_test, y_pred)
                st.metric("📈 R² Score", f"{r2:.4f}")
            st.markdown("---")
            col1, col2 = st.columns(2)
            with col1:
                st.markdown("### 📊 Predicciones vs Real")
                fig = px.scatter(
                    x=y_test,
                    y=y_pred,
                    title='Valores Predichos vs Valores Reales',
                    labels={'x': 'Valores Reales', 'y': 'Valores Predichos'},
                    color_discrete_sequence=['#667eea'],
                    opacity=0.6
                )
                fig.add_trace(
                    go.Scatter(
                        x=[y_test.min(), y_test.max()],
                        y=[y_test.min(), y_test.max()],
                        mode='lines',
                        name='Línea Perfecta',
                        line=dict(color='red', dash='dash')
                    )
                )
                st.plotly_chart(fig, use_container_width=True)
            with col2:
                st.markdown("### 📊 Distribución de Residuos")
                residuos = y_test - y_pred
                fig = px.histogram(
                    residuos,
                    nbins=50,
                    title='Distribución de Residuos',
                    color_discrete_sequence=['#764ba2']
                )
                st.plotly_chart(fig, use_container_width=True)
        elif model_data['type'] == 'clustering':
            st.markdown("### 🔍 Resultados de Clustering")
            clusters = model_data['clusters']
            col1, col2, col3 = st.columns(3)
            with col1:
                n_clusters_found = len(np.unique(clusters))
                st.metric("🎯 Clusters Encontrados", n_clusters_found)
            with col2:
                cluster_counts = pd.Series(clusters).value_counts()
                st.metric("📊 Cluster Más Grande", cluster_counts.max())
            with col3:
                if n_clusters_found > 1:
                    silhouette_avg = silhouette_score(model_data['X'], clusters)
                    st.metric("📈 Silhouette Score", f"{silhouette_avg:.4f}")
            st.markdown("---")
            st.markdown("### 📊 Distribución de Clusters")
            cluster_df = pd.DataFrame({
                'Cluster': clusters,
                'Count': 1
            })
            cluster_summary = cluster_df.groupby('Cluster').count().reset_index()
            fig = px.bar(
                cluster_summary,
                x='Cluster',
                y='Count',
                title='Número de Puntos por Cluster',
                color='Count',
                color_continuous_scale='Purples'
            )
            st.plotly_chart(fig, use_container_width=True)
            st.markdown("---")
            st.markdown("### 🎨 Visualización en 2D (PCA)")
            if model_data['X'].shape[1] > 2:
                pca = PCA(n_components=2)
                X_pca = pca.fit_transform(model_data['X'])
                fig = px.scatter(
                    x=X_pca[:, 0],
                    y=X_pca[:, 1],
                    color=clusters.astype(str),
                    title='Clusters en Espacio PCA',
                    labels={'x': 'PC1', 'y': 'PC2', 'color': 'Cluster'},
                    color_discrete_sequence=px.colors.qualitative.Vivid
                )
                st.plotly_chart(fig, use_container_width=True)
            # Mostrar tabla si existe
            if 'clustering_result' in st.session_state.ml_models:
                st.markdown("### 📋 Tabla con Asignación de Clusters")
                df_result = st.session_state.ml_models['clustering_result']
                st.dataframe(df_result, use_container_width=True)
        st.markdown("---")
        st.markdown("### 💾 Exportar Modelo")
        if st.button("💾 Descargar Modelo Entrenado", key="download_model"):
            try:
                buffer = BytesIO()
                pickle.dump(model_data, buffer)
                buffer.seek(0)
                st.download_button(
                    label="⬇️ Descargar Modelo (.pkl)",
                    data=buffer,
                    file_name=f"modelo_{model_data['algorithm'].replace(' ', '_')}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pkl",
                    mime="application/octet-stream"
                )
            except Exception as e:
                st.error(f"Error exportando modelo: {e}")
def seccion_geo_inteligencia():
    if not st.session_state.datasets:
        st.warning("⚠️ No hay datasets cargados.")
        return
    df = st.session_state.datasets[st.session_state.dataset_seleccionado]
    st.markdown("# 🌍 Geo Inteligencia")
    tab1, tab2, tab3 = st.tabs([
        "🗺️ Mapa de Riesgo",
        "📊 Análisis por Entidad",
        "🚨 Casos Críticos Georreferenciados"
    ])
    coordenadas_estados = {
        'AGUASCALIENTES': (21.88, -102.30), 'BAJA CALIFORNIA': (30.84, -115.28),
        'BAJA CALIFORNIA SUR': (26.04, -111.67), 'CAMPECHE': (19.83, -90.53),
        'CHIAPAS': (16.75, -93.11), 'CHIHUAHUA': (28.63, -106.08),
        'CIUDAD DE MEXICO': (19.43, -99.13), 'COAHUILA': (27.05, -101.70),
        'COLIMA': (19.24, -103.72), 'DURANGO': (24.02, -104.67),
        'GUANAJUATO': (21.02, -101.26), 'GUERRERO': (17.44, -99.54),
        'HIDALGO': (20.09, -98.76), 'JALISCO': (20.66, -103.35),
        'MEXICO': (19.28, -99.65), 'MICHOACAN': (19.57, -101.71),
        'MORELOS': (18.68, -99.10), 'NAYARIT': (21.75, -104.84),
        'NUEVO LEON': (25.59, -99.99), 'OAXACA': (17.07, -96.72),
        'PUEBLA': (19.04, -98.20), 'QUERETARO': (20.59, -100.39),
        'QUINTANA ROO': (19.18, -88.48), 'SAN LUIS POTOSI': (22.15, -100.98),
        'SINALOA': (25.00, -107.50), 'SONORA': (29.30, -110.33),
        'TABASCO': (17.84, -92.62), 'TAMAULIPAS': (24.27, -98.84),
        'TLAXCALA': (19.32, -98.24), 'VERACRUZ': (19.54, -96.91),
        'YUCATAN': (20.71, -89.09), 'ZACATECAS': (22.77, -102.58)
    }
    with tab1:
        st.markdown("### 🗺️ Mapa de Riesgo por Entidad Federativa")
        estado_col = None
        for posible_col in ['declaracion_situacionPatrimonial_datosEmpleoCargoComision_entidadFederativa',
                           'entidadFederativa', 'estado', 'Estado']:
            if posible_col in df.columns:
                estado_col = posible_col
                break
        if estado_col is None:
            st.warning("⚠️ No se encontró columna de estado/entidad federativa en el dataset.")
            return
        st.markdown("""
        <div class='info-box'>
            <p style='margin: 0; color: #667eea; font-weight: 600;'>
            🎯 Mapa de calor mostrando la cantidad de casos por estado
            </p>
        </div>
        """, unsafe_allow_html=True)
        df_estados = df[estado_col].value_counts().reset_index()
        df_estados.columns = ['Estado', 'Casos']
        df_estados['Estado_Normalizado'] = df_estados['Estado'].str.upper().str.strip()
        df_estados['Lat'] = df_estados['Estado_Normalizado'].map(lambda x: coordenadas_estados.get(x, (None, None))[0])
        df_estados['Lon'] = df_estados['Estado_Normalizado'].map(lambda x: coordenadas_estados.get(x, (None, None))[1])
        df_estados = df_estados.dropna(subset=['Lat', 'Lon'])
        if len(df_estados) == 0:
            st.warning("⚠️ No se pudieron georreferenciar los estados.")
            return
        try:
            valores_unicos = df_estados['Casos'].nunique()
            if valores_unicos <= 3:
                percentil_33 = df_estados['Casos'].quantile(0.33)
                percentil_67 = df_estados['Casos'].quantile(0.67)
                def clasificar_manual(x):
                    if x <= percentil_33:
                        return 'Bajo'
                    elif x <= percentil_67:
                        return 'Medio'
                    else:
                        return 'Alto'
                df_estados['Nivel_Riesgo'] = df_estados['Casos'].apply(clasificar_manual)
            else:
                df_estados['Nivel_Riesgo'] = pd.cut(
                    df_estados['Casos'],
                    bins=[0, df_estados['Casos'].quantile(0.33), df_estados['Casos'].quantile(0.67), df_estados['Casos'].max()],
                    labels=['Bajo', 'Medio', 'Alto'],
                    include_lowest=True,
                    duplicates='drop'
                )
        except Exception as e:
            st.warning(f"Usando clasificación simple por promedio: {e}")
            promedio = df_estados['Casos'].mean()
            df_estados['Nivel_Riesgo'] = df_estados['Casos'].apply(
                lambda x: 'Alto' if x > promedio * 1.2 else ('Medio' if x > promedio * 0.8 else 'Bajo')
            )
        color_map = {'Bajo': '#4CAF50', 'Medio': '#ff9800', 'Alto': '#f44336'}
        df_estados['Color'] = df_estados['Nivel_Riesgo'].map(color_map)
        fig = px.scatter_mapbox(
            df_estados,
            lat='Lat',
            lon='Lon',
            size='Casos',
            color='Nivel_Riesgo',
            hover_name='Estado',
            hover_data={'Casos': True, 'Lat': False, 'Lon': False, 'Nivel_Riesgo': True},
            color_discrete_map=color_map,
            size_max=50,
            zoom=4,
            center={'lat': 23.6345, 'lon': -102.5528},
            title='Mapa de Riesgo por Entidad Federativa'
        )
        fig.update_layout(
            mapbox_style='open-street-map',
            height=600
        )
        st.plotly_chart(fig, use_container_width=True)
        st.markdown("---")
        col1, col2, col3 = st.columns(3)
        alto_riesgo = len(df_estados[df_estados['Nivel_Riesgo'] == 'Alto'])
        medio_riesgo = len(df_estados[df_estados['Nivel_Riesgo'] == 'Medio'])
        bajo_riesgo = len(df_estados[df_estados['Nivel_Riesgo'] == 'Bajo'])
        with col1:
            st.markdown(f"""
            <div style='background: rgba(244, 67, 54, 0.1); padding: 1rem; border-radius: 10px; border-left: 4px solid #f44336;'>
                <p style='margin: 0; font-weight: 600; color: #f44336;'>🔴 Alto Riesgo</p>
                <p style='margin: 0.5rem 0 0 0; font-size: 1.5rem; color: #f44336;'>{alto_riesgo} estados</p>
            </div>
            """, unsafe_allow_html=True)
        with col2:
            st.markdown(f"""
            <div style='background: rgba(255, 152, 0, 0.1); padding: 1rem; border-radius: 10px; border-left: 4px solid #ff9800;'>
                <p style='margin: 0; font-weight: 600; color: #ff9800;'>🟡 Medio Riesgo</p>
                <p style='margin: 0.5rem 0 0 0; font-size: 1.5rem; color: #ff9800;'>{medio_riesgo} estados</p>
            </div>
            """, unsafe_allow_html=True)
        with col3:
            st.markdown(f"""
            <div style='background: rgba(76, 175, 80, 0.1); padding: 1rem; border-radius: 10px; border-left: 4px solid #4CAF50;'>
                <p style='margin: 0; font-weight: 600; color: #4CAF50;'>🟢 Bajo Riesgo</p>
                <p style='margin: 0.5rem 0 0 0; font-size: 1.5rem; color: #4CAF50;'>{bajo_riesgo} estados</p>
            </div>
            """, unsafe_allow_html=True)
    with tab2:
        st.markdown("### 📊 Análisis Detallado por Entidad")
        estado_col = None
        for posible_col in ['declaracion_situacionPatrimonial_datosEmpleoCargoComision_entidadFederativa',
                           'entidadFederativa', 'estado', 'Estado']:
            if posible_col in df.columns:
                estado_col = posible_col
                break
        if estado_col is None:
            st.warning("⚠️ No se encontró columna de estado/entidad federativa.")
            return
        estados_disponibles = sorted(df[estado_col].dropna().unique().tolist())
        estado_seleccionado = st.selectbox(
            "Seleccione una entidad federativa:",
            estados_disponibles,
            key="estado_analisis"
        )
        df_estado = df[df[estado_col] == estado_seleccionado]
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("📊 Total Registros", f"{len(df_estado):,}")
        with col2:
            if 'NombreCompleto' in df_estado.columns:
                servidores = df_estado['NombreCompleto'].nunique()
                st.metric("👥 Servidores Únicos", f"{servidores:,}")
        with col3:
            ingresos_cols = [col for col in df.columns if col.startswith('declaracion_situacionPatrimonial_ingresos_') 
                            and df[col].dtype in [np.float64, np.int64]]
            if ingresos_cols:
                total_ingresos = df_estado[ingresos_cols].sum().sum()
                st.metric("💰 Total Ingresos", f"${total_ingresos:,.2f}")
        with col4:
            bienes_cols = [col for col in df.columns if 'bienesInmuebles' in col and 'valorAdquisicion' in col]
            if bienes_cols:
                for col in bienes_cols:
                    df_estado[col] = pd.to_numeric(df_estado[col], errors='coerce').fillna(0)
                total_patrimonio = df_estado[bienes_cols].sum().sum()
                st.metric("🏛️ Total Patrimonio", f"${total_patrimonio:,.2f}")
        st.markdown("---")
        cargo_col = 'declaracion_situacionPatrimonial_datosEmpleoCargoComision_empleoCargoComision'
        if cargo_col in df_estado.columns:
            st.markdown("#### 📊 Top 10 Cargos Más Frecuentes")
            cargos = df_estado[cargo_col].value_counts().head(10)
            fig = px.bar(
                x=cargos.values,
                y=cargos.index,
                orientation='h',
                title=f'Cargos más frecuentes en {estado_seleccionado}',
                labels={'x': 'Cantidad', 'y': 'Cargo'},
                color=cargos.values,
                color_continuous_scale='Purples'
            )
            st.plotly_chart(fig, use_container_width=True)
        st.markdown("---")
        if ingresos_cols:
            st.markdown("#### 📈 Distribución de Ingresos")
            df_estado_temp = df_estado.copy()
            df_estado_temp['Total_Ingresos'] = df_estado_temp[ingresos_cols].sum(axis=1)
            fig = px.histogram(
                df_estado_temp,
                x='Total_Ingresos',
                nbins=50,
                title=f'Distribución de Ingresos en {estado_seleccionado}',
                color_discrete_sequence=['#667eea']
            )
            st.plotly_chart(fig, use_container_width=True)
    with tab3:
        st.markdown("### 🚨 Casos Críticos Georreferenciados")
        st.markdown("""
        <div class='info-box'>
            <p style='margin: 0; color: #f44336; font-weight: 600;'>
            ⚠️ Casos con ingresos en el percentil 90 o superior
            </p>
        </div>
        """, unsafe_allow_html=True)
        ingresos_cols = [col for col in df.columns if col.startswith('declaracion_situacionPatrimonial_ingresos_') 
                        and df[col].dtype in [np.float64, np.int64]]
        estado_col = None
        for posible_col in ['declaracion_situacionPatrimonial_datosEmpleoCargoComision_entidadFederativa',
                           'entidadFederativa', 'estado', 'Estado']:
            if posible_col in df.columns:
                estado_col = posible_col
                break
        if not ingresos_cols or estado_col is None:
            st.warning("⚠️ No se encontraron las columnas necesarias para el análisis.")
            return
        df_criticos = df.copy()
        df_criticos['IngresoTotal'] = df_criticos[ingresos_cols].sum(axis=1)
        umbral_critico = df_criticos['IngresoTotal'].quantile(0.90)
        df_criticos = df_criticos[df_criticos['IngresoTotal'] >= umbral_critico]
        casos_por_estado = df_criticos.groupby(estado_col).agg({
            'IngresoTotal': ['count', 'sum', 'mean']
        }).reset_index()
        casos_por_estado.columns = ['Estado', 'Casos_Criticos', 'Total_Ingresos', 'Promedio_Ingresos']
        casos_por_estado['Estado_Normalizado'] = casos_por_estado['Estado'].str.upper().str.strip()
        casos_por_estado['Lat'] = casos_por_estado['Estado_Normalizado'].map(lambda x: coordenadas_estados.get(x, (None, None))[0])
        casos_por_estado['Lon'] = casos_por_estado['Estado_Normalizado'].map(lambda x: coordenadas_estados.get(x, (None, None))[1])
        casos_por_estado = casos_por_estado.dropna(subset=['Lat', 'Lon'])
        if len(casos_por_estado) == 0:
            st.warning("⚠️ No se pudieron georreferenciar los casos críticos.")
            return
        fig = px.scatter_mapbox(
            casos_por_estado,
            lat='Lat',
            lon='Lon',
            size='Casos_Criticos',
            color='Casos_Criticos',
            hover_name='Estado',
            hover_data={
                'Casos_Criticos': True,
                'Total_Ingresos': ':,.2f',
                'Promedio_Ingresos': ':,.2f',
                'Lat': False,
                'Lon': False
            },
            color_continuous_scale='Reds',
            size_max=50,
            zoom=4,
            center={'lat': 23.6345, 'lon': -102.5528},
            title='Casos Críticos por Entidad Federativa'
        )
        fig.update_layout(
            mapbox_style='open-street-map',
            height=600
        )
        st.plotly_chart(fig, use_container_width=True)
        st.markdown("---")
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("🚨 Total Casos Críticos", f"{len(df_criticos):,}")
        with col2:
            estado_max = casos_por_estado.loc[casos_por_estado['Casos_Criticos'].idxmax(), 'Estado']
            st.metric("🏆 Estado con Más Casos", estado_max)
        with col3:
            promedio_ingresos = df_criticos['IngresoTotal'].mean()
            st.metric("💰 Ingreso Promedio Críticos", f"${promedio_ingresos:,.2f}")
        st.markdown("---")
        st.markdown("#### 📋 Lista de Casos Críticos (Top 50)")
        if 'NombreCompleto' in df_criticos.columns:
            df_top_criticos = df_criticos[['NombreCompleto', estado_col, 'IngresoTotal']].copy()
            df_top_criticos = df_top_criticos.sort_values('IngresoTotal', ascending=False).head(50)
            df_top_criticos['IngresoTotal'] = df_top_criticos['IngresoTotal'].apply(lambda x: f"${x:,.2f}")
            st.dataframe(df_top_criticos, use_container_width=True)
        # Botón de descarga
        if st.button("💾 Descargar Casos Críticos Georreferenciados"):
            try:
                output = BytesIO()
                df_para_excel = preparar_para_excel(df_criticos)
                with pd.ExcelWriter(output, engine='openpyxl') as writer:
                    df_para_excel.to_excel(writer, sheet_name='Casos Criticos', index=False)
                output.seek(0)
                st.download_button(
                    label="⬇️ Descargar Excel",
                    data=output,
                    file_name="casos_criticos_georreferenciados.xlsx",
                    mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
                )
            except Exception as e:
                st.error(f"Error exportando: {e}")
def seccion_evolucion_patrimonial():
    """
    Sección corregida para análisis de Evolución Patrimonial
    Calcula correctamente Ingresos Totales = Ingreso SP + Otros Ingresos
    Detecta incrementos inusuales y genera alertas
    """
    if not st.session_state.datasets:
        st.warning("⚠️ No hay datasets cargados. Por favor, cargue archivos en la sección Inicio.")
        return
    df = st.session_state.datasets[st.session_state.dataset_seleccionado]
    st.markdown("# 💰 Evolución Patrimonial - Detección de Enriquecimiento Oculto")
    st.markdown("""
    <div style='background: linear-gradient(135deg, rgba(244, 67, 54, 0.1) 0%, rgba(211, 47, 47, 0.1) 100%); 
    padding: 1.5rem; border-radius: 15px; border: 2px solid rgba(244, 67, 54, 0.3); margin-bottom: 2rem;'>
        <h3 style='color: #f44336; margin-top: 0;'>🎯 OBJETIVO PRINCIPAL DEL DATATÓN</h3>
        <p style='font-size: 1.1rem; margin-bottom: 0;'>
        Esta sección implementa el núcleo del sistema anticorrupción: detectar incrementos patrimoniales 
        que no pueden justificarse por los ingresos declarados de los servidores públicos.
        </p>
    </div>
    """, unsafe_allow_html=True)
    if 'NombreCompleto' not in df.columns:
        st.error("❌ No se encontró la columna 'NombreCompleto'. Verifique que los datos estén procesados correctamente.")
        return
    st.markdown("---")
    # Calcular métricas por persona si no existen
    if st.session_state.df_metricas_persona is None:
        with st.spinner("Calculando métricas patrimoniales..."):
            df_metricas = calcular_metricas_por_persona(df)
            st.session_state.df_metricas_persona = df_metricas
    else:
        df_metricas = st.session_state.df_metricas_persona
    if df_metricas.empty:
        st.error("❌ No se pudieron calcular las métricas. Verifique las columnas del dataset.")
        return
    # Calcular tabla de servidores únicos
    ingresos_cols = [col for col in df.columns if col.startswith('declaracion_situacionPatrimonial_ingresos_') 
                    and df[col].dtype in [np.float64, np.int64]]
    bienes_cols = [col for col in df.columns if 'bienesInmuebles' in col and 'valorAdquisicion' in col]
    vehiculos_cols = [col for col in df.columns if 'vehiculos' in col and 'valorAdquisicion' in col]
    muebles_cols = [col for col in df.columns if 'bienesMuebles' in col and 'valorAdquisicion' in col]
    inversiones_cols = [col for col in df.columns if 'inversiones' in col and 'saldoSituacionActual' in col]
    adeudos_cols = [col for col in df.columns if 'adeudos' in col and 'saldoInsolutoSituacionActual' in col]
    df_temp = df.copy()
    if bienes_cols:
        for col in bienes_cols:
            df_temp[col] = pd.to_numeric(df_temp[col], errors='coerce').fillna(0)
        df_temp['Suma_Bienes_Inmuebles'] = df_temp[bienes_cols].sum(axis=1)
    else:
        df_temp['Suma_Bienes_Inmuebles'] = 0
    if vehiculos_cols:
        for col in vehiculos_cols:
            df_temp[col] = pd.to_numeric(df_temp[col], errors='coerce').fillna(0)
        df_temp['Suma_Vehiculos'] = df_temp[vehiculos_cols].sum(axis=1)
    else:
        df_temp['Suma_Vehiculos'] = 0
    if muebles_cols:
        for col in muebles_cols:
            df_temp[col] = pd.to_numeric(df_temp[col], errors='coerce').fillna(0)
        df_temp['Suma_Muebles'] = df_temp[muebles_cols].sum(axis=1)
    else:
        df_temp['Suma_Muebles'] = 0
    if inversiones_cols:
        for col in inversiones_cols:
            df_temp[col] = pd.to_numeric(df_temp[col], errors='coerce').fillna(0)
        df_temp['Suma_Inversiones'] = df_temp[inversiones_cols].sum(axis=1)
    else:
        df_temp['Suma_Inversiones'] = 0
    if adeudos_cols:
        for col in adeudos_cols:
            df_temp[col] = pd.to_numeric(df_temp[col], errors='coerce').fillna(0)
        df_temp['Suma_Adeudos'] = df_temp[adeudos_cols].sum(axis=1)
    else:
        df_temp['Suma_Adeudos'] = 0
    df_temp['Patrimonio_Total'] = (
        df_temp['Suma_Bienes_Inmuebles'] + 
        df_temp['Suma_Vehiculos'] + 
        df_temp['Suma_Muebles'] + 
        df_temp['Suma_Inversiones'] - 
        df_temp['Suma_Adeudos']
    )
    df_temp['Numero_Bienes_Totales'] = (
        (df_temp[bienes_cols].notna().sum(axis=1) if bienes_cols else 0) +
        (df_temp[vehiculos_cols].notna().sum(axis=1) if vehiculos_cols else 0) +
        (df_temp[muebles_cols].notna().sum(axis=1) if muebles_cols else 0)
    )
    # Calcular ingresos totales
    if ingresos_cols:
        df_temp['Total_Ingresos'] = df_temp[ingresos_cols].sum(axis=1)
    else:
        df_temp['Total_Ingresos'] = 0
    # Agrupar por servidor
    metricas_patrimonio = df_temp.groupby('NombreCompleto').agg({
        'Patrimonio_Total': ['mean', 'max', 'min'],
        'Suma_Adeudos': 'mean',
        'Numero_Bienes_Totales': 'sum',
        'Total_Ingresos': 'sum'
    })
    metricas_patrimonio.columns = ['_'.join(col).strip() for col in metricas_patrimonio.columns.values]
    df_servidores = df_metricas.set_index('NombreCompleto').join(metricas_patrimonio)
    if 'Total_Ingresos_sum' in df_servidores.columns and 'Patrimonio_Total_mean' in df_servidores.columns:
        df_servidores['Ratio_Patrimonio_Ingreso'] = df_servidores['Patrimonio_Total_mean'] / (df_servidores['Total_Ingresos_sum'] + 1)
    if 'Patrimonio_Total_mean' in df_servidores.columns and 'Suma_Adeudos_mean' in df_servidores.columns:
        df_servidores['Ratio_Deuda_Activos'] = df_servidores['Suma_Adeudos_mean'] / (df_servidores['Patrimonio_Total_mean'] + 1)
    if 'declaracion_situacionPatrimonial_datosEmpleoCargoComision_fechaTomaPosesion' in df.columns:
        df_temp['fechaTomaPosesion'] = pd.to_datetime(df_temp['declaracion_situacionPatrimonial_datosEmpleoCargoComision_fechaTomaPosesion'], errors='coerce')
        años_cargo = df_temp.groupby('NombreCompleto')['fechaTomaPosesion'].agg(lambda x: (datetime.now() - x.min()).days / 365.25 if pd.notna(x.min()) else 0)
        df_servidores = df_servidores.join(años_cargo.rename('Años_en_Cargo'))
    df_servidores = df_servidores.reset_index()
    st.session_state.df_servidores_unicos = df_servidores
    # Generar alertas y anomalías
    df_anomalias = []
    df_alertas = []
    for nombre in df_servidores['NombreCompleto'].unique():
        df_persona = df[df['NombreCompleto'] == nombre].copy()
        if 'anioEjercicio' in df_persona.columns and ingresos_cols:
            df_persona_sorted = df_persona.sort_values('anioEjercicio')
            df_persona_sorted['Total_Ingresos'] = df_persona_sorted[ingresos_cols].sum(axis=1)
            if bienes_cols:
                for col in bienes_cols:
                    df_persona_sorted[col] = pd.to_numeric(df_persona_sorted[col], errors='coerce').fillna(0)
                df_persona_sorted['Total_Bienes_Inmuebles'] = df_persona_sorted[bienes_cols].sum(axis=1)
            if vehiculos_cols:
                for col in vehiculos_cols:
                    df_persona_sorted[col] = pd.to_numeric(df_persona_sorted[col], errors='coerce').fillna(0)
                df_persona_sorted['Total_Vehiculos'] = df_persona_sorted[vehiculos_cols].sum(axis=1)
            if inversiones_cols:
                for col in inversiones_cols:
                    df_persona_sorted[col] = pd.to_numeric(df_persona_sorted[col], errors='coerce').fillna(0)
                df_persona_sorted['Total_Inversiones'] = df_persona_sorted[inversiones_cols].sum(axis=1)
            if adeudos_cols:
                for col in adeudos_cols:
                    df_persona_sorted[col] = pd.to_numeric(df_persona_sorted[col], errors='coerce').fillna(0)
                df_persona_sorted['Total_Adeudos'] = df_persona_sorted[adeudos_cols].sum(axis=1)
            df_persona_sorted['Patrimonio_Neto'] = (
                df_persona_sorted.get('Total_Bienes_Inmuebles', 0) +
                df_persona_sorted.get('Total_Vehiculos', 0) +
                df_persona_sorted.get('Total_Inversiones', 0) -
                df_persona_sorted.get('Total_Adeudos', 0)
            )
            for idx in range(1, len(df_persona_sorted)):
                row_actual = df_persona_sorted.iloc[idx]
                prev_row = df_persona_sorted.iloc[idx-1]
                # Alertas de incremento de ingresos
                if pd.notna(row_actual.get('Total_Ingresos')) and pd.notna(prev_row.get('Total_Ingresos')) and prev_row['Total_Ingresos'] > 0:
                    inc_ingresos_pct = ((row_actual['Total_Ingresos'] - prev_row['Total_Ingresos']) / prev_row['Total_Ingresos']) * 100
                    if inc_ingresos_pct > 100:
                        df_alertas.append({
                            'NombreCompleto': nombre,
                            'Año': row_actual.get('anioEjercicio', 'N/A'),
                            'Tipo_Alerta': 'Incremento de Ingresos',
                            'Incremento_Pct': inc_ingresos_pct,
                            'Ingreso_Previo': prev_row['Total_Ingresos'],
                            'Ingreso_Actual': row_actual['Total_Ingresos']
                        })
                # Alertas de incremento patrimonial
                if pd.notna(row_actual.get('Patrimonio_Neto')) and pd.notna(prev_row.get('Patrimonio_Neto')) and prev_row['Patrimonio_Neto'] > 0:
                    inc_patrimonio_pct = ((row_actual['Patrimonio_Neto'] - prev_row['Patrimonio_Neto']) / prev_row['Patrimonio_Neto']) * 100
                    if inc_patrimonio_pct > 100:
                        df_anomalias.append({
                            'NombreCompleto': nombre,
                            'Año': row_actual.get('anioEjercicio', 'N/A'),
                            'Tipo_Anomalia': 'Incremento Patrimonial',
                            'Incremento_Pct': inc_patrimonio_pct,
                            'Patrimonio_Previo': prev_row['Patrimonio_Neto'],
                            'Patrimonio_Actual': row_actual['Patrimonio_Neto']
                        })
                    # Alerta de enriquecimiento oculto
                    inc_ingresos_pct = 0
                    if pd.notna(row_actual.get('Total_Ingresos')) and pd.notna(prev_row.get('Total_Ingresos')) and prev_row['Total_Ingresos'] > 0:
                        inc_ingresos_pct = ((row_actual['Total_Ingresos'] - prev_row['Total_Ingresos']) / prev_row['Total_Ingresos']) * 100
                    if inc_patrimonio_pct > 100 and inc_ingresos_pct < 50:
                        df_anomalias.append({
                            'NombreCompleto': nombre,
                            'Año': row_actual.get('anioEjercicio', 'N/A'),
                            'Tipo_Anomalia': 'Enriquecimiento Oculto',
                            'Incremento_Patrimonio_Pct': inc_patrimonio_pct,
                            'Incremento_Ingresos_Pct': inc_ingresos_pct,
                            'Patrimonio_Previo': prev_row['Patrimonio_Neto'],
                            'Patrimonio_Actual': row_actual['Patrimonio_Neto'],
                            'Ingreso_Previo': prev_row.get('Total_Ingresos', 0),
                            'Ingreso_Actual': row_actual.get('Total_Ingresos', 0)
                        })
    st.session_state.df_anomalias = pd.DataFrame(df_anomalias) if df_anomalias else pd.DataFrame()
    st.session_state.df_alertas = pd.DataFrame(df_alertas) if df_alertas else pd.DataFrame()
    st.markdown("### 👥 Tabla de Servidores Públicos Únicos")
    st.markdown(f"#### 📊 Total de Servidores Públicos Únicos: {len(df_servidores):,}")
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        promedio_declaraciones = df_servidores['n_declaraciones'].mean()
        st.metric("🔄 Promedio Declaraciones", f"{promedio_declaraciones:.2f}")
    with col2:
        if 'Total_Ingresos_sum' in df_servidores.columns:
            promedio_ingresos = df_servidores['Total_Ingresos_sum'].mean()
            st.metric("💰 Promedio Ingresos Total", f"${promedio_ingresos:,.2f}")
    with col3:
        if 'Patrimonio_Total_mean' in df_servidores.columns:
            promedio_patrimonio = df_servidores['Patrimonio_Total_mean'].mean()
            st.metric("📈 Promedio Máximo Ingresos", f"${promedio_patrimonio:,.2f}")
    with col4:
        alertas_totales = len(st.session_state.df_alertas)
        st.metric("🚨 Alertas Detectadas", f"{alertas_totales:,}")
    st.markdown("---")
    # Tabla con métricas formateadas
    df_display = df_servidores.copy()
    # Formatear columnas monetarias
    for col in ['ingreso_sp_promedio', 'otros_ingresos_promedio', 'ingresos_totales_promedio', 
               'max_ingresos_totales', 'min_ingresos_totales', 'ultimo_ingresos_totales', 'delta_max_abs',
               'Total_Ingresos_sum', 'Patrimonio_Total_mean', 'Patrimonio_Total_max', 'Patrimonio_Total_min']:
        if col in df_display.columns:
            df_display[col] = df_display[col].apply(lambda x: f"${x:,.2f}" if pd.notna(x) else "N/A")
    # Formatear porcentajes
    if 'delta_max_pct' in df_display.columns:
        df_display['delta_max_pct'] = df_display['delta_max_pct'].apply(lambda x: f"{x:.2f}%" if pd.notna(x) else "N/A")
    # Formatear booleanos
    if 'alerta_incremento_inusual' in df_display.columns:
        df_display['alerta_incremento_inusual'] = df_display['alerta_incremento_inusual'].apply(lambda x: "🚨 SÍ" if x else "✅ NO")
    st.dataframe(df_display, use_container_width=True, height=400)
    st.markdown("---")
    # Botones de descarga
    col1, col2, col3 = st.columns(3)
    with col1:
        if not st.session_state.df_alertas.empty:
            if st.button("💾 Descargar ALERTAS DETECTADAS"):
                output = BytesIO()
                df_para_excel = preparar_para_excel(st.session_state.df_alertas)
                with pd.ExcelWriter(output, engine='openpyxl') as writer:
                    df_para_excel.to_excel(writer, sheet_name='Alertas', index=False)
                output.seek(0)
                st.download_button(
                    label="⬇️ Descargar Excel - Alertas",
                    data=output,
                    file_name="alertas_detectadas.xlsx",
                    mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
                )
        else:
            st.info("ℹ️ No hay alertas detectadas")
    with col2:
        if not st.session_state.df_anomalias.empty:
            if st.button("💾 Descargar DETECCIÓN DE ANOMALÍAS"):
                output = BytesIO()
                df_para_excel = preparar_para_excel(st.session_state.df_anomalias)
                with pd.ExcelWriter(output, engine='openpyxl') as writer:
                    df_para_excel.to_excel(writer, sheet_name='Anomalias', index=False)
                output.seek(0)
                st.download_button(
                    label="⬇️ Descargar Excel - Anomalías",
                    data=output,
                    file_name="deteccion_anomalias.xlsx",
                    mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
                )
        else:
            st.info("ℹ️ No hay anomalías detectadas")
    with col3:
        df_alto_riesgo = df_servidores[df_servidores['alerta_incremento_inusual'] == True].head(50)
        if not df_alto_riesgo.empty:
            if st.button("💾 Descargar Tabla de Casos de Alto Riesgo (Top 50)"):
                output = BytesIO()
                df_para_excel = preparar_para_excel(df_alto_riesgo)
                with pd.ExcelWriter(output, engine='openpyxl') as writer:
                    df_para_excel.to_excel(writer, sheet_name='Alto Riesgo', index=False)
                output.seek(0)
                st.download_button(
                    label="⬇️ Descargar Excel - Alto Riesgo",
                    data=output,
                    file_name="casos_alto_riesgo.xlsx",
                    mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
                )
        else:
            st.info("ℹ️ No hay casos de alto riesgo")
    st.markdown("---")
    st.markdown("### 📈 Series Temporales Interactivas")
    col_ingreso_sp = 'declaracion_situacionPatrimonial_ingresos_remuneracionAnualCargoPublico_valor'
    col_otros_ingresos = 'declaracion_situacionPatrimonial_ingresos_otrosIngresosAnualesTotal_valor'
    col1, col2, col3 = st.columns(3)
    with col1:
        ente_col = 'declaracion_situacionPatrimonial_datosEmpleoCargoComision_nombreEntePublico'
        if ente_col in df.columns:
            # Normalizar valores antes de crear lista
            vals_ente = df[ente_col].dropna().astype(str).str.strip().replace('', np.nan).dropna().unique().tolist()
            entes = ['Todos'] + sorted(vals_ente, key=lambda x: x.lower())
            ente_seleccionado = st.selectbox("Filtrar por Ente Público:", entes, key="ente_filter_series")
    with col2:
        cargo_col = 'declaracion_situacionPatrimonial_datosEmpleoCargoComision_empleoCargoComision'
        if cargo_col in df.columns:
            # CORRECCIÓN: Normalizar valores mixtos (strings y números) antes de sorted
            vals_cargo = df[cargo_col].dropna().astype(str).str.strip().replace('', np.nan).dropna().unique().tolist()
            cargos = ['Todos'] + sorted(vals_cargo, key=lambda x: x.lower())
            cargo_seleccionado = st.selectbox("Filtrar por Cargo:", cargos, key="cargo_filter_series")
    with col3:
        nombres = ['Todos'] + sorted(df['NombreCompleto'].dropna().unique().tolist())
        nombre_seleccionado = st.selectbox("Filtrar por Nombre:", nombres, key="nombre_filter_series")
    df_filtrado = df.copy()
    if ente_seleccionado != 'Todos' and ente_col in df.columns:
        df_filtrado = df_filtrado[df_filtrado[ente_col] == ente_seleccionado]
    if cargo_seleccionado != 'Todos' and cargo_col in df.columns:
        df_filtrado = df_filtrado[df_filtrado[cargo_col] == cargo_seleccionado]
    if nombre_seleccionado != 'Todos':
        df_filtrado = df_filtrado[df_filtrado['NombreCompleto'] == nombre_seleccionado]
    st.markdown(f"**Registros filtrados:** {len(df_filtrado):,}")
    if len(df_filtrado) == 0:
        st.warning("⚠️ No hay datos para los filtros seleccionados.")
    else:
        if 'anioEjercicio' in df_filtrado.columns and ingresos_cols:
            df_temporal = df_filtrado.copy()
            # Asegurar que anioEjercicio es numérico
            df_temporal['anioEjercicio'] = pd.to_numeric(df_temporal['anioEjercicio'], errors='coerce')
            df_temporal = df_temporal.dropna(subset=['anioEjercicio'])
            # Calcular ingresos totales correctamente
            ingreso_sp_vals = pd.to_numeric(df_temporal[col_ingreso_sp], errors='coerce').fillna(0) if col_ingreso_sp in df_temporal.columns else 0
            otros_ingresos_vals = pd.to_numeric(df_temporal[col_otros_ingresos], errors='coerce').fillna(0) if col_otros_ingresos in df_temporal.columns else 0
            df_temporal['ingresos_totales_calc'] = ingreso_sp_vals + otros_ingresos_vals
            # Agrupar y promediar por año
            serie_temporal = df_temporal.groupby('anioEjercicio')['ingresos_totales_calc'].mean().reset_index()
            serie_temporal = serie_temporal.sort_values('anioEjercicio')
            # Verificar que hay datos
            if not serie_temporal.empty:
                fig = px.line(
                    serie_temporal,
                    x='anioEjercicio',
                    y='ingresos_totales_calc',
                    title='Evolución Temporal de Ingresos Promedio',
                    labels={'anioEjercicio': 'Año', 'ingresos_totales_calc': 'Ingresos Promedio (MXN)'},
                    markers=True,
                    line_shape='spline'
                )
                fig.update_traces(line_color='#667eea', marker=dict(size=10, color='#764ba2'))
                fig.update_layout(hovermode='x unified')
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.warning("⚠️ No hay datos suficientes para generar la serie temporal.")
    st.markdown("### 🔍 Análisis Individual Detallado")
    st.markdown("""
    <div class='info-box'>
        <p style='margin: 0; color: #667eea; font-weight: 600;'>
        🔎 Búsqueda Inteligente: Escriba el nombre del servidor público para ver su evolución patrimonial completa
        </p>
    </div>
    """, unsafe_allow_html=True)
    search_term = st.text_input(
        "Buscar servidor público:",
        placeholder="Escriba el nombre del servidor público...",
        key="search_servidor_individual"
    )
    if search_term and len(search_term) > 2:
        nombres_filtrados = df[df['NombreCompleto'].str.contains(search_term, case=False, na=False)]['NombreCompleto'].unique()
        if len(nombres_filtrados) > 0:
            st.markdown(f"**{len(nombres_filtrados)} resultados encontrados**")
            nombre_final = st.selectbox(
                "Seleccione el servidor público:",
                nombres_filtrados,
                key="select_servidor_individual_detailed"
            )
            df_persona = df[df['NombreCompleto'] == nombre_final].copy()
            st.markdown(f"### 👤 Análisis Detallado: {nombre_final}")
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.metric("🔄 Total Declaraciones", len(df_persona))
            with col2:
                if 'metadata_tipo' in df_persona.columns:
                    tipos = df_persona['metadata_tipo'].value_counts().to_dict()
                    tipo_mas_comun = max(tipos, key=tipos.get) if tipos else "N/A"
                    st.metric("📋 Tipo Más Común", tipo_mas_comun)
            with col3:
                if 'anioEjercicio' in df_persona.columns:
                    años = df_persona['anioEjercicio'].dropna()
                    if len(años) > 0:
                        st.metric("📅 Primer Año", int(años.min()))
            with col4:
                if 'anioEjercicio' in df_persona.columns:
                    años = df_persona['anioEjercicio'].dropna()
                    if len(años) > 0:
                        st.metric("📅 Último Año", int(años.max()))
            st.markdown("---")
            # Calcular evolución correctamente
            col_fecha_actualizacion = 'metadata_actualizacion'
            if col_fecha_actualizacion in df_persona.columns:
                df_persona_sorted = df_persona.sort_values(col_fecha_actualizacion)
                # Calcular ingresos totales correctamente
                ingreso_sp_vals = pd.to_numeric(df_persona_sorted[col_ingreso_sp], errors='coerce').fillna(0) if col_ingreso_sp in df_persona_sorted.columns else 0
                otros_ingresos_vals = pd.to_numeric(df_persona_sorted[col_otros_ingresos], errors='coerce').fillna(0) if col_otros_ingresos in df_persona_sorted.columns else 0
                df_persona_sorted['ingresos_totales_calc'] = ingreso_sp_vals + otros_ingresos_vals
                # Calcular incrementos
                df_persona_sorted['ingreso_prev'] = df_persona_sorted['ingresos_totales_calc'].shift(1)
                df_persona_sorted['delta_abs'] = df_persona_sorted['ingresos_totales_calc'] - df_persona_sorted['ingreso_prev']
                df_persona_sorted['delta_pct'] = ((df_persona_sorted['ingresos_totales_calc'] - df_persona_sorted['ingreso_prev']) / 
                                                  (df_persona_sorted['ingreso_prev'] + 1)) * 100
                st.markdown("#### 📊 Tabla Transpuesta de Evolución")
                columnas_mostrar = ['anioEjercicio', 'metadata_tipo', 'ingresos_totales_calc', 'delta_abs', 'delta_pct']
                columnas_mostrar = [c for c in columnas_mostrar if c in df_persona_sorted.columns]
                df_evolucion = df_persona_sorted[columnas_mostrar].copy()
                st.dataframe(df_evolucion, use_container_width=True)
                st.markdown("---")
                st.markdown("#### 🚨 Detección de Anomalías")
                alertas = []
                for idx in range(1, len(df_evolucion)):
                    row_actual = df_evolucion.iloc[idx]
                    if 'delta_pct' in df_evolucion.columns and pd.notna(row_actual.get('delta_pct')):
                        if row_actual['delta_pct'] > UMBRAL_ALERTA_INCREMENTO:
                            alertas.append({
                                'Año': row_actual.get('anioEjercicio', 'N/A'),
                                'Tipo': '💰 Incremento de Ingresos',
                                'Descripción': f"Incremento del {row_actual['delta_pct']:.2f}%",
                                'Nivel': '🔴 CRÍTICO' if row_actual['delta_pct'] > 200 else '🟡 ADVERTENCIA'
                            })
                if alertas:
                    st.markdown(f"**{len(alertas)} alertas detectadas**")
                    for alerta in alertas:
                        if '🔴' in alerta['Nivel']:
                            st.markdown(f"""
                            <div style='background: rgba(244, 67, 54, 0.1); padding: 1rem; border-radius: 10px; border-left: 4px solid #f44336; margin: 0.5rem 0;'>
                                <strong>{alerta['Nivel']} - {alerta['Tipo']} ({alerta['Año']})</strong><br>
                                {alerta['Descripción']}
                            </div>
                            """, unsafe_allow_html=True)
                        else:
                            st.markdown(f"""
                            <div style='background: rgba(255, 152, 0, 0.1); padding: 1rem; border-radius: 10px; border-left: 4px solid #ff9800; margin: 0.5rem 0;'>
                                <strong>{alerta['Nivel']} - {alerta['Tipo']} ({alerta['Año']})</strong><br>
                                {alerta['Descripción']}
                            </div>
                            """, unsafe_allow_html=True)
                else:
                    st.markdown("""
                    <div style='background: rgba(76, 175, 80, 0.1); padding: 1rem; border-radius: 10px; border-left: 4px solid #4CAF50;'>
                        <strong>✅ No se detectaron anomalías significativas</strong>
                    </div>
                    """, unsafe_allow_html=True)
        else:
            st.warning("⚠️ No se encontraron resultados para la búsqueda.")
    else:
        st.info("ℹ️ Escriba al menos 3 caracteres para buscar.")
    st.markdown("### 🚨 Sistema de Alertas y Semáforos de Riesgo")
    st.markdown("""
    <div style='background: linear-gradient(135deg, rgba(244, 67, 54, 0.1) 0%, rgba(211, 47, 47, 0.1) 100%); 
    padding: 1.5rem; border-radius: 15px; border: 2px solid rgba(244, 67, 54, 0.3);'>
        <h4 style='color: #f44336; margin-top: 0;'>🎯 Sistema de Detección Automatizada</h4>
        <p style='margin-bottom: 0;'>
        Clasificación automática de servidores públicos según nivel de riesgo de enriquecimiento oculto
        </p>
    </div>
    """, unsafe_allow_html=True)
    st.markdown("---")
    with st.spinner("Calculando niveles de riesgo..."):
        if st.session_state.df_servidores_unicos is not None:
            df_riesgo_agrupado = st.session_state.df_servidores_unicos.copy()
            # Clasificar riesgo según puntuación
            def clasificar_riesgo(row):
                puntos = 0
                # Puntos por ratio patrimonio/ingreso (si existe)
                if 'Total_Ingresos_sum' in row and row['Total_Ingresos_sum'] > 0:
                    if row.get('Patrimonio_Total_max', 0) > row['Total_Ingresos_sum'] * 10:
                        puntos += 3
                    elif row.get('Patrimonio_Total_max', 0) > row['Total_Ingresos_sum'] * 5:
                        puntos += 2
                # Puntos por incremento máximo porcentual
                if pd.notna(row.get('delta_max_pct')):
                    if row['delta_max_pct'] > 200:
                        puntos += 3
                    elif row['delta_max_pct'] > 100:
                        puntos += 2
                    elif row['delta_max_pct'] > 50:
                        puntos += 1
                # Puntos por alerta de incremento
                if row.get('alerta_incremento_inusual', False):
                    puntos += 2
                if puntos >= 6:
                    return '🔴 ALTO'
                elif puntos >= 3:
                    return '🟡 MEDIO'
                else:
                    return '🟢 BAJO'
            df_riesgo_agrupado['Nivel_Riesgo'] = df_riesgo_agrupado.apply(clasificar_riesgo, axis=1)
            col1, col2, col3, col4 = st.columns(4)
            total_analizado = len(df_riesgo_agrupado)
            alto_riesgo = len(df_riesgo_agrupado[df_riesgo_agrupado['Nivel_Riesgo'] == '🔴 ALTO'])
            medio_riesgo = len(df_riesgo_agrupado[df_riesgo_agrupado['Nivel_Riesgo'] == '🟡 MEDIO'])
            bajo_riesgo = len(df_riesgo_agrupado[df_riesgo_agrupado['Nivel_Riesgo'] == '🟢 BAJO'])
            with col1:
                st.metric("📊 Total Analizado", f"{total_analizado:,}")
            with col2:
                st.markdown(f"""
                <div style='background: rgba(244, 67, 54, 0.1); padding: 1rem; border-radius: 10px; border-left: 4px solid #f44336;'>
                    <p style='margin: 0; font-weight: 600; color: #f44336;'>🔴 Alto Riesgo</p>
                    <p style='margin: 0.5rem 0 0 0; font-size: 1.5rem; color: #f44336;'>{alto_riesgo:,} ({(alto_riesgo/total_analizado*100):.1f}%)</p>
                </div>
                """, unsafe_allow_html=True)
            with col3:
                st.markdown(f"""
                <div style='background: rgba(255, 152, 0, 0.1); padding: 1rem; border-radius: 10px; border-left: 4px solid #ff9800;'>
                    <p style='margin: 0; font-weight: 600; color: #ff9800;'>🟡 Medio Riesgo</p>
                    <p style='margin: 0.5rem 0 0 0; font-size: 1.5rem; color: #ff9800;'>{medio_riesgo:,} ({(medio_riesgo/total_analizado*100):.1f}%)</p>
                </div>
                """, unsafe_allow_html=True)
            with col4:
                st.markdown(f"""
                <div style='background: rgba(76, 175, 80, 0.1); padding: 1rem; border-radius: 10px; border-left: 4px solid #4CAF50;'>
                    <p style='margin: 0; font-weight: 600; color: #4CAF50;'>🟢 Bajo Riesgo</p>
                    <p style='margin: 0.5rem 0 0 0; font-size: 1.5rem; color: #4CAF50;'>{bajo_riesgo:,} ({(bajo_riesgo/total_analizado*100):.1f}%)</p>
                </div>
                """, unsafe_allow_html=True)
            st.markdown("---")
            st.markdown("#### 📋 Tabla de Casos de Alto Riesgo (Top 50)")
            df_alto_riesgo = df_riesgo_agrupado[df_riesgo_agrupado['Nivel_Riesgo'] == '🔴 ALTO'].head(50)
            if len(df_alto_riesgo) > 0:
                columnas_display = ['NombreCompleto', 'Total_Ingresos_sum', 'delta_max_pct', 'alerta_incremento_inusual', 'Nivel_Riesgo']
                columnas_display = [c for c in columnas_display if c in df_alto_riesgo.columns]
                st.dataframe(df_alto_riesgo[columnas_display], use_container_width=True)
            else:
                st.info("✅ No se detectaron casos de alto riesgo")
def seccion_nepotismo():
    if not st.session_state.datasets:
        st.warning("⚠️ No hay datasets cargados.")
        return
    df = st.session_state.datasets[st.session_state.dataset_seleccionado]
    st.markdown("# 🕸 Detección de Nepotismo")
    st.markdown("""
    <div class='info-box'>
        <h3 style='color: #ff9800; margin-top: 0;'>🔍 Análisis de Relaciones Sospechosas</h3>
        <p>Detección de posibles relaciones familiares o de compadrazgo dentro del mismo ente público
        mediante análisis de similitud de nombres y coincidencias de apellidos.</p>
    </div>
    """, unsafe_allow_html=True)
    if 'NombreCompleto' not in df.columns:
        st.error("❌ No se encontró la columna 'NombreCompleto'. Verifique que los datos estén procesados correctamente.")
        return
    st.markdown("---")
    df_coincidencias = pd.DataFrame()
    tab1, tab2, tab3 = st.tabs([
        "📊 Análisis de Similitud",
        "🕸 Grafos de Relación",
        "📋 Tabla de Casos Sospechosos"
    ])
    with tab1:
        st.markdown("### 📊 Análisis de Similitud de Nombres")
        umbral_similitud = st.slider(
            "Umbral de similitud (%):",
            50, 100, 85,
            help="Porcentaje mínimo de similitud para considerar una relación",
            key="umbral_similitud"
        )
        with st.spinner("Analizando similitudes..."):
            if 'declaracion_situacionPatrimonial_datosEmpleoCargoComision_nombreEntePublico' in df.columns:
                ente_col = 'declaracion_situacionPatrimonial_datosEmpleoCargoComision_nombreEntePublico'
                df_nombres = df[['NombreCompleto', ente_col]].drop_duplicates()
                coincidencias = []
                if FUZZY_AVAILABLE:
                    for ente in df_nombres[ente_col].unique():
                        if pd.isna(ente):
                            continue
                        df_ente = df_nombres[df_nombres[ente_col] == ente]
                        nombres = df_ente['NombreCompleto'].tolist()
                        for i in range(len(nombres)):
                            for j in range(i+1, len(nombres)):
                                if pd.notna(nombres[i]) and pd.notna(nombres[j]):
                                    similitud = fuzz.ratio(nombres[i], nombres[j])
                                    if similitud >= umbral_similitud:
                                        coincidencias.append({
                                            'Nombre 1': nombres[i],
                                            'Nombre 2': nombres[j],
                                            'Similitud (%)': similitud,
                                            'Ente Público': ente,
                                            'Tipo': 'Alta Similitud'
                                        })
                                    apellidos_1 = nombres[i].split()[1:] if len(nombres[i].split()) > 1 else []
                                    apellidos_2 = nombres[j].split()[1:] if len(nombres[j].split()) > 1 else []
                                    if apellidos_1 and apellidos_2:
                                        if any(ap in apellidos_2 for ap in apellidos_1):
                                            if similitud < umbral_similitud:
                                                coincidencias.append({
                                                    'Nombre 1': nombres[i],
                                                    'Nombre 2': nombres[j],
                                                    'Similitud (%)': similitud,
                                                    'Ente Público': ente,
                                                    'Tipo': 'Mismo Apellido'
                                                })
                    df_coincidencias = pd.DataFrame(coincidencias)
                    if len(df_coincidencias) > 0:
                        col1, col2, col3 = st.columns(3)
                        with col1:
                            st.metric("🔍 Total Coincidencias", len(df_coincidencias))
                        with col2:
                            alta_similitud = len(df_coincidencias[df_coincidencias['Tipo'] == 'Alta Similitud'])
                            st.metric("📊 Alta Similitud", alta_similitud)
                        with col3:
                            mismo_apellido = len(df_coincidencias[df_coincidencias['Tipo'] == 'Mismo Apellido'])
                            st.metric("👥 Mismo Apellido", mismo_apellido)
                        st.markdown("---")
                        st.markdown("#### 📊 Distribución por Tipo de Coincidencia")
                        fig = px.pie(
                            df_coincidencias,
                            names='Tipo',
                            title='Distribución de Tipos de Coincidencia',
                            color='Tipo',
                            color_discrete_map={
                                'Alta Similitud': '#ff9800',
                                'Mismo Apellido': '#f44336'
                            }
                        )
                        st.plotly_chart(fig, use_container_width=True)
                        st.markdown("---")
                        st.markdown("#### 📋 Top 20 Coincidencias por Similitud")
                        df_top = df_coincidencias.sort_values('Similitud (%)', ascending=False).head(20)
                        st.dataframe(df_top, use_container_width=True)
                        st.markdown("---")
                        st.markdown("#### 🏢 Entes con Más Coincidencias")
                        entes_coincidencias = df_coincidencias['Ente Público'].value_counts().head(10)
                        fig = px.bar(
                            x=entes_coincidencias.values,
                            y=entes_coincidencias.index,
                            orientation='h',
                            title='Top 10 Entes con Más Coincidencias',
                            labels={'x': 'Número de Coincidencias', 'y': 'Ente Público'},
                            color=entes_coincidencias.values,
                            color_continuous_scale='Oranges'
                        )
                        st.plotly_chart(fig, use_container_width=True)
                    else:
                        st.info("ℹ️ No se encontraron coincidencias con el umbral seleccionado.")
                else:
                    st.warning("⚠️ FuzzyWuzzy no está disponible. Instale con: pip install fuzzywuzzy python-Levenshtein")
            else:
                st.warning("⚠️ No se encontró la columna de Ente Público en el dataset.")
    with tab2:
        st.markdown("### 🕸 Grafos de Relación")
        if len(df_coincidencias) > 0 and NETWORKX_AVAILABLE:
            st.markdown("""
            <div class='info-box'>
                <p style='margin: 0; color: #ff9800; font-weight: 600;'>
                📊 Visualización de red de relaciones entre servidores públicos con coincidencias detectadas
                </p>
            </div>
            """, unsafe_allow_html=True)
            ente_seleccionado = st.selectbox(
                "Seleccione un ente público para visualizar:",
                df_coincidencias['Ente Público'].unique(),
                key="ente_grafo"
            )
            df_ente_grafo = df_coincidencias[df_coincidencias['Ente Público'] == ente_seleccionado]
            G = nx.Graph()
            for _, row in df_ente_grafo.iterrows():
                G.add_edge(
                    row['Nombre 1'], 
                    row['Nombre 2'],
                    weight=row['Similitud (%)'],
                    tipo=row['Tipo']
                )
            st.markdown(f"#### 🕸 Red de Relaciones: {ente_seleccionado}")
            st.markdown(f"**Nodos:** {G.number_of_nodes()} | **Conexiones:** {G.number_of_edges()}")
            if G.number_of_nodes() > 0:
                pos = nx.spring_layout(G, k=2, iterations=50)
                edge_x = []
                edge_y = []
                for edge in G.edges():
                    x0, y0 = pos[edge[0]]
                    x1, y1 = pos[edge[1]]
                    edge_x.extend([x0, x1, None])
                    edge_y.extend([y0, y1, None])
                edge_trace = go.Scatter(
                    x=edge_x, y=edge_y,
                    line=dict(width=2, color='#ff9800'),
                    hoverinfo='none',
                    mode='lines',
                    opacity=0.5
                )
                node_x = []
                node_y = []
                node_text = []
                node_size = []
                for node in G.nodes():
                    x, y = pos[node]
                    node_x.append(x)
                    node_y.append(y)
                    node_text.append(node)
                    node_size.append(G.degree(node) * 20 + 20)
                # Corregir colorbar sin titleside
                node_trace = go.Scatter(
                    x=node_x, y=node_y,
                    mode='markers+text',
                    hoverinfo='text',
                    text=node_text,
                    textposition="top center",
                    textfont=dict(size=10, color='#333'),
                    marker=dict(
                        showscale=True,
                        colorscale='Oranges',
                        size=node_size,
                        color=[G.degree(node) for node in G.nodes()],
                        colorbar=dict(
                            thickness=15,
                            title=dict(text='Conexiones'),  # Usar dict para título
                            xanchor='left',
                            # titleside removido
                        ),
                        line=dict(width=2, color='white')
                    )
                )
                fig = go.Figure(data=[edge_trace, node_trace],
                             layout=go.Layout(
                                title=f'Grafo de Relaciones - {ente_seleccionado}',
                                showlegend=False,
                                hovermode='closest',
                                margin=dict(b=0,l=0,r=0,t=40),
                                xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
                                yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
                                height=700
                                ))
                st.plotly_chart(fig, use_container_width=True)
                st.markdown("---")
                st.markdown("#### 📊 Estadísticas del Grafo")
                col1, col2, col3, col4 = st.columns(4)
                with col1:
                    st.metric("🔢 Nodos", G.number_of_nodes())
                with col2:
                    st.metric("🔗 Conexiones", G.number_of_edges())
                with col3:
                    densidad = nx.density(G)
                    st.metric("📊 Densidad", f"{densidad:.3f}")
                with col4:
                    grado_promedio = sum(dict(G.degree()).values()) / G.number_of_nodes()
                    st.metric("📈 Grado Promedio", f"{grado_promedio:.2f}")
                st.markdown("---")
                st.markdown("#### 👥 Nodos Más Conectados")
                grados = dict(G.degree())
                top_nodos = sorted(grados.items(), key=lambda x: x[1], reverse=True)[:10]
                df_top_nodos = pd.DataFrame(top_nodos, columns=['Nombre', 'Conexiones'])
                fig = px.bar(
                    df_top_nodos,
                    x='Conexiones',
                    y='Nombre',
                    orientation='h',
                    title='Top 10 Personas Más Conectadas',
                    color='Conexiones',
                    color_continuous_scale='Oranges'
                )
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.info("ℹ️ No hay suficientes datos para generar el grafo.")
        elif not NETWORKX_AVAILABLE:
            st.warning("⚠️ NetworkX no está disponible. Instale con: pip install networkx")
        else:
            st.info("ℹ️ No hay coincidencias detectadas. Ajuste el umbral en la pestaña 'Análisis de Similitud'.")
    with tab3:
        st.markdown("### 📋 Tabla de Casos Sospechosos")
        if len(df_coincidencias) > 0:
            st.markdown("""
            <div style='background: rgba(244, 67, 54, 0.1); padding: 1rem; border-radius: 10px; margin-bottom: 1rem;'>
                <p style='margin: 0; color: #f44336; font-weight: 600;'>
                ⚠️ Casos que requieren investigación adicional por posible nepotismo
                </p>
            </div>
            """, unsafe_allow_html=True)
            df_sospechosos = df_coincidencias.copy()
            def calcular_nivel_sospecha(row):
                puntos = 0
                if row['Similitud (%)'] >= 95:
                    puntos += 3
                elif row['Similitud (%)'] >= 90:
                    puntos += 2
                elif row['Similitud (%)'] >= 85:
                    puntos += 1
                if row['Tipo'] == 'Mismo Apellido':
                    puntos += 2
                if puntos >= 4:
                    return '🔴 CRÍTICO'
                elif puntos >= 2:
                    return '🟡 MEDIO'
                else:
                    return '🟢 BAJO'
            df_sospechosos['Nivel_Sospecha'] = df_sospechosos.apply(calcular_nivel_sospecha, axis=1)
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.metric("📊 Total Casos", len(df_sospechosos))
            with col2:
                criticos = len(df_sospechosos[df_sospechosos['Nivel_Sospecha'] == '🔴 CRÍTICO'])
                st.markdown(f"""
                <div style='background: rgba(244, 67, 54, 0.1); padding: 1rem; border-radius: 10px; border-left: 4px solid #f44336;'>
                    <p style='margin: 0; font-weight: 600; color: #f44336;'>🔴 Críticos</p>
                    <p style='margin: 0.5rem 0 0 0; font-size: 1.5rem; color: #f44336;'>{criticos}</p>
                </div>
                """, unsafe_allow_html=True)
            with col3:
                medios = len(df_sospechosos[df_sospechosos['Nivel_Sospecha'] == '🟡 MEDIO'])
                st.markdown(f"""
                <div style='background: rgba(255, 152, 0, 0.1); padding: 1rem; border-radius: 10px; border-left: 4px solid #ff9800;'>
                    <p style='margin: 0; font-weight: 600; color: #ff9800;'>🟡 Medios</p>
                    <p style='margin: 0.5rem 0 0 0; font-size: 1.5rem; color: #ff9800;'>{medios}</p>
                </div>
                """, unsafe_allow_html=True)
            with col4:
                bajos = len(df_sospechosos[df_sospechosos['Nivel_Sospecha'] == '🟢 BAJO'])
                st.markdown(f"""
                <div style='background: rgba(76, 175, 80, 0.1); padding: 1rem; border-radius: 10px; border-left: 4px solid #4CAF50;'>
                    <p style='margin: 0; font-weight: 600; color: #4CAF50;'>🟢 Bajos</p>
                    <p style='margin: 0.5rem 0 0 0; font-size: 1.5rem; color: #4CAF50;'>{bajos}</p>
                </div>
                """, unsafe_allow_html=True)
            st.markdown("---")
            filtro_nivel = st.selectbox(
                "Filtrar por nivel de sospecha:",
                ['Todos', '🔴 CRÍTICO', '🟡 MEDIO', '🟢 BAJO'],
                key="filtro_nivel_nepotismo"
            )
            if filtro_nivel == 'Todos':
                df_mostrar = df_sospechosos
            else:
                df_mostrar = df_sospechosos[df_sospechosos['Nivel_Sospecha'] == filtro_nivel]
            df_mostrar = df_mostrar.sort_values(['Nivel_Sospecha', 'Similitud (%)'], ascending=[True, False])
            st.markdown(f"#### 📋 Listado: {len(df_mostrar):,} casos")
            st.dataframe(df_mostrar, use_container_width=True, height=400)
            st.markdown("---")
            if st.button("💾 Exportar Casos Críticos"):
                try:
                    df_criticos = df_sospechosos[df_sospechosos['Nivel_Sospecha'] == '🔴 CRÍTICO']
                    output = BytesIO()
                    df_para_excel = preparar_para_excel(df_criticos)
                    with pd.ExcelWriter(output, engine='openpyxl') as writer:
                        df_para_excel.to_excel(writer, sheet_name='Casos Críticos', index=False)
                        resumen = pd.DataFrame({
                            'Nivel': ['🔴 CRÍTICO', '🟡 MEDIO', '🟢 BAJO'],
                            'Cantidad': [criticos, medios, bajos],
                            'Porcentaje': [
                                f"{(criticos/len(df_sospechosos)*100):.2f}%",
                                f"{(medios/len(df_sospechosos)*100):.2f}%",
                                f"{(bajos/len(df_sospechosos)*100):.2f}%"
                            ]
                        })
                        resumen.to_excel(writer, sheet_name='Resumen', index=False)
                    output.seek(0)
                    st.download_button(
                        label="⬇️ Descargar Excel",
                        data=output,
                        file_name="casos_nepotismo_criticos.xlsx",
                        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
                    )
                except Exception as e:
                    st.error(f"Error exportando: {e}")
        else:
            st.info("ℹ️ No hay coincidencias detectadas. Ajuste el umbral en la pestaña 'Análisis de Similitud'.")
def seccion_diccionario_datos():
    if not st.session_state.datasets:
        st.warning("⚠️ No hay datasets cargados. Por favor, cargue archivos en la sección Inicio.")
        return
    df = st.session_state.datasets[st.session_state.dataset_seleccionado]
    st.markdown("# 📋 Diccionario de Datos")
    st.markdown("""
    <div class='info-box'>
        <h3 style='color: #667eea; margin-top: 0;'>📖 Información Completa del Dataset</h3>
        <p>Diccionario completo de todas las columnas con estadísticas y codificación aplicada</p>
    </div>
    """, unsafe_allow_html=True)
    st.markdown("---")
    tab1, tab2 = st.tabs([
        "📊 Diccionario Completo",
        "🔢 Diccionario de Codificación"
    ])
    with tab1:
        st.markdown("### 📊 Información Detallada por Columna")
        diccionario = []
        for col in df.columns:
            tipo_dato = str(df[col].dtype)
            valores_nulos = df[col].isnull().sum()
            porcentaje_nulos = (valores_nulos / len(df)) * 100
            valores_unicos = df[col].nunique()
            muestra = df[col].dropna().head(3).tolist()
            muestra_str = ', '.join([str(x)[:50] for x in muestra])
            diccionario.append({
                'Columna': col,
                'Tipo de Dato': tipo_dato,
                'Valores Nulos': valores_nulos,
                '% Nulos': f"{porcentaje_nulos:.2f}%",
                'Valores Únicos': valores_unicos,
                'Muestra': muestra_str
            })
        df_diccionario = pd.DataFrame(diccionario)
        st.markdown(f"**Total de columnas:** {len(df_diccionario)}")
        st.dataframe(df_diccionario, use_container_width=True, height=500)
        st.markdown("---")
        col1, col2 = st.columns(2)
        with col1:
            st.markdown("#### 📊 Estadísticas Generales")
            col1a, col1b, col1c = st.columns(3)
            with col1a:
                st.metric("📋 Total Columnas", len(df.columns))
            with col1b:
                columnas_con_nulos = (df.isnull().sum() > 0).sum()
                st.metric("❌ Columnas con Nulos", columnas_con_nulos)
            with col1c:
                promedio_nulos = df.isnull().sum().mean()
                st.metric("📊 Promedio Nulos", f"{promedio_nulos:.2f}")
        with col2:
            st.markdown("#### 🎯 Distribución de Tipos de Datos")
            tipos_datos = df_diccionario['Tipo de Dato'].value_counts()
            fig = px.pie(
                values=tipos_datos.values,
                names=tipos_datos.index,
                title='Distribución de Tipos de Datos',
                color_discrete_sequence=px.colors.sequential.Purp
            )
            st.plotly_chart(fig, use_container_width=True)
        st.markdown("---")
        if st.button("💾 Exportar Diccionario a Excel"):
            try:
                output = BytesIO()
                df_para_excel = preparar_para_excel(df_diccionario)
                with pd.ExcelWriter(output, engine='openpyxl') as writer:
                    df_para_excel.to_excel(writer, sheet_name='Diccionario', index=False)
                    if st.session_state.encoding_dicts:
                        dict_rows = []
                        for col, mappings in st.session_state.encoding_dicts.items():
                            for original, codigo in mappings.items():
                                dict_rows.append({
                                    'Columna': col,
                                    'Valor Original': original,
                                    'Código': codigo
                                })
                        if dict_rows:
                            df_encoding = pd.DataFrame(dict_rows)
                            df_encoding.to_excel(writer, sheet_name='Codificación', index=False)
                output.seek(0)
                st.download_button(
                    label="⬇️ Descargar Excel",
                    data=output,
                    file_name="diccionario_datos.xlsx",
                    mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
                )
            except Exception as e:
                st.error(f"Error exportando: {e}")
    with tab2:
        st.markdown("### 🔢 Diccionario de Codificación de Variables")
        if not st.session_state.encoding_dicts:
            st.info("ℹ️ No hay variables codificadas. Primero genere el dataset codificado en la sección de Machine Learning.")
            st.markdown("""
            ### ¿Qué es la Codificación de Variables?
            La codificación de variables es el proceso de convertir datos categóricos (texto) a valores numéricos
            para que puedan ser utilizados en algoritmos de Machine Learning.
            **Ejemplo:**
            - **Original:** Casa, Terreno, Parcela
            - **Codificado:** 1, 2, 3
            **Ventajas:**
            - ✅ Permite usar algoritmos de ML
            - ✅ Mejora el rendimiento
            - ✅ Facilita cálculos matemáticos
            - ✅ Reduce el tamaño de los datos
            """)
        else:
            st.markdown("#### 📖 Columnas Codificadas")
            st.metric("📊 Total Columnas Codificadas", len(st.session_state.encoding_dicts))
            columna_seleccionada = st.selectbox(
                "Seleccione una columna para ver su diccionario:",
                list(st.session_state.encoding_dicts.keys()),
                key="dict_col_select"
            )
            if columna_seleccionada:
                st.markdown(f"#### 🔍 Diccionario: {columna_seleccionada}")
                mapping = st.session_state.encoding_dicts[columna_seleccionada]
                df_mapping = pd.DataFrame([
                    {"Valor Original": k, "Código Numérico": v}
                    for k, v in mapping.items()
                ]).sort_values('Código Numérico')
                col1, col2 = st.columns(2)
                with col1:
                    st.metric("📊 Valores Únicos", len(mapping))
                with col2:
                    st.metric("🔢 Rango de Códigos", f"0 - {len(mapping)-1}")
                st.dataframe(df_mapping, use_container_width=True, height=400)
                st.markdown("---")
                fig = px.bar(
                    df_mapping,
                    x='Código Numérico',
                    y='Valor Original',
                    orientation='h',
                    title=f'Codificación de {columna_seleccionada}',
                    color='Código Numérico',
                    color_continuous_scale='Purples'
                )
                st.plotly_chart(fig, use_container_width=True)
def seccion_reporte_ejecutivo():
    if not st.session_state.datasets:
        st.warning("⚠️ No hay datasets cargados. Por favor, cargue archivos en la sección Inicio.")
        return
    df = st.session_state.datasets[st.session_state.dataset_seleccionado]
    st.markdown("# 📄 Reporte Ejecutivo")
    st.markdown("""
    <div class='info-box'>
        <h3 style='color: #667eea; margin-top: 0;'>📊 Resumen Ejecutivo del Análisis Anticorrupción</h3>
        <p>Dashboard consolidado con los hallazgos más importantes del análisis</p>
    </div>
    """, unsafe_allow_html=True)
    st.markdown("---")
    st.markdown("## 📊 Métricas Generales")
    col1, col2, col3, col4, col5 = st.columns(5)
    with col1:
        st.metric("📄 Total Registros", f"{len(df):,}")
    with col2:
        if 'NombreCompleto' in df.columns:
            servidores_unicos = df['NombreCompleto'].nunique()
            st.metric("👥 Servidores Únicos", f"{servidores_unicos:,}")
    with col3:
        if 'anioEjercicio' in df.columns:
            años = df['anioEjercicio'].dropna().unique()
            st.metric("📅 Años Analizados", len(años))
    with col4:
        ente_col = 'declaracion_situacionPatrimonial_datosEmpleoCargoComision_nombreEntePublico'
        if ente_col in df.columns:
            entes = df[ente_col].nunique()
            st.metric("🏢 Entes Públicos", f"{entes:,}")
    with col5:
        memoria = df.memory_usage(deep=True).sum() / (1024**2)
        st.metric("💾 Datos Procesados", f"{memoria:.2f} MB")
    st.markdown("---")
    st.markdown("## 🚨 Hallazgos Principales")
    ingresos_cols = [col for col in df.columns if col.startswith('declaracion_situacionPatrimonial_ingresos_') 
                    and df[col].dtype in [np.float64, np.int64]]
    if ingresos_cols and 'NombreCompleto' in df.columns:
        df_temp = df.copy()
        df_temp['Total_Ingresos'] = df_temp[ingresos_cols].sum(axis=1)
        col1, col2 = st.columns(2)
        with col1:
            st.markdown("### 💰 Top 10 Mayores Ingresos")
            top_ingresos = df_temp.groupby('NombreCompleto')['Total_Ingresos'].sum().sort_values(ascending=False).head(10)
            df_top_ingresos = pd.DataFrame({
                'Servidor Público': top_ingresos.index,
                'Ingresos Totales': [f"${x:,.2f}" for x in top_ingresos.values]
            })
            st.dataframe(df_top_ingresos, use_container_width=True)
        with col2:
            st.markdown("### 📈 Evolución de Ingresos Promedio")
            if 'anioEjercicio' in df.columns:
                evolucion = df_temp.groupby('anioEjercicio')['Total_Ingresos'].mean().reset_index()
                fig = px.line(
                    evolucion,
                    x='anioEjercicio',
                    y='Total_Ingresos',
                    title='Ingresos Promedio por Año',
                    markers=True,
                    line_shape='spline'
                )
                fig.update_traces(line_color='#667eea', marker=dict(size=10, color='#764ba2'))
                st.plotly_chart(fig, use_container_width=True)
    st.markdown("---")
    st.markdown("## 🎯 Recomendaciones")
    recomendaciones = [
        {
            'Prioridad': '🔴 ALTA',
            'Área': 'Evolución Patrimonial',
            'Recomendación': 'Investigar casos con incremento patrimonial superior al 200% sin justificación por ingresos'
        },
        {
            'Prioridad': '🟡 MEDIA',
            'Área': 'Nepotismo',
            'Recomendación': 'Revisar coincidencias de apellidos en mismo ente público'
        },
        {
            'Prioridad': '🟢 BAJA',
            'Área': 'Declaraciones',
            'Recomendación': 'Verificar completitud de declaraciones en servidores con alta rotación'
        }
    ]
    df_recomendaciones = pd.DataFrame(recomendaciones)
    st.dataframe(df_recomendaciones, use_container_width=True)
    st.markdown("---")
    # Botones de descarga para hallazgos clave
    st.markdown("## 📥 Descargar Hallazgos Clave")
    col1, col2, col3 = st.columns(3)
    with col1:
        if 'NombreCompleto' in df.columns and ingresos_cols:
            top_ingresos = df_temp.groupby('NombreCompleto')['Total_Ingresos'].sum().sort_values(ascending=False).head(10).reset_index()
            if st.button("📊 Descargar Top 10 Ingresos"):
                output = BytesIO()
                df_para_excel = preparar_para_excel(top_ingresos)
                with pd.ExcelWriter(output, engine='openpyxl') as writer:
                    df_para_excel.to_excel(writer, sheet_name='Top 10 Ingresos', index=False)
                output.seek(0)
                st.download_button(
                    label="⬇️ Descargar Top 10 Ingresos (Excel)",
                    data=output,
                    file_name="top_10_ingresos.xlsx",
                    mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
                )
    with col2:
        if st.session_state.df_alertas is not None and not st.session_state.df_alertas.empty:
            if st.button("🚨 Descargar Alertas Detectadas"):
                output = BytesIO()
                df_para_excel = preparar_para_excel(st.session_state.df_alertas)
                with pd.ExcelWriter(output, engine='openpyxl') as writer:
                    df_para_excel.to_excel(writer, sheet_name='Alertas', index=False)
                output.seek(0)
                st.download_button(
                    label="⬇️ Descargar Alertas (Excel)",
                    data=output,
                    file_name="alertas_ejecutivo.xlsx",
                    mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
                )
    with col3:
        if st.session_state.df_anomalias is not None and not st.session_state.df_anomalias.empty:
            if st.button("⚠️ Descargar Anomalías Detectadas"):
                output = BytesIO()
                df_para_excel = preparar_para_excel(st.session_state.df_anomalias)
                with pd.ExcelWriter(output, engine='openpyxl') as writer:
                    df_para_excel.to_excel(writer, sheet_name='Anomalías', index=False)
                output.seek(0)
                st.download_button(
                    label="⬇️ Descargar Anomalías (Excel)",
                    data=output,
                    file_name="anomalias_ejecutivo.xlsx",
                    mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
                )
    st.markdown("---")
    if st.button("📄 Generar Reporte Completo PDF"):
        st.info("🚧 Funcionalidad de generación de PDF en desarrollo")
def seccion_borrar_datos():
    st.markdown("# 🗑️ Borrar Datos y Sesión")
    st.markdown("""
    <div style='background: linear-gradient(135deg, rgba(244, 67, 54, 0.1) 0%, rgba(211, 47, 47, 0.1) 100%); 
    padding: 1.5rem; border-radius: 15px; border: 2px solid rgba(244, 67, 54, 0.3);'>
        <h3 style='color: #f44336; margin-top: 0;'>⚠️ ADVERTENCIA</h3>
        <p style='margin-bottom: 0;'>
        Esta acción eliminará TODOS los datos cargados y reiniciará la sesión.
        Esta operación NO se puede deshacer.
        </p>
    </div>
    """, unsafe_allow_html=True)
    st.markdown("---")
    if st.session_state.datasets:
        st.markdown("### 📊 Datos Actualmente Cargados")
        for nombre, df in st.session_state.datasets.items():
            col1, col2, col3 = st.columns(3)
            with col1:
                st.write(f"**📁 {nombre}**")
            with col2:
                st.write(f"{len(df):,} registros")
            with col3:
                memoria = df.memory_usage(deep=True).sum() / (1024**2)
                st.write(f"{memoria:.2f} MB")
        st.markdown("---")
        confirmacion = st.text_input(
            "Para confirmar, escriba 'BORRAR' en mayúsculas:",
            key="confirmacion_borrar"
        )
        if confirmacion == "BORRAR":
            if st.button("🗑️ CONFIRMAR BORRADO", type="primary"):
                st.session_state.datasets = {}
                st.session_state.dataset_seleccionado = None
                st.session_state.encoding_dicts = {}
                st.session_state.df_encoded = None
                st.session_state.ml_models = {}
                st.session_state.df_procesado = None
                st.session_state.df_metricas_persona = None
                st.session_state.df_servidores_unicos = None
                st.session_state.df_anomalias = None
                st.session_state.df_alertas = None
                st.success("✅ Todos los datos han sido eliminados exitosamente.")
                st.balloons()
                st.rerun()
        else:
            st.info("ℹ️ Escriba 'BORRAR' para habilitar el botón de confirmación.")
    else:
        st.info("ℹ️ No hay datos cargados en la sesión actual.")

def run_checks():
    """
    Función de validación para verificar cálculos de métricas
    Imprime resumen para persona de ejemplo
    """
    if not st.session_state.datasets:
        print("No hay datasets cargados")
        return
    df = st.session_state.datasets[st.session_state.dataset_seleccionado]
    if 'NombreCompleto' not in df.columns:
        print("NombreCompleto no encontrado en dataset")
        return
    print("=" * 80)
    print("VALIDACIÓN DE CÁLCULOS - MÉTRICAS PATRIMONIALES")
    print("=" * 80)
    # Seleccionar persona de ejemplo
    persona_ejemplo = df['NombreCompleto'].value_counts().head(1).index[0]
    print(f"\nPersona de ejemplo: {persona_ejemplo}")
    print("-" * 80)
    df_persona = df[df['NombreCompleto'] == persona_ejemplo].copy()
    print(f"Total declaraciones: {len(df_persona)}")
    # Verificar columnas de ingresos
    col_ingreso_sp = 'declaracion_situacionPatrimonial_ingresos_remuneracionAnualCargoPublico_valor'
    col_otros_ingresos = 'declaracion_situacionPatrimonial_ingresos_otrosIngresosAnualesTotal_valor'
    if col_ingreso_sp in df_persona.columns:
        ingreso_sp_vals = pd.to_numeric(df_persona[col_ingreso_sp], errors='coerce').fillna(0)
        print(f"\nIngreso SP (Servidor Público):")
        print(f"  - Promedio: ${ingreso_sp_vals.mean():,.2f}")
        print(f"  - Máximo: ${ingreso_sp_vals.max():,.2f}")
        print(f"  - Mínimo: ${ingreso_sp_vals.min():,.2f}")
    if col_otros_ingresos in df_persona.columns:
        otros_ingresos_vals = pd.to_numeric(df_persona[col_otros_ingresos], errors='coerce').fillna(0)
        print(f"\nOtros Ingresos:")
        print(f"  - Promedio: ${otros_ingresos_vals.mean():,.2f}")
        print(f"  - Máximo: ${otros_ingresos_vals.max():,.2f}")
        print(f"  - Mínimo: ${otros_ingresos_vals.min():,.2f}")
    # Calcular ingresos totales correctamente
    ingreso_sp_vals = pd.to_numeric(df_persona[col_ingreso_sp], errors='coerce').fillna(0) if col_ingreso_sp in df_persona.columns else 0
    otros_ingresos_vals = pd.to_numeric(df_persona[col_otros_ingresos], errors='coerce').fillna(0) if col_otros_ingresos in df_persona.columns else 0
    df_persona['ingresos_totales_calc'] = ingreso_sp_vals + otros_ingresos_vals
    print(f"\nIngresos Totales (Calculado = SP + Otros):")
    print(f"  - Promedio: ${df_persona['ingresos_totales_calc'].mean():,.2f}")
    print(f"  - Máximo: ${df_persona['ingresos_totales_calc'].max():,.2f}")
    print(f"  - Mínimo: ${df_persona['ingresos_totales_calc'].min():,.2f}")
    # Verificar incrementos
    col_fecha = 'metadata_actualizacion'
    if col_fecha in df_persona.columns:
        df_persona[col_fecha] = pd.to_datetime(df_persona[col_fecha], utc=True, errors='coerce')
        df_persona_sorted = df_persona.sort_values(col_fecha)
        df_persona_sorted['ingreso_prev'] = df_persona_sorted['ingresos_totales_calc'].shift(1)
        df_persona_sorted['delta_abs'] = df_persona_sorted['ingresos_totales_calc'] - df_persona_sorted['ingreso_prev']
        df_persona_sorted['delta_pct'] = ((df_persona_sorted['ingresos_totales_calc'] - df_persona_sorted['ingreso_prev']) / 
                                          (df_persona_sorted['ingreso_prev'] + 1)) * 100
        incrementos_detectados = df_persona_sorted[df_persona_sorted['delta_pct'] > UMBRAL_ALERTA_INCREMENTO]
        print(f"\nIncrementos detectados (umbral: {UMBRAL_ALERTA_INCREMENTO}%):")
        print(f"  - Total: {len(incrementos_detectados)}")
        if len(incrementos_detectados) > 0:
            print(f"  - Máximo incremento: {incrementos_detectados['delta_pct'].max():.2f}%")
            print(f"  - Promedio incrementos: {incrementos_detectados['delta_pct'].mean():.2f}%")
    print("\n" + "=" * 80)
    print("VALIDACIÓN COMPLETADA")
    print("=" * 80)

def main():
    st.sidebar.markdown("""
    <div class='hero-header' style='padding: 1.5rem; margin-bottom: 1rem;'>
        <div class='brand-title' style='font-size: 2rem;'>🔍 AURORA ETHICS</div>
    </div>
    """, unsafe_allow_html=True)
    uploaded_files = st.sidebar.file_uploader(
        "📂 Seleccione archivos CSV/Excel",
        type=['csv', 'xls', 'xlsx'],
        accept_multiple_files=True,
        help="Carga archivos sin límite de tamaño"
    )
    if uploaded_files:
        for uploaded_file in uploaded_files:
            if uploaded_file.name not in st.session_state.datasets:
                with st.spinner(f"Procesando {uploaded_file.name}..."):
                    df = procesar_archivo_cargado(uploaded_file)
                    if df is not None:
                        st.session_state.datasets[uploaded_file.name] = df
                        st.sidebar.success(f"✅ {uploaded_file.name}")
    if st.session_state.datasets:
        st.markdown("""
        <div class='hero-header'>
            <div class='brand-title'>🔍 AURORA ETHICS</div>
            <p class='quote-text'>"La transparencia es el mejor antídoto contra la corrupción"</p>
            <p class='author-text'>Marco Antonio Sánchez Cedillo & Saul Avila Hernández</p>
            <p class='signature-text'>Datatón Anticorrupción México 2025 - SESNA</p>
        </div>
        """, unsafe_allow_html=True)
        dataset_names = list(st.session_state.datasets.keys())
        st.session_state.dataset_seleccionado = st.selectbox(
            "📁 Seleccione el dataset a analizar:",
            dataset_names,
            key="dataset_main_selector"
        )
        st.markdown("---")
        tab1, tab2, tab3, tab4, tab5, tab6, tab7, tab8, tab9, tab10, tab11 = st.tabs([
            "🏠 Inicio",
            "🔍 EDA",
            "📈 Visualización Global",
            "🎨 Personalizado",
            "🤖 Machine Learning",
            "🌍 Geo Inteligencia",
            "💰 Evolución Patrimonial",
            "🕸 Nepotismo",
            "📋 Diccionario",
            "📄 Reporte Ejecutivo",
            "🗑️ Borrar Datos"
        ])
        with tab1:
            seccion_inicio()
        with tab2:
            seccion_eda()
        with tab3:
            seccion_visualizacion_global()
        with tab4:
            seccion_visualizacion_personalizada()
        with tab5:
            seccion_machine_learning()
        with tab6:
            seccion_geo_inteligencia()
        with tab7:
            seccion_evolucion_patrimonial()
        with tab8:
            seccion_nepotismo()
        with tab9:
            seccion_diccionario_datos()
        with tab10:
            seccion_reporte_ejecutivo()
        with tab11:
            seccion_borrar_datos()
    else:
        st.markdown("""
        <div class='hero-header'>
            <div class='brand-title'>🔍 AURORA ETHICS</div>
            <p class='quote-text'>"La transparencia es el mejor antídoto contra la corrupción"</p>
            <p class='author-text'>Marco Antonio Sánchez Cedillo & Saul Avila Hernández</p>
            <p class='signature-text'>Datatón Anticorrupción México 2025 - SESNA</p>
        </div>
        """, unsafe_allow_html=True)
        st.markdown("---")
        st.info("👈 **Por favor, cargue uno o más archivos desde el panel lateral para comenzar el análisis**")

if __name__ == "__main__":
    main()