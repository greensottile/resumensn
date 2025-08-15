import os
import streamlit as st
import pandas as pd
import plotly.express as px
from datetime import datetime

# ==============================
# Config
# ==============================
st.set_page_config(page_title="Dashboard de Gastos (AFIP Recibidos)", layout="wide")

# ---------- Acceso con código (Opción 1)
def gate():
    # Persistimos que ya pasó el gate en la sesión
    if st.session_state.get("auth_ok"):
        return
    st.markdown("### 🔒 Acceso")
    pwd = st.text_input("Ingresá el **código de acceso**", type="password", help="Pedile al admin la clave.")
    correct = st.secrets.get("APP_PASS", "")
    if pwd == "":
        st.stop()
    if pwd != correct:
        st.error("Código incorrecto.")
        st.stop()
    st.session_state["auth_ok"] = True
    st.rerun()

gate()  # <- bloquea el resto de la app si la clave no coincide

# ==============================
# Parámetros y helpers
# ==============================
DEFAULT_CSV = "comprobantes_agosto_2025.csv"  # si lo subís al repo, usá este nombre
MAP_RUBRO = {
    "TEKNAL S A": "Alimento / Nutrición",
    "NUTRIFARMS S.R.L.": "Alimento / Nutrición",
    "FUTURO GAS S.A.": "Energía / Gas",
    "VIAL CRESPO SERVICIOS SRL": "Servicios / Mantenimiento",
    "ESPERANZA DISTRIBUCIONES SRL": "Insumos / Veterinaria",
    "WITPEL S.R.L.": "Insumos / Varios",
    "AGROCERES PIC ARGENTINA S A": "Genética",
    "FERRERO FABIO": "Compra de ganado / Hacienda",
    "ZILLI LUCAS MARIANO": "Servicios veterinarios",
    "PEDRO AGULLO SRL": "Transporte / Logística",
    "BONARDO CARLOS ALBERTO": "Transporte / Logística",
    "EBERHARDT AUGUSTO MARTIN": "Transporte / Logística",
}

def to_num(x):
    if pd.isna(x): return 0.0
    s = str(x).strip().replace("$","").replace("AR$","").replace("ARS","").replace(" ", "")
    # miles . y decimales , -> 1.234.567,89
    if "," in s and "." in s and s.rfind(",") > s.rfind("."):
        s = s.replace(".", "").replace(",", ".")
    else:
        s = s.replace(",", ".")
    try:
        return float(s)
    except:
        return 0.0

@st.cache_data
def load_data_from_df(df_in: pd.DataFrame) -> pd.DataFrame:
    df = df_in.copy()

    # Renombrar columnas a nombres más cómodos si vienen del CSV de AFIP
    rename_map = {
        "Denominación Emisor": "Proveedor",
        "Importe Total": "ImporteTotal",
    }
    df = df.rename(columns={k: v for k, v in rename_map.items() if k in df.columns})

    # Parseo de fecha en formato día/mes/año (dd/mm/yyyy)
    if "Fecha" not in df.columns:
        # Permitir "Fecha Emisión" u otros nombres comunes
        for alt_name in ["Fecha Emisión", "FechaEmision", "fecha", "FECHA"]:
            if alt_name in df.columns:
                df["Fecha"] = df[alt_name]
                break

    df["Fecha"] = pd.to_datetime(df["Fecha"], errors="coerce", dayfirst=True)

    # Conversión numérica robusta
    for col in ["Imp. Neto Gravado","Imp. Neto No Gravado","Imp. Op. Exentas","Otros Tributos","IVA","ImporteTotal"]:
        if col in df.columns:
            df[col] = df[col].apply(to_num)
        else:
            df[col] = 0.0

    # Clasificación por rubro
    df["Proveedor"] = df.get("Proveedor", "").fillna("Sin identificar")
    df["Rubro"] = df["Proveedor"].map(MAP_RUBRO).fillna("Sin clasificar")

    # Período para ordenar correctamente por mes
    df = df.dropna(subset=["Fecha"])
    if len(df) == 0:
        return df
    df["Mes_dt"] = df["Fecha"].dt.to_period("M")
    df["Mes"] = df["Mes_dt"].astype(str)

    return df

@st.cache_data
def to_csv_bytes(dfin):
    return dfin.to_csv(index=False).encode("utf-8")

# ==============================
# Entrada de datos
# ==============================
st.sidebar.header("Datos")
uploaded = st.sidebar.file_uploader("Subí tu CSV (AFIP)", type=["csv"], help="Podés omitir si dejaste el CSV en el repositorio.")

df_source = None
if uploaded is not None:
    try:
        df_source = pd.read_csv(uploaded)
    except Exception as e:
        st.sidebar.error(f"No se pudo leer el CSV subido: {e}")

if df_source is None:
    if os.path.exists(DEFAULT_CSV):
        df_source = pd.read_csv(DEFAULT_CSV)
        st.sidebar.info(f"Usando archivo del repo: `{DEFAULT_CSV}`")
    else:
        st.warning("No se encontró un CSV. Subí uno desde la barra lateral o agregalo al repo como "
                   f"`{DEFAULT_CSV}` y volvé a cargar.")
        st.stop()

df = load_data_from_df(df_source)

if len(df) == 0:
    st.warning("No hay filas con fechas válidas en el CSV.")
    st.stop()

# ==============================
# Filtros
# ==============================
st.sidebar.header("Filtros")

min_date = df["Fecha"].min()
max_date = df["Fecha"].max()
date_range = st.sidebar.date_input("Rango de fechas", value=(min_date, max_date))
if isinstance(date_range, tuple) and len(date_range) == 2:
    df = df[(df["Fecha"] >= pd.to_datetime(date_range[0])) & (df["Fecha"] <= pd.to_datetime(date_range[1]))]

proveedores = ["(Todos)"] + sorted([p for p in df["Proveedor"].dropna().unique()])
prov_sel = st.sidebar.selectbox("Proveedor", proveedores)
if prov_sel != "(Todos)":
    df = df[df["Proveedor"] == prov_sel]

rubros = ["(Todos)"] + sorted([r for r in df["Rubro"].dropna().unique()])
rubro_sel = st.sidebar.selectbox("Rubro", rubros)
if rubro_sel != "(Todos)":
    df = df[df["Rubro"] == rubro_sel]

if len(df) == 0:
    st.info("No hay datos que cumplan los filtros seleccionados.")
    st.stop()

# ==============================
# KPIs
# ==============================
total = df["ImporteTotal"].sum()
cant = len(df)
ticket = total / cant if cant else 0.0
iva = df["IVA"].sum()

st.title("📊 Dashboard de Gastos (AFIP - Comprobantes Recibidos)")
c1, c2, c3, c4 = st.columns(4)
c1.metric("💸 Total Gastado", f"${total:,.2f}")
c2.metric("🧾 Cant. de Facturas", f"{cant}")
c3.metric("🧮 Ticket Promedio", f"${ticket:,.2f}")
c4.metric("🧾 IVA (Crédito)", f"${iva:,.2f}")

# ==============================
# Gráficos
# ==============================
colA, colB = st.columns(2)

# Top Proveedores
g_prov = (df.groupby("Proveedor", as_index=False)["ImporteTotal"]
            .sum()
            .sort_values("ImporteTotal", ascending=False)
            .head(15))
fig_prov = px.bar(g_prov, x="Proveedor", y="ImporteTotal", title="Gasto por Proveedor (Top 15)")
fig_prov.update_layout(xaxis_tickangle=-45)
colA.plotly_chart(fig_prov, use_container_width=True)

# Distribución por Rubro
g_rubro = (df.groupby("Rubro", as_index=False)["ImporteTotal"]
             .sum()
             .sort_values("ImporteTotal", ascending=False))
fig_rubro = px.pie(g_rubro, names="Rubro", values="ImporteTotal", title="Distribución por Rubro", hole=0.35)
colB.plotly_chart(fig_rubro, use_container_width=True)

# Evolución por Mes (orden real por período)
g_mes = (df.groupby("Mes_dt", as_index=False)["ImporteTotal"]
           .sum()
           .sort_values("Mes_dt"))
g_mes["Mes"] = g_mes["Mes_dt"].astype(str)
st.plotly_chart(
    px.line(g_mes, x="Mes", y="ImporteTotal", title="Evolución Mensual", markers=True),
    use_container_width=True
)

# ==============================
# Detalle + descarga
# ==============================
st.subheader("Detalle (filtrado)")
st.dataframe(df.sort_values("Fecha", ascending=False), use_container_width=True)

st.download_button("⬇️ Descargar CSV filtrado", data=to_csv_bytes(df),
                   file_name="gastos_filtrados.csv", mime="text/csv")
