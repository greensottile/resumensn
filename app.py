import os, re, io, hashlib
import streamlit as st
import pandas as pd
import plotly.express as px
from datetime import datetime

# ========= Config =========
st.set_page_config(page_title="Dashboard de Gastos (AFIP + CSVs + PDFs)", layout="wide")

# ---------- Gate por clave (Opción 1)
def gate():
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

gate()


# ========= Helpers =========
DEFAULT_CSV = "comprobantes_agosto_2025.csv"  # CSV opcional en el repo

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

def digits_only(s):
    return re.sub(r"\D", "", str(s)) if pd.notna(s) else ""

def make_key_from_row(r):
    """Clave fuerte para dedupe:
       1) CUIT + PtoVta + Nro
       2) CUIT + CAE
       3) CUIT + Fecha + Total
       4) Proveedor + Fecha + Total (fallback)
    """
    cuit = digits_only(r.get("CUIT_Emisor",""))
    pto  = digits_only(r.get("PtoVta",""))
    nro  = digits_only(r.get("NroCpbte",""))
    cae  = digits_only(r.get("CAE",""))
    fecha = r.get("Fecha", pd.NaT)
    total = r.get("ImporteTotal", 0.0)

    if cuit and pto and nro:
        return f"{cuit}-{pto.zfill(4)}-{nro.zfill(8)}"
    if cuit and cae:
        return f"{cuit}-{cae}"
    if cuit and pd.notna(fecha) and pd.notna(total) and float(total) > 0:
        return f"{cuit}-{pd.to_datetime(fecha).strftime('%Y%m%d')}-{float(total):.2f}"
    prov = str(r.get("Proveedor","")).strip()
    return f"{prov}-{pd.to_datetime(fecha).strftime('%Y%m%d') if pd.notna(fecha) else 'NA'}-{float(total):.2f}"

def make_row_id(r):
    """ID estable para exclusiones manuales (hash de campos clave)."""
    raw = "|".join([
        str(r.get("Clave","")),
        str(r.get("Proveedor","")),
        str(r.get("Fecha","")),
        f"{float(r.get('ImporteTotal',0.0)):.2f}",
        str(r.get("Fuente","")),
    ])
    return hashlib.md5(raw.encode("utf-8")).hexdigest()

def normalize_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Normaliza columnas de CSVs: nombres, fechas dd/mm/yyyy, números, rubro, período y clave."""
    df = df.copy()

    # Renombrar columnas comunes (AFIP y variantes)
    rename_map = {
        "Denominación Emisor": "Proveedor",
        "Importe Total": "ImporteTotal",
        "Fecha Emisión": "Fecha",
        "FechaEmision": "Fecha",
        "FECHA": "Fecha",
        "fecha": "Fecha",
        "CUIT": "CUIT_Emisor",
        "Número Desde": "NroCpbte",
        "Numero Desde": "NroCpbte",
        "Nro Desde": "NroCpbte",
        "Punto de Venta": "PtoVta",
        "Punto De Venta": "PtoVta",
        "Pto Vta": "PtoVta",
    }
    df = df.rename(columns={k: v for k, v in rename_map.items() if k in df.columns})

    # Proveedor
    if "Proveedor" not in df.columns:
        for alt in ["Denominación Receptor", "Razon Social", "Razón Social", "Cliente/Proveedor"]:
            if alt in df.columns:
                df["Proveedor"] = df[alt]
                break
    if "Proveedor" not in df.columns:
        df["Proveedor"] = "Sin identificar"

    # Fechas dd/mm/yyyy
    if "Fecha" in df.columns:
        df["Fecha"] = pd.to_datetime(df["Fecha"], errors="coerce", dayfirst=True)
    else:
        df["Fecha"] = pd.NaT

    # Numéricos
    for col in ["Imp. Neto Gravado","Imp. Neto No Gravado","Imp. Op. Exentas","Otros Tributos","IVA","ImporteTotal"]:
        if col in df.columns:
            df[col] = df[col].apply(to_num)
        else:
            df[col] = 0.0

    # Campos estándar para dedupe
    if "CUIT_Emisor" not in df.columns:
        df["CUIT_Emisor"] = ""
    if "PtoVta" not in df.columns:
        df["PtoVta"] = ""
    if "NroCpbte" not in df.columns:
        df["NroCpbte"] = ""
    if "CAE" not in df.columns:
        df["CAE"] = ""

    # Rubro / Período
    df["Proveedor"] = df["Proveedor"].fillna("Sin identificar")
    df["Rubro"] = df["Proveedor"].map(MAP_RUBRO).fillna("Sin clasificar")

    df = df.dropna(subset=["Fecha"])
    if len(df) == 0:
        df["Mes_dt"] = pd.PeriodIndex([], freq="M")
        df["Mes"] = ""
        df["Clave"] = ""
        df["RowID"] = ""
        return df

    df["Mes_dt"] = df["Fecha"].dt.to_period("M")
    df["Mes"] = df["Mes_dt"].astype(str)

    # Clave fuerte + RowID
    df["Clave"] = df.apply(make_key_from_row, axis=1)
    df["RowID"] = df.apply(make_row_id, axis=1)
    return df

@st.cache_data
def to_csv_bytes(dfin: pd.DataFrame):
    return dfin.to_csv(index=False).encode("utf-8")


# ========= PDFs =========
try:
    import fitz  # PyMuPDF
except Exception:
    fitz = None

def extract_text_from_pdf(file_bytes: bytes) -> str:
    if fitz is None:
        return ""
    text = ""
    with fitz.open(stream=file_bytes, filetype="pdf") as doc:
        for page in doc:
            text += page.get_text("text") + "\n"
    return text

def find_first_date(text: str):
    # dd/mm/yyyy o dd-mm-yyyy
    candidates = re.findall(r'(\d{1,2}[/-]\d{1,2}[/-]\d{2,4})', text)
    for c in candidates:
        try:
            d = pd.to_datetime(c, dayfirst=True, errors="raise")
            if 2015 <= d.year <= 2035:
                return d
        except:
            pass
    return pd.NaT

def find_amount(text: str, keys):
    lines = text.splitlines()
    pat_num = re.compile(r'[\$]?\s*\d{1,3}(\.\d{3})*(,\d{2})|\d+(\,\d{2})?')
    for line in lines:
        low = line.lower()
        if any(k in low for k in keys):
            m = pat_num.search(line)
            if m:
                return to_num(m.group(0))
    totals = re.findall(r'(?i)\btotal\b[^\d]{0,20}([\$]?\s*\d[\d\.\,]*)', text)
    if totals:
        return to_num(totals[-1])
    return 0.0

def find_cuit(text: str):
    m = re.search(r'(?i)CUIT[^\d]*([0-9]{2}-?[0-9]{8}-?[0-9])', text)
    if not m:
        m = re.search(r'\b([0-9]{2}-?[0-9]{8}-?[0-9])\b', text)
    if m:
        digits = re.sub(r'\D', '', m.group(1))
        return f"{digits[:2]}-{digits[2:10]}-{digits[10:]}"
    return ""

def find_pto_vta_y_nro(text: str):
    m1 = re.search(r'(?i)pto\.?\s*de\s*venta\s*(\d+).*?(?i)(comp\.?\s*nro\.?|nro\.?)\s*(\d+)', text)
    m2 = re.search(r'(?i)factura\s*[a-z]?\s*.*?(\d{4})-(\d{8})', text)
    if m1:
        return (m1.group(1), m1.group(2))
    if m2:
        return (m2.group(1), m2.group(2))
    return ("","")

def find_cae(text: str):
    m = re.search(r'(?i)\bCAE\b[^\d]*([0-9]{8,14})', text)
    return m.group(1) if m else ""

def parse_invoice_text(text: str, known_vendors):
    fecha = find_first_date(text)
    total = find_amount(text, ["importe total", "total a pagar", "total a facturar", "total"])
    iva   = find_amount(text, ["iva 21", "iva 10,5", "iva 10.5", "iva", "impuesto al valor agregado"])
    neto  = find_amount(text, ["neto gravado", "imp. neto gravado", "subtotal"])
    if neto == 0 and total and iva:
        neto = max(total - iva, 0)

    cuit = find_cuit(text)
    pto, nro = find_pto_vta_y_nro(text)
    cae = find_cae(text)

    # Proveedor por heurística simple
    low = text.lower()
    proveedor = None
    for name in known_vendors:
        if name.lower() in low:
            proveedor = name
            break
    if not proveedor:
        m = re.search(r'(?i)raz[oó]n\s+social[:\s]*([A-Z0-9\-\.\&\s]+)', text)
        if m:
            proveedor = m.group(1).strip()
    if not proveedor:
        proveedor = "Sin identificar"

    row = {
        "Proveedor": proveedor,
        "Fecha": fecha,
        "ImporteTotal": float(total or 0.0),
        "IVA": float(iva or 0.0),
        "Imp. Neto Gravado": float(neto or 0.0),
        "Imp. Neto No Gravado": 0.0,
        "Imp. Op. Exentas": 0.0,
        "Otros Tributos": 0.0,
        "CUIT_Emisor": cuit,
        "PtoVta": pto,
        "NroCpbte": nro,
        "CAE": cae,
        "Fuente": "PDF subido",
    }
    row["Mes_dt"] = pd.to_datetime(row["Fecha"], errors="coerce").to_period("M") if pd.notna(row["Fecha"]) else pd.NaT
    row["Mes"] = str(row["Mes_dt"]) if pd.notna(row["Fecha"]) else ""
    row["Clave"] = make_key_from_row(row)
    row["RowID"] = make_row_id(row)
    return row


# ========= Entrada de datos =========
st.sidebar.header("Datos")

# CSVs múltiples
uploaded_files = st.sidebar.file_uploader(
    "Subí uno o varios CSV (AFIP)",
    type=["csv"],
    accept_multiple_files=True,
    help="Podés subir varios meses y se consolidan."
)

use_default = st.sidebar.checkbox(
    f"Incluir CSV del repo (`{DEFAULT_CSV}`) si existe",
    value=True
)

frames = []

# CSVs subidos
if uploaded_files:
    for f in uploaded_files:
        try:
            df_tmp = pd.read_csv(f)
            df_tmp["Fuente"] = f.name
            frames.append(df_tmp)
        except Exception as e:
            st.sidebar.error(f"No se pudo leer `{f.name}`: {e}")

# CSV del repo
if use_default and os.path.exists(DEFAULT_CSV):
    try:
        df_repo = pd.read_csv(DEFAULT_CSV)
        df_repo["Fuente"] = DEFAULT_CSV
        frames.append(df_repo)
    except Exception as e:
        st.sidebar.error(f"No se pudo leer `{DEFAULT_CSV}` del repo: {e}")

# Normalizar CSVs
df_csv_all = pd.DataFrame()
if frames:
    raw_all = pd.concat(frames, ignore_index=True, sort=False)
    df_csv_all = normalize_columns(raw_all)

# PDFs (acumulados en sesión)
if "pdf_df" not in st.session_state:
    st.session_state["pdf_df"] = pd.DataFrame(columns=[
        "Proveedor","Fecha","ImporteTotal","IVA","Imp. Neto Gravado","Imp. Neto No Gravado",
        "Imp. Op. Exentas","Otros Tributos","CUIT_Emisor","PtoVta","NroCpbte","CAE",
        "Fuente","Mes_dt","Mes","Clave","RowID"
    ])

with st.sidebar.expander("➕ Subir facturas PDF (uno o varios)", expanded=False):
    try:
        import fitz  # ensure available
        pdf_files = st.file_uploader("Seleccioná PDF(s)", type=["pdf"], accept_multiple_files=True)
        if pdf_files:
            rows = []
            for f in pdf_files:
                try:
                    text = extract_text_from_pdf(f.read())
                    rows.append(parse_invoice_text(text, list(MAP_RUBRO.keys())))
                except Exception as e:
                    st.warning(f"No se pudo procesar {f.name}: {e}")
            if rows:
                new_pdf_df = pd.DataFrame(rows)
                # Dedupe dentro de PDFs por Clave
                new_pdf_df = new_pdf_df.drop_duplicates(subset=["Clave"], keep="first")
                st.session_state["pdf_df"] = pd.concat(
                    [st.session_state["pdf_df"], new_pdf_df],
                    ignore_index=True
                ).drop_duplicates(subset=["Clave"], keep="first")
                st.success(f"PDFs agregados. Totales en memoria: {len(st.session_state['pdf_df'])}")
        c1, c2 = st.columns(2)
        if c1.button("🧹 Limpiar PDFs cargados"):
            st.session_state["pdf_df"] = st.session_state["pdf_df"].iloc[0:0]
            st.rerun()
        if c2.button("👀 Ver PDFs cargados"):
            st.dataframe(
                st.session_state["pdf_df"][["Proveedor","Fecha","ImporteTotal","CUIT_Emisor","PtoVta","NroCpbte","CAE","Clave"]],
                use_container_width=True
            )
    except Exception:
        st.info("Para analizar PDFs se requiere PyMuPDF. Asegurate de tener 'pymupdf' en requirements.txt.")

# ========= Consolidado + Dedupe fuerte =========
parts = []
if not df_csv_all.empty:
    parts.append(df_csv_all)
if not st.session_state["pdf_df"].empty:
    parts.append(st.session_state["pdf_df"])

if not parts:
    st.warning(
        "No se encontraron datos. Subí CSVs desde la barra lateral, agregá "
        f"`{DEFAULT_CSV}` al repo o cargá PDFs."
    )
    st.stop()

consolidated = pd.concat(parts, ignore_index=True, sort=False)

# Asegurar columnas mínimas
for col in ["Fecha","Mes_dt","Mes","ImporteTotal","Proveedor","Rubro","Clave","IVA","RowID","CUIT_Emisor","PtoVta","NroCpbte","CAE","Fuente"]:
    if col not in consolidated.columns:
        if col in ["ImporteTotal","IVA"]:
            consolidated[col] = 0.0
        else:
            consolidated[col] = ""

# Recalcular Mes si faltan valores
if consolidated["Mes"].eq("").any():
    consolidated["Mes_dt"] = pd.to_datetime(consolidated["Fecha"], errors="coerce").dt.to_period("M")
    consolidated["Mes"] = consolidated["Mes_dt"].astype(str)

# Dedupe por Clave (fuerte)
count_before = len(consolidated)
consolidated = consolidated.drop_duplicates(subset=["Clave"], keep="first")
count_after = len(consolidated)
dups_removed = count_before - count_after

# ========= Duplicados potenciales (mismo total y ±1 día) =========
# Bucket por CUIT (si hay), total redondeado y ventana de fecha (-1, 0, +1)
c = consolidated.copy()
c["CUIT_norm"] = c["CUIT_Emisor"].apply(digits_only)
c["TotalRound"] = c["ImporteTotal"].round(2)
c["FechaDay"] = pd.to_datetime(c["Fecha"], errors="coerce").dt.floor("D")

# filas que no tengan fecha o total no se pueden comparar
c = c.dropna(subset=["FechaDay"])
c = c[c["TotalRound"].notna()]

# armamos buckets: día, día-1, día+1
tmp0 = c[["RowID","CUIT_norm","Proveedor","TotalRound","FechaDay"]].copy()
tmp0["Bucket"] = tmp0["FechaDay"]

tmpm = tmp0.copy(); tmpm["Bucket"] = tmpm["FechaDay"] - pd.Timedelta(days=1)
tmpp = tmp0.copy(); tmpp["Bucket"] = tmpm["FechaDay"] + pd.Timedelta(days=2)  # +1 desde el original (0 + 2 por haber restado 1 en tmpm)

# NOTA: la línea anterior crea +1 real respecto a tmp0:  (FechaDay-1)+2 => FechaDay+1

cand = pd.concat([tmp0, tmpm, tmpp], ignore_index=True)

# Si hay CUIT, agrupamos por CUIT; si no, por Proveedor (fallback)
cand["ClaveGrupo"] = cand.apply(
    lambda r: f"C-{r['CUIT_norm']}-{r['TotalRound']:.2f}-{r['Bucket'].date()}"
              if r["CUIT_norm"] else f"P-{r['Proveedor']}-{r['TotalRound']:.2f}-{r['Bucket'].date()}",
    axis=1
)

grp = cand.groupby("ClaveGrupo")["RowID"].nunique()
grupo_validos = set(grp[grp > 1].index)

pot = cand[cand["ClaveGrupo"].isin(grupo_validos)].merge(
    c[["RowID","Proveedor","Fecha","ImporteTotal","CUIT_Emisor","PtoVta","NroCpbte","CAE","Fuente","Clave"]],
    on="RowID",
    how="left"
)

# Nos quedamos con 1 fila por RowID (puede aparecer en varios buckets)
pot_unique = pot.sort_values("Bucket").drop_duplicates(subset=["RowID"], keep="first").copy()
pot_unique.rename(columns={"Bucket":"GrupoFecha"}, inplace=True)
pot_unique["GrupoFecha"] = pot_unique["GrupoFecha"].dt.date

# Estado de exclusiones manuales
if "manual_exclude_ids" not in st.session_state:
    st.session_state["manual_exclude_ids"] = set()

with st.expander("🔍 Duplicados potenciales (mismo total y ±1 día)", expanded=False):
    if pot_unique.empty:
        st.write("No se detectaron duplicados potenciales con los criterios actuales.")
    else:
        st.caption(
            "Criterio: mismo **Total** (redondeado a 2 decimales) y **misma fecha**, "
            "o dentro de una ventana de **±1 día**. Se prioriza CUIT; si no hay, se usa Proveedor."
        )
        st.write(f"Grupos detectados: **{len(grupo_validos)}** · Registros potencialmente duplicados: **{len(pot_unique)}**")

        # Checkbox por fila
        pot_show = pot_unique[[
            "RowID","ClaveGrupo","GrupoFecha","Proveedor","Fecha","ImporteTotal",
            "CUIT_Emisor","PtoVta","NroCpbte","CAE","Fuente","Clave"
        ]].copy()
        pot_show.insert(0, "Excluir", pot_show["RowID"].isin(st.session_state["manual_exclude_ids"]))

        edited = st.data_editor(
            pot_show,
            use_container_width=True,
            hide_index=True,
            column_config={
                "Excluir": st.column_config.CheckboxColumn(help="Marcá los que quieras excluir del consolidado."),
                "ImporteTotal": st.column_config.NumberColumn(format="%.2f"),
                "Fecha": st.column_config.DatetimeColumn(format="DD/MM/YYYY"),
            }
        )

        # Botones
        c1, c2, c3 = st.columns(3)
        if c1.button("Aplicar exclusiones"):
            to_exclude = set(edited.loc[edited["Excluir"], "RowID"].tolist())
            st.session_state["manual_exclude_ids"] = to_exclude
            st.success(f"Guardadas {len(to_exclude)} exclusiones manuales.")
            st.rerun()
        if c2.button("Limpiar exclusiones manuales"):
            st.session_state["manual_exclude_ids"] = set()
            st.rerun()
        st.download_button(
            "⬇️ Descargar duplicados potenciales (CSV)",
            data=to_csv_bytes(pot_show.drop(columns=["Excluir"])),
            file_name="duplicados_potenciales.csv",
            mime="text/csv"
        )

# Aplicar exclusiones manuales al consolidado
if st.session_state["manual_exclude_ids"]:
    consolidated = consolidated[~consolidated["RowID"].isin(st.session_state["manual_exclude_ids"])]

# ========= Resumen de fuentes =========
with st.sidebar.expander("Resumen de archivos cargados", expanded=False):
    try:
        resumen = consolidated.groupby("Fuente", as_index=False).agg(
            Registros=("Proveedor","count"),
            Desde=("Fecha","min"),
            Hasta=("Fecha","max"),
            Total=("ImporteTotal","sum")
        )
        st.dataframe(resumen, use_container_width=True)
    except Exception:
        st.write("No disponible.")

# ========= Filtros =========
st.sidebar.header("Filtros")

min_date = pd.to_datetime(consolidated["Fecha"]).min()
max_date = pd.to_datetime(consolidated["Fecha"]).max()
date_range = st.sidebar.date_input("Rango de fechas", value=(min_date, max_date))

df = consolidated.copy()
if isinstance(date_range, tuple) and len(date_range) == 2:
    df = df[(pd.to_datetime(df["Fecha"]) >= pd.to_datetime(date_range[0])) &
            (pd.to_datetime(df["Fecha"]) <= pd.to_datetime(date_range[1]))]

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

# ========= KPIs =========
total = df["ImporteTotal"].sum()
cant = len(df)
ticket = total / cant if cant else 0.0
iva = df["IVA"].sum()

st.title("📊 Dashboard de Gastos (AFIP + CSVs + PDFs)")
st.caption(
    f"Consolidado con deduplicado por clave. Duplicados descartados automáticamente: **{dups_removed}**. "
    f"Exclusiones manuales aplicadas: **{len(st.session_state.get('manual_exclude_ids', []))}**."
)
c1, c2, c3, c4 = st.columns(4)
c1.metric("💸 Total Gastado", f"${total:,.2f}")
c2.metric("🧾 Cant. de Facturas", f"{cant}")
c3.metric("🧮 Ticket Promedio", f"${ticket:,.2f}")
c4.metric("🧾 IVA (Crédito)", f"${iva:,.2f}")

# ========= Gráficos =========
colA, colB = st.columns(2)

g_prov = (df.groupby("Proveedor", as_index=False)["ImporteTotal"]
            .sum()
            .sort_values("ImporteTotal", ascending=False)
            .head(15))
fig_prov = px.bar(g_prov, x="Proveedor", y="ImporteTotal", title="Gasto por Proveedor (Top 15)")
fig_prov.update_layout(xaxis_tickangle=-45)
colA.plotly_chart(fig_prov, use_container_width=True)

g_rubro = (df.groupby("Rubro", as_index=False)["ImporteTotal"]
             .sum()
             .sort_values("ImporteTotal", ascending=False))
fig_rubro = px.pie(g_rubro, names="Rubro", values="ImporteTotal", title="Distribución por Rubro", hole=0.35)
colB.plotly_chart(fig_rubro, use_container_width=True)

g_mes = (df.groupby("Mes_dt", as_index=False)["ImporteTotal"]
           .sum()
           .sort_values("Mes_dt"))
g_mes["Mes"] = g_mes["Mes_dt"].astype(str)
st.plotly_chart(
    px.line(g_mes, x="Mes", y="ImporteTotal", title="Evolución Mensual", markers=True),
    use_container_width=True
)

# ========= Detalle + descarga =========
st.subheader("Detalle (filtrado)")
st.dataframe(df.sort_values("Fecha", ascending=False), use_container_width=True)

st.download_button("⬇️ Descargar CSV filtrado", data=to_csv_bytes(df),
                   file_name="gastos_filtrados.csv", mime="text/csv")
