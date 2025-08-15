import os, re, io, hashlib, base64, unicodedata
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
# Extras (EPE)
MAP_RUBRO.update({
    "EMP.PROVINCIAL DE LA ENERGÍA DE SANTA FE": "Energía / Electricidad",
    "EMPRESA PROVINCIAL DE LA ENERGIA": "Energía / Electricidad",
    "EPE": "Energía / Electricidad",
})

def to_num(x):
    """Convierte a float aceptando formatos AR (1.234.567,89) y US (1,234,567.89)."""
    if pd.isna(x):
        return 0.0
    s = str(x).strip().replace("AR$","").replace("ARS","").replace("$","").replace(" ", "")
    if "," in s and "." in s:
        if s.rfind(".") > s.rfind(","):
            s = s.replace(",", "")
        else:
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
       3) PtoVta + Nro (sin CUIT)
       4) CUIT + Fecha + Total
       5) Proveedor + Fecha + Total (fallback)
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
    if pto and nro:
        return f"{pto.zfill(4)}-{nro.zfill(8)}"
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

# ==== Normalizador flexible para CSVs (agnóstico a nombres) ====
def _norm(s: str) -> str:
    s = str(s or "").strip().lower()
    s = unicodedata.normalize("NFKD", s).encode("ascii","ignore").decode("ascii")
    return s

def _pick_col(df, candidates=None, regex=None):
    cols = { _norm(c): c for c in df.columns }
    if candidates:
        for name in candidates:
            key = _norm(name)
            if key in cols:
                return cols[key]
    if regex:
        import re as _re
        for c in df.columns:
            if _re.search(regex, _norm(c)):
                return c
    return None

def normalize_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Normaliza columnas de CSVs heterogéneos: nombres, fechas dd/mm/yyyy, números, rubro, período y clave."""
    df = df.copy()

    # Detectar columnas principales (muchos alias)
    col_prov = _pick_col(df,
        ["Proveedor","Denominación Emisor","Denominacion Emisor","Razon Social","Razon Social Emisor",
         "Emisor","Nombre","Cliente/Proveedor"],
        regex=r"(proveedor|emisor|razon|cliente)"
    )
    col_fecha = _pick_col(df,
        ["Fecha","Fecha Emisión","FechaEmision","FECHA","Emision","Fecha Comprobante","Date"],
        regex=r"(fecha|emision|date)"
    )
    col_total = _pick_col(df,
        ["Importe Total","ImporteTotal","Total","Monto","Importe","Total Factura","Total a Pagar",
         "Total con IVA","IMPORTE TOTAL A PAGAR"],
        regex=r"(importe|monto|total)"
    )
    col_cuit  = _pick_col(df,
        ["CUIT","CUIT Emisor","CUIT_Emisor","C.U.I.T","CUIT Proveedor"],
        regex=r"(cuit)"
    )
    col_pto   = _pick_col(df,
        ["PtoVta","Punto de Venta","Pto Vta","PV","Pto"],
        regex=r"(pto|punto.*venta|ptovta|pv)"
    )
    col_nro   = _pick_col(df,
        ["NroCpbte","Nro Comprobante","Numero","Número","Comprobante","Nro Factura","Factura"],
        regex=r"(n(ro|um)|comprob|factur)"
    )
    col_cae   = _pick_col(df, ["CAE","Cod. Autorizacion","Codigo Autorizacion"], regex=r"(cae|autoriz)")

    # Estándar
    df["Proveedor"] = df[col_prov].astype(str).fillna("Sin identificar") if col_prov else "Sin identificar"
    df["Fecha"] = pd.to_datetime(df[col_fecha], errors="coerce", dayfirst=True) if col_fecha else pd.NaT

    if col_total:
        df["ImporteTotal"] = df[col_total].apply(to_num)
    else:
        # Si no hay total claro, elegir la columna numérica con mayor suma
        numcands = []
        for c in df.columns:
            try:
                s = df[c].apply(to_num)
                if s.sum() > 0:
                    numcands.append((s.sum(), c, s))
            except Exception:
                pass
        if numcands:
            numcands.sort(reverse=True, key=lambda x: x[0])
            df["ImporteTotal"] = numcands[0][2]
        else:
            df["ImporteTotal"] = 0.0

    for col in ["Imp. Neto Gravado","Imp. Neto No Gravado","Imp. Op. Exentas","Otros Tributos","IVA"]:
        if col in df.columns:
            df[col] = df[col].apply(to_num)
        else:
            df[col] = 0.0

    df["CUIT_Emisor"] = df[col_cuit].astype(str) if col_cuit else ""
    df["PtoVta"] = df[col_pto].astype(str) if col_pto else ""
    df["NroCpbte"] = df[col_nro].astype(str) if col_nro else ""
    df["CAE"] = df[col_cae].astype(str) if col_cae else ""

    # Si existe "0001-00123456" en alguna columna, separarlo
    if (not col_pto) or (not col_nro):
        pat = re.compile(r"(\d{3,4})\s*[-–]\s*(\d{6,10})")
        cols_scan = [col_nro] if col_nro else []
        if not cols_scan:
            cols_scan = [c for c in df.columns if re.search(r"(comprob|factur|nro|numero)", _norm(c))]
            if not cols_scan:
                cols_scan = [c for c in df.columns if df[c].dtype == object]
        pto_list, nro_list = [], []
        for _, rr in df.iterrows():
            found = ("","")
            for c in cols_scan:
                m = pat.search(str(rr.get(c, "")))
                if m:
                    found = (m.group(1), m.group(2)); break
            pto_list.append(found[0]); nro_list.append(found[1])
        if not col_pto: df["PtoVta"] = df.get("PtoVta","")
        if not col_nro: df["NroCpbte"] = df.get("NroCpbte","")
        df.loc[df["PtoVta"].eq(""), "PtoVta"] = pto_list
        df.loc[df["NroCpbte"].eq(""), "NroCpbte"] = nro_list

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

# --- Parser mejorado (EPE & co) ---
def find_issue_date(text: str):
    m = re.search(
        r'Fecha\s*(?:de\s*Emisi[oó]n)?\s*[:\-]\s*(\d{1,2}[\/\-]\d{1,2}[\/\-]\d{2,4})',
        text, flags=re.IGNORECASE
    )
    if m:
        try:
            return pd.to_datetime(m.group(1), dayfirst=True, errors="raise")
        except:
            pass
    for c in re.findall(r'(\d{1,2}[\/\-]\d{1,2}[\/\-]\d{2,4})', text):
        try:
            d = pd.to_datetime(c, dayfirst=True, errors="raise")
            if 2015 <= d.year <= 2035:
                return d
        except:
            pass
    return pd.NaT

def find_amount(text: str, keys):
    # 1) Preferencia por "IMPORTE TOTAL A PAGAR"
    m = re.search(r'(?i)importe\s+total\s+a\s+pagar[^\d]{0,40}([\$]?\s*\d[\d\.\,\s]*)', text)
    if m:
        return to_num(m.group(1))
    # 2) Claves conocidas línea por línea
    lines = text.splitlines()
    pat_num = re.compile(r'[\$]?\s*\d{1,3}([.,]\d{3})*([.,]\d{2})|\d+([.,]\d{2})?')
    for line in lines:
        low = line.lower()
        if any(k in low for k in keys):
            m = pat_num.search(line)
            if m:
                return to_num(m.group(0))
    # 3) Total genérico (último del documento)
    totals = re.findall(r'(?i)\btotal\b[^\d]{0,30}([\$]?\s*\d[\d\.,\s]*)', text)
    if totals:
        return to_num(totals[-1])
    return 0.0

def find_cuit(text: str):
    m = re.search(
        r'C\.?\s*U\.?\s*I\.?\s*T\.?\s*[:\s-]*([0-9]{2}-?[0-9]{8}-?[0-9])',
        text, flags=re.IGNORECASE
    )
    if not m:
        m = re.search(r'\b([0-9]{2}-?[0-9]{8}-?[0-9])\b', text)
    if m:
        digits = re.sub(r'\D', '', m.group(1))
        return f"{digits[:2]}-{digits[2:10]}-{digits[10:]}"
    return ""

def find_pto_vta_y_nro(text: str):
    # 1) "Pto de Venta 0001 ... Comp. Nro 00000012"
    m = re.search(
        r'pto\.?\s*de\s*venta\s*(\d+).*?(comp\.?\s*nro\.?|nro\.?)\s*(\d+)',
        text, flags=re.IGNORECASE | re.DOTALL
    )
    if m:
        return (m.group(1), m.group(3))
    # 2) "Factura(s) ... 0001 - 00000012" (espacios opcionales)
    m = re.search(
        r'facturas?\s*[A-Z]?\s*.*?(\d{4})\s*[-–]\s*(\d{6,10})',
        text, flags=re.IGNORECASE | re.DOTALL
    )
    if m:
        return (m.group(1), m.group(2))
    # 3) Línea con indicadores y guion con espacios
    for line in text.splitlines():
        if re.search(r'(fact|fc|comprob|pto\.?\s*de\s*venta|n[º°o]|no\.)', line, flags=re.IGNORECASE):
            mm = re.search(r'(\d{4})\s*[-–]\s*(\d{6,10})', line, flags=re.IGNORECASE)
            if mm:
                return (mm.group(1), mm.group(2))
    # 4) Patrón global
    mm = re.search(r'(\d{4})\s*[-–]\s*(\d{6,10})', text, flags=re.IGNORECASE)
    if mm:
        return (mm.group(1), mm.group(2))
    return ("", "")

def find_cae(text: str):
    m = re.search(r'\bCAE\b[^\d]*([0-9]{8,14})', text, flags=re.IGNORECASE)
    return m.group(1) if m else ""

def parse_invoice_text(text: str, known_vendors):
    fecha = find_issue_date(text)
    total = find_amount(text, ["importe total", "total a pagar", "total a facturar", "total"])
    iva   = find_amount(text, ["iva 21", "iva 10,5", "iva 10.5", "iva", "impuesto al valor agregado"])
    neto  = find_amount(text, ["neto gravado", "imp. neto gravado", "subtotal"])
    if neto == 0 and total and iva:
        neto = max(total - iva, 0)

    cuit = find_cuit(text)
    pto, nro = find_pto_vta_y_nro(text)
    cae = find_cae(text)

    low = text.lower()
    proveedor = None
    for name in MAP_RUBRO.keys():
        if name.lower() in low:
            proveedor = name
            break
    if not proveedor:
        m = re.search(r'raz[oó]n\s+social[:\s]*([A-Z0-9\-\.\&\s]+)', text, flags=re.IGNORECASE)
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
        "Fuente": "PDF Gmail",
    }
    row["Mes_dt"] = pd.to_datetime(row["Fecha"], errors="coerce").to_period("M") if pd.notna(row["Fecha"]) else pd.NaT
    row["Mes"] = str(row["Mes_dt"]) if pd.notna(row["Fecha"]) else ""
    row["Clave"] = make_key_from_row(row)
    row["RowID"] = make_row_id(row)
    return row


# ========= INTEGRACIÓN GMAIL =========
try:
    from googleapiclient.discovery import build as gbuild
    from google.auth.transport.requests import Request as GRequest
    from google.oauth2.credentials import Credentials as GCredentials
except Exception:
    gbuild = None

def gmail_creds_from_secrets():
    if gbuild is None:
        st.warning("Faltan librerías de Google. Agregá google-api-python-client, google-auth, google-auth-oauthlib a requirements.txt")
        return None
    info = st.secrets.get("gmail_oauth")
    if not info:
        st.info("Configurá en **Secrets** el bloque [gmail_oauth] con client_id, client_secret y refresh_token.")
        return None
    scopes = ["https://www.googleapis.com/auth/gmail.readonly"]
    creds = GCredentials(
        token=None,
        refresh_token=info.get("refresh_token"),
        token_uri=info.get("token_uri","https://oauth2.googleapis.com/token"),
        client_id=info.get("client_id"),
        client_secret=info.get("client_secret"),
        scopes=scopes
    )
    try:
        creds.refresh(GRequest())
    except Exception as e:
        st.error(f"No se pudo refrescar el token de Gmail: {e}")
        return None
    return creds

def gmail_service(creds):
    return gbuild("gmail", "v1", credentials=creds, cache_discovery=False)

def gmail_iter_messages(svc, q, max_pages=5):
    user_id = "me"
    token = None
    pages = 0
    while True:
        resp = svc.users().messages().list(userId=user_id, q=q, maxResults=500, pageToken=token).execute()
        for m in resp.get("messages", []):
            yield m["id"]
        token = resp.get("nextPageToken")
        pages += 1
        if not token or pages >= max_pages:
            break

def msg_headers_dict(payload):
    out = {}
    for h in payload.get("headers", []):
        name = (h.get("name") or "").lower()
        val = h.get("value") or ""
        if name in ("from","subject","date"):
            out[name] = val
    return out

def iter_pdf_parts(payload):
    stack = [payload]
    while stack:
        p = stack.pop()
        if "parts" in p:
            stack.extend(p["parts"])
        filename = p.get("filename") or ""
        mime = p.get("mimeType") or ""
        body = p.get("body", {})
        att_id = body.get("attachmentId")
        if att_id and (filename.lower().endswith(".pdf") or mime == "application/pdf"):
            yield att_id, filename

def gmail_download_attachment(svc, msg_id, att_id):
    att = svc.users().messages().attachments().get(userId="me", messageId=msg_id, id=att_id).execute()
    data = att.get("data")
    return base64.urlsafe_b64decode(data.encode("utf-8")) if data else b""


# ========= Entrada de datos =========
st.sidebar.header("Datos")

# CSVs múltiples (lector tolerante)
uploaded_files = st.sidebar.file_uploader(
    "Subí uno o varios CSV (AFIP/otros)",
    type=["csv"],
    accept_multiple_files=True,
    help="Podés subir varios meses y se consolidan."
)

use_default = st.sidebar.checkbox(
    f"Incluir CSV del repo (`{DEFAULT_CSV}`) si existe",
    value=True
)

frames = []

# CSVs subidos (con fallback a autodetect sep / latin-1)
if uploaded_files:
    for f in uploaded_files:
        try:
            df_tmp = pd.read_csv(f)
        except Exception:
            f.seek(0)
            df_tmp = pd.read_csv(f, sep=None, engine="python", encoding="latin-1")
        df_tmp["Fuente"] = f.name
        frames.append(df_tmp)

# CSV del repo
if use_default and os.path.exists(DEFAULT_CSV):
    try:
        df_repo = pd.read_csv(DEFAULT_CSV)
    except Exception:
        df_repo = pd.read_csv(DEFAULT_CSV, sep=None, engine="python", encoding="latin-1")
    df_repo["Fuente"] = DEFAULT_CSV
    frames.append(df_repo)

# --------- Importar desde Gmail ----------
with st.sidebar.expander("📥 Importar desde Gmail (PDF adjuntos)", expanded=False):
    st.caption("Requiere `gmail_oauth` en **Secrets** (client_id, client_secret, refresh_token).")
    dflt_start = datetime(datetime.now().year, 1, 1).date()
    d_from = st.date_input("Desde fecha", value=dflt_start, help="Filtro: after:YYYY/MM/DD")
    remitentes = st.text_input("Remitentes (coma)", value="", help="Dejalo vacío para traer de cualquiera.")
    asunto = st.text_input("Asunto contiene", value="", help="Vacío = sin filtro de asunto.")
    max_pages = st.number_input("Máx. páginas a leer (x500 mensajes)", min_value=1, max_value=20, value=3)
    if st.button("🔎 Buscar y procesar Gmail"):
        creds = gmail_creds_from_secrets()
        if creds is None:
            st.stop()
        svc = gmail_service(creds)
        # Query
        q = f"has:attachment filename:pdf after:{d_from.strftime('%Y/%m/%d')} "
        if remitentes.strip():
            ors = " OR ".join([f"from:{x.strip()}" for x in remitentes.split(",") if x.strip()])
            if ors: q += f"({ors}) "
        if asunto.strip():
            ors = " OR ".join([f"subject:{x.strip()}" for x in asunto.split(",") if x.strip()])
            if ors: q += f"({ors}) "
        st.write(f"Query: `{q}`")

        progress = st.progress(0)
        processed, added = 0, 0
        rows = []
        try:
            msg_ids = list(gmail_iter_messages(svc, q, max_pages=int(max_pages)))
            total_msgs = len(msg_ids)
            for i, mid in enumerate(msg_ids, start=1):
                msg = svc.users().messages().get(userId="me", id=mid, format="full").execute()
                payload = msg.get("payload", {})
                headers = msg_headers_dict(payload)
                for att_id, fname in iter_pdf_parts(payload):
                    data = gmail_download_attachment(svc, mid, att_id)
                    if not data:
                        continue
                    if fitz is None:
                        st.warning("PyMuPDF no está instalado; no se pueden leer PDFs.")
                        continue
                    text = extract_text_from_pdf(data)
                    row = parse_invoice_text(text, list(MAP_RUBRO.keys()))
                    row["Fuente"] = f"Gmail:{headers.get('from','?')} · {headers.get('subject','')}"
                    rows.append(row)
                    added += 1
                processed += 1
                progress.progress(min(i/total_msgs, 1.0))
        except Exception as e:
            st.error(f"Error leyendo Gmail: {e}")

        if rows:
            new_pdf_df = pd.DataFrame(rows).drop_duplicates(subset=["Clave"], keep="first")
            if "pdf_df" not in st.session_state:
                st.session_state["pdf_df"] = pd.DataFrame(columns=[
                    "Proveedor","Fecha","ImporteTotal","IVA","Imp. Neto Gravado","Imp. Neto No Gravado",
                    "Imp. Op. Exentas","Otros Tributos","CUIT_Emisor","PtoVta","NroCpbte","CAE",
                    "Fuente","Mes_dt","Mes","Clave","RowID"
                ])
            before = len(st.session_state["pdf_df"])
            st.session_state["pdf_df"] = pd.concat(
                [st.session_state["pdf_df"], new_pdf_df],
                ignore_index=True
            ).drop_duplicates(subset=["Clave"], keep="first")
            after = len(st.session_state["pdf_df"])
            st.success(f"📬 Procesados {processed} mensajes. Agregados {added} PDFs (únicos: {after-before}).")
            st.rerun()
        else:
            st.info("No se encontraron PDFs nuevos con la query indicada.")

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
                new_pdf_df = pd.DataFrame(rows).drop_duplicates(subset=["Clave"], keep="first")
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
        f"`{DEFAULT_CSV}` al repo o cargá PDFs/Gmail."
    )
    st.stop()

consolidated = pd.concat(parts, ignore_index=True, sort=False)

# Asegurar columnas mínimas
for col in ["Fecha","Mes_dt","Mes","ImporteTotal","Proveedor","Rubro","Clave","IVA","RowID",
            "CUIT_Emisor","PtoVta","NroCpbte","CAE","Fuente"]:
    if col not in consolidated.columns:
        if col in ["ImporteTotal","IVA"]:
            consolidated[col] = 0.0
        else:
            consolidated[col] = ""

# Recalcular Mes si faltan valores
if consolidated["Mes"].eq("").any():
    consolidated["Mes_dt"] = pd.to_datetime(consolidated["Fecha"], errors="coerce").to_period("M")
    consolidated["Mes"] = consolidated["Mes_dt"].astype(str)

# Dedupe por Clave (fuerte)
count_before = len(consolidated)
consolidated = consolidated.drop_duplicates(subset=["Clave"], keep="first")
count_after = len(consolidated)
dups_removed = count_before - count_after

# ========= Duplicados potenciales (mismo total y ±1 día) =========
c = consolidated.copy()
c["CUIT_norm"] = c["CUIT_Emisor"].apply(digits_only)
c["TotalRound"] = c["ImporteTotal"].round(2)
c["FechaDay"] = pd.to_datetime(c["Fecha"], errors="coerce").dt.floor("D")

c = c.dropna(subset=["FechaDay"])
c = c[c["TotalRound"].notna()]

tmp0 = c[["RowID","CUIT_norm","Proveedor","TotalRound","FechaDay"]].copy()
tmp0["Bucket"] = tmp0["FechaDay"]

tmpm1 = tmp0.copy(); tmpm1["Bucket"] = tmp0["FechaDay"] - pd.Timedelta(days=1)
tmpp1 = tmp0.copy(); tmpp1["Bucket"] = tmp0["FechaDay"] + pd.Timedelta(days=1)

cand = pd.concat([tmp0, tmpm1, tmpp1], ignore_index=True)

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

pot_unique = pot.sort_values("Bucket").drop_duplicates(subset=["RowID"], keep="first").copy()
pot_unique.rename(columns={"Bucket":"GrupoFecha"}, inplace=True)
pot_unique["GrupoFecha"] = pot_unique["GrupoFecha"].dt.date

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

# ========= Filtros globales =========
st.sidebar.header("Filtros")

min_date = pd.to_datetime(consolidated["Fecha"]).min()
max_date = pd.to_datetime(consolidated["Fecha"]).max()
date_range = st.sidebar.date_input("Rango de fechas (global)", value=(min_date, max_date))

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

# ========= Gráficos (con selectores propios) =========
st.subheader("Gráficos")

global_min = pd.to_datetime(df["Fecha"]).min().date()
global_max = pd.to_datetime(df["Fecha"]).max().date()

colA, colB = st.columns(2)

# Top Proveedores (rango propio)
with colA:
    st.markdown("**Gasto por Proveedor (Top 15)**")
    rng_prov = st.date_input("Rango de fechas (Top Proveedores)", value=(global_min, global_max), key="rng_prov")
    if isinstance(rng_prov, tuple) and len(rng_prov) == 2:
        d_from, d_to = pd.to_datetime(rng_prov[0]), pd.to_datetime(rng_prov[1])
        df_prov = df[(pd.to_datetime(df["Fecha"]) >= d_from) & (pd.to_datetime(df["Fecha"]) <= d_to)]
    else:
        df_prov = df
    if df_prov.empty:
        st.info("No hay datos en ese rango.")
    else:
        g_prov = (df_prov.groupby("Proveedor", as_index=False)["ImporteTotal"]
                    .sum()
                    .sort_values("ImporteTotal", ascending=False)
                    .head(15))
        fig_prov = px.bar(g_prov, x="Proveedor", y="ImporteTotal", title="Gasto por Proveedor (Top 15)")
        fig_prov.update_layout(xaxis_tickangle=-45)
        st.plotly_chart(fig_prov, use_container_width=True)

# Distribución por Rubro (rango propio)
with colB:
    st.markdown("**Distribución por Rubro**")
    rng_rubro = st.date_input("Rango de fechas (Rubro)", value=(global_min, global_max), key="rng_rubro")
    if isinstance(rng_rubro, tuple) and len(rng_rubro) == 2:
        d_from, d_to = pd.to_datetime(rng_rubro[0]), pd.to_datetime(rng_rubro[1])
        df_rubro = df[(pd.to_datetime(df["Fecha"]) >= d_from) & (pd.to_datetime(df["Fecha"]) <= d_to)]
    else:
        df_rubro = df
    if df_rubro.empty:
        st.info("No hay datos en ese rango.")
    else:
        g_rubro = (df_rubro.groupby("Rubro", as_index=False)["ImporteTotal"]
                    .sum()
                    .sort_values("ImporteTotal", ascending=False))
        fig_rubro = px.pie(g_rubro, names="Rubro", values="ImporteTotal", title="Distribución por Rubro", hole=0.35)
        st.plotly_chart(fig_rubro, use_container_width=True)

# Evolución por Mes (usa filtro global)
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
