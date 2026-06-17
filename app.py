import math
import unicodedata
from pathlib import Path

import numpy as np
import pandas as pd
import geopandas as gpd
import streamlit as st
import folium
from folium.features import GeoJsonTooltip, GeoJsonPopup
from streamlit_folium import st_folium
from shapely.geometry import Point

try:
    import osmnx as ox
    import networkx as nx
    OSMNX_AVAILABLE = True
except Exception:
    OSMNX_AVAILABLE = False

try:
    import pulp
    PULP_AVAILABLE = True
except Exception:
    PULP_AVAILABLE = False

APP_DIR = Path(__file__).parent
DATA_DIR = APP_DIR / "data"
LOGO_PATH = DATA_DIR / "logo_bler.png"
VENTAS_PATH = DATA_DIR / "ventas_clientes_asesor_zona.xlsx"

st.set_page_config(
    page_title="UPZ Bogota | Pinturas BLER",
    page_icon="🗺️",
    layout="wide",
)

st.markdown(
    """
    <style>
    .main .block-container {padding-top: 1.2rem; padding-bottom: 2rem; max-width: 1500px;}
    h1, h2, h3 {letter-spacing: -0.02em;}
    .brand-card {
        background: linear-gradient(90deg, #ffffff 0%, #f8fbff 100%);
        border: 1px solid #e7edf3;
        border-radius: 16px;
        padding: 12px 18px;
        margin-bottom: 8px;
    }
    .info-card {
        background: rgba(255,255,255,0.04);
        border: 1px solid rgba(148,163,184,0.25);
        border-radius: 14px;
        padding: 18px 20px;
        margin-top: 12px;
    }
    .cluster-card {
        background: rgba(31, 41, 55, 0.04);
        border: 1px solid rgba(100, 116, 139, 0.25);
        border-radius: 16px;
        padding: 18px 20px;
        margin-top: 14px;
    }
    .small-note {color:#64748b; font-size: 0.90rem;}
    div[data-testid="stVerticalBlock"] > div:has(iframe) {overflow: visible;}
    iframe {border-radius: 12px;}
    </style>
    """,
    unsafe_allow_html=True,
)

NAME_COL = "NOMBRE"
CLUSTER_COL = "cluster_k4"
VARIABLE_LABELS = {
    "NOMBRE": "UPZ",
    "numhogares": "Hogares",
    "area": "Área",
    "densidadPoblacional": "Densidad poblacional",
    "estrato_asignado": "Estrato asignado",
    "personasReal": "Personas",
    "num_ferreterias": "Ferreterías",
    "num_tiendasPinturas": "Tiendas de pinturas",
    "num_construcciones": "Comercios de construcción",
    "num_industriales": "Comercios industriales",
    "num_clientes": "Clientes",
    "primary": "Vías primarias",
    "secondary": "Vías secundarias",
    "tertiary": "Vías terciarias",
    "trunk": "Vías troncales",
    "valResidencial": "Valor residencial",
    "valComercial": "Valor comercial",
    "indiceaccesibilidad": "Índice de accesibilidad",
    "LOCALIDAD": "Localidad",
    "upz_count": "Número de UPZ",
    "ventas_2024": "Ventas reales 2024",
    "ventas_2025": "Ventas reales 2025",
    "crecimiento_ventas": "Crecimiento de ventas",
    "participacion_zona_2025": "Participación zona 2025",
    "cluster_k4": "Cluster oficial",
    "cluster_k5": "Cluster k=5",
    "cluster_gmm": "Cluster GMM",
    "cluster_hca": "Cluster jerarquico",
    "cluster_hca_mapped": "Cluster jerarquico ajustado",
    "consistente": "Cluster consistente",
    "silhouette": "Silhouette",
}

CLUSTER_VARIABLES = {
    "cluster_k4", "cluster_k5", "cluster_gmm", "cluster_hca",
    "cluster_hca_mapped", "consistente", "silhouette"
}

BASE_POPUP_FIELDS = [
    NAME_COL, "LOCALIDAD", "numhogares", "personasReal", "estrato_asignado", "num_clientes",
    "num_ferreterias", "indiceaccesibilidad", CLUSTER_COL
]
LOCALIDAD_POPUP_FIELDS = [
    "LOCALIDAD", "upz_count", "numhogares", "personasReal", "num_clientes",
    "num_ferreterias", "ventas_2025", "ventas_2024", "crecimiento_ventas"
]
CLUSTER_COLORS = {0: "#0B4EA2", 1: "#E42313", 2: "#F59E0B", 3: "#22C55E"}


def normalize_match(s):
    if pd.isna(s):
        return ""
    text = str(s).strip().upper()
    # Algunas fuentes geograficas llegan con codificacion mojibake, por ejemplo RÃO en lugar de RIO.
    text = text.replace("Ã", "I").replace("Ã\u008d", "I")
    text = unicodedata.normalize("NFKD", text).encode("ascii", "ignore").decode("ascii")
    return " ".join(text.replace("-", " ").split())


def build_upz_localidad_map():
    mapping = {
        # 01 Usaquén
        "PASEO DE LOS LIBERTADORES": "Usaquén", "VERBENAL": "Usaquén", "LA URIBE": "Usaquén",
        "LOS CEDROS": "Usaquén", "SAN CRISTOBAL NORTE": "Usaquén", "TOBERIN": "Usaquén",
        "COUNTRY CLUB": "Usaquén", "SANTA BARBARA": "Usaquén", "USAQUEN": "Usaquén",
        # 02 Chapinero
        "EL REFUGIO": "Chapinero", "SAN ISIDRO PATIOS": "Chapinero", "PARDO RUBIO": "Chapinero",
        "CHICO LAGO": "Chapinero", "CHAPINERO": "Chapinero",
        # 03 Santa Fe
        "SAGRADO CORAZON": "Santa Fe", "LA MACARENA": "Santa Fe", "LAS NIEVES": "Santa Fe",
        "LAS CRUCES": "Santa Fe", "LOURDES": "Santa Fe",
        # 04 San Cristóbal
        "SAN BLAS": "San Cristóbal", "SOSIEGO": "San Cristóbal", "20 DE JULIO": "San Cristóbal",
        "LA GLORIA": "San Cristóbal", "LOS LIBERTADORES": "San Cristóbal", "RAMAJAL": "San Cristóbal",
        "JUAN REY": "San Cristóbal", "SAN ISIDRO": "San Cristóbal",
        # 05 Usme
        "LA FLORA": "Usme", "DANUBIO": "Usme", "ALFONSO LOPEZ": "Usme",
        "PARQUE ENTRE NUBES": "Usme", "PARQUE ENTRENUBES": "Usme", "NUEVO MUZU": "Usme",
        "GRAN YOMASA": "Usme", "COMUNEROS": "Usme", "CIUDAD USME": "Usme", "UPR RIO TUNJUELO": "Usme",
        # 06 Tunjuelito
        "VENECIA": "Tunjuelito", "TUNJUELITO": "Tunjuelito",
        # 07 Bosa
        "APOGEO": "Bosa", "BOSA OCCIDENTAL": "Bosa", "BOSA CENTRAL": "Bosa",
        "EL PORVENIR": "Bosa", "TINTAL SUR": "Bosa",
        # 08 Kennedy
        "AMERICAS": "Kennedy", "CARVAJAL": "Kennedy", "CASTILLA": "Kennedy", "KENNEDY CENTRAL": "Kennedy",
        "TIMIZA": "Kennedy", "PATIO BONITO": "Kennedy", "TINTAL NORTE": "Kennedy", "CALANDAIMA": "Kennedy",
        "CORABASTOS": "Kennedy", "GRAN BRITALIA": "Kennedy", "LAS MARGARITAS": "Kennedy", "BAVARIA": "Kennedy",
        # 09 Fontibón
        "FONTIBON": "Fontibón", "FONTIBON CENTRO": "Fontibón", "SAN PABLO": "Fontibón", "FONTIBON SAN PABLO": "Fontibón",
        "ZONA FRANCA": "Fontibón", "CIUDAD SALITRE OCCIDENTAL": "Fontibón", "GRANJAS DE TECHO": "Fontibón",
        "MODELIA": "Fontibón", "CAPELLANIA": "Fontibón", "AEROPUERTO EL DORADO": "Fontibón",
        # 10 Engativá
        "LAS FERIAS": "Engativá", "MINUTO DE DIOS": "Engativá", "BOYACA REAL": "Engativá",
        "SANTA CECILIA": "Engativá", "JARDIN BOTANICO": "Engativá", "NORMANDIA": "Engativá",
        "GARCES NAVAS": "Engativá", "ALAMOS": "Engativá", "ENGATIVA": "Engativá", "BOLIVIA": "Engativá",
        # 11 Suba
        "LA ACADEMIA": "Suba", "GUAYMARAL": "Suba", "SAN JOSE DE BAVARIA": "Suba", "BRITALIA": "Suba",
        "EL PRADO": "Suba", "LA ALHAMBRA": "Suba", "CASABLANCA SUBA": "Suba", "CASA BLANCA SUBA": "Suba",
        "NIZA": "Suba", "LA FLORESTA": "Suba", "SUBA CENTRO": "Suba", "SUBA": "Suba", "EL RINCON": "Suba",
        "TIBABUYES": "Suba", "UPR ZONA NORTE": "Suba",
        # 12 Barrios Unidos
        "LOS ANDES": "Barrios Unidos", "DOCE DE OCTUBRE": "Barrios Unidos", "LOS ALCAZARES": "Barrios Unidos",
        "PARQUE SALITRE": "Barrios Unidos",
        # 13 Teusaquillo
        "GALERIAS": "Teusaquillo", "TEUSAQUILLO": "Teusaquillo", "PARQUE SIMON BOLIVAR": "Teusaquillo",
        "PARQUE SIMON BOLIVAR CAN": "Teusaquillo", "LA ESMERALDA": "Teusaquillo", "QUINTA PAREDES": "Teusaquillo",
        "CIUDAD SALITRE ORIENTAL": "Teusaquillo",
        # 14 Los Mártires
        "LA SABANA": "Los Mártires", "SANTA ISABEL": "Los Mártires",
        # 15 Antonio Nariño
        "RESTREPO": "Antonio Nariño", "CIUDAD JARDIN": "Antonio Nariño",
        # 16 Puente Aranda
        "CIUDAD MONTES": "Puente Aranda", "MUZU": "Puente Aranda", "SAN RAFAEL": "Puente Aranda",
        "ZONA INDUSTRIAL": "Puente Aranda", "PUENTE ARANDA": "Puente Aranda",
        # 17 La Candelaria
        "LA CANDELARIA": "La Candelaria",
        # 18 Rafael Uribe Uribe
        "SAN JOSE SUR": "Rafael Uribe Uribe", "SAN JOSE": "Rafael Uribe Uribe", "QUIROGA": "Rafael Uribe Uribe",
        "MARCO FIDEL SUAREZ": "Rafael Uribe Uribe", "MARRUECOS": "Rafael Uribe Uribe", "DIANA TURBAY": "Rafael Uribe Uribe",
        "MOLINOS": "Rafael Uribe Uribe", "CARACAS": "Rafael Uribe Uribe",
        # 19 Ciudad Bolívar
        "EL MOCHUELO": "Ciudad Bolívar", "MONTEBLANCO": "Ciudad Bolívar", "MONTE BLANCO": "Ciudad Bolívar",
        "EL TESORO": "Ciudad Bolívar", "ISMAEL PERDOMO": "Ciudad Bolívar", "JERUSALEN": "Ciudad Bolívar", "JERUSALEM": "Ciudad Bolívar",
        "ARBORIZADORA": "Ciudad Bolívar", "SAN FRANCISCO": "Ciudad Bolívar", "LUCERO": "Ciudad Bolívar",
        # 20 Sumapaz
        "UPR RIO BLANCO": "Sumapaz", "UPR RIO SUMAPAZ": "Sumapaz",
    }
    return {normalize_match(k): v for k, v in mapping.items()}


UPZ_TO_LOCALIDAD = build_upz_localidad_map()
LOCALIDAD_ALIASES = {
    "USAQUEN": "Usaquén", "SANTAFE": "Santa Fe", "SANTA FE": "Santa Fe", "SAN CRISTOBAL": "San Cristóbal",
    "CIUDAD BOLIVAR": "Ciudad Bolívar", "FONTIBON": "Fontibón", "ENGATIVA": "Engativá",
    "MARTIRES": "Los Mártires", "LOS MARTIRES": "Los Mártires", "ANTONIO NARINO": "Antonio Nariño",
    "CANDELARIA": "La Candelaria", "LA CANDELARIA": "La Candelaria",
    "RAFAEL URIBE URIBE": "Rafael Uribe Uribe", "BARRIOS UNIDOS": "Barrios Unidos",
}


def normalize_name(s):
    if pd.isna(s):
        return s
    return str(s).strip().upper()


def pretty_col(col):
    return VARIABLE_LABELS.get(col, str(col).replace("_", " ").strip().title())


def pretty_dataframe(df, columns=None):
    out = df.copy()
    if columns:
        cols = [c for c in columns if c in out.columns]
        out = out[cols]
    out = out.rename(columns={c: pretty_col(c) for c in out.columns})
    return out


@st.cache_data(show_spinner="Cargando y uniendo datos...")
def load_data():
    shp_path = DATA_DIR / "pensionadosupz_042023.shp"
    variables_path = DATA_DIR / "dataset_final.csv"
    clusters_path = DATA_DIR / "upz_clusterizada.csv"

    upz = gpd.read_file(shp_path)
    variables = pd.read_csv(variables_path)
    clusters = pd.read_csv(clusters_path)

    upz["_key"] = upz[NAME_COL].map(normalize_name)
    variables["_key"] = variables[NAME_COL].map(normalize_name)
    clusters["_key"] = clusters[NAME_COL].map(normalize_name)

    cluster_cols = [
        "_key", "indiceaccesibilidad", "cluster_k4", "cluster_k5", "silhouette",
        "cluster_gmm", "cluster_hca", "cluster_hca_mapped", "consistente"
    ]
    cluster_cols = [c for c in cluster_cols if c in clusters.columns]

    gdf = upz.merge(variables.drop(columns=[NAME_COL], errors="ignore"), on="_key", how="left")
    gdf = gdf.merge(clusters[cluster_cols], on="_key", how="left", suffixes=("", "_cluster"))

    if "indiceaccesibilidad_cluster" in gdf.columns and "indiceaccesibilidad" not in gdf.columns:
        gdf["indiceaccesibilidad"] = gdf["indiceaccesibilidad_cluster"]
    elif "indiceaccesibilidad_cluster" in gdf.columns:
        gdf["indiceaccesibilidad"] = gdf["indiceaccesibilidad"].fillna(gdf["indiceaccesibilidad_cluster"])
        gdf = gdf.drop(columns=["indiceaccesibilidad_cluster"])

    if gdf.crs is None:
        gdf = gdf.set_crs("EPSG:4686")
    gdf = gdf.to_crs("EPSG:4326")

    projected = gdf.to_crs("EPSG:3116")
    reps = projected.representative_point().to_crs("EPSG:4326")
    gdf["lon"] = reps.x
    gdf["lat"] = reps.y
    gdf["LOCALIDAD"] = gdf[NAME_COL].map(lambda x: UPZ_TO_LOCALIDAD.get(normalize_match(x), "Sin localidad"))

    for col in VARIABLE_LABELS:
        if col in gdf.columns and col != NAME_COL:
            
            converted = pd.to_numeric(gdf[col], errors="coerce")
            if converted.notna().sum() > 0:
                gdf[col] = converted

    return gdf


@st.cache_data(show_spinner=False)
def load_ventas_bogota():
    if not VENTAS_PATH.exists():
        return pd.DataFrame()
    try:
        raw = pd.read_excel(VENTAS_PATH, sheet_name="Detalle Vendedores", header=None)
    except Exception:
        return pd.DataFrame()

    marker_mask = raw.apply(lambda row: row.astype(str).str.contains("ASESOR", case=False, na=False).any() and row.astype(str).str.contains("BOGOTA", case=False, na=False).any(), axis=1)
    start = int(marker_mask[marker_mask].index.min()) if marker_mask.any() else 24
    block = raw.iloc[start:, 11:17].copy()
    block.columns = ["Asesor", "Localidad/Zona", "Ventas 2024", "Ventas 2025", "Crecimiento", "Participacion Zona 2025"]
    block["Asesor"] = block["Asesor"].ffill()
    block = block.dropna(subset=["Localidad/Zona"])
    block = block[block["Localidad/Zona"].astype(str).str.strip().ne("")]
    for c in ["Ventas 2024", "Ventas 2025", "Crecimiento", "Participacion Zona 2025"]:
        block[c] = pd.to_numeric(block[c], errors="coerce")
    block = block.dropna(subset=["Ventas 2025"])
    block["Localidad/Zona"] = block["Localidad/Zona"].astype(str).str.strip()
    block["Asesor"] = block["Asesor"].astype(str).str.strip()
    block["localidad_key"] = block["Localidad/Zona"].map(normalize_match)
    block["Localidad"] = block["localidad_key"].map(lambda k: LOCALIDAD_ALIASES.get(k, str(block.loc[block["localidad_key"] == k, "Localidad/Zona"].iloc[0]).strip().title()))
    return block.reset_index(drop=True)


def ventas_por_localidad(ventas):
    if ventas.empty:
        return pd.DataFrame(columns=["LOCALIDAD", "asesores", "ventas_2024", "ventas_2025", "crecimiento_ventas", "participacion_zona_2025"])
    df = ventas.copy()
    df["LOCALIDAD"] = df["localidad_key"].map(lambda k: LOCALIDAD_ALIASES.get(k, df.loc[df["localidad_key"] == k, "Localidad"].iloc[0]))
    grouped = df.groupby("LOCALIDAD", as_index=False).agg(
        asesores=("Asesor", lambda x: ", ".join(sorted(set([str(v) for v in x if pd.notna(v)])))),
        ventas_2024=("Ventas 2024", "sum"),
        ventas_2025=("Ventas 2025", "sum"),
        participacion_zona_2025=("Participacion Zona 2025", "sum"),
    )
    grouped["crecimiento_ventas"] = np.where(
        grouped["ventas_2024"] > 0,
        (grouped["ventas_2025"] - grouped["ventas_2024"]) / grouped["ventas_2024"],
        np.nan,
    )
    return grouped


def build_localidad_gdf(gdf, ventas):
    base = gdf[gdf["LOCALIDAD"].ne("Sin localidad")].copy()
    if base.empty:
        return gpd.GeoDataFrame(columns=["LOCALIDAD", "geometry"], geometry="geometry", crs=gdf.crs)
    sum_cols = [
        "numhogares", "personasReal", "num_clientes", "num_ferreterias", "num_tiendasPinturas",
        "num_construcciones", "num_industriales", "primary", "secondary", "tertiary", "trunk",
        "valResidencial", "valComercial", "area"
    ]
    mean_cols = ["densidadPoblacional", "estrato_asignado", "indiceaccesibilidad"]
    agg = {c: "sum" for c in sum_cols if c in base.columns}
    agg.update({c: "mean" for c in mean_cols if c in base.columns})
    localidad_gdf = base.dissolve(by="LOCALIDAD", aggfunc=agg).reset_index()
    counts = base.groupby("LOCALIDAD").size().rename("upz_count").reset_index()
    localidad_gdf = localidad_gdf.merge(counts, on="LOCALIDAD", how="left")
    ventas_resumen = ventas_por_localidad(ventas)
    if not ventas_resumen.empty:
        localidad_gdf = localidad_gdf.merge(ventas_resumen, on="LOCALIDAD", how="left")
    return localidad_gdf


def localidad_from_click(localidad_gdf, click_info):
    if not click_info or localidad_gdf.empty:
        return None
    lat = click_info.get("lat")
    lng = click_info.get("lng")
    if lat is None or lng is None:
        return None
    p = Point(lng, lat)
    hits = localidad_gdf[localidad_gdf.geometry.contains(p)]
    if hits.empty:
        proj = localidad_gdf.to_crs("EPSG:3116")
        p_proj = gpd.GeoSeries([p], crs="EPSG:4326").to_crs("EPSG:3116").iloc[0]
        idx = proj.geometry.distance(p_proj).idxmin()
        return localidad_gdf.loc[idx, "LOCALIDAD"]
    return hits.iloc[0]["LOCALIDAD"]


def format_value(v):
    if pd.isna(v):
        return "Sin dato"
    if isinstance(v, (int, np.integer)):
        return f"{v:,.0f}"
    if isinstance(v, (float, np.floating)):
        if abs(v) >= 1000:
            return f"{v:,.0f}"
        return f"{v:,.2f}"
    return str(v)


def format_money(v):
    if pd.isna(v):
        return "Sin dato"
    return "$ " + f"{float(v):,.0f}".replace(",", ".")


def base_map(location=(4.65, -74.1), zoom_start=11):
    return folium.Map(location=location, zoom_start=zoom_start, tiles="CartoDB positron", control_scale=True)


def gdf_for_folium(gdf, cols=None):
    cols = cols or [NAME_COL, "geometry"]
    keep = [c for c in cols if c in gdf.columns]
    if "geometry" not in keep:
        keep.append("geometry")
    out = gdf[keep].copy()
    for c in out.columns:
        if c != "geometry":
            out[c] = out[c].astype(object).where(pd.notnull(out[c]), None)
    return out


def add_upz_geojson(m, gdf, color="#2c7fb8", fill_opacity=0.55, fields=None, name="UPZ"):
    fields = [f for f in (fields or BASE_POPUP_FIELDS) if f in gdf.columns]
    aliases = [pretty_col(f) for f in fields]
    folium.GeoJson(
        gdf_for_folium(gdf, fields + ["geometry"]),
        name=name,
        style_function=lambda x: {
            "fillColor": color,
            "color": "#4a5568",
            "weight": 0.7,
            "fillOpacity": fill_opacity,
        },
        highlight_function=lambda x: {"weight": 3, "color": "#111827", "fillOpacity": 0.8},
        tooltip=GeoJsonTooltip(fields=fields, aliases=aliases, sticky=True),
        popup=GeoJsonPopup(fields=fields, aliases=aliases, max_width=350),
    ).add_to(m)


def upz_from_click(gdf, click_info):
    if not click_info:
        return None
    lat = click_info.get("lat")
    lng = click_info.get("lng")
    if lat is None or lng is None:
        return None
    p = Point(lng, lat)
    hits = gdf[gdf.geometry.contains(p)]
    if hits.empty:
        proj = gdf.to_crs("EPSG:3116")
        p_proj = gpd.GeoSeries([p], crs="EPSG:4326").to_crs("EPSG:3116").iloc[0]
        idx = proj.geometry.distance(p_proj).idxmin()
        return gdf.loc[idx, NAME_COL]
    return hits.iloc[0][NAME_COL]


@st.cache_data(show_spinner=False)
def approximate_isochrone(lat, lon, time_minutes=15, speed_kmh=25):
    radius_m = (speed_kmh * 1000 / 60) * time_minutes
    p = gpd.GeoSeries([Point(lon, lat)], crs="EPSG:4326").to_crs("EPSG:3116")
    poly = p.buffer(radius_m).convex_hull.to_crs("EPSG:4326").iloc[0]
    return poly


@st.cache_data(show_spinner="Calculando isocrona con OSMnx...")
def osmnx_isochrone(lat, lon, time_minutes=15, network_type="drive"):
    if not OSMNX_AVAILABLE:
        raise RuntimeError("OSMnx no esta disponible en este entorno.")
    dist = max(3000, int(time_minutes * 650))
    G = ox.graph_from_point((lat, lon), dist=dist, network_type=network_type, simplify=True)
    G = ox.add_edge_speeds(G)
    G = ox.add_edge_travel_times(G)
    center_node = ox.distance.nearest_nodes(G, lon, lat)
    subgraph = nx.ego_graph(G, center_node, radius=time_minutes * 60, distance="travel_time")
    nodes, _ = ox.graph_to_gdfs(subgraph)
    if len(nodes) == 0:
        return approximate_isochrone(lat, lon, time_minutes, 20)
    return nodes.unary_union.convex_hull


def calculate_isochrone(lat, lon, time_minutes, use_osmnx, network_type, speed_kmh):
    if use_osmnx:
        try:
            return osmnx_isochrone(lat, lon, time_minutes, network_type), "OSMnx"
        except Exception as exc:
            st.warning(f"OSMnx fallo y se uso una aproximacion por distancia. Detalle: {exc}")
    return approximate_isochrone(lat, lon, time_minutes, speed_kmh), "Aproximada"


def coverage_from_polygon(gdf, polygon):
    points = gpd.GeoDataFrame(gdf[[NAME_COL, "numhogares", "num_clientes"]].copy(), geometry=gdf.geometry.representative_point(), crs=gdf.crs)
    covered = points[points.geometry.within(polygon)].copy()
    total_hogares = gdf["numhogares"].sum(skipna=True)
    total_clientes = gdf["num_clientes"].sum(skipna=True) if "num_clientes" in gdf.columns else np.nan
    hogares = covered["numhogares"].sum(skipna=True)
    clientes = covered["num_clientes"].sum(skipna=True) if "num_clientes" in covered.columns else np.nan
    return covered, {
        "upz_cubiertas": len(covered),
        "hogares_cubiertos": hogares,
        "pct_hogares": hogares / total_hogares if total_hogares else np.nan,
        "clientes_cubiertos": clientes,
        "pct_clientes": clientes / total_clientes if total_clientes and not pd.isna(total_clientes) else np.nan,
    }


def build_coverage_matrix(gdf, time_minutes, speed_kmh):
    radius_m = (speed_kmh * 1000 / 60) * time_minutes
    proj = gdf.to_crs("EPSG:3116").copy()
    pts = proj.geometry.representative_point()
    coords = np.array([(p.x, p.y) for p in pts])
    dist = np.sqrt(((coords[:, None, :] - coords[None, :, :]) ** 2).sum(axis=2))
    return dist <= radius_m


def solve_mclp(gdf, p, time_minutes, speed_kmh, weight_col="numhogares"):
    coverage = build_coverage_matrix(gdf, time_minutes, speed_kmh)
    weights = pd.to_numeric(gdf[weight_col], errors="coerce").fillna(0).to_numpy()
    n = len(gdf)

    if PULP_AVAILABLE:
        model = pulp.LpProblem("MCLP_UPZ_Bogota", pulp.LpMaximize)
        x = pulp.LpVariable.dicts("x", range(n), lowBound=0, upBound=1, cat="Binary")
        y = pulp.LpVariable.dicts("y", range(n), lowBound=0, upBound=1, cat="Binary")
        model += pulp.lpSum(weights[i] * y[i] for i in range(n))
        model += pulp.lpSum(x[j] for j in range(n)) <= p
        for i in range(n):
            covering_sites = [j for j in range(n) if coverage[i, j]]
            if covering_sites:
                model += y[i] <= pulp.lpSum(x[j] for j in covering_sites)
            else:
                model += y[i] == 0
        model.solve(pulp.PULP_CBC_CMD(msg=False))
        selected = [j for j in range(n) if pulp.value(x[j]) is not None and pulp.value(x[j]) > 0.5]
    else:
        selected = []
        covered_mask = np.zeros(n, dtype=bool)
        for _ in range(p):
            best_j, best_gain = None, -1
            for j in range(n):
                if j in selected:
                    continue
                new_cover = coverage[:, j] & (~covered_mask)
                gain = weights[new_cover].sum()
                if gain > best_gain:
                    best_j, best_gain = j, gain
            if best_j is None:
                break
            selected.append(best_j)
            covered_mask |= coverage[:, best_j]

    covered = np.any(coverage[:, selected], axis=1) if selected else np.zeros(n, dtype=bool)
    return selected, np.where(covered)[0].tolist()


def cluster_conclusions(gdf):
    vars_for_profile = [
        "numhogares", "densidadPoblacional", "estrato_asignado", "num_ferreterias",
        "num_tiendasPinturas", "num_construcciones", "num_industriales",
        "num_clientes", "valResidencial", "valComercial", "indiceaccesibilidad"
    ]
    vars_for_profile = [v for v in vars_for_profile if v in gdf.columns]
    profile = gdf.groupby(CLUSTER_COL)[vars_for_profile].mean(numeric_only=True).round(2)
    global_mean = gdf[vars_for_profile].mean(numeric_only=True)
    global_std = gdf[vars_for_profile].std(numeric_only=True).replace(0, np.nan)
    z = (profile - global_mean) / global_std

    lines = {}
    for cluster in profile.index:
        vals = z.loc[cluster].dropna().sort_values(ascending=False)
        high = [pretty_col(c) for c in vals.head(3).index if vals[c] > 0.35]
        low_vals = vals.sort_values().head(2)
        low = [pretty_col(c) for c in low_vals.index if low_vals[c] < -0.35]
        text = []
        if high:
            text.append("Rasgos altos: " + ", ".join(high))
        if low:
            text.append("Rasgos bajos: " + ", ".join(low))
        if not text:
            text.append("Perfil cercano al promedio general de las UPZ.")
        lines[int(cluster)] = " ".join(text)
    return profile, lines


def show_brand_header():
    h1, h2 = st.columns([1, 5])
    with h1:
        if LOGO_PATH.exists():
            st.image(str(LOGO_PATH), use_container_width=True)
    with h2:
        st.title("UPZ Bogota: variables, isocronas y optimizacion")
        st.caption("Aplicacion interactiva para Pinturas BLER. Cluster oficial: K-Means k=4.")


def cluster_detail_card(cluster_gdf, selected_cluster, profile, conclusions):
    sub = cluster_gdf[cluster_gdf[CLUSTER_COL] == selected_cluster]
    row = profile.loc[selected_cluster]
    st.markdown(f"### Cluster {int(selected_cluster)}")
    c = st.columns(4)
    c[0].metric("UPZ del cluster", f"{len(sub):,}")
    c[1].metric("Hogares promedio", format_value(row.get("numhogares", np.nan)))
    c[2].metric("Clientes promedio", format_value(row.get("num_clientes", np.nan)))
    c[3].metric("Accesibilidad promedio", format_value(row.get("indiceaccesibilidad", np.nan)))
    st.info(conclusions.get(int(selected_cluster), "Sin conclusion disponible."))
    with st.expander("Ver UPZ de este cluster"):
        st.dataframe(
            pretty_dataframe(sub.sort_values(NAME_COL), [NAME_COL, "numhogares", "num_clientes", "estrato_asignado", "indiceaccesibilidad"]),
            use_container_width=True,
            hide_index=True,
        )


gdf = load_data()
ventas_bogota = load_ventas_bogota()
localidad_gdf = build_localidad_gdf(gdf, ventas_bogota)
center = (float(gdf["lat"].mean()), float(gdf["lon"].mean()))

show_brand_header()

missing_clusters = int(gdf[CLUSTER_COL].isna().sum()) if CLUSTER_COL in gdf.columns else len(gdf)
cols = st.columns(4)
cols[0].metric("UPZ en mapa", f"{len(gdf):,}")
cols[1].metric("UPZ con cluster k4", f"{len(gdf) - missing_clusters:,}")
cols[2].metric("Hogares", format_value(gdf["numhogares"].sum()))
cols[3].metric("Clientes", format_value(gdf["num_clientes"].sum() if "num_clientes" in gdf.columns else np.nan))

view1, view2, view3 = st.tabs(["1. Variables por UPZ", "2. Isocronas", "3. Resultados"])

with view1:
    st.subheader("Mapa de calor por variable")
    nivel = st.radio("Nivel de visualización", ["UPZ", "Localidad"], horizontal=True)

    if nivel == "UPZ":
        numeric_vars = [
            c for c in VARIABLE_LABELS
            if c in gdf.columns
            and c not in CLUSTER_VARIABLES
            and pd.api.types.is_numeric_dtype(gdf[c])
        ]
        variable = st.selectbox("Variable a visualizar", numeric_vars, format_func=pretty_col, key="var_upz")

        m = base_map(center, 11)
        choropleth_data = gdf[[NAME_COL, variable, "geometry"]].dropna(subset=[variable]).copy()
        folium.Choropleth(
            geo_data=choropleth_data.to_json(),
            name=pretty_col(variable),
            data=choropleth_data,
            columns=[NAME_COL, variable],
            key_on=f"feature.properties.{NAME_COL}",
            fill_color="YlOrRd",
            fill_opacity=0.75,
            line_opacity=0.45,
            nan_fill_color="#f1f5f9",
            legend_name=pretty_col(variable),
        ).add_to(m)
        add_upz_geojson(m, gdf, color="#ffffff", fill_opacity=0.02, name="Detalle UPZ")
        folium.LayerControl().add_to(m)

        map_state = st_folium(m, height=650, use_container_width=True, returned_objects=["last_clicked"])
        selected_name = upz_from_click(gdf, map_state.get("last_clicked")) if map_state else None
        if not selected_name:
            selected_name = st.selectbox("Selecciona una UPZ para ver detalle", sorted(gdf[NAME_COL].dropna().unique()), key="select_upz_detail")
        row = gdf[gdf[NAME_COL] == selected_name].iloc[0]

        st.markdown("### Detalle de la UPZ seleccionada")
        d1, d2 = st.columns([1.2, 1], gap="large")
        with d1:
            st.markdown(f"#### {selected_name}")
            detail_fields = ["LOCALIDAD", variable, "numhogares", "personasReal", "num_clientes", "estrato_asignado", "num_ferreterias", "indiceaccesibilidad", CLUSTER_COL]
            seen = set()
            for f in detail_fields:
                if f in gdf.columns and f not in seen:
                    st.write(f"**{pretty_col(f)}:** {format_value(row[f])}")
                    seen.add(f)
        with d2:
            st.markdown("#### Ventas de la localidad")
            loc = row.get("LOCALIDAD", None)
            venta_loc = ventas_por_localidad(ventas_bogota)
            if loc and not venta_loc.empty and loc in venta_loc["LOCALIDAD"].values:
                vz = venta_loc[venta_loc["LOCALIDAD"] == loc].iloc[0]
                st.metric("Ventas reales 2025", format_money(vz["ventas_2025"]))
                st.write(f"**Asesor(es):** {vz['asesores']}")
                st.write(f"**Ventas 2024:** {format_money(vz['ventas_2024'])}")
                st.write(f"**Crecimiento:** {vz['crecimiento_ventas']:.1%}" if pd.notna(vz['crecimiento_ventas']) else "**Crecimiento:** Sin dato")
            else:
                st.info("No hay ventas asociadas a esta localidad en el archivo comercial.")

    else:
        localidad_vars = [
            "ventas_2025", "ventas_2024", "crecimiento_ventas", "participacion_zona_2025",
            "numhogares", "personasReal", "num_clientes", "num_ferreterias", "indiceaccesibilidad", "upz_count"
        ]
        localidad_vars = [c for c in localidad_vars if c in localidad_gdf.columns and pd.api.types.is_numeric_dtype(localidad_gdf[c])]
        variable = st.selectbox("Variable a visualizar", localidad_vars, format_func=pretty_col, key="var_localidad")

        m = base_map(center, 11)
        choropleth_data = localidad_gdf[["LOCALIDAD", variable, "geometry"]].dropna(subset=[variable]).copy()
        if choropleth_data.empty:
            add_upz_geojson(m, gdf, color="#e5e7eb", fill_opacity=0.15, name="UPZ")
            st.warning("La variable seleccionada no tiene datos suficientes para pintar el mapa.")
        else:
            folium.Choropleth(
                geo_data=choropleth_data.to_json(),
                name=pretty_col(variable),
                data=choropleth_data,
                columns=["LOCALIDAD", variable],
                key_on="feature.properties.LOCALIDAD",
                fill_color="YlGnBu",
                fill_opacity=0.78,
                line_opacity=0.55,
                nan_fill_color="#f1f5f9",
                legend_name=pretty_col(variable),
            ).add_to(m)
            fields = [f for f in LOCALIDAD_POPUP_FIELDS if f in localidad_gdf.columns]
            folium.GeoJson(
                gdf_for_folium(localidad_gdf, fields + ["geometry"]),
                name="Detalle localidad",
                style_function=lambda x: {"fillColor": "#ffffff", "color": "#334155", "weight": 1.0, "fillOpacity": 0.03},
                highlight_function=lambda x: {"weight": 3, "color": "#111827", "fillOpacity": 0.15},
                tooltip=GeoJsonTooltip(fields=fields, aliases=[pretty_col(f) for f in fields], sticky=True),
                popup=GeoJsonPopup(fields=fields, aliases=[pretty_col(f) for f in fields], max_width=420),
            ).add_to(m)
        folium.LayerControl().add_to(m)

        map_state = st_folium(m, height=650, use_container_width=True, returned_objects=["last_clicked"])
        selected_loc = localidad_from_click(localidad_gdf, map_state.get("last_clicked")) if map_state else None
        if not selected_loc:
            selected_loc = st.selectbox("Selecciona una localidad para ver detalle", sorted(localidad_gdf["LOCALIDAD"].dropna().unique()), key="select_localidad_detail")
        loc_row = localidad_gdf[localidad_gdf["LOCALIDAD"] == selected_loc].iloc[0]

        st.markdown("### Detalle de la localidad seleccionada")
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Localidad", selected_loc)
        c2.metric("UPZ", format_value(loc_row.get("upz_count", np.nan)))
        c3.metric("Ventas reales 2025", format_money(loc_row.get("ventas_2025", np.nan)))
        c4.metric("Crecimiento", f"{loc_row.get('crecimiento_ventas', np.nan):.1%}" if pd.notna(loc_row.get("crecimiento_ventas", np.nan)) else "Sin dato")

        d1, d2 = st.columns([1, 1], gap="large")
        with d1:
            st.markdown("#### Indicadores territoriales")
            for f in ["numhogares", "personasReal", "num_clientes", "num_ferreterias", "estrato_asignado", "indiceaccesibilidad"]:
                if f in localidad_gdf.columns:
                    st.write(f"**{pretty_col(f)}:** {format_value(loc_row.get(f, np.nan))}")
        with d2:
            st.markdown("#### Información comercial")
            st.write(f"**Asesor(es):** {loc_row.get('asesores', 'Sin dato') if pd.notna(loc_row.get('asesores', np.nan)) else 'Sin dato'}")
            st.write(f"**Ventas reales 2024:** {format_money(loc_row.get('ventas_2024', np.nan))}")
            st.write(f"**Ventas reales 2025:** {format_money(loc_row.get('ventas_2025', np.nan))}")
            st.write(f"**Participación zona 2025:** {loc_row.get('participacion_zona_2025', np.nan):.2%}" if pd.notna(loc_row.get("participacion_zona_2025", np.nan)) else "**Participación zona 2025:** Sin dato")

        with st.expander("Ver UPZ dentro de esta localidad"):
            sub = gdf[gdf["LOCALIDAD"] == selected_loc].sort_values(NAME_COL)
            st.dataframe(pretty_dataframe(sub, [NAME_COL, "numhogares", "num_clientes", CLUSTER_COL]), use_container_width=True, hide_index=True)

        unmatched = ventas_por_localidad(ventas_bogota)
        if not unmatched.empty:
            unmatched = unmatched[~unmatched["LOCALIDAD"].isin(localidad_gdf["LOCALIDAD"])]
            if not unmatched.empty:
                with st.expander("Zonas comerciales del Excel que no son localidades de Bogotá"):
                    st.dataframe(pretty_dataframe(unmatched, ["LOCALIDAD", "asesores", "ventas_2025", "ventas_2024", "crecimiento_ventas"]), use_container_width=True, hide_index=True)

with view2:
    st.subheader("Isocrona dinamica por UPZ")
    c1, c2, c3, c4 = st.columns([2, 1, 1, 1])
    selected_upz = c1.selectbox("UPZ origen", sorted(gdf[NAME_COL].dropna().unique()), key="iso_upz")
    time_minutes = c2.slider("Minutos", 5, 30, 15, 5)
    speed_kmh = c3.slider("Velocidad aprox. km/h", 10, 40, 25, 5)
    use_osmnx = c4.checkbox("Usar OSMnx", value=False, help="Mas realista, pero puede ser lento o fallar en Streamlit Cloud.")
    network_type = st.selectbox("Tipo de red OSMnx", ["drive", "walk", "bike"], index=0, disabled=not use_osmnx)

    row = gdf[gdf[NAME_COL] == selected_upz].iloc[0]
    poly, method = calculate_isochrone(row["lat"], row["lon"], time_minutes, use_osmnx, network_type, speed_kmh)
    covered, metrics = coverage_from_polygon(gdf, poly)

    kpis = st.columns(4)
    kpis[0].metric("Metodo", method)
    kpis[1].metric("UPZ cubiertas", f"{metrics['upz_cubiertas']:,}")
    kpis[2].metric("Hogares cubiertos", format_value(metrics["hogares_cubiertos"]), f"{metrics['pct_hogares']:.1%}")
    kpis[3].metric("Clientes cubiertos", format_value(metrics["clientes_cubiertos"]), f"{metrics['pct_clientes']:.1%}" if not pd.isna(metrics["pct_clientes"]) else None)

    m2 = base_map((row["lat"], row["lon"]), 12)
    add_upz_geojson(m2, gdf, color="#e5e7eb", fill_opacity=0.25, name="UPZ")
    if not covered.empty:
        add_upz_geojson(m2, gdf.loc[covered.index], color="#60a5fa", fill_opacity=0.55, name="UPZ cubiertas")
    folium.GeoJson(
        gpd.GeoDataFrame({"nombre": [f"Isocrona {time_minutes} min"], "geometry": [poly]}, crs="EPSG:4326").to_json(),
        name="Isocrona / envolvente convexa",
        style_function=lambda x: {"fillColor": "#f97316", "color": "#c2410c", "weight": 2, "fillOpacity": 0.28},
    ).add_to(m2)
    folium.Marker([row["lat"], row["lon"]], popup=selected_upz, tooltip="Origen").add_to(m2)
    folium.LayerControl().add_to(m2)
    st_folium(m2, height=650, use_container_width=True)

    with st.expander("Ver UPZ cubiertas"):
        st.dataframe(pretty_dataframe(covered.sort_values(NAME_COL), [NAME_COL, "numhogares", "num_clientes"]), use_container_width=True, hide_index=True)

with view3:
    st.subheader("Resultados: clusterizacion y optimizacion")
    mode = st.radio("Modo", ["Clusterizacion", "Optimizacion"], horizontal=True)

    if mode == "Clusterizacion":
        cluster_gdf = gdf.dropna(subset=[CLUSTER_COL]).copy()
        cluster_gdf[CLUSTER_COL] = cluster_gdf[CLUSTER_COL].astype(int)
        m3 = base_map(center, 11)
        for cl, sub in cluster_gdf.groupby(CLUSTER_COL):
            add_upz_geojson(m3, sub, color=CLUSTER_COLORS.get(int(cl), "#64748b"), fill_opacity=0.65, name=f"Cluster {int(cl)}")
        folium.LayerControl().add_to(m3)
        cluster_state = st_folium(m3, height=620, use_container_width=True, returned_objects=["last_clicked"])

        clicked_upz = upz_from_click(cluster_gdf, cluster_state.get("last_clicked")) if cluster_state else None
        clicked_cluster = None
        if clicked_upz:
            clicked_cluster = int(cluster_gdf.loc[cluster_gdf[NAME_COL] == clicked_upz, CLUSTER_COL].iloc[0])

        counts = cluster_gdf[CLUSTER_COL].value_counts().sort_index()
        metric_cols = st.columns(len(counts))
        for i, (cl, cnt) in enumerate(counts.items()):
            metric_cols[i].metric(f"Cluster {int(cl)}", f"{cnt} UPZ")

        profile, conclusions = cluster_conclusions(cluster_gdf)
        clusters = sorted(cluster_gdf[CLUSTER_COL].dropna().unique().astype(int))
        default_idx = clusters.index(clicked_cluster) if clicked_cluster in clusters else 0
        selected_cluster = st.selectbox("Selecciona o espicha un cluster en el mapa", clusters, index=default_idx, format_func=lambda c: f"Cluster {int(c)}")
        if clicked_upz:
            st.caption(f"Ultima UPZ espichada: {clicked_upz} - Cluster {clicked_cluster}")
        cluster_detail_card(cluster_gdf, int(selected_cluster), profile, conclusions)

    else:
        st.markdown("La optimizacion selecciona UPZ candidatas para maximizar la cobertura ponderada.")
        c1, c2, c3 = st.columns(3)
        p = c1.slider("Numero maximo de ubicaciones", 1, 12, 4)
        opt_time = c2.slider("Tiempo de cobertura", 5, 30, 15, 5, key="opt_time")
        opt_speed = c3.slider("Velocidad aprox. km/h", 10, 40, 25, 5, key="opt_speed")
        weight_col = st.selectbox("Variable objetivo", [c for c in ["numhogares", "num_clientes", "personasReal"] if c in gdf.columns], format_func=pretty_col)

        if st.button("Correr optimizacion", type="primary"):
            selected_idx, covered_idx = solve_mclp(gdf.reset_index(drop=True), p, opt_time, opt_speed, weight_col)
            opt_gdf = gdf.reset_index(drop=True)
            selected = opt_gdf.iloc[selected_idx].copy()
            covered = opt_gdf.iloc[covered_idx].copy()

            total_weight = pd.to_numeric(opt_gdf[weight_col], errors="coerce").fillna(0).sum()
            covered_weight = pd.to_numeric(covered[weight_col], errors="coerce").fillna(0).sum()
            st.session_state["opt_result"] = {
                "selected_names": selected[NAME_COL].tolist(),
                "covered_names": covered[NAME_COL].tolist(),
                "covered_weight": covered_weight,
                "total_weight": total_weight,
                "weight_col": weight_col,
                "time": opt_time,
                "speed": opt_speed,
            }

        result = st.session_state.get("opt_result")
        if result:
            selected = gdf[gdf[NAME_COL].isin(result["selected_names"])]
            covered = gdf[gdf[NAME_COL].isin(result["covered_names"])]
            k = st.columns(4)
            k[0].metric("Ubicaciones seleccionadas", len(selected))
            k[1].metric("UPZ cubiertas", len(covered))
            k[2].metric(f"{pretty_col(result['weight_col'])} cubiertos", format_value(result["covered_weight"]))
            k[3].metric("Cobertura objetivo", f"{result['covered_weight'] / result['total_weight']:.1%}" if result["total_weight"] else "Sin dato")

            m4 = base_map(center, 11)
            add_upz_geojson(m4, gdf, color="#e5e7eb", fill_opacity=0.2, name="UPZ")
            add_upz_geojson(m4, covered, color="#93c5fd", fill_opacity=0.55, name="UPZ cubiertas")
            for _, r in selected.iterrows():
                poly = approximate_isochrone(r["lat"], r["lon"], result["time"], result["speed"])
                folium.GeoJson(
                    gpd.GeoDataFrame({"geometry": [poly]}, crs="EPSG:4326").to_json(),
                    style_function=lambda x: {"fillColor": "#fb923c", "color": "#ea580c", "weight": 1.5, "fillOpacity": 0.22},
                ).add_to(m4)
                folium.Marker([r["lat"], r["lon"]], tooltip=r[NAME_COL], popup=r[NAME_COL], icon=folium.Icon(color="red", icon="star")).add_to(m4)
            folium.LayerControl().add_to(m4)
            st_folium(m4, height=620, use_container_width=True)

            csel, ccov = st.columns(2)
            csel.markdown("### UPZ seleccionadas")
            csel.dataframe(pretty_dataframe(selected.sort_values(NAME_COL), [NAME_COL, "numhogares", "num_clientes", CLUSTER_COL]), use_container_width=True, hide_index=True)
            ccov.markdown("### UPZ cubiertas")
            ccov.dataframe(pretty_dataframe(covered.sort_values(NAME_COL), [NAME_COL, "numhogares", "num_clientes", CLUSTER_COL]), use_container_width=True, hide_index=True)
        else:
            st.info("Configura los parametros y presiona 'Correr optimizacion'.")
