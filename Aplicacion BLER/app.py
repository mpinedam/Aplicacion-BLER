import math
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

st.set_page_config(
    page_title="UPZ Bogota | Isocronas y Optimizacion",
    page_icon="🗺️",
    layout="wide",
)

st.markdown(
    """
    <style>
    .main .block-container {padding-top: 1.5rem; padding-bottom: 2rem;}
    .metric-card {background:#f7f9fb;border:1px solid #e7edf3;border-radius:12px;padding:14px;}
    h1, h2, h3 {letter-spacing: -0.02em;}
    </style>
    """,
    unsafe_allow_html=True,
)

NAME_COL = "NOMBRE"
CLUSTER_COL = "cluster_k4"
VARIABLE_LABELS = {
    "numhogares": "Hogares",
    "area": "Area",
    "densidadPoblacional": "Densidad poblacional",
    "estrato_asignado": "Estrato asignado",
    "personasReal": "Personas",
    "num_ferreterias": "Ferreterias",
    "num_tiendasPinturas": "Tiendas de pinturas",
    "num_construcciones": "Comercios construccion",
    "num_industriales": "Comercios industriales",
    "num_clientes": "Clientes",
    "primary": "Vias primary",
    "secondary": "Vias secondary",
    "tertiary": "Vias tertiary",
    "trunk": "Vias trunk",
    "valResidencial": "Valor residencial",
    "valComercial": "Valor comercial",
    "indiceaccesibilidad": "Indice de accesibilidad",
}
BASE_POPUP_FIELDS = [
    NAME_COL, "numhogares", "personasReal", "estrato_asignado", "num_clientes",
    "num_ferreterias", "indiceaccesibilidad", CLUSTER_COL
]


def normalize_name(s):
    if pd.isna(s):
        return s
    return str(s).strip().upper()


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

    # Evita columnas duplicadas: el CSV de clusters solo aporta campos de clusterizacion e indice.
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

    # Geometria a WGS84 para Folium.
    if gdf.crs is None:
        gdf = gdf.set_crs("EPSG:4686")
    gdf = gdf.to_crs("EPSG:4326")

    # Centroides robustos desde CRS proyectado.
    projected = gdf.to_crs("EPSG:3116")
    reps = projected.representative_point().to_crs("EPSG:4326")
    gdf["lon"] = reps.x
    gdf["lat"] = reps.y

    for col in ["numhogares", "personasReal", "num_clientes", "cluster_k4"]:
        if col in gdf.columns:
            gdf[col] = pd.to_numeric(gdf[col], errors="coerce")

    return gdf


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
    aliases = [VARIABLE_LABELS.get(f, f) for f in fields]
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
        # Fallback: UPZ mas cercana al punto clickeado.
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
    # Distancia de descarga amplia segun velocidad supuesta de ciudad.
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
    # Matriz aproximada para optimizacion: distancia euclidiana sobre EPSG:3116.
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
        # Greedy fallback.
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
        high = [VARIABLE_LABELS.get(c, c) for c in vals.head(3).index if vals[c] > 0.35]
        low_vals = vals.sort_values().head(2)
        low = [VARIABLE_LABELS.get(c, c) for c in low_vals.index if low_vals[c] < -0.35]
        text = []
        if high:
            text.append("Fortalezas/rasgos altos: " + ", ".join(high))
        if low:
            text.append("Rasgos bajos: " + ", ".join(low))
        if not text:
            text.append("Perfil cercano al promedio general de las UPZ.")
        lines[int(cluster)] = " ".join(text)
    return profile, lines


gdf = load_data()
center = (float(gdf["lat"].mean()), float(gdf["lon"].mean()))

st.title("🗺️ UPZ Bogota: variables, isocronas y optimizacion")
st.caption("Version funcional inicial. Cluster oficial: K-Means k=4 (`cluster_k4`).")

missing_clusters = int(gdf[CLUSTER_COL].isna().sum()) if CLUSTER_COL in gdf.columns else len(gdf)
cols = st.columns(4)
cols[0].metric("UPZ en mapa", f"{len(gdf):,}")
cols[1].metric("UPZ con cluster k4", f"{len(gdf) - missing_clusters:,}")
cols[2].metric("Hogares", format_value(gdf["numhogares"].sum()))
cols[3].metric("Clientes", format_value(gdf["num_clientes"].sum() if "num_clientes" in gdf.columns else np.nan))

view1, view2, view3 = st.tabs(["1. Variables por UPZ", "2. Isocronas", "3. Resultados"])

with view1:
    st.subheader("Mapa de calor por variable")
    numeric_vars = [c for c in VARIABLE_LABELS if c in gdf.columns and pd.api.types.is_numeric_dtype(gdf[c])]
    variable = st.selectbox("Variable a visualizar", numeric_vars, format_func=lambda x: VARIABLE_LABELS.get(x, x))
    left, right = st.columns([3, 1])

    m = base_map(center, 11)
    choropleth_data = gdf[[NAME_COL, variable, "geometry"]].dropna(subset=[variable]).copy()
    folium.Choropleth(
        geo_data=choropleth_data.to_json(),
        name=VARIABLE_LABELS.get(variable, variable),
        data=choropleth_data,
        columns=[NAME_COL, variable],
        key_on=f"feature.properties.{NAME_COL}",
        fill_color="YlOrRd",
        fill_opacity=0.75,
        line_opacity=0.45,
        nan_fill_color="#f1f5f9",
        legend_name=VARIABLE_LABELS.get(variable, variable),
    ).add_to(m)
    add_upz_geojson(m, gdf, color="#ffffff", fill_opacity=0.02, name="Detalle UPZ")
    folium.LayerControl().add_to(m)
    map_state = st_folium(m, height=650, width=None, returned_objects=["last_clicked"])

    selected_name = upz_from_click(gdf, map_state.get("last_clicked")) if map_state else None
    with right:
        st.markdown("### Detalle UPZ")
        if not selected_name:
            selected_name = st.selectbox("Selecciona una UPZ", sorted(gdf[NAME_COL].dropna().unique()))
        row = gdf[gdf[NAME_COL] == selected_name].iloc[0]
        st.markdown(f"**{selected_name}**")
        detail_fields = [variable, "numhogares", "personasReal", "num_clientes", "estrato_asignado", "num_ferreterias", "indiceaccesibilidad", CLUSTER_COL]
        for f in detail_fields:
            if f in gdf.columns:
                st.write(f"**{VARIABLE_LABELS.get(f, f)}:** {format_value(row[f])}")

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
    st_folium(m2, height=650, width=None)

    with st.expander("Ver UPZ cubiertas"):
        st.dataframe(covered[[NAME_COL, "numhogares", "num_clientes"]].sort_values(NAME_COL), use_container_width=True)

with view3:
    st.subheader("Resultados: clusterizacion y optimizacion")
    mode = st.radio("Modo", ["Clusterizacion", "Optimizacion"], horizontal=True)

    if mode == "Clusterizacion":
        cluster_gdf = gdf.dropna(subset=[CLUSTER_COL]).copy()
        cluster_gdf[CLUSTER_COL] = cluster_gdf[CLUSTER_COL].astype(int)
        m3 = base_map(center, 11)
        cluster_colors = {0: "#1f77b4", 1: "#d62728", 2: "#e377c2", 3: "#17becf"}
        for cl, sub in cluster_gdf.groupby(CLUSTER_COL):
            add_upz_geojson(m3, sub, color=cluster_colors.get(int(cl), "#64748b"), fill_opacity=0.65, name=f"Cluster {int(cl)}")
        folium.LayerControl().add_to(m3)
        st_folium(m3, height=620, width=None)

        counts = cluster_gdf[CLUSTER_COL].value_counts().sort_index()
        metric_cols = st.columns(len(counts))
        for i, (cl, cnt) in enumerate(counts.items()):
            metric_cols[i].metric(f"Cluster {int(cl)}", f"{cnt} UPZ")

        profile, conclusions = cluster_conclusions(cluster_gdf)
        st.markdown("### Perfil promedio por cluster")
        st.dataframe(profile, use_container_width=True)
        st.markdown("### Conclusiones principales")
        for cl in sorted(conclusions):
            st.write(f"**Cluster {cl}:** {conclusions[cl]}")

    else:
        st.markdown("La optimizacion selecciona UPZ candidatas para maximizar la cobertura ponderada.")
        c1, c2, c3 = st.columns(3)
        p = c1.slider("Numero maximo de ubicaciones", 1, 12, 4)
        opt_time = c2.slider("Tiempo de cobertura", 5, 30, 15, 5, key="opt_time")
        opt_speed = c3.slider("Velocidad aprox. km/h", 10, 40, 25, 5, key="opt_speed")
        weight_col = st.selectbox("Variable objetivo", [c for c in ["numhogares", "num_clientes", "personasReal"] if c in gdf.columns], format_func=lambda x: VARIABLE_LABELS.get(x, x))

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
            k[2].metric(f"{VARIABLE_LABELS.get(result['weight_col'], result['weight_col'])} cubiertos", format_value(result["covered_weight"]))
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
            st_folium(m4, height=620, width=None)

            csel, ccov = st.columns(2)
            csel.markdown("### UPZ seleccionadas")
            csel.dataframe(selected[[NAME_COL, "numhogares", "num_clientes", CLUSTER_COL]].sort_values(NAME_COL), use_container_width=True)
            ccov.markdown("### UPZ cubiertas")
            ccov.dataframe(covered[[NAME_COL, "numhogares", "num_clientes", CLUSTER_COL]].sort_values(NAME_COL), use_container_width=True)
        else:
            st.info("Configura los parametros y presiona 'Correr optimizacion'.")
