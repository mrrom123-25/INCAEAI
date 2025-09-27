# sustainable_energy_dashboard.py
import streamlit as st
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from pathlib import Path
import glob

import plotly.express as px
import plotly.graph_objects as go
import plotly.io as pio

# ================== PAGE CONFIG & THEME ==================
st.set_page_config(page_title="Global Sustainable Energy Dashboard", layout="wide")
pio.templates.default = "plotly_dark"

st.markdown("""
<style>
  .block-container { padding-top: 0.75rem; padding-bottom: 2rem; }
  .kpi-card { background: #0b1220; border: 1px solid #1f2a44; border-radius: 14px; padding: 14px 16px; }
  .kpi-title { color: #e5eefc; font-weight: 700; font-size: 15px; letter-spacing: .2px; }
  .kpi-sub { color: #92a4c0; font-size: 12px; }
  .stProgress > div > div > div > div { background-color: #10b981 !important; }
  .hint { color:#93a2bd; font-size:12px; }
  .cluster-card { background:#0e1628; border:1px solid #213150; border-radius:14px; padding:14px; }
  .mini { color:#9DB0CE; font-size:12px; }
</style>
""", unsafe_allow_html=True)

# ================== COLUMN KEYS ==================
COL = {
    "ENTITY": "Entity",
    "YEAR": "Year",
    "ACCESS_ELECTRICITY": "Access to electricity (% of population)",
    "CLEAN_FUELS": "Access to clean fuels for cooking",
    "RENEW_CAP_PER_CAP": "Renewable-electricity-generating-capacity-per-capita",
    "FLOWS_USD": "Financial flows to developing countries (US $)",
    "RENEW_SHARE_TFEC": "Renewable energy share in the total final energy consumption (%)",
    "ELEC_FOSSIL_TWH": "Electricity from fossil fuels (TWh)",
    "ELEC_NUCLEAR_TWH": "Electricity from nuclear (TWh)",
    "ELEC_RENEW_TWH": "Electricity from renewables (TWh)",
    "LOW_CARBON_ELEC_PCT": "Low-carbon electricity (% electricity)",
    "PRIMARY_PER_CAP_KWH": "Primary energy consumption per capita (kWh/person)",
    "ENERGY_INTENSITY": "Energy intensity level of primary energy (MJ/$2017 PPP GDP)",
    "CO2_KT": "Value_co2_emissions_kt_by_country",
    "CO2_PC": "Value_co2_emissions (metric tons per capita)",
    "RENEW_EQ_PRIMARY": "Renewables (% equivalent primary energy)",
    "GDP_GROWTH": "gdp_growth",
    "GDP_PER_CAPITA": "gdp_per_capita",
    "DENSITY": "Density\n(P/Km2)",
    "LAND_KM2": "Land Area(Km2)",
    "LAT": "Latitude",
    "LON": "Longitude",
}
REQUIRED_COLS_BASE = [
    COL["ENTITY"], COL["YEAR"],
    COL["ACCESS_ELECTRICITY"], COL["CLEAN_FUELS"],
    COL["RENEW_SHARE_TFEC"],
    COL["ELEC_FOSSIL_TWH"], COL["ELEC_NUCLEAR_TWH"], COL["ELEC_RENEW_TWH"],
    COL["LOW_CARBON_ELEC_PCT"],
    COL["ENERGY_INTENSITY"],
    COL["GDP_PER_CAPITA"],
]
CO2_ACCEPT_ANY = [COL["CO2_KT"], COL["CO2_PC"]]

# ================== REGION MAP (expanded) ==================
BASE_REGION_MAP = {
    # --- Africa (excerpt) ---
    "Algeria":"Africa","Angola":"Africa","Benin":"Africa","Botswana":"Africa","Burkina Faso":"Africa","Burundi":"Africa",
    "Cabo Verde":"Africa","Cameroon":"Africa","Central African Republic":"Africa","Chad":"Africa","Comoros":"Africa",
    "Congo":"Africa","Congo, Dem. Rep.":"Africa","Democratic Republic of the Congo":"Africa",
    "Côte d’Ivoire":"Africa","Cote d'Ivoire":"Africa","Djibouti":"Africa","Egypt":"Africa","Equatorial Guinea":"Africa",
    "Eritrea":"Africa","Eswatini":"Africa","Swaziland":"Africa","Ethiopia":"Africa","Gabon":"Africa","Gambia":"Africa",
    "Ghana":"Africa","Guinea":"Africa","Guinea-Bissau":"Africa","Kenya":"Africa","Lesotho":"Africa","Liberia":"Africa",
    "Libya":"Africa","Madagascar":"Africa","Malawi":"Africa","Mali":"Africa","Mauritania":"Africa","Mauritius":"Africa",
    "Morocco":"Africa","Mozambique":"Africa","Namibia":"Africa","Niger":"Africa","Nigeria":"Africa","Rwanda":"Africa",
    "Sao Tome and Principe":"Africa","Senegal":"Africa","Seychelles":"Africa","Sierra Leone":"Africa","Somalia":"Africa",
    "South Africa":"Africa","South Sudan":"Africa","Sudan":"Africa","Tanzania":"Africa","Togo":"Africa","Tunisia":"Africa",
    "Uganda":"Africa","Zambia":"Africa","Zimbabwe":"Africa",
    # --- Americas (excerpt) ---
    "United States":"North America","Canada":"North America","Mexico":"North America","Costa Rica":"North America",
    "Guatemala":"North America","Honduras":"North America","El Salvador":"North America","Nicaragua":"North America",
    "Panama":"North America","Dominican Republic":"North America","Jamaica":"North America","Haiti":"North America",
    "Cuba":"North America","Trinidad and Tobago":"North America","Barbados":"North America","Bahamas":"North America",
    "Belize":"North America","Grenada":"North America","Saint Lucia":"North America","Antigua and Barbuda":"North America",
    "Saint Kitts and Nevis":"North America","Saint Vincent and the Grenadines":"North America",
    "Argentina":"South America","Bolivia":"South America","Brazil":"South America","Chile":"South America",
    "Colombia":"South America","Ecuador":"South America","Guyana":"South America","Paraguay":"South America",
    "Peru":"South America","Suriname":"South America","Uruguay":"South America","Venezuela":"South America",
    # --- Asia (excerpt) ---
    "Afghanistan":"Asia","Armenia":"Asia","Azerbaijan":"Asia","Bahrain":"Asia","Bangladesh":"Asia","Bhutan":"Asia",
    "Brunei Darussalam":"Asia","Cambodia":"Asia","China":"Asia","Georgia":"Asia","India":"Asia","Indonesia":"Asia",
    "Iran, Islamic Republic of":"Asia","Iraq":"Asia","Israel":"Asia","Japan":"Asia","Jordan":"Asia","Kazakhstan":"Asia",
    "Kuwait":"Asia","Kyrgyzstan":"Asia","Lao PDR":"Asia","Lebanon":"Asia","Malaysia":"Asia","Maldives":"Asia",
    "Mongolia":"Asia","Myanmar":"Asia","Nepal":"Asia","Oman":"Asia","Pakistan":"Asia","Palestine":"Asia",
    "Philippines":"Asia","Qatar":"Asia","Republic of Korea":"Asia","South Korea":"Asia","Saudi Arabia":"Asia",
    "Singapore":"Asia","Sri Lanka":"Asia","Syria":"Asia","Tajikistan":"Asia","Thailand":"Asia","Timor-Leste":"Asia",
    "Turkey":"Asia","Türkiye":"Asia","Turkmenistan":"Asia","United Arab Emirates":"Asia","Uzbekistan":"Asia",
    "Viet Nam":"Asia","Vietnam":"Asia","Yemen":"Asia",
    # --- Europe (excerpt) ---
    "Albania":"Europe","Andorra":"Europe","Austria":"Europe","Belarus":"Europe","Belgium":"Europe","Bosnia and Herzegovina":"Europe",
    "Bulgaria":"Europe","Croatia":"Europe","Cyprus":"Europe","Czechia":"Europe","Czech Republic":"Europe","Denmark":"Europe",
    "Estonia":"Europe","Finland":"Europe","France":"Europe","Germany":"Europe","Greece":"Europe","Hungary":"Europe",
    "Iceland":"Europe","Ireland":"Europe","Italy":"Europe","Kosovo":"Europe","Latvia":"Europe","Liechtenstein":"Europe",
    "Lithuania":"Europe","Luxembourg":"Europe","Malta":"Europe","Moldova":"Europe","Monaco":"Europe","Montenegro":"Europe",
    "Netherlands":"Europe","North Macedonia":"Europe","Norway":"Europe","Poland":"Europe","Portugal":"Europe","Romania":"Europe",
    "Russia":"Europe","San Marino":"Europe","Serbia":"Europe","Slovakia":"Europe","Slovenia":"Europe","Spain":"Europe",
    "Sweden":"Europe","Switzerland":"Europe","Ukraine":"Europe","United Kingdom":"Europe","UK":"Europe",
    # --- Oceania ---
    "Australia":"Oceania","New Zealand":"Oceania","Fiji":"Oceania","Kiribati":"Oceania","Marshall Islands":"Oceania",
    "Micronesia (Federated States of)":"Oceania","Nauru":"Oceania","Palau":"Oceania","Papua New Guinea":"Oceania",
    "Samoa":"Oceania","Solomon Islands":"Oceania","Tonga":"Oceania","Tuvalu":"Oceania","Vanuatu":"Oceania",
}

def region_from_sources(df: pd.DataFrame) -> pd.Series:
    region_cols = [c for c in df.columns if str(c).strip().lower() in {
        "region","continent","un region","un_region","who region","who_region","worldbank region","worldbank_region"
    }]
    if region_cols:
        s = df[region_cols[0]].astype(str).str.strip().replace({"nan": np.nan})
        if s.notna().any():
            return s.fillna("Other/Unknown")
    return df[COL["ENTITY"]].map(lambda x: BASE_REGION_MAP.get(str(x), "Other/Unknown"))

# ================== LOADING HELPERS ==================
SCRIPT_DIR = Path(__file__).parent
PRIMARY_NAME = "global-data-on-sustainable-energy (1).csv"

def find_local_csv() -> Path | None:
    for c in [SCRIPT_DIR / PRIMARY_NAME, SCRIPT_DIR / "data" / PRIMARY_NAME]:
        if c.exists(): return c
    for pat in [str(SCRIPT_DIR / "global*on*sustainable*energy*(1).csv"),
                str(SCRIPT_DIR / "data" / "global*on*sustainable*energy*(1).csv")]:
        hits = glob.glob(pat)
        if hits: return Path(hits[0])
    return None

@st.cache_data
def load_default_or_fail() -> pd.DataFrame:
    p = find_local_csv()
    if not p: return pd.DataFrame()
    return pd.read_csv(p)

def validate_columns(df: pd.DataFrame) -> list[str]:
    missing = [c for c in REQUIRED_COLS_BASE if c not in df.columns]
    if all([c not in df.columns for c in CO2_ACCEPT_ANY]):
        missing.append(f"CO₂ column (need '{COL['CO2_KT']}' OR '{COL['CO2_PC']}')")
    return missing

ALIAS = {
    "Access to clean fuels for cooking (% of population)": COL["CLEAN_FUELS"],
    "Value_co2_emissions (metric tons per capita)": COL["CO2_PC"],
    "Value_co2_emissions_kt_by_country": COL["CO2_KT"],
}
def ensure_alias_columns(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    for k, v in ALIAS.items():
        if k in df.columns and v not in df.columns:
            df.rename(columns={k: v}, inplace=True)
    return df

# ================== DATA QA ==================
PCT_COLS = [COL["ACCESS_ELECTRICITY"], COL["CLEAN_FUELS"], COL["RENEW_SHARE_TFEC"], COL["LOW_CARBON_ELEC_PCT"]]
NONNEG_COLS = [COL["ELEC_FOSSIL_TWH"], COL["ELEC_NUCLEAR_TWH"], COL["ELEC_RENEW_TWH"], COL["RENEW_CAP_PER_CAP"], COL["FLOWS_USD"]]
NUMERIC_CANDIDATES = list({*PCT_COLS, *NONNEG_COLS, COL["GDP_PER_CAPITA"], COL["ENERGY_INTENSITY"], COL["CO2_KT"], COL["CO2_PC"]})

def _to_numeric(df: pd.DataFrame, cols: list[str]) -> pd.DataFrame:
    df = df.copy()
    for c in cols:
        if c in df.columns: df[c] = pd.to_numeric(df[c], errors="coerce")
    return df

def _dedupe_entity_year(df: pd.DataFrame) -> pd.DataFrame:
    if COL["ENTITY"] not in df.columns or COL["YEAR"] not in df.columns: return df
    df = df.copy()
    score_cols = [c for c in [*PCT_COLS, COL["GDP_PER_CAPITA"], COL["ENERGY_INTENSITY"], COL["CO2_KT"], COL["CO2_PC"]] if c in df.columns]
    df["_nn"] = df[score_cols].notna().sum(axis=1)
    rs = COL["RENEW_SHARE_TFEC"]
    df["_rs"] = df[rs].fillna(-1) if rs in df.columns else -1
    df = (df.sort_values([COL["ENTITY"], COL["YEAR"], "_nn", "_rs"], ascending=[True, True, False, False])
            .drop_duplicates(subset=[COL["ENTITY"], COL["YEAR"]], keep="first")
            .drop(columns=["_nn","_rs"]))
    return df

def clean_dataframe(raw: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    df = ensure_alias_columns(raw.copy())
    df = _to_numeric(df, NUMERIC_CANDIDATES)
    pct_clip, nonneg_fix = {}, {}
    for c in PCT_COLS:
        if c in df.columns:
            before = df[c].copy()
            df[c] = df[c].clip(0, 100)
            pct_clip[c] = int((before != df[c]).fillna(False).sum())
    for c in NONNEG_COLS:
        if c in df.columns:
            before = df[c].copy()
            df[c] = df[c].clip(lower=0)
            nonneg_fix[c] = int((before != df[c]).fillna(False).sum())
    if COL["GDP_PER_CAPITA"] in df.columns: df.loc[df[COL["GDP_PER_CAPITA"]] < 0, COL["GDP_PER_CAPITA"]] = np.nan
    if COL["ENERGY_INTENSITY"] in df.columns: df.loc[df[COL["ENERGY_INTENSITY"]] < 0, COL["ENERGY_INTENSITY"]] = np.nan
    pre = len(df); df = _dedupe_entity_year(df); dedup_removed = pre - len(df)
    return df, {"pct_clipped": pct_clip, "nonneg_fixed": nonneg_fix, "dedup_removed": dedup_removed}

# ================== SIDEBAR ==================
st.sidebar.title("Data & Filters")
uploaded = st.sidebar.file_uploader("Upload CSV", type=["csv"], key="data_upl")
df = pd.read_csv(uploaded) if uploaded is not None else load_default_or_fail()
if df.empty:
    st.sidebar.warning(f"No CSV found. Place **{PRIMARY_NAME}** next to this script (or in ./data/) or upload it.")
    st.stop()
st.sidebar.success("Data loaded.")

# year & schema
if COL["YEAR"] in df.columns:
    df = df[pd.to_numeric(df[COL["YEAR"]], errors="coerce").notna()].copy()
    df[COL["YEAR"]] = df[COL["YEAR"]].astype(int)
missing = validate_columns(df)
if missing:
    st.error("Missing required columns: " + ", ".join(missing)); st.stop()

# Clean + Region
df, QA = clean_dataframe(df)
df["Region(_auto_)"] = region_from_sources(df).fillna("Other/Unknown")

# Filters
all_years = sorted(df[COL["YEAR"]].unique().tolist())
sel_year = st.sidebar.slider("Year", int(min(all_years)), int(max(all_years)), int(max(all_years)), step=1)
regions = ["All"] + sorted(df["Region(_auto_)"].unique().tolist())
selected_region = st.sidebar.selectbox("Region", regions, index=0)
all_countries = sorted(df[COL["ENTITY"]].dropna().unique().tolist())
country_search = st.sidebar.text_input("Quick country search")
country_options = [c for c in all_countries if country_search.lower() in c.lower()] if country_search else all_countries
selected_countries = st.sidebar.multiselect("Countries (leave empty = All)", options=country_options, default=[])

compare_mode = st.sidebar.checkbox("Compare to another year")
comp_year = None
if compare_mode:
    comp_year = st.sidebar.slider("Compare Year", int(min(all_years)), int(max(all_years)), int(min(all_years)))

kpi_agg = st.sidebar.radio("KPI aggregation for percentages", ["Mean", "Median"], horizontal=True)
top_n = st.sidebar.slider("Top N for leaderboards/flows", 5, 25, 10, step=1)

def _apply_filters(df_in: pd.DataFrame) -> pd.DataFrame:
    d = df_in.copy()
    if selected_region != "All": d = d[d["Region(_auto_)"] == selected_region]
    if selected_countries: d = d[d[COL["ENTITY"]].isin(selected_countries)]
    return d

df_f = _apply_filters(df)

# ================== TITLE & QA ==================
st.title("Global Sustainable Energy Dashboard")
st.caption("Interactive Plotly visuals, robust data QA, and auto region mapping.")

if (sum(QA.get("pct_clipped", {}).values()) + sum(QA.get("nonneg_fixed", {}).values()) + QA.get("dedup_removed", 0)) > 0:
    with st.expander("Data QA adjustments applied (click for details)"):
        st.write(f"- (Entity, Year) duplicates removed: **{QA['dedup_removed']}**")
        if any(v>0 for v in QA["pct_clipped"].values()):
            st.write("Percentages clipped to [0,100]:"); st.json({k:v for k,v in QA["pct_clipped"].items() if v>0})
        if any(v>0 for v in QA["nonneg_fixed"].values()):
            st.write("Non-negative fixes (set <0 → 0):"); st.json({k:v for k,v in QA["nonneg_fixed"].items() if v>0})

# ================== KPI HELPERS ==================
@st.cache_data
def compute_kpis(df_all: pd.DataFrame, year: int, agg: str):
    d = df_all[df_all[COL["YEAR"]] == year].copy()
    if d.empty:
        return dict(electricity=np.nan, clean=np.nan, renewshare=np.nan, lowcarbon=np.nan,
                    co2_mt=None, co2pc=None, energy_intensity=np.nan, gdp_pc=np.nan, flows_musd=np.nan)
    agg_func = np.nanmean if agg == "Mean" else np.nanmedian
    electricity = float(agg_func(pd.to_numeric(d[COL["ACCESS_ELECTRICITY"]], errors="coerce")))
    clean = float(agg_func(pd.to_numeric(d[COL["CLEAN_FUELS"]], errors="coerce")))
    renewshare = float(agg_func(pd.to_numeric(d[COL["RENEW_SHARE_TFEC"]], errors="coerce")))
    lowcarbon = float(agg_func(pd.to_numeric(d[COL["LOW_CARBON_ELEC_PCT"]], errors="coerce")))
    energy_intensity = float(agg_func(pd.to_numeric(d[COL["ENERGY_INTENSITY"]], errors="coerce")))
    gdp_pc = float(agg_func(pd.to_numeric(d[COL["GDP_PER_CAPITA"]], errors="coerce")))
    flows_musd = pd.to_numeric(d[COL["FLOWS_USD"]], errors="coerce").sum() / 1e6 if COL["FLOWS_USD"] in d.columns else np.nan
    co2_mt, co2pc = None, None
    if COL["CO2_KT"] in d.columns and d[COL["CO2_KT"]].notna().any():
        co2_mt = pd.to_numeric(d[COL["CO2_KT"]], errors="coerce").sum() / 1000.0  # kt -> Mt
    elif COL["CO2_PC"] in d.columns and d[COL["CO2_PC"]].notna().any():
        co2pc = float(agg_func(pd.to_numeric(d[COL["CO2_PC"]], errors="coerce")))
    return dict(electricity=electricity, clean=clean, renewshare=renewshare, lowcarbon=lowcarbon,
                co2_mt=co2_mt, co2pc=co2pc, energy_intensity=energy_intensity, gdp_pc=gdp_pc, flows_musd=flows_musd)

def progress_bar(pct: float):
    if np.isnan(pct): pct = 0.0
    pct = max(0.0, min(100.0, pct))
    st.progress(pct/100.0, text=f"{pct:.1f}%")

# ---------- Helpers de fallback ----------
def nearest_year_with(df_in: pd.DataFrame, metric_col: str, target_year: int) -> int:
    """Primer año (por distancia) con al menos 1 valor no nulo en 'metric_col' bajo los filtros actuales."""
    d = df_in[[COL["YEAR"], metric_col]].copy()
    d = d[pd.to_numeric(d[metric_col], errors="coerce").notna()]
    if d.empty: return target_year
    years = d[COL["YEAR"]].unique().tolist()
    years_sorted = sorted(years, key=lambda y: (abs(y - target_year), y))
    return int(years_sorted[0])

def nearest_year_with_all(df_in: pd.DataFrame, metric_cols: list[str], target_year: int) -> int:
    """Primer año con filas que tengan TODOS los 'metric_cols' no nulos."""
    d = df_in[[COL["YEAR"], *metric_cols]].copy()
    for c in metric_cols:
        d = d[pd.to_numeric(d[c], errors="coerce").notna()]
    if d.empty: return target_year
    years = d[COL["YEAR"]].unique().tolist()
    years_sorted = sorted(years, key=lambda y: (abs(y - target_year), y))
    return int(years_sorted[0])

# ================== KPI SECTION ==================
kpi_vals = compute_kpis(df_f, sel_year, kpi_agg)
st.subheader("Key Indicators")
c1, c2, c3, c4 = st.columns(4)
with c1:
    st.markdown('<div class="kpi-card"><div class="kpi-title">🔌 Access to Electricity</div></div>', unsafe_allow_html=True)
    st.metric("", f"{kpi_vals['electricity']:.1f}%"); progress_bar(kpi_vals['electricity'])
with c2:
    st.markdown('<div class="kpi-card"><div class="kpi-title">🍳 Clean Cooking Fuels</div></div>', unsafe_allow_html=True)
    st.metric("", f"{kpi_vals['clean']:.1f}%"); progress_bar(kpi_vals['clean'])
with c3:
    st.markdown('<div class="kpi-card"><div class="kpi-title">☀️ Renewable Energy Share</div></div>', unsafe_allow_html=True)
    st.metric("", f"{kpi_vals['renewshare']:.1f}%"); progress_bar(kpi_vals['renewshare'])
with c4:
    st.markdown('<div class="kpi-card"><div class="kpi-title">🌿 Low-carbon Electricity</div></div>', unsafe_allow_html=True)
    st.metric("", f"{kpi_vals['lowcarbon']:.1f}%"); progress_bar(kpi_vals['lowcarbon'])
st.markdown(f'<div class="hint">Percent KPIs use <b>{kpi_agg.lower()}</b> across your current selection.</div>', unsafe_allow_html=True)

# ========= (1) Panel de Cobertura =========
st.markdown("### Cobertura de datos en el año seleccionado")
coverage_cols = [
    COL["RENEW_SHARE_TFEC"], COL["LOW_CARBON_ELEC_PCT"], COL["ENERGY_INTENSITY"],
    COL["ELEC_FOSSIL_TWH"], COL["ELEC_NUCLEAR_TWH"], COL["ELEC_RENEW_TWH"]
]
co2_col_cov = COL["CO2_KT"] if COL["CO2_KT"] in df.columns else (COL["CO2_PC"] if COL["CO2_PC"] in df.columns else None)
if co2_col_cov: coverage_cols.append(co2_col_cov)

cov_year = df_f[df_f[COL["YEAR"]] == sel_year]
cov_rows = []
for c in coverage_cols:
    if c in cov_year.columns:
        cov_rows.append({"Metric": c, "Non-null": int(pd.to_numeric(cov_year[c], errors="coerce").notna().sum()), "Rows in selection": len(cov_year)})
cov = pd.DataFrame(cov_rows)
st.dataframe(cov, use_container_width=True, height=220)

# ========= Comparison Panel (ALL KPIs/Metrics when toggled) =========
if compare_mode and comp_year is not None:
    st.markdown("### Period Comparison")
    kpi_base = compute_kpis(df_f, comp_year, kpi_agg)
    # Percent metrics chart
    pct_df = pd.DataFrame({
        "Metric": ["Electricity Access", "Clean Fuels", "Renewable Share", "Low-carbon Electricity"],
        str(comp_year): [kpi_base['electricity'], kpi_base['clean'], kpi_base['renewshare'], kpi_base['lowcarbon']],
        str(sel_year):  [kpi_vals['electricity'], kpi_vals['clean'], kpi_vals['renewshare'], kpi_vals['lowcarbon']]
    })
    pct_df = pct_df.melt(id_vars="Metric", var_name="Year", value_name="Percent")
    fig_pct = px.bar(pct_df, x="Percent", y="Metric", color="Year", barmode="group", orientation="h",
                     labels={"Percent":"%", "Metric":""})
    fig_pct.update_layout(height=360, legend_title="")
    st.plotly_chart(fig_pct, use_container_width=True)

    # Level metrics chart
    level_names, base_vals, cur_vals = [], [], []
    if kpi_vals["co2_mt"] is not None or kpi_base["co2_mt"] is not None:
        level_names.append("CO₂ (Mt)"); base_vals.append(kpi_base["co2_mt"] or 0); cur_vals.append(kpi_vals["co2_mt"] or 0)
    elif kpi_vals["co2pc"] is not None or kpi_base["co2pc"] is not None:
        level_names.append("CO₂ (t/person)"); base_vals.append(kpi_base["co2pc"] or 0); cur_vals.append(kpi_vals["co2pc"] or 0)
    level_names += ["Energy Intensity (MJ/$)", "GDP per Capita (USD)", "Financial Flows (MUSD)"]
    base_vals += [kpi_base["energy_intensity"], kpi_base["gdp_pc"], kpi_base["flows_musd"]]
    cur_vals  += [kpi_vals["energy_intensity"], kpi_vals["gdp_pc"], kpi_vals["flows_musd"]]

    lvl_df = pd.DataFrame({"Metric": level_names, str(comp_year): base_vals, str(sel_year): cur_vals})
    lvl_df = lvl_df.melt(id_vars="Metric", var_name="Year", value_name="Value")
    fig_lvl = px.bar(lvl_df, x="Value", y="Metric", color="Year", barmode="group", orientation="h",
                     labels={"Value":"Value", "Metric":""})
    fig_lvl.update_layout(height=380, legend_title="")
    st.plotly_chart(fig_lvl, use_container_width=True)

# ================== TABS ==================
tab1, tab2, tab3, tab4, tab5 = st.tabs(["📈 Trends", "🏆 Leaders & Flows", "📊 Analysis", "🔮 Forecast", "📄 Data"])

# ---------- Tab 1: Trends ----------
with tab1:
    st.subheader("Energy Source Evolution (TWh)")
    d = df_f.groupby(COL["YEAR"], as_index=False).agg(
        Fossil=(COL["ELEC_FOSSIL_TWH"], "sum"),
        Nuclear=(COL["ELEC_NUCLEAR_TWH"], "sum"),
        Renewables=(COL["ELEC_RENEW_TWH"], "sum"),
    ).sort_values(COL["YEAR"])
    if d.empty:
        st.info("No data available for energy transition.")
    else:
        dfm = d.melt(id_vars=[COL["YEAR"]], var_name="Source", value_name="TWh")
        fig = px.area(dfm, x=COL["YEAR"], y="TWh", color="Source", labels={COL["YEAR"]:"Year"})
        if compare_mode and comp_year is not None:
            for yv in [sel_year, comp_year]:
                fig.add_vline(x=float(yv), line_dash="dash", opacity=0.5)
        fig.update_layout(height=420, legend_title="")
        st.plotly_chart(fig, use_container_width=True)

# ---------- Tab 2: Leaders & Flows ----------
with tab2:
    left, right = st.columns((1,1), gap="large")
    with left:
        st.subheader("Top Renewable Leaders — (cleaned)")
        # Fallback para ranking de renovables
        year_use = sel_year
        tmp = df_f[df_f[COL["YEAR"]] == year_use][COL["RENEW_SHARE_TFEC"]]
        if pd.to_numeric(tmp, errors="coerce").dropna().empty:
            year_use = nearest_year_with(df_f, COL["RENEW_SHARE_TFEC"], sel_year)
            st.caption(f"⚠️ Sin datos en {sel_year} para renovables; mostrando año más cercano: {year_use}")
        sub = df_f[df_f[COL["YEAR"]] == year_use][
            [COL["ENTITY"], COL["RENEW_SHARE_TFEC"], COL["LOW_CARBON_ELEC_PCT"], "Region(_auto_)"]
        ].copy()
        sub[COL["RENEW_SHARE_TFEC"]] = pd.to_numeric(sub[COL["RENEW_SHARE_TFEC"]], errors="coerce").clip(0, 100)
        if COL["LOW_CARBON_ELEC_PCT"] in sub.columns:
            sub[COL["LOW_CARBON_ELEC_PCT"]] = pd.to_numeric(sub[COL["LOW_CARBON_ELEC_PCT"]], errors="coerce").clip(0, 100)
        sub = sub.dropna(subset=[COL["RENEW_SHARE_TFEC"]])
        if sub.empty:
            st.info("No renewable-share observations for this slice.")
        else:
            d_rank = sub.sort_values(COL["RENEW_SHARE_TFEC"], ascending=False).head(top_n)
            fig = px.bar(d_rank[::-1], x=COL["RENEW_SHARE_TFEC"], y=COL["ENTITY"], color="Region(_auto_)",
                         orientation="h", labels={COL["RENEW_SHARE_TFEC"]:"Renewable share (%)", COL["ENTITY"]:""})
            fig.update_layout(height=520, legend_title="")
            st.plotly_chart(fig, use_container_width=True)
            st.dataframe(
                d_rank.rename(columns={
                    COL["ENTITY"]:"Country",
                    COL["RENEW_SHARE_TFEC"]:"Renewable Share (%)",
                    COL["LOW_CARBON_ELEC_PCT"]:"Low-carbon (% electricity)"
                }).reset_index(drop=True),
                use_container_width=True, height=320
            )
    with right:
        st.subheader("Financial Flows")
        # Fallback para flujos
        year_use_flows = sel_year
        tmpf = df_f[df_f[COL["YEAR"]] == year_use_flows][COL["FLOWS_USD"]]
        if pd.to_numeric(tmpf, errors="coerce").dropna().empty:
            year_use_flows = nearest_year_with(df_f, COL["FLOWS_USD"], sel_year)
            st.caption(f"⚠️ Sin datos en {sel_year} para flujos; mostrando año más cercano: {year_use_flows}")
        flows = df_f[df_f[COL["YEAR"]] == year_use_flows][[COL["ENTITY"], COL["FLOWS_USD"], "Region(_auto_)"]].copy()
        flows[COL["FLOWS_USD"]] = pd.to_numeric(flows[COL["FLOWS_USD"]], errors="coerce")
        flows = flows.dropna()
        if flows.empty:
            st.info("No financial flow data for selected year.")
        else:
            flows["Flow_MUSD"] = flows[COL["FLOWS_USD"]] / 1e6
            flows = flows.sort_values("Flow_MUSD", ascending=False).head(top_n)
            fig = px.bar(flows[::-1], x="Flow_MUSD", y=COL["ENTITY"], orientation="h",
                         color="Region(_auto_)", labels={"Flow_MUSD":"Million USD", COL["ENTITY"]:""})
            fig.update_layout(height=520, legend_title="")
            st.plotly_chart(fig, use_container_width=True)

# ---------- Tab 3: Analysis ----------
with tab3:
    left, right = st.columns((1,1), gap="large")
    with left:
        st.subheader("CO₂ Emissions vs Energy Intensity")
        use_log_y = st.checkbox("Log scale for CO₂ axis", value=True, key="logco2")
        top_emitters = st.slider("Show top N emitters", 10, 100, 40, step=5, key="topemit")
        if COL["CO2_KT"] in df_f.columns and df_f[COL["CO2_KT"]].notna().any():
            co2_col, y_label, scaler = COL["CO2_KT"], "CO₂ (Mt)", 1000.0
        elif COL["CO2_PC"] in df_f.columns and df_f[COL["CO2_PC"]].notna().any():
            co2_col, y_label, scaler = COL["CO2_PC"], "CO₂ (t per capita)", None
        else:
            co2_col, y_label, scaler = None, "", None

        # Fallback que exige ambas columnas no nulas en el año
        year_use_scatter = sel_year
        if co2_col is not None:
            needed_cols = [COL["ENERGY_INTENSITY"], co2_col]
            dtest = df_f[df_f[COL["YEAR"]] == year_use_scatter][needed_cols].apply(pd.to_numeric, errors="coerce")
            if dtest.dropna().empty:
                year_use_scatter = nearest_year_with_all(df_f, needed_cols, sel_year)
                st.caption(f"⚠️ Sin datos completos en {sel_year} para la dispersión; mostrando {year_use_scatter}")
        dyear = df_f[df_f[COL["YEAR"]] == year_use_scatter].copy()
        if co2_col is None or dyear.empty:
            st.info("No data for scatter.")
        else:
            cols_needed = [COL["ENTITY"], COL["ENERGY_INTENSITY"], co2_col, "Region(_auto_)"]
            dyear = dyear[cols_needed].replace([np.inf, -np.inf], np.nan).dropna()
            if dyear.empty:
                st.info("No data for scatter.")
            else:
                co2_vals = pd.to_numeric(dyear[co2_col], errors="coerce")
                dyear = dyear[co2_vals.notna()].copy()
                dyear["CO2_Y"] = (co2_vals / scaler) if scaler else co2_vals
                dyear = dyear.sort_values("CO2_Y", ascending=False).head(top_emitters)
                fig = px.scatter(dyear, x=COL["ENERGY_INTENSITY"], y="CO2_Y", color="Region(_auto_)",
                                 hover_name=COL["ENTITY"],
                                 labels={COL["ENERGY_INTENSITY"]:"Energy Intensity (MJ/$2017 PPP GDP)", "CO2_Y": y_label})
                if use_log_y: fig.update_yaxes(type="log")
                fig.update_traces(marker=dict(size=10, opacity=0.9, line=dict(width=0.5, color="rgba(255,255,255,0.4)")))
                fig.update_layout(height=460, legend_title="")
                st.plotly_chart(fig, use_container_width=True)
    with right:
        st.subheader("Key Metric Correlations")
        metrics = {
            "GDP per Capita": COL["GDP_PER_CAPITA"],
            "Renewable Share": COL["RENEW_SHARE_TFEC"],
            "Energy Intensity": COL["ENERGY_INTENSITY"],
        }
        if COL["CO2_KT"] in df_f.columns and df_f[COL["CO2_KT"]].notna().any():
            metrics["CO₂ (kt)"] = COL["CO2_KT"]
        elif COL["CO2_PC"] in df_f.columns and df_f[COL["CO2_PC"]].notna().any():
            metrics["CO₂ (t per cap)"] = COL["CO2_PC"]

        # Intento en año seleccionado
        dsel = df_f[df_f[COL["YEAR"]] == sel_year].copy()
        def corr_rows(dframe):
            rows, keys = [], list(metrics.keys())
            for i in range(len(keys)):
                for j in range(i+1, len(keys)):
                    a = pd.to_numeric(dframe[metrics[keys[i]]], errors="coerce")
                    b = pd.to_numeric(dframe[metrics[keys[j]]], errors="coerce")
                    both = pd.concat([a,b], axis=1).dropna()
                    if len(both)>=3: rows.append({"pair": f"{keys[i]} ↔ {keys[j]}", "r": float(both.corr().iloc[0,1])})
            return rows

        rows = corr_rows(dsel)

        # Fallback: buscar el año con mayor cantidad de pares válidos si el seleccionado no tiene
        if not rows:
            best_year, best_count = sel_year, -1
            for y in all_years:
                dtry = df_f[df_f[COL["YEAR"]]==y]
                rs = corr_rows(dtry)
                if len(rs) > best_count:
                    best_count, best_year, best_rows = len(rs), y, rs
            if best_count > 0:
                rows = best_rows
                st.caption(f"⚠️ Correlaciones insuficientes en {sel_year}; mostrando el año más informativo: {best_year}")

        if not rows:
            st.info("Not enough data for correlation pairs.")
        else:
            corr_df = pd.DataFrame(rows).sort_values("r", key=lambda s: s.abs(), ascending=False)
            fig = px.bar(corr_df[::-1], x="r", y="pair", orientation="h",
                         labels={"r":"Correlation (r)", "pair":""},
                         color=corr_df["r"].apply(lambda v: "Positive" if v>=0 else "Negative"),
                         color_discrete_map={"Positive":"#10B981","Negative":"#EF4444"})
            fig.update_layout(height=460, legend_title="")
            st.plotly_chart(fig, use_container_width=True)

    st.markdown("---")
    st.subheader("AI Country Clustering (K-means) — with Named Groups")
    feats_df = df_f[df_f[COL["YEAR"]] == sel_year][[
        COL["ACCESS_ELECTRICITY"], COL["RENEW_SHARE_TFEC"], COL["GDP_PER_CAPITA"], COL["LOW_CARBON_ELEC_PCT"],
        COL["ENTITY"], "Region(_auto_)"
    ]].dropna()
    if len(feats_df) < 6:
        # Fallback: intenta el año más cercano con >=6 filas completas en features
        years_valid = []
        for y in all_years:
            tmp = df_f[df_f[COL["YEAR"]]==y][[
                COL["ACCESS_ELECTRICITY"], COL["RENEW_SHARE_TFEC"], COL["GDP_PER_CAPITA"], COL["LOW_CARBON_ELEC_PCT"],
                COL["ENTITY"], "Region(_auto_)"
            ]].dropna()
            years_valid.append((y, len(tmp)))
        years_valid = [(y,n) for y,n in years_valid if n>=6]
        if years_valid:
            year_use_cluster = sorted(years_valid, key=lambda t:(abs(t[0]-sel_year), -t[1]))[0][0]
            st.caption(f"⚠️ Pocas filas para clustering en {sel_year}; usando {year_use_cluster}")
            feats_df = df_f[df_f[COL["YEAR"]] == year_use_cluster][[
                COL["ACCESS_ELECTRICITY"], COL["RENEW_SHARE_TFEC"], COL["GDP_PER_CAPITA"], COL["LOW_CARBON_ELEC_PCT"],
                COL["ENTITY"], "Region(_auto_)"
            ]].dropna()
        else:
            st.info("Not enough data to cluster countries.")
    if len(feats_df) >= 6:
        k = min(4, max(2, len(feats_df)//3))
        km = KMeans(n_clusters=k, n_init="auto", random_state=42)
        X = feats_df[[COL["ACCESS_ELECTRICITY"], COL["RENEW_SHARE_TFEC"], COL["GDP_PER_CAPITA"], COL["LOW_CARBON_ELEC_PCT"]]].values
        labels = km.fit_predict(X)
        feats_df["cluster"] = labels

        # --- Name clusters by simple rules on their means ---
        summary = feats_df.groupby("cluster").agg(
            N=(COL["ENTITY"], "count"),
            elec=(COL["ACCESS_ELECTRICITY"], "mean"),
            ren=(COL["RENEW_SHARE_TFEC"], "mean"),
            lowc=(COL["LOW_CARBON_ELEC_PCT"], "mean"),
            gdp=(COL["GDP_PER_CAPITA"], "mean"),
        )
        def name_row(r):
            if r.elec >= 90 and r.ren >= 35: return "Leaders"
            if r.elec >= 85 and r.ren < 35:  return "Developed Mixed"
            if r.elec >= 60 and r.ren >= 20: return "Emerging Transitioners"
            return "Early-Stage Access"
        names = summary.apply(name_row, axis=1)
        name_map = {i:names.loc[i] for i in summary.index}
        feats_df["ClusterName"] = feats_df["cluster"].map(name_map)

        # PCA for visualization
        pca = PCA(n_components=2, random_state=42)
        pc = pca.fit_transform(X)
        feats_df["PC1"] = pc[:,0]; feats_df["PC2"] = pc[:,1]

        fig = px.scatter(
            feats_df, x="PC1", y="PC2", color="ClusterName",
            hover_data={COL["ENTITY"]:True, "Region(_auto_)":True,
                        COL["ACCESS_ELECTRICITY"]:':.1f', COL["RENEW_SHARE_TFEC"]:':.1f',
                        COL["LOW_CARBON_ELEC_PCT"]:':.1f', COL["GDP_PER_CAPITA"]:':.0f'},
            labels={"ClusterName":"Cluster"},
        )
        fig.update_traces(marker=dict(size=10, line=dict(width=0.5, color="rgba(255,255,255,0.5)")))
        fig.update_layout(height=520, legend_title="")
        st.plotly_chart(fig, use_container_width=True)

        # Pretty cluster cards
        st.markdown("#### Cluster Summaries")
        for cname, grp in feats_df.groupby("ClusterName"):
            c = grp[[COL["ACCESS_ELECTRICITY"], COL["RENEW_SHARE_TFEC"], COL["LOW_CARBON_ELEC_PCT"], COL["GDP_PER_CAPITA"]]].mean()
            countries = ", ".join(grp[COL["ENTITY"]].tolist()[:15]) + ("..." if len(grp)>15 else "")
            col = st.container()
            with col:
                st.markdown(f"""
                <div class="cluster-card">
                  <div style="font-weight:700;font-size:16px">{cname} <span class="mini">({len(grp)} countries)</span></div>
                  <div class="mini" style="margin-top:6px">Avg access: {c[COL["ACCESS_ELECTRICITY"]]:.1f}% | Avg renew: {c[COL["RENEW_SHARE_TFEC"]]:.1f}% | Low-carbon: {c[COL["LOW_CARBON_ELEC_PCT"]]:.1f}% | GDP pc: ${c[COL["GDP_PER_CAPITA"]]:.0f}</div>
                  <div class="mini" style="margin-top:6px"><b>Examples:</b> {countries}</div>
                </div>
                """, unsafe_allow_html=True)

# ---------- Tab 4: Forecast ----------
with tab4:
    st.subheader("Renewable Energy Forecast to 2030")
    g = df_f.groupby(COL["YEAR"], as_index=False)[COL["RENEW_SHARE_TFEC"]].mean().dropna().sort_values(COL["YEAR"])
    if len(g) < 2:
        # Fallback: usa el rango completo sin promediar si hay muy pocos puntos por año
        st.info("Not enough data to forecast.")
    else:
        X = g[[COL["YEAR"]]].values; y = g[COL["RENEW_SHARE_TFEC"]].values
        lr = LinearRegression().fit(X, y)
        last_year = int(g[COL["YEAR"]].max())
        future_years = np.arange(last_year + 1, 2030 + 1)
        y_fit = lr.predict(X)
        y_fore = lr.predict(future_years.reshape(-1,1)) if len(future_years) else np.array([])
        r2 = lr.score(X, y)

        fig = go.Figure()
        fig.add_trace(go.Scatter(x=g[COL["YEAR"]], y=y, mode="lines+markers", name="Actual"))
        fig.add_trace(go.Scatter(x=g[COL["YEAR"]], y=y_fit, mode="lines", name="Predicted (fit)", line=dict(dash="dash")))
        if len(y_fore): fig.add_trace(go.Scatter(x=future_years, y=y_fore, mode="lines", name="Forecast", line=dict(dash="dot")))
        fig.update_layout(height=440, xaxis_title="Year", yaxis_title="Renewable share (%)", legend_title="")
        st.plotly_chart(fig, use_container_width=True)
        if len(y_fore): st.caption(f"R² = {r2:.3f} — 2030 projection: {float(y_fore[-1]):.1f}%")

# ---------- Tab 5: Data ----------
with tab5:
    st.subheader("Filtered Data")
    year_min = sel_year if not compare_mode else min(sel_year, comp_year or sel_year)
    year_max = sel_year if not compare_mode else max(sel_year, comp_year or sel_year)
    show_df = df_f[df_f[COL["YEAR"]].between(year_min, year_max)]
    st.dataframe(show_df, use_container_width=True, height=460)
    st.download_button("Download filtered CSV", data=show_df.to_csv(index=False), file_name="filtered_energy_data.csv", mime="text/csv")

st.markdown("---")
st.caption("Cobertura visible y fallback automático por vista evitan pantallas vacías cuando el año seleccionado tiene huecos.")