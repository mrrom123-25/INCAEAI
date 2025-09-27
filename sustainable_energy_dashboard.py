# sustainable_energy_dashboard.py
import streamlit as st
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
from pathlib import Path
import glob

# =============== PAGE CONFIG & STYLE ===============
st.set_page_config(page_title="Global Sustainable Energy Dashboard", layout="wide")

st.markdown("""
<style>
  .block-container { padding-top: 0.75rem; padding-bottom: 2rem; }
  .kpi-card {
    background: #0b1220; border: 1px solid #1f2a44; border-radius: 14px; padding: 14px 16px;
  }
  .kpi-title { color: #e5eefc; font-weight: 700; font-size: 15px; letter-spacing: .2px; }
  .kpi-sub { color: #92a4c0; font-size: 12px; }
  .stProgress > div > div > div > div { background-color: #10b981 !important; }
  .hint { color:#93a2bd; font-size:12px; }
</style>
""", unsafe_allow_html=True)

# =============== COLUMN KEYS (MUST MATCH CSV) ===============
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

# =============== INTERNAL REGION MAP (expanded but not exhaustive) ===============
BASE_REGION_MAP = {
    # Africa
    "Algeria":"Africa","Angola":"Africa","Benin":"Africa","Botswana":"Africa","Burkina Faso":"Africa","Burundi":"Africa",
    "Cabo Verde":"Africa","Cape Verde":"Africa","Cameroon":"Africa","Central African Republic":"Africa","Chad":"Africa",
    "Comoros":"Africa","Congo":"Africa","Congo, Dem. Rep.":"Africa","Democratic Republic of the Congo":"Africa",
    "Côte d’Ivoire":"Africa","Cote d'Ivoire":"Africa","Ivory Coast":"Africa","Djibouti":"Africa","Egypt":"Africa",
    "Equatorial Guinea":"Africa","Eritrea":"Africa","Eswatini":"Africa","Swaziland":"Africa","Ethiopia":"Africa",
    "Gabon":"Africa","Gambia":"Africa","Ghana":"Africa","Guinea":"Africa","Guinea-Bissau":"Africa","Kenya":"Africa",
    "Lesotho":"Africa","Liberia":"Africa","Libya":"Africa","Madagascar":"Africa","Malawi":"Africa","Mali":"Africa",
    "Mauritania":"Africa","Mauritius":"Africa","Morocco":"Africa","Mozambique":"Africa","Namibia":"Africa","Niger":"Africa",
    "Nigeria":"Africa","Rwanda":"Africa","Sao Tome and Principe":"Africa","Senegal":"Africa","Seychelles":"Africa",
    "Sierra Leone":"Africa","Somalia":"Africa","South Africa":"Africa","South Sudan":"Africa","Sudan":"Africa",
    "Tanzania":"Africa","United Republic of Tanzania":"Africa","Togo":"Africa","Tunisia":"Africa","Uganda":"Africa",
    "Zambia":"Africa","Zimbabwe":"Africa",
    # Americas
    "Antigua and Barbuda":"North America","Bahamas":"North America","Barbados":"North America","Belize":"North America",
    "Canada":"North America","Costa Rica":"North America","Cuba":"North America","Dominica":"North America",
    "Dominican Republic":"North America","El Salvador":"North America","Grenada":"North America","Guatemala":"North America",
    "Haiti":"North America","Honduras":"North America","Jamaica":"North America","Mexico":"North America",
    "Nicaragua":"North America","Panama":"North America","Saint Kitts and Nevis":"North America",
    "Saint Lucia":"North America","Saint Vincent and the Grenadines":"North America","Trinidad and Tobago":"North America",
    "United States":"North America",
    "Argentina":"South America","Bolivia":"South America","Plurinational State of Bolivia":"South America","Brazil":"South America",
    "Chile":"South America","Colombia":"South America","Ecuador":"South America","Guyana":"South America","Paraguay":"South America",
    "Peru":"South America","Suriname":"South America","Uruguay":"South America","Venezuela":"South America",
    # Asia
    "Afghanistan":"Asia","Armenia":"Asia","Azerbaijan":"Asia","Bahrain":"Asia","Bangladesh":"Asia","Bhutan":"Asia",
    "Brunei Darussalam":"Asia","Brunei":"Asia","Cambodia":"Asia","China":"Asia","Georgia":"Asia","India":"Asia",
    "Indonesia":"Asia","Iran, Islamic Republic of":"Asia","Iran (Islamic Republic of)":"Asia","Iraq":"Asia",
    "Israel":"Asia","Japan":"Asia","Jordan":"Asia","Kazakhstan":"Asia","Kuwait":"Asia","Kyrgyzstan":"Asia","Lao PDR":"Asia",
    "Lao People's Democratic Republic":"Asia","Lebanon":"Asia","Malaysia":"Asia","Maldives":"Asia","Mongolia":"Asia",
    "Myanmar":"Asia","Nepal":"Asia","Oman":"Asia","Pakistan":"Asia","Palestine":"Asia","Philippines":"Asia","Qatar":"Asia",
    "Republic of Korea":"Asia","South Korea":"Asia","Saudi Arabia":"Asia","Singapore":"Asia","Sri Lanka":"Asia",
    "Syrian Arab Republic":"Asia","Syria":"Asia","Tajikistan":"Asia","Thailand":"Asia","Timor-Leste":"Asia",
    "Turkey":"Asia","Türkiye":"Asia","Turkmenistan":"Asia","United Arab Emirates":"Asia","Uzbekistan":"Asia",
    "Viet Nam":"Asia","Vietnam":"Asia","Yemen":"Asia",
    # Europe
    "Albania":"Europe","Andorra":"Europe","Austria":"Europe","Belarus":"Europe","Belgium":"Europe",
    "Bosnia and Herzegovina":"Europe","Bulgaria":"Europe","Croatia":"Europe","Cyprus":"Europe","Czechia":"Europe",
    "Czech Republic":"Europe","Denmark":"Europe","Estonia":"Europe","Finland":"Europe","France":"Europe","Germany":"Europe",
    "Greece":"Europe","Hungary":"Europe","Iceland":"Europe","Ireland":"Europe","Italy":"Europe","Kosovo":"Europe",
    "Latvia":"Europe","Liechtenstein":"Europe","Lithuania":"Europe","Luxembourg":"Europe","Malta":"Europe",
    "Moldova":"Europe","Republic of Moldova":"Europe","Monaco":"Europe","Montenegro":"Europe","Netherlands":"Europe",
    "North Macedonia":"Europe","Norway":"Europe","Poland":"Europe","Portugal":"Europe","Romania":"Europe",
    "Russian Federation":"Europe","Russia":"Europe","San Marino":"Europe","Serbia":"Europe","Slovakia":"Europe",
    "Slovenia":"Europe","Spain":"Europe","Sweden":"Europe","Switzerland":"Europe","Ukraine":"Europe",
    "United Kingdom":"Europe","UK":"Europe",
    # Oceania
    "Australia":"Oceania","New Zealand":"Oceania","Fiji":"Oceania","Kiribati":"Oceania","Marshall Islands":"Oceania",
    "Micronesia (Federated States of)":"Oceania","Nauru":"Oceania","Palau":"Oceania","Papua New Guinea":"Oceania",
    "Samoa":"Oceania","Solomon Islands":"Oceania","Tonga":"Oceania","Tuvalu":"Oceania","Vanuatu":"Oceania",
}

def region_from_sources(df: pd.DataFrame) -> pd.Series:
    # 1) Use a region/continent column if present in the CSV.
    region_cols = [c for c in df.columns if str(c).strip().lower() in {
        "region","continent","un region","un_region","who region","who_region","worldbank region","worldbank_region"
    }]
    if region_cols:
        s = df[region_cols[0]].astype(str).str.strip()
        s = s.replace({"nan": np.nan})
        if s.notna().any():
            return s.fillna("Other/Unknown")
    # 2) Otherwise map via internal dictionary.
    return df[COL["ENTITY"]].map(lambda x: BASE_REGION_MAP.get(str(x), "Other/Unknown"))

# =============== LOADING HELPERS ===============
SCRIPT_DIR = Path(__file__).parent
PRIMARY_NAME = "global-data-on-sustainable-energy (1).csv"

def find_local_csv() -> Path | None:
    candidates = [SCRIPT_DIR / PRIMARY_NAME, SCRIPT_DIR / "data" / PRIMARY_NAME]
    for c in candidates:
        if c.exists():
            return c
    for pat in [
        str(SCRIPT_DIR / "global*on*sustainable*energy*(1).csv"),
        str(SCRIPT_DIR / "data" / "global*on*sustainable*energy*(1).csv"),
    ]:
        hits = glob.glob(pat)
        if hits:
            return Path(hits[0])
    return None

@st.cache_data
def load_default_or_fail() -> pd.DataFrame:
    p = find_local_csv()
    if not p:
        return pd.DataFrame()
    return pd.read_csv(p)

def validate_columns(df: pd.DataFrame) -> list[str]:
    missing = [c for c in REQUIRED_COLS_BASE if c not in df.columns]
    if all([c not in df.columns for c in CO2_ACCEPT_ANY]):
        missing.append("CO₂ column (need one of: "
                       f"'{COL['CO2_KT']}' OR '{COL['CO2_PC']}')")
    return missing

# Aliases for robustness
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

# =============== DATA QA & SANITY LAYER ===============
PCT_COLS = [
    COL["ACCESS_ELECTRICITY"], COL["CLEAN_FUELS"],
    COL["RENEW_SHARE_TFEC"], COL["LOW_CARBON_ELEC_PCT"],
]
NONNEG_COLS = [
    COL["ELEC_FOSSIL_TWH"], COL["ELEC_NUCLEAR_TWH"],
    COL["ELEC_RENEW_TWH"], COL["RENEW_CAP_PER_CAP"], COL["FLOWS_USD"],
]
NUMERIC_CANDIDATES = list({
    *PCT_COLS, *NONNEG_COLS, COL["GDP_PER_CAPITA"], COL["ENERGY_INTENSITY"], COL["CO2_KT"], COL["CO2_PC"]
})

def _to_numeric(df: pd.DataFrame, cols: list[str]) -> pd.DataFrame:
    df = df.copy()
    for c in cols:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")
    return df

def _dedupe_entity_year(df: pd.DataFrame) -> pd.DataFrame:
    if COL["ENTITY"] not in df.columns or COL["YEAR"] not in df.columns:
        return df
    df = df.copy()
    score_cols = [c for c in [*PCT_COLS, COL["GDP_PER_CAPITA"], COL["ENERGY_INTENSITY"], COL["CO2_KT"], COL["CO2_PC"]] if c in df.columns]
    df["_nn"] = df[score_cols].notna().sum(axis=1)
    rs = COL["RENEW_SHARE_TFEC"]
    df["_rs"] = df[rs].fillna(-1) if rs in df.columns else -1
    df = (
        df.sort_values([COL["ENTITY"], COL["YEAR"], "_nn", "_rs"], ascending=[True, True, False, False])
          .drop_duplicates(subset=[COL["ENTITY"], COL["YEAR"]], keep="first")
          .drop(columns=["_nn", "_rs"])
    )
    return df

def clean_dataframe(raw: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    df = raw.copy()
    df = _to_numeric(df, NUMERIC_CANDIDATES)
    pct_clip_counts, nonneg_fix_counts = {}, {}

    for c in PCT_COLS:
        if c in df.columns:
            before = df[c].copy()
            df[c] = df[c].clip(0, 100)
            pct_clip_counts[c] = int((before != df[c]).fillna(False).sum())
    for c in NONNEG_COLS:
        if c in df.columns:
            before = df[c].copy()
            df[c] = df[c].clip(lower=0)
            nonneg_fix_counts[c] = int((before != df[c]).fillna(False).sum())

    if COL["GDP_PER_CAPITA"] in df.columns:
        df.loc[df[COL["GDP_PER_CAPITA"]] < 0, COL["GDP_PER_CAPITA"]] = np.nan
    if COL["ENERGY_INTENSITY"] in df.columns:
        df.loc[df[COL["ENERGY_INTENSITY"]] < 0, COL["ENERGY_INTENSITY"]] = np.nan

    pre = len(df)
    df = _dedupe_entity_year(df)
    dedup_removed = pre - len(df)
    qa = {"pct_clipped": pct_clip_counts, "nonneg_fixed": nonneg_fix_counts, "dedup_removed": dedup_removed}
    return df, qa

# =============== SIDEBAR (DATA + FILTERS) ===============
st.sidebar.title("Data & Filters")
uploaded = st.sidebar.file_uploader("Upload CSV", type=["csv"], key="data_upl")

if uploaded is not None:
    try:
        df = pd.read_csv(uploaded)
        st.sidebar.success("Using uploaded file.")
    except Exception as e:
        st.sidebar.error(f"Error reading uploaded CSV: {e}")
        st.stop()
else:
    df = load_default_or_fail()
    if df.empty:
        st.sidebar.warning(
            "No uploaded file and no local CSV found.\n\n"
            f"Put **{PRIMARY_NAME}** next to this script (or in ./data/) or upload it here."
        )
        st.stop()
    else:
        st.sidebar.success("Loaded local CSV file.")

df = ensure_alias_columns(df)

if COL["YEAR"] in df.columns:
    df = df[pd.to_numeric(df[COL["YEAR"]], errors="coerce").notna()].copy()
    df[COL["YEAR"]] = df[COL["YEAR"]].astype(int)

missing = validate_columns(df)
if missing:
    st.error("Missing required columns: " + ", ".join(missing))
    st.stop()

df, QA = clean_dataframe(df)

# Build Region column (existing column if present, else internal map)
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
    if selected_region != "All":
        d = d[d["Region(_auto_)"] == selected_region]
    if selected_countries:
        d = d[d[COL["ENTITY"]].isin(selected_countries)]
    return d

df_f = _apply_filters(df)

# =============== TITLE & QA ===============
st.title("Global Sustainable Energy Dashboard")
st.caption("ODS 7: Affordable and Clean Energy Analytics — regions auto-organized from your data or an internal map.")

issues = (sum(QA["pct_clipped"].values()) if isinstance(QA.get("pct_clipped"), dict) else 0) \
         + (sum(QA["nonneg_fixed"].values()) if isinstance(QA.get("nonneg_fixed"), dict) else 0) \
         + QA.get("dedup_removed", 0)
if issues > 0:
    with st.expander("Data QA adjustments applied (click for details)"):
        st.write(f"- (Entity, Year) duplicates removed: **{QA['dedup_removed']}**")
        fixed = {k: v for k, v in QA["pct_clipped"].items() if v > 0}
        if fixed:
            st.write("Percentages clipped to [0,100]:")
            st.json(fixed)
        fixed = {k: v for k, v in QA["nonneg_fixed"].items() if v > 0}
        if fixed:
            st.write("Non-negative fixes (set <0 → 0):")
            st.json(fixed)

# =============== KPI SECTION ===============
@st.cache_data
def compute_kpis(df_all: pd.DataFrame, year: int, agg: str):
    d = df_all[df_all[COL["YEAR"]] == year].copy()
    if d.empty:
        return dict(electricity=0, clean=0, renewshare=0, co2_mt=None, co2pc=None)
    agg_func = np.nanmean if agg == "Mean" else np.nanmedian
    electricity = float(agg_func(pd.to_numeric(d[COL["ACCESS_ELECTRICITY"]], errors="coerce")))
    clean = float(agg_func(pd.to_numeric(d[COL["CLEAN_FUELS"]], errors="coerce")))
    renewshare = float(agg_func(pd.to_numeric(d[COL["RENEW_SHARE_TFEC"]], errors="coerce")))
    co2_mt, co2pc = None, None
    if COL["CO2_KT"] in d.columns and d[COL["CO2_KT"]].notna().any():
        co2_mt = pd.to_numeric(d[COL["CO2_KT"]], errors="coerce").sum() / 1000.0
    elif COL["CO2_PC"] in d.columns and d[COL["CO2_PC"]].notna().any():
        co2pc = float(agg_func(pd.to_numeric(d[COL["CO2_PC"]], errors="coerce")))
    return dict(electricity=electricity, clean=clean, renewshare=renewshare, co2_mt=co2_mt, co2pc=co2pc)

def progress_bar(pct: float):
    pct = max(0.0, min(100.0, pct if pd.notnull(pct) else 0.0))
    st.progress(pct / 100.0, text=f"{pct:.1f}%")

kpi_vals = compute_kpis(df_f, sel_year, kpi_agg)
st.subheader("Key Indicators")
c1, c2, c3, c4 = st.columns(4)
with c1:
    st.markdown('<div class="kpi-card"><div class="kpi-title">🔌 Access to Electricity</div></div>', unsafe_allow_html=True)
    st.metric("", f"{kpi_vals['electricity']:.1f}%")
    progress_bar(kpi_vals['electricity'])
with c2:
    st.markdown('<div class="kpi-card"><div class="kpi-title">🍳 Clean Cooking Fuels</div></div>', unsafe_allow_html=True)
    st.metric("", f"{kpi_vals['clean']:.1f}%")
    progress_bar(kpi_vals['clean'])
with c3:
    st.markdown('<div class="kpi-card"><div class="kpi-title">☀️ Renewable Energy Share</div></div>', unsafe_allow_html=True)
    st.metric("", f"{kpi_vals['renewshare']:.1f}%")
    progress_bar(kpi_vals['renewshare'])
with c4:
    st.markdown('<div class="kpi-card"><div class="kpi-title">🌍 CO₂</div><div class="kpi-sub">Sum (Mt) if total is available; otherwise average per-capita</div></div>', unsafe_allow_html=True)
    if kpi_vals["co2_mt"] is not None:
        st.metric("", f"{kpi_vals['co2_mt']:.1f} Mt")
    elif kpi_vals["co2pc"] is not None:
        st.metric("", f"{kpi_vals['co2pc']:.2f} t/person")
    else:
        st.metric("", "—")
st.markdown(f'<div class="hint">Percent KPIs use <b>{kpi_agg.lower()}</b> across the current selection.</div>', unsafe_allow_html=True)

# =============== TABS ===============
tab1, tab2, tab3, tab4, tab5 = st.tabs(["📈 Trends", "🏆 Leaders & Flows", "📊 Analysis", "🔮 Forecast", "📄 Data"])

# ---------- Tab 1: Trends ----------
with tab1:
    st.subheader("Energy Source Evolution (TWh)")
    d = df_f.groupby(COL["YEAR"], as_index=False).agg(
        fossil=(COL["ELEC_FOSSIL_TWH"], "sum"),
        nuclear=(COL["ELEC_NUCLEAR_TWH"], "sum"),
        renew=(COL["ELEC_RENEW_TWH"], "sum"),
    ).sort_values(COL["YEAR"])
    if d.empty:
        st.info("No data available for energy transition.")
    else:
        fig, ax = plt.subplots(figsize=(10, 5))
        x = d[COL["YEAR"]].values
        y = np.vstack([d["renew"].values, d["nuclear"].values, d["fossil"].values])
        ax.stackplot(x, y, labels=["Renewables", "Nuclear", "Fossil"])
        if compare_mode and comp_year is not None:
            for yv in [sel_year, comp_year]:
                if yv in x:
                    ax.axvline(x=yv, linestyle="--", alpha=0.5)
        ax.set_xlabel("Year"); ax.set_ylabel("TWh"); ax.legend(loc="upper left")
        st.pyplot(fig)

# ---------- Tab 2: Leaders & Flows ----------
with tab2:
    left, right = st.columns((1, 1), gap="large")

    with left:
        st.subheader(f"Top Renewable Leaders — {sel_year} (cleaned)")
        sub = df_f[df_f[COL["YEAR"]] == sel_year][[COL["ENTITY"], COL["RENEW_SHARE_TFEC"], COL["LOW_CARBON_ELEC_PCT"]]].copy()
        sub[COL["RENEW_SHARE_TFEC"]] = pd.to_numeric(sub[COL["RENEW_SHARE_TFEC"]], errors="coerce").clip(0, 100)
        if COL["LOW_CARBON_ELEC_PCT"] in sub.columns:
            sub[COL["LOW_CARBON_ELEC_PCT"]] = pd.to_numeric(sub[COL["LOW_CARBON_ELEC_PCT"]], errors="coerce").clip(0, 100)
        sub = sub.dropna(subset=[COL["RENEW_SHARE_TFEC"]])
        if sub.empty:
            st.info("No renewable-share observations for this slice.")
        else:
            d_rank = sub.sort_values(COL["RENEW_SHARE_TFEC"], ascending=False).head(top_n)
            fig, ax = plt.subplots(figsize=(7, 6))
            ax.barh(d_rank[COL["ENTITY"]][::-1], d_rank[COL["RENEW_SHARE_TFEC"]][::-1])
            ax.set_xlabel("Renewable share (%)")
            st.pyplot(fig)

            table = d_rank.rename(columns={
                COL["ENTITY"]: "Country",
                COL["RENEW_SHARE_TFEC"]: "Renewable Share (%)",
                COL["LOW_CARBON_ELEC_PCT"]: "Low-carbon (% electricity)"
            }).reset_index(drop=True)
            st.dataframe(table, use_container_width=True)
            st.download_button("Download leader table",
                               data=table.to_csv(index=False),
                               file_name=f"top_renewables_{sel_year}.csv",
                               mime="text/csv")

    with right:
        st.subheader(f"Financial Flows — {sel_year}")
        flows = df_f[df_f[COL["YEAR"]] == sel_year][[COL["ENTITY"], COL["FLOWS_USD"]]].copy()
        flows[COL["FLOWS_USD"]] = pd.to_numeric(flows[COL["FLOWS_USD"]], errors="coerce")
        flows = flows.dropna()
        if flows.empty:
            st.info("No financial flow data for selected year.")
        else:
            flows["Flow_MUSD"] = flows[COL["FLOWS_USD"]] / 1e6
            flows = flows.sort_values("Flow_MUSD", ascending=False).head(top_n)
            fig, ax = plt.subplots(figsize=(7, 5))
            ax.barh(flows[COL["ENTITY"]][::-1], flows["Flow_MUSD"][::-1])
            ax.set_xlabel("Million USD")
            st.pyplot(fig)
            st.download_button("Download flows CSV",
                               data=flows.to_csv(index=False),
                               file_name=f"financial_flows_{sel_year}.csv",
                               mime="text/csv")

# ---------- Tab 3: Analysis ----------
with tab3:
    left, right = st.columns((1, 1), gap="large")

    with left:
        st.subheader(f"CO₂ Emissions vs Energy Intensity — {sel_year}")
        if COL["CO2_KT"] in df_f.columns and df_f[COL["CO2_KT"]].notna().any():
            co2_col, y_label, scaler = COL["CO2_KT"], "CO₂ (Mt)", 1000.0
        elif COL["CO2_PC"] in df_f.columns and df_f[COL["CO2_PC"]].notna().any():
            co2_col, y_label, scaler = COL["CO2_PC"], "CO₂ (t per capita)", None
        else:
            co2_col, y_label, scaler = None, "", None

        d = df_f[df_f[COL["YEAR"]] == sel_year]
        if co2_col is None or d.empty:
            st.info("No data for scatter.")
        else:
            d = d[[COL["ENTITY"], COL["ENERGY_INTENSITY"], co2_col]].replace([np.inf, -np.inf], np.nan).dropna()
            if d.empty:
                st.info("No data for scatter.")
            else:
                if scaler:
                    d["CO2_Y"] = pd.to_numeric(d[co2_col], errors="coerce") / scaler
                else:
                    d["CO2_Y"] = pd.to_numeric(d[co2_col], errors="coerce")
                fig, ax = plt.subplots(figsize=(7.5, 5.2))
                ax.scatter(pd.to_numeric(d[COL["ENERGY_INTENSITY"]], errors="coerce"), d["CO2_Y"])
                ax.set_xlabel("Energy Intensity (MJ/$2017 PPP GDP)")
                ax.set_ylabel(y_label)
                for _, r in d.iterrows():
                    ax.annotate(
                        str(r[COL["ENTITY"]]),
                        (float(r[COL["ENERGY_INTENSITY"]]), float(r["CO2_Y"])),
                        fontsize=7, alpha=0.6
                    )
                st.pyplot(fig)

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

        d = df_f[df_f[COL["YEAR"]] == sel_year].copy()
        pairs_show = []
        keys = list(metrics.keys())
        for i in range(len(keys)):
            for j in range(i + 1, len(keys)):
                a = pd.to_numeric(d[metrics[keys[i]]], errors="coerce")
                b = pd.to_numeric(d[metrics[keys[j]]], errors="coerce")
                both = pd.concat([a, b], axis=1).dropna()
                if len(both) >= 3:
                    r = float(both.corr().iloc[0, 1])
                    pairs_show.append((keys[i], keys[j], r))
        if not pairs_show:
            st.info("Not enough data for correlation pairs.")
        else:
            for m1, m2, r in sorted(pairs_show, key=lambda x: -abs(x[2])):
                bar = "█" * int(20 * abs(r))
                st.write(f"**{m1} ↔ {m2}** — r = {r:+.3f}  |{bar}")

    st.markdown("---")
    st.subheader("AI Country Clustering (K-means)")
    feats = df_f[df_f[COL["YEAR"]] == sel_year][[
        COL["ACCESS_ELECTRICITY"], COL["RENEW_SHARE_TFEC"], COL["GDP_PER_CAPITA"], COL["LOW_CARBON_ELEC_PCT"]
    ]].dropna()
    if len(feats) < 6:
        st.info("Not enough data to cluster countries.")
    else:
        k = min(4, max(2, len(feats) // 3))
        km = KMeans(n_clusters=k, n_init="auto", random_state=42)
        labels = km.fit_predict(feats.values)
        d2 = df_f[df_f[COL["YEAR"]] == sel_year].loc[feats.index, [COL["ENTITY"]]].copy()
        d2["cluster"] = labels
        for cid in sorted(d2["cluster"].unique()):
            names = d2[d2["cluster"] == cid][COL["ENTITY"]].tolist()
            st.write(f"- **Cluster {cid+1}** ({len(names)}): {', '.join(names[:20])}{'...' if len(names) > 20 else ''}")

# ---------- Tab 4: Forecast ----------
with tab4:
    st.subheader("Renewable Energy Forecast to 2030")
    d = df_f.groupby(COL["YEAR"], as_index=False)[COL["RENEW_SHARE_TFEC"]].mean().dropna().sort_values(COL["YEAR"])
    if len(d) < 2:
        st.info("Not enough data to forecast.")
    else:
        X = d[[COL["YEAR"]]].values
        y = d[COL["RENEW_SHARE_TFEC"]].values
        lr = LinearRegression().fit(X, y)
        last_year = int(d[COL["YEAR"]].max())
        future_years = np.arange(last_year + 1, 2030 + 1).reshape(-1, 1)
        y_pred_hist = lr.predict(X)
        y_pred_future = lr.predict(future_years) if len(future_years) > 0 else np.array([])
        r2 = lr.score(X, y)

        fig, ax = plt.subplots(figsize=(9, 4.8))
        ax.plot(d[COL["YEAR"]], y, marker="o", label="Actual")
        ax.plot(d[COL["YEAR"]], y_pred_hist, linestyle="--", label="Predicted (fit)")
        if len(y_pred_future) > 0:
            ax.plot(future_years.flatten(), y_pred_future, linestyle=":", label="Forecast")
        ax.set_xlabel("Year"); ax.set_ylabel("Renewable share (%)"); ax.legend()
        st.pyplot(fig)
        if len(y_pred_future) > 0:
            st.caption(f"R² = {r2:.3f} — 2030 projection: {float(y_pred_future[-1]):.1f}%")

# ---------- Tab 5: Data ----------
with tab5:
    st.subheader("Filtered Data")
    year_min = sel_year if not compare_mode else min(sel_year, comp_year or sel_year)
    year_max = sel_year if not compare_mode else max(sel_year, comp_year or sel_year)
    show_df = df_f[df_f[COL["YEAR"]].between(year_min, year_max)]
    st.dataframe(show_df, use_container_width=True, height=440)
    st.download_button("Download filtered CSV",
                       data=show_df.to_csv(index=False),
                       file_name="filtered_energy_data.csv",
                       mime="text/csv")

st.markdown("---")
st.caption("Regions are auto-assigned from your CSV (if present) or an internal map — no extra uploads needed.")