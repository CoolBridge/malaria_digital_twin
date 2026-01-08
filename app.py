import streamlit as st
import pandas as pd
import smtplib
from email.message import EmailMessage
import os
import sklearn
import joblib
from PIL import Image
import streamlit as st

# ----------------------------------
# App Entry State
# ----------------------------------
if "entered_app" not in st.session_state:
    st.session_state.entered_app = False
# ----------------------------------
# Welcome / Landing Page
# ----------------------------------
if not st.session_state.entered_app:
    st.set_page_config(page_title="Malaria Digital Twin", layout="centered")


    st.markdown("## 🦟 Malaria Digital Twin for Nigeria")
    st.markdown(
        """
        ### Evidence-Driven Intelligence for Malaria Control

        This platform integrates **34+ years of environmental data**,  
        **malaria rapid diagnostic test (RDT) outcomes**, and  
        **machine-learning risk models**  
        to support **smarter, data-aligned malaria interventions**.

        **What you can explore inside:**
        - 🗺️ **Geospatial risk maps** at cluster & LGA level  
        - 🧠 **Risk–RDT mismatch detection** (surveillance gaps & emerging hotspots)  
        - 📊 **Temporal trends & stability rankings** across years  
        - 🏘️ **LGA priority summaries** for decision support  
        - ⚙️ **Model-informed insights** aligned with real-world field data  

        This tool is designed for:
        **public health programs, researchers, NGOs, donors, and policy teams**
        seeking **transparent, explainable malaria risk intelligence**.
        
        
        
        ---
        ### 👤 About the Researcher

        **Daniel Onimisi**  
        *Independent Health Data Scientist & Malaria Modeling Researcher*  

        **Specialties**
        - Malaria risk modeling & digital twins  
        - Climate–health analytics (NDVI, CHIRPS, ERA5)  
        - Spatial epidemiology & decision intelligence  
        - Machine learning for public health systems  

        📧 **Contact:** `da.zx@outlook.com`  
       

        ---
        
        """
    )

    st.markdown("---")

    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        if st.button("🚀 Enter Dashboard", use_container_width=True):
            st.session_state.entered_app = True
            st.rerun()

    st.stop()

st.markdown(
    """
    <style>
    .nav-alert {
        background-color: #8b0000;
        color: white;
        padding: 12px 16px;
        border-radius: 8px;
        font-size: 15px;
        font-weight: 600;
        margin-bottom: 15px;
        text-align: center;
    }
    .nav-alert span {
        font-size: 18px;
        margin-right: 6px;
    }
    </style>

    <div class="nav-alert">
        <span>☰</span> Tap the menu (top-left) to explore maps, rankings, and policy insights
    </div>
    """,
    unsafe_allow_html=True
)

def render_scrollable_table(df, height=420):
    """
    Renders a styled pandas DataFrame as a scrollable HTML table
    with NO download button.
    """
    html = df.to_html(index=False)
    scrollable_html = f"""
    <div style="
        max-height: {height}px;
        overflow-y: auto;
        border: 1px solid #444;
        border-radius: 6px;
        padding: 6px;
    ">
        {html}
    </div>
    """
    st.markdown(scrollable_html, unsafe_allow_html=True)

# ===============================
# 📧 EMAIL CONFIG (GMAIL)
# ===============================



from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText

def send_data_access_email(name, email, affiliation, purpose, message):
    sender_email = st.secrets["DATA_ACCESS_SENDER_EMAIL"]
    sender_password = st.secrets["DATA_ACCESS_EMAIL_PASSWORD"]
    recipient_email = st.secrets["DATA_ACCESS_RECEIVER_EMAIL"]

    subject = "🦟 New Malaria Data Access Request"

    body = f"""
New Data Access Request Received

Name: {name}
Email: {email}
Affiliation: {affiliation}

Purpose:
{purpose}

Message:
{message}
"""

    msg = MIMEMultipart()
    msg["From"] = sender_email
    msg["To"] = recipient_email
    msg["Subject"] = subject
    msg.attach(MIMEText(body, "plain"))

    with smtplib.SMTP_SSL("smtp.gmail.com", 465) as server:
        server.login(sender_email, sender_password)
        server.send_message(msg)



st.set_page_config(
    page_title="Nigeria Malaria Digital Twin",
    layout="wide"
)
# ==========================================
# 📂 Load & Prepare Data (GLOBAL)
# ==========================================

@st.cache_data
def load_data():
    df = pd.read_csv("data/mis_clusters_with_predictions.csv")

    # Standardize column names
    df.columns = df.columns.str.strip()

    # ---- Rename columns to analysis-standard names ----
    df = df.rename(columns={
        "rdt_result": "rdt_prevalence",
        "LATNUM": "latitude",
        "LONGNUM": "longitude"
    })

    # ---- Convert numeric fields safely ----
    df["rdt_prevalence"] = pd.to_numeric(df["rdt_prevalence"], errors="coerce")
    df["pred_proba"] = pd.to_numeric(df["pred_proba"], errors="coerce")
    df["year"] = pd.to_numeric(df["year"], errors="coerce")

    return df


# 🔁 Load once, used everywhere
data = load_data()

# ==========================================
# 🧠 GLOBAL CLUSTER-LEVEL MISMATCH BUILDER
# ==========================================
def build_cluster_mismatch(df, risk_q50, risk_q75, rdt_q50, rdt_q75):

    def classify(row):
        risk = row["pred_proba"]
        rdt = row["rdt_prevalence"]

        if pd.isna(risk) or pd.isna(rdt):
            return "⚪ Insufficient data"

        if rdt >= rdt_q75 and risk < risk_q50:
            return "🟥 High RDT – Low Model Risk (Surveillance gap)"

        elif rdt < rdt_q50 and risk >= risk_q75:
            return "🟧 Low RDT – High Model Risk (Emerging hotspot)"

        elif rdt >= rdt_q75 and risk >= risk_q75:
            return "🟩 High RDT – High Risk (True hotspot)"

        else:
            return "🟦 Low RDT – Low Risk"

    df = df.copy()
    df["mismatch_flag"] = df.apply(classify, axis=1)
    return df

# ===============================
# 🧠 GLOBAL SAFE DATA VIEWS
# ===============================

# Default year (used when maps not active)
default_year = int(data["year"].dropna().max())

# Always available filtered dataset
map_data = data[data["year"] == default_year].copy()

map_data["rdt_prevalence"] = pd.to_numeric(
    map_data["rdt_prevalence"], errors="coerce"
)
# ==========================================
# 🏘️ GLOBAL LGA SUMMARY BUILDER  ✅ PASTE HERE
# ==========================================
def build_lga_summary(df):
    lga = (
        df
        .groupby(["State", "LGA"], as_index=False)
        .agg(
            mean_risk=("pred_proba", "mean"),
            clusters=("cluster_id", "nunique"),
            mean_rdt_prevalence=("rdt_prevalence", "mean")
        )
    )

    lga = lga[lga["clusters"] >= 5]

    def classify_mismatch(row):
        if row["mean_rdt_prevalence"] >= 0.30 and row["mean_risk"] >= 0.50:
            return "🟩 High–High (Immediate Intervention)"
        elif row["mean_rdt_prevalence"] >= 0.30 and row["mean_risk"] < 0.50:
            return "🟧 High RDT – Low Risk (Underestimated)"
        elif row["mean_rdt_prevalence"] < 0.30 and row["mean_risk"] >= 0.50:
            return "🟦 Low RDT – High Risk (Emerging / Surveillance)"
        else:
            return "⬜ Low–Low (Routine Monitoring)"

    lga["risk_rdt_flag"] = lga.apply(classify_mismatch, axis=1)

    return lga.sort_values("mean_risk", ascending=False)

st.sidebar.title("📌 Navigation")

section = st.sidebar.radio(
    "Go to:",
    [
        "🏠 Overview",
        "🗺️ Risk & Burden Maps",
        "🧪 Confirmed RDT Burden",
        "🧠 Risk–RDT Mismatch",
        "📊 Trends & Rankings",
        "🔮 Predictions (Coming Soon)",
        "🔐 Data Access Request",
        "ℹ️ Methodology"
    ]
)



if section == "🏠 Overview":

    st.title("🦟 Nigeria Malaria Risk Digital Twin")
    st.caption("Public Health Decision Intelligence Platform | Daniel Onimisi  | da.zx@outlook.com")

    st.markdown("""
    This platform provides **decision-ready malaria risk intelligence** for Nigeria,
    combining **machine-learning models** with nationally representative
    **DHS/MIS survey data** and **long-term environmental signals**.

    The system integrates:
    - **Household and biomarker data** from DHS/MIS surveys  
    - **34 years of rainfall (CHIRPS)** and **vegetation (NDVI)** trends  
    - **Spatial and temporal modeling** to estimate malaria risk at cluster, LGA, and state levels

    Outputs are designed to support:
    - **Program planning and targeting**
    - **Intervention prioritization**
    - **Burden–risk reconciliation (observed vs modeled patterns)**
    - **Evidence-based policy and funding decisions**

    All results are **aggregated, non-identifiable**, and intended for
    **responsible analytical and policy use**.
    """)
    st.info(
        "ℹ️ This platform is **not a surveillance system** and does **not replace routine reporting**. "
        "It is designed to **complement existing malaria control and elimination efforts** "
        "by supporting strategic planning, prioritization, and resource allocation."
    )


    # ------------------
    # KPI STRIP
    # ------------------
    k1, k2, k3 = st.columns(3)
    k1.metric("Survey Clusters", data["cluster_id"].nunique())
    k2.metric("Years Covered", f"{int(data.year.min())} – {int(data.year.max())}")
    k3.metric("Model", "Random Forest")

    # ------------------
    # TABS
    # ------------------
    tab1, tab2, tab3 = st.tabs([
        "📈 Model Explainability",
        "🗺️ National Risk Pattern",
        "ℹ️ How to Use This Tool"
    ])

    with tab1:
        c1, c2 = st.columns(2)
        with c1:
            st.image("figures/shap_summary.png", use_container_width=True)
        with c2:
            st.image("figures/feature_importance.png", use_container_width=True)

    with tab2:
        st.image("figures/nigeria_malaria_risk_v2.png", use_container_width=True)

    with tab3:
        st.markdown("""
        **Who this is for**
        - National Malaria Programs
        - Donors & partners
        - Researchers & policy analysts

        **What this is NOT**
        - Case reporting system
        - Diagnostic replacement
        - Real-time surveillance
        """)

import folium
from streamlit_folium import st_folium

if section == "🗺️ Risk & Burden Maps":
        st.title("🗺️ Malaria Risk & Burden Maps")

        selected_year = st.selectbox(
            "Select survey year",
            sorted(data["year"].unique())
        )

        map_data = data[data["year"] == selected_year].copy()

        map_data["rdt_prevalence"] = pd.to_numeric(
            map_data["rdt_prevalence"], errors="coerce"
        )
        # ===============================
        # Dynamic year-specific thresholds
        # ===============================
        risk_q25 = map_data["pred_proba"].quantile(0.25)
        risk_q50 = map_data["pred_proba"].quantile(0.50)
        risk_q75 = map_data["pred_proba"].quantile(0.75)

        rdt_q50 = map_data["rdt_prevalence"].quantile(0.50)
        rdt_q75 = map_data["rdt_prevalence"].quantile(0.75)

        tab1, tab2, tab3 = st.tabs([
            "🧠 Model Risk(Cluster Level)",
            "🧪 Observed RDT Burden",
            "⚠️ Risk–RDT Mismatch"
        ])

        with tab1:

            st.markdown(
                f"""
                **Interpretation**  
                Each point represents a **survey cluster** for **{selected_year}**.
                Color reflects the **model-estimated probability** that an individual in the cluster
                tests malaria-positive, conditioned on climate, environment, and intervention coverage.
                """
            )


            # Center map on Nigeria
            m = folium.Map(
                location=[9.0, 8.0],
                zoom_start=6,
                tiles="cartodbpositron"
            )


            # Color scale based on probability
            def risk_color(prob):
                if prob < 0.1:
                    return "green"
                elif prob < 0.25:
                    return "lightgreen"
                elif prob < 0.4:
                    return "orange"
                elif prob < 0.6:
                    return "red"
                else:
                    return "darkred"


            # Add points
            for _, row in map_data.iterrows():
                if pd.notnull(row["latitude"]) and pd.notnull(row["longitude"]):
                    folium.CircleMarker(
                        location=[row["latitude"], row["longitude"]],
                        radius=4,
                        color=risk_color(row["pred_proba"]),
                        fill=True,
                        fill_opacity=0.7,
                        popup=f"""
                        <b>State:</b> {row['State']}<br>
                        <b>LGA:</b> {row['LGA']}<br>
                        <b>Risk Class:</b> {int(row['pred_class'])}<br>
                        <b>Probability:</b> {row['pred_proba']:.2f}
                        """
                    ).add_to(m)

            legend_html = """
            <div style="
            position: fixed;
            bottom: 50px;
            left: 50px;
            width: 220px;
            height: 200px;
            background-color: black;
            border:2px solid grey;
            z-index:9999;
            font-size:14px;
            padding: 10px;
            ">
            <b>Malaria Risk Probability</b><br>
            <i style="background:green;width:10px;height:10px;display:inline-block;"></i> Very Low (&lt; 0.10)<br>
            <i style="background:lightgreen;width:10px;height:10px;display:inline-block;"></i> Low (0.10–0.2.5)<br>
            <i style="background:orange;width:10px;height:10px;display:inline-block;"></i> Moderate (0.25–0.40)<br>
            <i style="background:red;width:10px;height:10px;display:inline-block;"></i> High (0.40–0.60)<br>
            <i style="background:darkred;width:10px;height:10px;display:inline-block;"></i> Very High (&gt; 0.60)</div>
            """
            m.get_root().html.add_child(folium.Element(legend_html))

            st.write("Rendering map for", selected_year)

            st_folium(
                m,
                width=1200,
                height=600,
                key="risk_map_" + str(selected_year),
                returned_objects=[]
            )

        with tab2:

            st.header("🧪 Observed Malaria Positivity (RDT-Based Map)")
            rdt_map_data = (
                map_data
                .groupby(
                    ["cluster_id", "latitude", "longitude"],
                    as_index=False
                )
                .agg(
                    mean_rdt=("rdt_prevalence", "mean"),
                    mean_risk=("pred_proba", "mean"),
                    State=("State", "first"),
                    LGA=("LGA", "first")
                )
            )
            m_rdt = folium.Map(
                location=[9.0, 8.0],
                zoom_start=6,
                tiles="cartodbpositron"
            )
            for _, row in rdt_map_data.iterrows():
                if pd.notnull(row["latitude"]) and pd.notnull(row["longitude"]):
                    folium.CircleMarker(
                        location=[row["latitude"], row["longitude"]],
                        radius=5,
                        color="red" if row["mean_rdt"] >= 0.4 else "orange",
                        fill=True,
                        fill_opacity=0.7,
                        popup=f"""
                        <b>State:</b> {row['State']}<br>
                        <b>LGA:</b> {row['LGA']}<br>
                        <b>RDT Positivity:</b> {row['mean_rdt']:.2f}<br>
                        <b>Model Risk:</b> {row['mean_risk']:.2f}
                        """
                    ).add_to(m_rdt)

            st.write("Rendering map for", selected_year)
            legend_rdt = f"""
               <div style="
               position: fixed;
               bottom: 50px;
               left: 50px;
               width: 240px;
               background-color: black;
               border: 2px solid grey;
               z-index: 9999;
               font-size: 13px;
               padding: 10px;
               ">
               <b>Observed RDT Positivity ({selected_year})</b><br>
               🔴 High (≥ {rdt_q75:.2f})<br>
               🟠 Moderate ({rdt_q50:.2f}–{rdt_q75:.2f})<br>
               🟢 Low (&lt; {rdt_q50:.2f})
               </div>
               """

            m_rdt.get_root().html.add_child(folium.Element(legend_rdt))
            st_folium(
                m_rdt,
                width=1200,
                height=600,
                key="rdt_map_" + str(selected_year),
                returned_objects=[]
            )

        # ==========================================
        # 🧠 Risk–RDT Mismatch Map (Cluster Level)
        # ==========================================

        with tab3:

            st.header("🧠 Risk–RDT Mismatch Map")

            st.markdown(
                f"""
                    **Purpose**  
                    Identifies clusters where **observed malaria burden (RDT positivity)**
                    does **not align** with **model-estimated risk** for **{selected_year}**.
                    These represent surveillance gaps, emerging hotspots, or confirmed priority zones.
                    """
            )
            # Define mismatch logic at CLUSTER level
            def mismatch_flag_cluster(row):
                risk = row["pred_proba"]
                rdt = row["rdt_prevalence"]

                if pd.isna(risk) or pd.isna(rdt):
                    return "no_data"

                if rdt >= rdt_q75 and risk < risk_q50:
                    return "high_rdt_low_risk"

                elif rdt < rdt_q50 and risk >= risk_q75:
                    return "low_rdt_high_risk"

                elif rdt >= rdt_q75 and risk >= risk_q75:
                    return "high_high"

                else:
                    return "low_low"


            map_data["mismatch_flag"] = map_data.apply(
                mismatch_flag_cluster,
                axis=1
            )
            mismatch_label = {
                "high_rdt_low_risk": "🟥 High RDT – Low Model Risk (Surveillance gap)",
                "low_rdt_high_risk": "🟧 Low RDT – High Model Risk (Emerging hotspot)",
                "high_high": "🟩 High RDT – High Risk (Confirmed hotspot)",
                "low_low": "🟦 Low RDT – Low Risk",
                "no_data": "⚪ Insufficient data"
            }


            def mismatch_color(flag):
                return {
                    "high_rdt_low_risk": "red",
                    "low_rdt_high_risk": "orange",
                    "high_high": "darkgreen",
                    "low_low": "blue",
                    "no_data": "gray"
                }.get(flag, "gray")


            # Create map
            m_mismatch = folium.Map(
                location=[9.0, 8.0],
                zoom_start=6,
                tiles="cartodbpositron"
            )

            for _, row in map_data.iterrows():
                if pd.notnull(row["latitude"]) and pd.notnull(row["longitude"]):
                    folium.CircleMarker(
                        location=[row["latitude"], row["longitude"]],
                        radius=5,
                        color=mismatch_color(row["mismatch_flag"]),
                        fill=True,
                        fill_opacity=0.7,
                        popup=f"""
                        <b>State:</b> {row['State']}<br>
                        <b>LGA:</b> {row['LGA']}<br>
                        <b>RDT Positivity:</b> {row['rdt_prevalence']:.2f}<br>
                        <b>Model Risk:</b> {row['pred_proba']:.2f}<br>
                        <b>Category:</b> {mismatch_label[row['mismatch_flag']]}
                        """
                    ).add_to(m_mismatch)

            st.write("Rendering map for", selected_year)
            legend_mismatch = f"""
            <div style="
            position: fixed;
            bottom: 50px;
            left: 50px;
            width: 320px;
            background-color: black;
            border: 2px solid grey;
            z-index: 9999;
            font-size: 13px;
            padding: 10px;
            ">
            <b>Risk–RDT Mismatch ({selected_year})</b><br>
            🟥 High RDT ≥ {rdt_q75:.2f} & Low Risk &lt; {risk_q50:.2f}<br>
            🟧 Low RDT &lt; {rdt_q50:.2f} & High Risk ≥ {risk_q75:.2f}<br>
            🟩 High RDT ≥ {rdt_q75:.2f} & High Risk ≥ {risk_q75:.2f}<br>
            🟦 Low RDT &lt; {rdt_q50:.2f} & Low Risk &lt; {risk_q75:.2f}
            </div>
            """

            m_mismatch.get_root().html.add_child(folium.Element(legend_mismatch))

            st_folium(
                    m_mismatch,
                    width=1200,
                    height=600,
                    key=f"mismatch_map_{selected_year}"
                )
            st.markdown("---")
            st.subheader("🏘️ LGA-Level Risk–RDT Priority Summary")

            # ===============================
            # LGA-level aggregation
            # ===============================
            lga_summary = (
                map_data
                .groupby(["State", "LGA"], as_index=False)
                .agg(
                    mean_risk=("pred_proba", "mean"),
                    mean_rdt=("rdt_prevalence", "mean"),
                    clusters=("cluster_id", "nunique")
                )
            )

            # Minimum evidence filter
            st.write("LGA cluster counts:", lga_summary["clusters"].describe())



            # ===============================
            # LGA-level classification
            # ===============================
            def classify_lga(row):
                if row["mean_rdt"] >= rdt_q75 and row["mean_risk"] >= risk_q75:
                    return "🟩 High RDT – High Risk (Immediate Intervention)"
                elif row["mean_rdt"] >= rdt_q75 and row["mean_risk"] < risk_q50:
                    return "🟥 High RDT – Low Risk (Underestimated)"
                elif row["mean_rdt"] < rdt_q50 and row["mean_risk"] >= risk_q75:
                    return "🟧 Low RDT – High Risk (Emerging)"
                else:
                    return "🟦 Low RDT – Low Risk (Routine Monitoring)"


            lga_summary["Priority Category"] = lga_summary.apply(classify_lga, axis=1)

            # Sort by highest risk
            lga_summary = lga_summary.sort_values("mean_risk", ascending=False)

            # ===============================
            # Display table
            # ===============================
            st.markdown(
                lga_summary
                .style
                .format({
                    "mean_risk": "{:.2f}",
                    "mean_rdt": "{:.2f}"
                })
                .to_html(),
                unsafe_allow_html=True
            )





# ==========================================
# 🚦 Risk–RDT Mismatch Summary (Policy View)
# ==========================================
if section == "🧠 Risk–RDT Mismatch":

    st.subheader("🚦 Risk–RDT Mismatch Summary (Policy View)")

    lga_summary = build_lga_summary(map_data)
    mismatch_policy = (
        lga_summary
        .groupby("risk_rdt_flag", as_index=False)
        .agg(
            LGAs=("LGA", "nunique"),
            Avg_RDT=("mean_rdt_prevalence", "mean"),
            Avg_Risk=("mean_risk", "mean"),
            Avg_Clusters=("clusters", "mean")
        )
        .sort_values("LGAs", ascending=False)
    )

    st.caption(
        "Summary of Local Government Areas by malaria burden–risk alignment. "
        "Used for intervention planning, surveillance prioritization, and funding allocation."
    )

    st.markdown(
        mismatch_policy
        .style
        .format({
            "Avg_RDT": "{:.2f}",
            "Avg_Risk": "{:.2f}",
            "Avg_Clusters": "{:.1f}"
        })
        .to_html(),
        unsafe_allow_html=True
    )

# ==========================================
# ⏳ Time-Trend Analysis (Year-over-Year)
# ==========================================
if section == "📊 Trends & Rankings":
    st.header("📊 Summary Statistics")
    # ==========================================
    # 🏛️ State-Level Malaria Risk Ranking
    # ==========================================

    st.header("🏛️ State-Level Malaria Risk Ranking")

    # ------------------------------------------
    # 1️⃣ Build state summary
    # ------------------------------------------
    state_summary = (
        map_data
        .groupby("State", as_index=False)
        .agg(
            mean_risk=("pred_proba", "mean"),
            mean_rdt_prevalence=("rdt_prevalence", "mean"),
            clusters=("cluster_id", "nunique")
        )
    )


    # ------------------------------------------
    # 2️⃣ Risk–RDT policy classification
    # ------------------------------------------
    def classify_state_mismatch(row):
        if row["mean_rdt_prevalence"] >= 0.30 and row["mean_risk"] >= 0.50:
            return "🟩 High–High (Immediate Intervention)"
        elif row["mean_rdt_prevalence"] >= 0.30 and row["mean_risk"] < 0.50:
            return "🟧 High RDT – Low Risk (Underestimated)"
        elif row["mean_rdt_prevalence"] < 0.30 and row["mean_risk"] >= 0.50:
            return "🟦 Low RDT – High Risk (Emerging)"
        else:
            return "⬜ Low–Low (Routine Monitoring)"


    state_summary["risk_rdt_flag"] = state_summary.apply(
        classify_state_mismatch,
        axis=1
    )

    # ------------------------------------------
    # 3️⃣ Ranking mode toggle
    # ------------------------------------------
    ranking_mode = st.radio(
        "Select ranking method",
        [
            "🧭 Policy Priority (Risk–RDT Logic)",
            "📈 Pure Model Risk (Predicted Only)"
        ],
        horizontal=True
    )

    # ------------------------------------------
    # 4️⃣ Apply sorting logic
    # ------------------------------------------
    if ranking_mode == "🧭 Policy Priority (Risk–RDT Logic)":
        ranked_states = state_summary.sort_values(
            ["risk_rdt_flag", "mean_risk"],
            ascending=[True, False]
        )
    else:
        ranked_states = state_summary.sort_values(
            "mean_risk",
            ascending=False
        )


    # ------------------------------------------
    # 5️⃣ Color styling for policy flags
    # ------------------------------------------
    def highlight_risk_flag(val):
        if "High–High" in val:
            return "background-color: #d73027; color: white;"  # Red
        elif "High RDT – Low Risk" in val:
            return "background-color: #fc8d59; color: black;"  # Orange
        elif "Low RDT – High Risk" in val:
            return "background-color: #4575b4; color: white;"  # Blue
        else:
            return "background-color: #e0e0e0; color: black;"  # Grey


    # ------------------------------------------
    # 6️⃣ Display table (policy-grade)
    # ------------------------------------------
    render_scrollable_table(
        ranked_states
        .style
        .format({
            "mean_risk": "{:.2f}",
            "mean_rdt_prevalence": "{:.2f}"
        })
        .applymap(
            highlight_risk_flag,
            subset=["risk_rdt_flag"]
        )
    )

    st.subheader("⚙️ Analysis Scope")

    analysis_scope = st.radio(
        "Select ranking scope:",
        ["Single Year (Map Selection)", "All Years (Stability Ranking)"],
        horizontal=True
    )

    k1, k2, k3 = st.columns(3)

    k1.metric(
        "Mean Predicted Risk",
        f"{map_data['pred_proba'].mean():.2f}"
    )

    k2.metric(
        "High-Risk Clusters (>0.6)",
        int((map_data["pred_proba"] > 0.6).sum())
    )

    k3.metric(
        "Clusters Displayed",
        len(map_data)
    )

    st.header("⏳ Malaria Risk & Burden Trends Over Time")

    trend_summary = (
        data
        .groupby("year", as_index=False)
        .agg(
            Avg_RDT=("rdt_prevalence", "mean"),
            Avg_Model_Risk=("pred_proba", "mean"),
            Clusters=("cluster_id", "nunique")
        )
        .sort_values("year")
    )

    # Compute year-over-year change
    trend_summary["Δ RDT YoY"] = trend_summary["Avg_RDT"].diff()
    trend_summary["Δ Risk YoY"] = trend_summary["Avg_Model_Risk"].diff()

    st.caption(
        "National-level year-over-year trends in observed malaria positivity "
        "and model-estimated risk."
    )

    st.table(
        trend_summary.style.format({
            "Avg_RDT": "{:.2f}",
            "Avg_Model_Risk": "{:.2f}",
            "Δ RDT YoY": "{:+.2f}",
            "Δ Risk YoY": "{:+.2f}"
        })
    )
    st.divider()
    st.header("🏘️ LGA-Level Priority Ranking")
    if analysis_scope == "Single Year (Map Selection)":
        lga_base = map_data.copy()
        st.caption(f"📅 Rankings based on survey year {int(map_data['year'].iloc[0])}")
    else:
        lga_base = data.copy()
        st.caption("📅 Rankings based on all available survey years")

    states = ["All States"] + sorted(lga_base["State"].dropna().unique())

    selected_state = st.selectbox(
    "Filter by State",
    states
    )

    if selected_state != "All States":
        lga_base = lga_base[lga_base["State"] == selected_state]

    lga_summary = (
        lga_base
        .groupby(["State", "LGA"], as_index=False)
        .agg(
            mean_risk=("pred_proba", "mean"),
            clusters=("cluster_id", "nunique"),
            mean_rdt_prevalence=("rdt_prevalence", "mean")
        )
    )




    #lga_summary = lga_summary[lga_summary["clusters"] >= min_clusters]


    # For stricter analysis, use:
    #lga_summary = lga_summary[lga_summary["clusters"] >= 10]
    # ===============================
    # 🧠 Risk–RDT mismatch classification (LGA-level)
    # ===============================

    def classify_mismatch(row):
        if row["mean_rdt_prevalence"] >= 0.30 and row["mean_risk"] >= 0.50:
            return "🟩 High–High (Immediate Intervention)"
        elif row["mean_rdt_prevalence"] >= 0.30 and row["mean_risk"] < 0.50:
            return "🟧 High RDT – Low Risk (Underestimated)"
        elif row["mean_rdt_prevalence"] < 0.30 and row["mean_risk"] >= 0.50:
            return "🟦 Low RDT – High Risk (Emerging / Surveillance)"
        else:
            return "⬜ Low–Low (Routine Monitoring)"

    lga_summary["risk_rdt_flag"] = lga_summary.apply(classify_mismatch, axis=1)

    # Now it is safe to rank

    lga_summary = lga_summary.sort_values("mean_risk", ascending=False)

    st.markdown(
        f"""
        <div style="
            max-height: 360px;
            overflow-y: auto;
            border: 1px solid #444;
            border-radius: 6px;
            padding: 6px;
        ">
            {
        lga_summary
        .style
        .format({
            "mean_risk": "{:.2f}",
            "mean_rdt_prevalence": "{:.2f}"
        })
        .to_html()
        }
        </div>
        """,
        unsafe_allow_html=True
    )

# ===============================
# 🧪 RDT BURDEN — CONFIRMED CASES
# ===============================
if section == "🧪 Confirmed RDT Burden":

    st.header("🧪 LGAs with Highest RDT Positivity (Confirmed Burden)")

    rdt_hotspots = (
        map_data
        .groupby(["State", "LGA"], as_index=False)
        .agg(
            mean_rdt_prevalence=("rdt_prevalence", "mean"),
            clusters=("cluster_id", "nunique"),
            mean_risk=("pred_proba", "mean")
        )
    )

    # 🔹 Table 1: Raw burden (NO cluster filter)
    rdt_hotspots = rdt_hotspots.sort_values(
        "mean_rdt_prevalence", ascending=False
    )
    def confidence_tier(n):
        if n >= 10:
            return "High"
        elif n >= 5:
            return "Medium"
        else:
            return "Low"

    rdt_hotspots["confidence"] = rdt_hotspots["clusters"].apply(confidence_tier)
    # -------------------------------
    # Risk–RDT mismatch classification
    # -------------------------------
    def risk_rdt_flag(risk, rdt):
        if rdt >= 0.3 and risk < 0.3:
            return "🟥 High RDT / Low Model Risk"
        elif rdt < 0.2 and risk >= 0.5:
            return "🟧 Low RDT / High Model Risk"
        elif rdt >= 0.3 and risk >= 0.5:
            return "🟩 High RDT / High Risk (True Hotspot)"
        else:
            return "🟦 Low RDT / Low Risk"

    rdt_hotspots["risk_rdt_flag"] = rdt_hotspots.apply(
        lambda x: risk_rdt_flag(x["mean_risk"], x["mean_rdt_prevalence"]),
        axis=1
    )

    st.caption(
        "Ranked strictly by observed RDT positivity. Confidence reflects number of survey clusters."
    )

    render_scrollable_table(
        rdt_hotspots
        .head(20)
        .style
        .format({
            "mean_rdt_prevalence": "{:.2f}",
            "mean_risk": "{:.2f}"
        })
    )

    # 🔹 Table 2: High-confidence burden (programmatic use)
    st.subheader("🧪 High-RDT LGAs (High Confidence Only)")

    rdt_confident = rdt_hotspots[rdt_hotspots["clusters"] >= 5]

    render_scrollable_table(
        rdt_confident
        .style
        .format({
            "mean_rdt_prevalence": "{:.2f}",
            "mean_risk": "{:.2f}"
        })
    )

    st.subheader("🚨 Top 10 Highest-Risk LGAs")

    lga_summary = build_lga_summary(map_data)

    # Format values first
    lga_summary_fmt = lga_summary.copy()
    lga_summary_fmt["mean_risk"] = lga_summary_fmt["mean_risk"].round(2)
    lga_summary_fmt["mean_rdt_prevalence"] = lga_summary_fmt["mean_rdt_prevalence"].round(2)

    # Render scrollable table (first 10 visible, rest scroll)
    render_scrollable_table(
        lga_summary_fmt,
        height=350
    )

# ==========================================
# 🔐 Data & Insight Access Request
# ==========================================
if section == "🔐 Data Access Request":

    st.header("🔐 Request Access to Data & Decision Products")

    st.markdown(
        """
        Due to data sensitivity and responsible-use requirements, 
        downloads are **not publicly available**.
    
        Researchers, NGOs, donors, and government partners may request:
        - Aggregated datasets
        - Policy briefs
        - Intervention targeting outputs
        - Time-trend and comparative analyses
        - Technical documentation
        """
    )

    st.markdown(
        """
        ### 🧭 Responsible Access & Sustainability Notice
        This platform operates as a **public-interest decision intelligence system**
        supporting malaria control, health-system planning, and evidence-based policy.
    
        Core summaries and policy insights are provided for **academic, NGO, and government use**.
        Requests involving **custom analysis, extended time-series data, or program-specific outputs**
        may be supported through **cost-recovery or institutional partnerships**, particularly for
        donor-funded or consulting applications.
    
        **No fees are required** for public-interest or planning-focused requests.
        """
    )

    # ------------------------------------------
    # 📋 Access Request Form
    # ------------------------------------------
    with st.form("access_request_form"):

        st.subheader("👤 Requester Information")
        name = st.text_input("Full Name *")
        email = st.text_input("Email Address *")
        affiliation = st.text_input("Organization / Affiliation *")

        st.subheader(" Intended Use Classification")
        purpose = st.text_area(
            "Purpose of Request *",
            placeholder="Explain why you are requesting access"
        )

        st.subheader("🧭 Intended Use Classification")
        access_type = st.selectbox(
            "Intended use category *",
            [
                "Academic / Research",
                "NGO / Programmatic planning",
                "Government decision support",
                "Donor-funded project",
                "Commercial / Consulting use"
            ]
        )

        st.subheader("📦 Requested Products")
        request_items = st.multiselect(
            "What would you like access to?",
            [
                "LGA-level malaria risk & burden tables",
                "Risk–RDT mismatch analysis",
                "Intervention priority lists",
                "Time-trend (year-over-year) malaria analysis",
                "Technical methodology & model documentation",
                "Custom analysis (please specify below)"
            ]
        )

        st.subheader("📝 Additional Notes")
        message = st.text_area(
                "For custom analysis or extended datasets, "
                "please describe scope, geography, time period, "
                "and intended application."
            )


        st.subheader("🤝 Support Platform Sustainability (Optional)")
        support_option = st.radio(
            "How would you like to support this work?",
            [
                "No contribution at this time",
                "Voluntary contribution (cost recovery)",
                "Institutional support / grant alignment",
                "Open to discussion if custom work is required"
            ]
        )

        st.caption(
            "Voluntary contributions help maintain data pipelines, "
            "model updates, and policy translation. "
            "They are **never required** for public-interest use."
        )

        #message = st.text_area(
         #   "Additional Message (optional)"
        #)
        submitted = st.form_submit_button("📩 Submit Request")

    # ------------------------------------------
    # 🧾 Form Validation & Secure Logging
    # ------------------------------------------
    if submitted:
        if not name or not email or not affiliation or not purpose:
            st.error("❌ Please complete all required fields.")
            st.stop()

        try:
            send_data_access_email(
                name=name,
                email=email,
                affiliation=affiliation,
                purpose=purpose,
                message=message
            )
            st.success("✅ Request submitted successfully. Email sent.")
        except Exception as e:
            st.error(f"❌ Email failed: {e}")
            st.stop()

        request_log = pd.DataFrame([{
                "name": name,
                "email": email,
                "affiliation": affiliation,
                "access_type": access_type,
                "requested_items": ", ".join(request_items),
                "support_option": support_option,
                "notes": notes,
                "timestamp": pd.Timestamp.now()
            }])

        try:
                request_log.to_csv(
                    "access_requests_log.csv",
                    mode="a",
                    header=not pd.io.common.file_exists("access_requests_log.csv"),
                    index=False
                )
        except Exception:
                pass  # Fail silently in cloud or restricted environments
if section == "ℹ️ Methodology":

    st.title("ℹ️ Methodology")

    st.markdown("""
    This platform implements a **data-fusion and machine-learning framework**
    designed to support **strategic malaria control, targeting, and surveillance**
    at national and sub-national levels.

    The methodology prioritizes **scientific rigor, transparency, and policy usability**,
    ensuring outputs can inform **programmatic decision-making** without replacing
    routine surveillance systems.
    """)

    st.subheader("1️⃣ Data Sources & Integration")

    st.markdown("""
    Multiple high-quality datasets are harmonized at the **survey-cluster level**:

    **Health & Biomarker Data**
    - Demographic and Health Surveys (DHS)
    - Malaria Indicator Surveys (MIS)
    - Rapid Diagnostic Test (RDT) positivity

    **Environmental & Climate Data**
    - **CHIRPS**: 34-year historical rainfall time series
    - **NDVI**: Vegetation dynamics and land-surface proxies
    - Temporal aggregation aligned to survey timing

    All datasets are spatially joined using cluster geocoordinates
    and temporally aligned to preserve epidemiological validity.
    """)

    st.subheader("2️⃣ Analytical Unit & Spatial Resolution")

    st.markdown("""
    - **Primary unit**: Survey cluster
    - **Secondary aggregation**: Local Government Area (LGA) and State
    - Outputs reflect **population-weighted and evidence-filtered summaries**
      to avoid instability from sparse data

    Minimum-evidence thresholds are applied before ranking LGAs,
    ensuring interpretability and responsible use.
    """)

    st.subheader("3️⃣ Predictive Modeling Framework")

    st.markdown("""
    A **Random Forest classifier** is used to estimate the probability
    of malaria positivity at the cluster level.

    **Model characteristics**
    - Non-linear, interaction-aware learning
    - Robust to multicollinearity
    - Suitable for heterogeneous epidemiological environments

    **Target variable**
    - Binary malaria positivity (RDT-based)

    **Predictors include**
    - Climate and vegetation indicators
    - Temporal features
    - Contextual environmental covariates
    """)

    st.subheader("4️⃣ Validation & Interpretability")

    st.markdown("""
    To ensure generalizability and scientific credibility:

    - **Temporal cross-validation** is applied across survey years
    - Performance assessed using probability-based metrics
    - **SHAP values** used to interpret feature contributions

    This enables:
    - Transparent explanation of model behavior
    - Trust from policymakers and technical reviewers
    """)

    st.subheader("5️⃣ Risk–Burden Reconciliation (Decision Intelligence Layer)")

    st.markdown("""
    Model-estimated risk is explicitly compared with **observed RDT positivity**
    to identify actionable mismatches:

    - **High Risk – High RDT**: Immediate intervention priority
    - **High RDT – Low Risk**: Potential underestimation or data gaps
    - **Low RDT – High Risk**: Emerging or surveillance-priority areas
    - **Low–Low**: Routine monitoring zones

    This layer transforms predictive analytics into
    **programmatically meaningful decision intelligence**.
    """)

    st.subheader("6️⃣ Responsible Use & Limitations")

    st.markdown("""
    This platform **does not replace routine surveillance systems**
    or official reporting channels.

    Instead, it:
    - Complements national malaria programs
    - Supports strategic planning and targeting
    - Highlights areas requiring further investigation

    Outputs are **aggregated, non-identifiable, and ethically constrained**
    to support responsible public-health use.
    """)
# ==========================================
# 🔮 Future Malaria Risk Predictions
# ==========================================
if section == "🔮 Predictions (Coming Soon)":

    st.title("🔮 Future Malaria Risk & RDT Predictions")

    st.markdown("""
    ### 🚧 Section Under Development

    This module will provide **forward-looking malaria intelligence**
    by projecting **malaria risk** and **expected RDT positivity**
    over the coming years using:

    - Climate trend extrapolation (rainfall, NDVI, temperature)
    - Temporal machine-learning models
    - Scenario-based intervention assumptions
    - Stability-weighted historical patterns
    """)

    st.info(
        "🔧 **Coming Soon**: This section is currently in development. "
        "All results shown here in future releases will be **model-driven estimates**, "
        "not real-time surveillance data."
    )

    st.markdown("""
    ---
    ### 🔍 Planned Capabilities

    ✅ National & state-level malaria risk forecasts  
    ✅ LGA-level projected intervention priority zones  
    ✅ Expected RDT positivity under climate scenarios  
    ✅ Risk trajectory comparisons (historical vs projected)  
    ✅ Uncertainty bands & confidence tiers  

    ---
    ### 🧪 Why This Matters

    Predictive malaria intelligence enables:
    - **Earlier intervention planning**
    - **Smarter allocation of limited resources**
    - **Climate-informed malaria preparedness**
    - **Proactive surveillance instead of reactive response**
    """)

    st.warning(
        "⚠️ Forecast outputs will be released only after full validation "
        "and alignment with responsible-use guidelines."
    )
