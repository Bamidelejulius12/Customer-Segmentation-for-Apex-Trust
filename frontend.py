import streamlit as st
import requests
import pandas as pd
import base64
from PIL import Image
import io
import re

# CONFIG
API_BASE = "http://localhost:8000"

st.set_page_config(
    page_title="Customer Segmentation Dashboard",
    layout="wide"
)

# SAAS STYLING
st.markdown("""
<style>

/* Main background */
.main {
    background-color: #0e1117;
}

/* Sidebar */
[data-testid="stSidebar"] {
    background-color: #111827;
    padding: 20px;
}

/* Sidebar title */
.sidebar-title {
    font-size: 20px;
    font-weight: 700;
    color: white;
    margin-bottom: 20px;
}

/* Navigation */
div[role="radiogroup"] > label {
    background: #1f2937;
    padding: 12px 14px;
    border-radius: 10px;
    margin-bottom: 8px;
    border: 1px solid transparent;
    transition: all 0.2s ease;
}

div[role="radiogroup"] > label:hover {
    background: #374151;
    border: 1px solid #4ECDC4;
}

div[role="radiogroup"] > label[data-selected="true"] {
    background: #4ECDC4;
    color: black !important;
    font-weight: 600;
}

/* Hide radio circle */
div[role="radiogroup"] input {
    display: none;
}

/* KPI Cards */
.kpi-card {
    background: #1f2937;
    padding: 20px;
    border-radius: 12px;
    border: 1px solid #2d3748;
}

.kpi-title {
    font-size: 14px;
    color: #9ca3af;
}

.kpi-value {
    font-size: 28px;
    font-weight: bold;
    color: white;
}

/* Spacing */
.block-container {
    padding-top: 1rem;
}

</style>
""", unsafe_allow_html=True)

# SIDEBAR
st.sidebar.markdown('<div class="sidebar-title"> Customer Behaviour Tracking Dashboard</div>', unsafe_allow_html=True)

page = st.sidebar.radio(
    "",
    ["Dashboard", "Segments"]
)

page = page.split(" ")[-1]

st.sidebar.markdown("---")

if st.sidebar.button("Retrain Model", use_container_width=True):
    requests.get(f"{API_BASE}/retrain")
    st.sidebar.success("Model retrained successfully")

# HELPERS
def display_base64_image(b64_string):
    img = base64.b64decode(b64_string)
    return Image.open(io.BytesIO(img))

# LOAD DATA
@st.cache_data
def load_segments():
    res = requests.get(f"{API_BASE}/segments")
    data = res.json()

    segmented_df = pd.DataFrame(data["segmented_data"])
    cluster_df = pd.DataFrame(data["cluster_summary"])

    return segmented_df, cluster_df


@st.cache_data
def load_dashboard():
    res = requests.get(f"{API_BASE}/dashboard")
    return res.text


segmented_df, cluster_df = load_segments()

# SEGMENTS PAGE
if page == "Segments":
    st.markdown("#Customer Segments")

    st.markdown("### Segmented Data")
    st.dataframe(segmented_df, use_container_width=True)

    csv = segmented_df.to_csv(index=False).encode("utf-8")
    st.download_button(
        "Download CSV",
        csv,
        "segmented_rfm.csv",
        "text/csv"
    )

    st.markdown("---")

    st.markdown("### Cluster Summary")
    st.dataframe(cluster_df, use_container_width=True)

# DASHBOARD PAGE
elif page == "Dashboard":
    st.markdown("# Customer Analytics")

    html = load_dashboard()
    images = re.findall(r"base64,(.*?)\"", html)


    # KPI SECTION

    st.markdown("### Overview")

    total_customers = cluster_df["Customer_Count"].sum()
    total_revenue = (cluster_df["Avg_Monetary"] * cluster_df["Customer_Count"]).sum()
    avg_value = cluster_df["Avg_Monetary"].mean()
    num_segments = len(cluster_df)

    col1, col2, col3, col4 = st.columns(4)

    def kpi_card(title, value):
        st.markdown(f"""
            <div class="kpi-card">
                <div class="kpi-title">{title}</div>
                <div class="kpi-value">{value}</div>
            </div>
        """, unsafe_allow_html=True)

    with col1:
        kpi_card("Customers", f"{int(total_customers):,}")
    with col2:
        kpi_card("Revenue", f"{total_revenue:,.0f}")
    with col3:
        kpi_card("Avg Value", f"{avg_value:,.2f}")
    with col4:
        kpi_card("Segments", f"{num_segments}")

    st.markdown("---")

    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Customer Distribution")
        st.image(display_base64_image(images[0]), use_container_width=True)

    with col2:
        st.subheader("Segment Size")
        st.image(display_base64_image(images[2]), use_container_width=True)
       


    col3, col4 = st.columns(2)

    with col3:
        st.subheader("RFM Analysis")
        st.image(display_base64_image(images[1]), use_container_width=True)

    with col4:
        st.subheader("Revenue vs Customers")
        st.image(display_base64_image(images[4]), use_container_width=True)


    # ROW 3

    col5, = st.columns(1)  # Note the comma to unpack

    with col5:
        st.subheader("Revenue Distribution")
        st.image(display_base64_image(images[3]), use_container_width=True)

    # RADAR CHART

    st.markdown("### Segment Profile")

    radar_html = html.split("<h2>Radar Chart</h2>")[1]
    st.components.v1.html(radar_html, height=500)