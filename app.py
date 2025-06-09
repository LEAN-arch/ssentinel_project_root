# sentinel_project_root/app.py
# SME PLATINUM STANDARD - APPLICATION ENTRY POINT

import logging
import sys
from pathlib import Path
import html

# --- CRITICAL: Add project root to path for module resolution ---
# This ensures that 'config', 'data_processing', etc. can be imported.
try:
    _project_root = Path(__file__).resolve().parent
    if str(_project_root) not in sys.path:
        sys.path.insert(0, str(_project_root))

    # --- Import Core Application Modules ---
    import streamlit as st
    from packaging import version
    
    from config import settings
    from visualization import load_and_inject_css, set_plotly_theme
    
except ImportError as e:
    # This block provides a more helpful error message if crucial imports fail.
    print(f"FATAL ERROR in app.py: A core module failed to import.", file=sys.stderr)
    print(f"This is often due to an incomplete setup or running from the wrong directory.", file=sys.stderr)
    print(f"1. Ensure you have run 'scripts/setup.sh' to install dependencies.", file=sys.stderr)
    print(f"2. Run the app from the project root: `cd sentinel_project_root && streamlit run app.py`", file=sys.stderr)
    print(f"\nPython Path: {sys.path}", file=sys.stderr)
    print(f"Original ImportError: {e}", file=sys.stderr)
    sys.exit(1)

# --- Global Configuration ---
logging.basicConfig(
    level=settings.LOG_LEVEL,
    format=settings.LOG_FORMAT,
    datefmt=settings.LOG_DATE_FORMAT,
    handlers=[logging.StreamHandler(sys.stdout)],
    force=True
)
logger = logging.getLogger(__name__)

st.set_page_config(
    page_title=f"{settings.APP_NAME} - Overview",
    page_icon=str(settings.APP_LOGO_SMALL_PATH),
    layout="wide",
    initial_sidebar_state="expanded",
    menu_items={
        "Get Help": f"mailto:{settings.SUPPORT_CONTACT_INFO}",
        "Report a bug": f"mailto:{settings.SUPPORT_CONTACT_INFO}?subject=Bug Report - {settings.APP_NAME} v{settings.APP_VERSION}",
        "About": f"### {settings.APP_NAME} (v{settings.APP_VERSION})\n{settings.APP_FOOTER_TEXT}"
    }
)

# Apply global visual styles and themes
load_and_inject_css(settings.STYLE_CSS_PATH)
set_plotly_theme()

# --- Application Header ---
header_cols = st.columns([0.1, 0.9])
with header_cols[0]:
    st.image(str(settings.APP_LOGO_LARGE_PATH), width=110)
with header_cols[1]:
    st.title(settings.APP_NAME)
    st.subheader("Actionable Intelligence for Resilient Health Systems")
st.divider()

# --- Welcome & System Description ---
st.markdown(f"""
### Welcome to the {html.escape(settings.APP_NAME)}
This application demonstrates an **edge-first health intelligence system** designed for **maximum clinical and 
operational actionability** in resource-constrained environments. It transforms diverse data sources 
into life-saving, workflow-integrated decisions.
""")

st.info("""
**💡 How to Use This Demo:** The dashboards linked below simulate the views available to **Supervisors, Clinic Managers, and Health Analysts**. 
The primary interface for frontline workers (e.g., CHWs) is a separate, dedicated native application on their device, which is not part of this web demo.
""", icon="ℹ️")
st.divider()

# --- Dynamic Page Navigation Links ---
st.header("Explore Simulated Role-Specific Dashboards")

pages = {
    "Field Operations": {
        "file": "pages/01_Field_Operations.py",
        "icon": "🧑‍⚕️",
        "desc": "A supervisor's view of team performance, high-risk patients, and emerging local health signals."
    },
    "Clinic Dashboard": {
        "file": "pages/02_Clinic_Dashboard.py",
        "icon": "🏥",
        "desc": "A clinic manager's dashboard for service efficiency, care quality, and facility safety."
    },
    "Supply Chain": {
        "file": "pages/03_Supply_Chain.py",
        "icon": "📦",
        "desc": "A logistics view for monitoring stock levels and forecasting supply needs."
    },
    "Population Analytics": {
        "file": "pages/04_Population_Analytics.py",
        "icon": "📊",
        "desc": "An analyst's tool for epidemiological deep dives and health equity analysis."
    }
}

nav_cols = st.columns(2)
col_idx = 0
for page_name, page_info in pages.items():
    if Path(page_info["file"]).exists():
        with nav_cols[col_idx % 2]:
            with st.container(border=True):
                st.subheader(f"{page_info['icon']} {page_name}")
                st.markdown(f"<small>{page_info['desc']}</small>", unsafe_allow_html=True)
                st.page_link(page_info["file"], label=f"Explore {page_name} View", use_container_width=True, icon="➡️")
        col_idx += 1
    else:
        logger.warning(f"Navigation page not found, skipping link: {page_info['file']}")

st.divider()

# --- Sidebar ---
with st.sidebar:
    st.header(f"{settings.APP_NAME}")
    st.caption(f"v{settings.APP_VERSION}")
    st.divider()
    st.info("Navigate between simulated dashboards using the links above or the list on the left.")
    st.divider()
    st.markdown(f"**{html.escape(settings.ORGANIZATION_NAME)}**")
    st.markdown(f"Contact: <a href='mailto:{settings.SUPPORT_CONTACT_INFO}'>{settings.SUPPORT_CONTACT_INFO}</a>", unsafe_allow_html=True)
    st.caption(settings.APP_FOOTER_TEXT)

logger.info("Main application page loaded successfully.")

