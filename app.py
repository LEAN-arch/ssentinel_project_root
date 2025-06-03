# Minimal main/main/app.py for testing
import streamlit as st
import sys
import os
import logging

# --- Robust Path Setup ---
_current_file_dir_app_main = os.path.dirname(os.path.abspath(__file__))
_project_root_dir_app_main = os.path.abspath(os.path.join(_current_file_dir_app_main, os.pardir, os.pardir))
if _project_root_dir_app_main not in sys.path:
    sys.path.insert(0, _project_root_dir_app_main)
    print(f"INFO: Added to sys.path: {_project_root_dir_app_main}", file=sys.stderr) # Log path add

# Attempt to import settings first, as it's fundamental
try:
    from config import settings
    print(f"INFO: Successfully imported config.settings. APP_NAME: {settings.APP_NAME}", file=sys.stderr)
except ImportError as e_cfg:
    print(f"FATAL: Failed to import config.settings: {e_cfg}", file=sys.stderr)
    # st.error(f"FATAL: Failed to import config.settings: {e_cfg}") # Might not be available if st fails
    sys.exit(1) # Hard exit if config can't load
except Exception as e_other_cfg:
    print(f"FATAL: Error during config.settings import or access: {e_other_cfg}", file=sys.stderr)
    sys.exit(1)

st.set_page_config(page_title=f"Test - {settings.APP_NAME}", layout="wide")
st.title(f"Minimal Test App - {settings.APP_NAME}")
st.write("If you see this, Streamlit server started and basic config loaded.")
st.write(f"Project Root (determined): {_project_root_dir_app_main}")
st.write(f"Python sys.path: {sys.path}")

# Try importing other modules one by one to find the culprit
try:
    from visualization import plots
    st.write("Successfully imported visualization.plots")
    # plots.set_sentinel_plotly_theme() # Call functions if needed
    # st.write("Plotly theme applied.")
except ImportError as e:
    st.error(f"Failed to import visualization.plots: {e}")
    print(f"ERROR: Failed to import visualization.plots: {e}", file=sys.stderr)
except Exception as e_viz:
    st.error(f"Error during visualization import/setup: {e_viz}")
    print(f"ERROR: Error during visualization import/setup: {e_viz}", file=sys.stderr)

# --- Global Logging Configuration ---
logging.basicConfig(
    level=getattr(logging, str(settings.LOG_LEVEL).upper(), logging.INFO),
    format=settings.LOG_FORMAT,
    datefmt=settings.LOG_DATE_FORMAT,
    handlers=[logging.StreamHandler(sys.stdout)],
    force=True
)
logger = logging.getLogger(__name__)

# --- Apply Plotly Theme Globally ---
try:
    set_sentinel_plotly_theme()
    logger.info("Sentinel Plotly theme applied globally from main/main/app.py.")
except Exception as e_theme_app_main:
    logger.error(f"Failed to apply Sentinel Plotly theme in main/main/app.py: {e_theme_app_main}", exc_info=True)

# --- Page Configuration (must be the first Streamlit command) ---
page_icon_main_app = settings.APP_LOGO_SMALL_PATH
if not os.path.exists(page_icon_main_app):
    logger.warning(f"Small app logo for page icon not found at '{page_icon_main_app}'. Using fallback '🌍'.")
    page_icon_main_app = "🌍"

st.set_page_config(
    page_title=f"{settings.APP_NAME} - System Overview",
    page_icon=page_icon_main_app,
    layout="wide",
    initial_sidebar_state="expanded",
    menu_items={
        'Get Help': f"mailto:{settings.SUPPORT_CONTACT_INFO}?subject=Help Request - {settings.APP_NAME}",
        'Report a bug': f"mailto:{settings.SUPPORT_CONTACT_INFO}?subject=Bug Report - {settings.APP_NAME} v{settings.APP_VERSION}",
        'About': f"""
### {settings.APP_NAME} (v{settings.APP_VERSION})
{settings.APP_FOOTER_TEXT}

An Edge-First Health Intelligence Co-Pilot designed for resource-limited environments.
This demonstrator showcases higher-level views. Frontline workers use dedicated native PED apps.
"""
    }
)

# --- CSS Loading ---
@st.cache_resource
def load_global_styles_from_main_app(css_path: str): # Renamed function for clarity
    if os.path.exists(css_path):
        try:
            with open(css_path, "r", encoding="utf-8") as f_css_content:
                st.markdown(f'<style>{f_css_content.read()}</style>', unsafe_allow_html=True)
            logger.info(f"Global CSS styles loaded by main/main/app.py from: {css_path}")
        except Exception as e_css_load_main:
            logger.error(f"Error reading global CSS file {css_path} in main/main/app.py: {e_css_load_main}", exc_info=True)
            st.error("Critical error: Application styles could not be loaded.")
    else:
        logger.warning(f"Global CSS file not found by main/main/app.py: {css_path}.")

if settings.STYLE_CSS_PATH_WEB:
    load_global_styles_from_main_app(settings.STYLE_CSS_PATH_WEB)
else:
    logger.info("No global CSS path configured in settings. Skipping custom CSS load in main/main/app.py.")

# --- Main Application Header ---
header_cols_main_page = st.columns([0.12, 0.88])
with header_cols_main_page[0]:
    logo_header_main_page = settings.APP_LOGO_LARGE_PATH
    if not os.path.exists(logo_header_main_page):
        logo_header_main_page = settings.APP_LOGO_SMALL_PATH
    
    if os.path.exists(logo_header_main_page):
        st.image(logo_header_main_page, width=90)
    else:
        st.markdown("### 🌍", unsafe_allow_html=True)
with header_cols_main_page[1]:
    st.title(settings.APP_NAME)
    st.subheader("Transforming Data into Lifesaving Action at the Edge")
st.divider()

# --- Enhanced Welcome & System Description ---
st.markdown(f"""
    ## Welcome to the Sentinel Health Co-Pilot
    
    Sentinel is an **edge-first health intelligence system** designed for **maximum clinical and 
    operational actionability** in resource-limited, high-risk LMIC environments. It converts 
    diverse data sources—wearables, IoT, contextual inputs—into life-saving, workflow-integrated 
    decisions, even with **minimal or no internet connectivity.**
""")

st.markdown("#### Core Principles Guiding Sentinel:")
core_principles_cols_main_app = st.columns(2)
core_principles_data_main_app = [
    ("📶 **Offline-First Operations**", "On-device Edge AI on Personal Edge Devices (PEDs) ensures critical functionality without continuous connectivity."),
    ("🎯 **Action-Oriented Intelligence**", "Every insight aims to trigger a clear, targeted response relevant to frontline workflows."),
    ("🧑‍🤝‍🧑 **Human-Centered Design**", "Interfaces are optimized for low-literacy, high-stress users, prioritizing immediate understanding and ease of use on PEDs."),
    ("🔗 **Resilience & Scalability**", "Modular design allows scaling from individual PEDs to facility and regional views, with robust, flexible data synchronization mechanisms.")
]

for idx_core_principle, (title_core_principle, desc_core_principle) in enumerate(core_principles_data_main_app):
    with core_principles_cols_main_app[idx_core_principle % 2]:
        st.markdown(f"##### {title_core_principle}")
        st.markdown(f"<small>{desc_core_principle}</small>", unsafe_allow_html=True)
        if idx_core_principle < len(core_principles_data_main_app) - 2 : 
             st.markdown("<div style='margin-bottom: 1rem;'></div>", unsafe_allow_html=True)

st.markdown("""
    ---
    👈 **Navigate via the sidebar** to explore simulated web dashboards for various operational tiers. 
    These views represent perspectives of **Supervisors, Clinic Managers, or District Health Officers (DHOs)**. 
    The primary interface for frontline workers (e.g., CHWs) is a dedicated native application on their 
    Personal Edge Device (PED), tailored for their specific operational context.
""")
st.info(
    "💡 **Note:** This web application serves as a high-level demonstrator for the Sentinel system's "
    "data processing capabilities and the types of aggregated views available to management and strategic personnel."
)
st.divider()

# --- Simulated Role-Specific Views Section (Navigation to Pages) ---
st.header("Explore Simulated Role-Specific Dashboards")
st.caption("These views demonstrate information available at higher tiers. Frontline workers use dedicated PED apps.")

role_navigation_config_main_app = [
    {"title": "🧑‍⚕️ CHW Operations & Field Support", "desc": "Supervisor view: team performance, CHW support, local epi signals.", "page": "pages/chw_dashboard.py"},
    {"title": "🏥 Clinic Operations & Environment", "desc": "Clinic Manager view: service efficiency, care quality, resources, facility safety.", "page": "pages/clinic_dashboard.py"},
    {"title": "🗺️ District Health Strategic Overview", "desc": "DHO view: population health, resource allocation, strategic interventions.", "page": "pages/district_dashboard.py"},
    {"title": "📊 Population Health Analytics", "desc": "Analyst view: in-depth epi/systems analysis, SDOH impacts, equity.", "page": "pages/population_dashboard.py"},
]

nav_cols_main_app = st.columns(2)
for i_nav_main, nav_item_main_cfg in enumerate(role_navigation_config_main_app):
    with nav_cols_main_app[i_nav_main % 2]:
        with st.container(border=True):
            st.subheader(nav_item_main_cfg["title"])
            st.markdown(f"<small>{nav_item_main_cfg['desc']}</small>", unsafe_allow_html=True)
            st.page_link(nav_item_main_cfg["page"], label=f"Explore {nav_item_main_cfg['title'].split('(')[0].strip()}", icon="➡️", use_container_width=True)
            st.markdown("<div style='margin-bottom: 0.5rem;'></div>", unsafe_allow_html=True)
st.divider()

# --- Key Capabilities Reimagined Section ---
st.header(f"{settings.APP_NAME} - Key Capabilities Reimagined")
capabilities_data_main_app = [
    ("🛡️ Frontline Worker Safety & Support", "Real-time vitals/environmental monitoring, fatigue detection, safety nudges on PEDs."),
    ("🌍 Offline-First Edge AI", "On-device intelligence for alerts, prioritization, guidance without continuous connectivity."),
    ("⚡ Actionable, Contextual Insights", "Raw data to clear, role-specific recommendations integrated into field workflows."),
    ("🤝 Human-Centered & Accessible UX", "Pictogram UIs, voice/tap commands, local language support for low-literacy, high-stress users on PEDs."),
    ("📡 Resilient Data Synchronization", "Flexible data sharing (Bluetooth, QR, SD card, SMS, opportunistic IP) across devices/tiers."),
    ("🌱 Scalable & Interoperable Architecture", "Modular design (personal to national), FHIR/HL7 considerations for integration.")
]
cap_rows_main_app = [st.columns(3), st.columns(3)]
all_cap_cols_main_app = cap_rows_main_app[0] + cap_rows_main_app[1]
for i_cap_main, (cap_title_item, cap_desc_item) in enumerate(capabilities_data_main_app):
    if i_cap_main < len(all_cap_cols_main_app):
        with all_cap_cols_main_app[i_cap_main]:
            st.markdown(f"##### {cap_title_item}")
            st.markdown(f"<small>{cap_desc_item}</small>", unsafe_allow_html=True)
            st.markdown("<div style='margin-bottom: 1.2rem;'></div>", unsafe_allow_html=True)
st.divider()

# --- Link to the Glossary page ---
with st.expander("📜 **System Glossary & Terminology**", expanded=False):
    st.markdown(
        f"Explore definitions for key terms, metrics, and system components specific to the {settings.APP_NAME}."
    )
    st.page_link("pages/glossary_page.py", label="Go to Glossary", icon="📚")

# --- Sidebar Content ---
st.sidebar.header(f"{settings.APP_NAME}")
st.sidebar.divider()
st.sidebar.markdown("#### About This Demonstrator:")
st.sidebar.info(
    "This web app simulates higher-level dashboards. "
    "Frontline worker interaction occurs on dedicated Personal Edge Devices (PEDs)."
)
st.sidebar.markdown("---")
st.sidebar.page_link("pages/glossary_page.py", label="📜 System Glossary", icon="📚")
st.sidebar.divider()
st.sidebar.markdown(f"**{settings.ORGANIZATION_NAME}**")
st.sidebar.markdown(f"Support: [{settings.SUPPORT_CONTACT_INFO}](mailto:{settings.SUPPORT_CONTACT_INFO})")
st.sidebar.divider()
st.sidebar.caption(settings.APP_FOOTER_TEXT)

logger.info(f"{settings.APP_NAME} (v{settings.APP_VERSION}) - System Overview page (main/main/app.py) loaded.")
