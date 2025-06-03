# sentinel_project_root/main/main/app.py
# Minimal Streamlit app for testing Sentinel Health Co-Pilot

import streamlit as st
import sys
import os
import logging
from pathlib import Path

# --- Robust Path Setup ---
_current_file_dir = Path(__file__).parent.resolve()
_project_root_dir = _current_file_dir.parent.parent.resolve()
if str(_project_root_dir) not in sys.path:
    sys.path.insert(0, str(_project_root_dir))
    print(f"INFO: Added to sys.path: {_project_root_dir}", file=sys.stderr)

# --- Import Settings ---
try:
    from config import settings
    print(f"INFO: Successfully imported config.settings. APP_NAME: {settings.APP_NAME}", file=sys.stderr)
except ImportError as e:
    print(f"FATAL: Failed to import config.settings: {e}", file=sys.stderr)
    sys.exit(1)
except Exception as e:
    print(f"FATAL: Error during config.settings import: {e}", file=sys.stderr)
    sys.exit(1)

# --- Validate Streamlit Version ---
import streamlit
if streamlit.__version__ < "1.30.0":
    print(f"ERROR: Streamlit version {streamlit.__version__} is too old. Requires 1.30.0+", file=sys.stderr)
    sys.exit(1)

# --- Global Logging Configuration ---
valid_log_levels = {"DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"}
log_level = str(settings.LOG_LEVEL).upper()
if log_level not in valid_log_levels:
    print(f"WARN: Invalid LOG_LEVEL '{log_level}' in settings. Using INFO.", file=sys.stderr)
    log_level = "INFO"

logging.basicConfig(
    level=getattr(logging, log_level, logging.INFO),
    format=settings.LOG_FORMAT,
    datefmt=settings.LOG_DATE_FORMAT,
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger(__name__)

# --- Page Configuration (Single Call) ---
page_icon = Path(settings.APP_LOGO_SMALL_PATH).resolve()
if not page_icon.exists():
    logger.warning(f"Page icon not found at '{page_icon}'. Using fallback '🌍'.")
    page_icon = "🌍"

st.set_page_config(
    page_title=f"{settings.APP_NAME} - System Overview",
    page_icon=str(page_icon) if isinstance(page_icon, Path) else page_icon,
    layout="wide",
    initial_sidebar_state="expanded",
    menu_items={
        "Get Help": f"mailto:{settings.SUPPORT_CONTACT_INFO}?subject=Help Request - {settings.APP_NAME}",
        "Report a bug": f"mailto:{settings.SUPPORT_CONTACT_INFO}?subject=Bug Report - {settings.APP_NAME} v{settings.APP_VERSION}",
        "About": f"""
### {settings.APP_NAME} (v{settings.APP_VERSION})
{settings.APP_FOOTER_TEXT}

An Edge-First Health Intelligence Co-Pilot designed for resource-limited environments.
This demonstrator showcases higher-level views. Frontline workers use dedicated native PED apps.
"""
    }
)

# --- Apply Plotly Theme ---
try:
    from visualization.plots import set_sentinel_plotly_theme
    set_sentinel_plotly_theme()
    logger.debug("Sentinel Plotly theme applied globally.")
except ImportError as e:
    logger.error(f"Failed to import visualization.plots: {e}", exc_info=True)
    st.error(f"Failed to import visualization.plots: {e}")
except Exception as e:
    logger.error(f"Error applying Plotly theme: {e}", exc_info=True)
    st.error(f"Error applying Plotly theme: {e}")

# --- CSS Loading ---
@st.cache_resource
def load_global_styles(css_path: str):
    css_path = Path(css_path).resolve()
    if css_path.exists():
        try:
            with open(css_path, "r", encoding="utf-8") as f:
                st.markdown(f'<style>{f.read()}</style>', unsafe_allow_html=True)
            logger.debug(f"Global CSS styles loaded from: {css_path}")
        except Exception as e:
            logger.error(f"Error reading CSS file {css_path}: {e}", exc_info=True)
            st.error("Critical error: Application styles could not be loaded.")
    else:
        logger.warning(f"CSS file not found: {css_path}")

if settings.STYLE_CSS_PATH_WEB:
    load_global_styles(settings.STYLE_CSS_PATH_WEB)
else:
    logger.debug("No CSS path configured. Skipping custom CSS load.")

# --- Main Application Header ---
header_cols = st.columns([0.15, 0.85])
with header_cols[0]:
    logo_path = Path(settings.APP_LOGO_LARGE_PATH).resolve()
    if not logo_path.exists():
        logo_path = Path(settings.APP_LOGO_SMALL_PATH).resolve()
    if logo_path.exists():
        st.image(str(logo_path), width=90)
    else:
        st.markdown("### 🌍", unsafe_allow_html=True)
with header_cols[1]:
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
core_principles_cols = st.columns(2)
core_principles_data = [
    ("📶 **Offline-First Operations**", "On-device Edge AI on Personal Edge Devices (PEDs) ensures critical functionality without continuous connectivity."),
    ("🎯 **Action-Oriented Intelligence**", "Every insight aims to trigger a clear, targeted response relevant to frontline workflows."),
    ("🧑‍🤝‍🧑 **Human-Centered Design**", "Interfaces are optimized for low-literacy, high-stress users, prioritizing immediate understanding and ease of use on PEDs."),
    ("🔗 **Resilience & Scalability**", "Modular design allows scaling from individual PEDs to facility and regional views, with robust, flexible data synchronization mechanisms.")
]

for idx, (title, desc) in enumerate(core_principles_data):
    with core_principles_cols[idx % 2]:
        st.markdown(f"##### {title}")
        st.markdown(f"<small>{desc}</small>", unsafe_allow_html=True)
        if idx < len(core_principles_data) - 1:
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

# --- Simulated Role-Specific Views Section ---
st.header("Explore Simulated Role-Specific Dashboards")
st.caption("These views demonstrate information available at higher tiers. Frontline workers use dedicated PED apps.")

role_navigation_config = [
    {"title": "🧑‍⚕️ CHW Operations & Field Support", "desc": "Supervisor view: team performance, CHW support, local epi signals.", "page": "pages/chw_dashboard.py"},
    {"title": "🏥 Clinic Operations & Environment", "desc": "Clinic Manager view: service efficiency, care quality, resources, facility safety.", "page": "pages/clinic_dashboard.py"},
    {"title": "🗺️ District Health Strategic Overview", "desc": "DHO view: population health, resource allocation, strategic interventions.", "page": "pages/district_dashboard.py"},
    {"title": "📊 Population Health Analytics", "desc": "Analyst view: in-depth epi/systems analysis, SDOH impacts, equity.", "page": "pages/population_dashboard.py"},
]

nav_cols = st.columns(2)
for i, nav_item in enumerate(role_navigation_config):
    page_path = Path(_project_root_dir, nav_item["page"])
    if not page_path.exists():
        logger.warning(f"Navigation page not found: {page_path}")
        continue
    with nav_cols[i % 2]:
        with st.container(border=True):
            st.subheader(nav_item["title"])
            st.markdown(f"<small>{nav_item['desc']}</small>", unsafe_allow_html=True)
            st.page_link(str(page_path), label=f"Explore {nav_item['title'].split('(')[0].strip()}", icon="➡️", use_container_width=True)
            st.markdown("<div style='margin-bottom: 0.5rem;'></div>", unsafe_allow_html=True)
st.divider()

# --- Key Capabilities Reimagined Section ---
st.header(f"{settings.APP_NAME} - Key Capabilities Reimagined")
capabilities_data = [
    ("🛡️ Frontline Worker Safety & Support", "Real-time vitals/environmental monitoring, fatigue detection, safety nudges on PEDs."),
    ("🌍 Offline-First Edge AI", "On-device intelligence for alerts, prioritization, guidance without continuous connectivity."),
    ("⚡ Actionable, Contextual Insights", "Raw data to clear, role-specific recommendations integrated into field workflows."),
    ("🤝 Human-Centered & Accessible UX", "Pictogram UIs, voice/tap commands, local language support for low-literacy, high-stress users on PEDs."),
    ("📡 Resilient Data Synchronization", "Flexible data sharing (Bluetooth, QR, SD card, SMS, opportunistic IP) across devices/tiers."),
    ("🌱 Scalable & Interoperable Architecture", "Modular design (personal to national), FHIR/HL7 considerations for integration.")
]
cap_rows = [st.columns(3), st.columns(3)]
all_cap_cols = cap_rows[0] + cap_rows[1]
for i, (cap_title, cap_desc) in enumerate(capabilities_data):
    if i < len(all_cap_cols):
        with all_cap_cols[i]:
            st.markdown(f"##### {cap_title}")
            st.markdown(f"<small>{cap_desc}</small>", unsafe_allow_html=True)
            st.markdown("<div style='margin-bottom: 1.2rem;'></div>", unsafe_allow_html=True)
st.divider()

# --- Link to the Glossary Page ---
glossary_path = Path(_project_root_dir, "pages/glossary_page.py")
with st.expander("📜 **System Glossary & Terminology**", expanded=False):
    st.markdown(f"Explore definitions for key terms, metrics, and system components specific to {settings.APP_NAME}.")
    if glossary_path.exists():
        st.page_link(str(glossary_path), label="Go to Glossary", icon="📚")
    else:
        logger.warning(f"Glossary page not found: {glossary_path}")
        st.markdown("Glossary page unavailable.")

# --- Sidebar Content ---
st.sidebar.header(f"{settings.APP_NAME}")
st.sidebar.divider()
st.sidebar.markdown("#### About This Demonstrator:")
st.sidebar.info(
    "This web app simulates higher-level dashboards. "
    "Frontline worker interaction occurs on dedicated Personal Edge Devices (PEDs)."
)
st.sidebar.markdown("---")
if glossary_path.exists():
    st.sidebar.page_link(str(glossary_path), label="📜 System Glossary", icon="📚")
st.sidebar.divider()
st.sidebar.markdown(f"**{settings.ORGANIZATION_NAME}**")
st.sidebar.markdown(f"Support: [{settings.SUPPORT_CONTACT_INFO}](mailto:{settings.SUPPORT_CONTACT_INFO})")
st.sidebar.divider()
st.sidebar.caption(settings.APP_FOOTER_TEXT)

logger.info(f"{settings.APP_NAME} (v{settings.APP_VERSION}) - System Overview page loaded.")
