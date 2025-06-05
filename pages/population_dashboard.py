# sentinel_project_root/app.py
# Main Streamlit application for Sentinel Health Co-Pilot Demonstrator.

import streamlit as st
import sys
import logging
from pathlib import Path

# --- Robust Path Setup ---
_current_app_file_dir = Path(__file__).parent.resolve()
# Assuming app.py is in the project root for this setup
_project_root_dir = _current_app_file_dir

# If app.py is in a subdirectory like 'main_app', adjust root:
# _project_root_dir = _current_app_file_dir.parent # If app.py is in 'main_app/'
# For the provided structure, app.py seems to be at the root.

if str(_project_root_dir) not in sys.path:
    sys.path.insert(0, str(_project_root_dir))
    # Use print for initial setup messages before logging is configured
    print(f"INFO: Added project root to sys.path: {_project_root_dir}", file=sys.stderr)

# --- Import Settings ---
try:
    from config import settings
    print(f"INFO: Successfully imported config.settings. APP_NAME: {settings.APP_NAME}", file=sys.stderr)
except ImportError as e_cfg_app:
    print(f"FATAL: Failed to import config.settings in app.py: {e_cfg_app}", file=sys.stderr)
    print(f"PYTHONPATH for app.py: {sys.path}", file=sys.stderr)
    print(f"Calculated project root: {_project_root_dir}", file=sys.stderr)
    sys.exit(1)
except Exception as e_generic_cfg:
    print(f"FATAL: Generic error during config.settings import in app.py: {e_generic_cfg}", file=sys.stderr)
    sys.exit(1)


# --- Global Logging Configuration ---
valid_log_levels_app = {"DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"}
log_level_app_str = str(settings.LOG_LEVEL).upper() # Get from settings.py
if log_level_app_str not in valid_log_levels_app:
    print(f"WARN: Invalid LOG_LEVEL '{log_level_app_str}' in settings. Defaulting to INFO.", file=sys.stderr)
    log_level_app_str = "INFO"

logging.basicConfig(
    level=getattr(logging, log_level_app_str, logging.INFO), # Use validated log level
    format=settings.LOG_FORMAT, # Get from settings.py
    datefmt=settings.LOG_DATE_FORMAT, # Get from settings.py
    handlers=[logging.StreamHandler(sys.stdout)], # Stream to stdout for Streamlit
    force=True # Override any previous basicConfig
)
logger = logging.getLogger(__name__) # Logger for this app.py


# --- Streamlit Version Check ---
try:
    import streamlit
    # Example: Check for a minimum version if specific features are used
    # (e.g., st.container(border=True) needs 1.25.0+)
    major, minor, patch_str = streamlit.__version__.split('.')
    patch = int(patch_str.split('-')[0]) # Handle potential dev versions like 1.30.0-dev
    if not (int(major) >= 1 and int(minor) >= 30):
        warn_msg_st_ver = f"Streamlit version {streamlit.__version__} is older than recommended (1.30.0+). Some UI features like `st.container(border=True)` or `st.page_link` might not work as expected or require alternatives."
        logger.warning(warn_msg_st_ver)
        # Try to show in UI if st is available
        try: st.sidebar.warning(warn_msg_st_ver)
        except: pass
except ImportError:
    logger.critical("Streamlit library not found. Sentinel Web UI cannot run.")
    sys.exit("Streamlit library not found. Please install it via `pip install streamlit`.")


# --- Page Configuration (must be the first Streamlit command) ---
page_icon_path_obj = Path(settings.APP_LOGO_SMALL_PATH) # settings.APP_LOGO_SMALL_PATH is already string
if not page_icon_path_obj.is_absolute():
    page_icon_path_obj = (_project_root_dir / settings.APP_LOGO_SMALL_PATH).resolve()

final_page_icon_str: str
if page_icon_path_obj.exists() and page_icon_path_obj.is_file():
    final_page_icon_str = str(page_icon_path_obj)
else:
    logger.warning(f"Page icon not found at '{page_icon_path_obj}'. Using fallback emoji '🌍'.")
    final_page_icon_str = "🌍"

st.set_page_config(
    page_title=f"{settings.APP_NAME} - System Overview",
    page_icon=final_page_icon_str,
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

# --- Apply Plotly Theme Globally ---
try:
    from visualization.plots import set_sentinel_plotly_theme
    set_sentinel_plotly_theme()
    logger.debug("Sentinel Plotly theme applied globally from app.py.")
except ImportError as e_theme_imp_app:
    logger.error(f"Failed to import set_sentinel_plotly_theme in app.py: {e_theme_imp_app}", exc_info=True)
    st.error("Critical error: Plotting theme could not be applied. Visualizations may be affected.")
except Exception as e_theme_apply_app:
    logger.error(f"Error applying Plotly theme in app.py: {e_theme_apply_app}", exc_info=True)
    st.error("Error applying visualization theme. Charts might not display correctly.")

# --- Global CSS Loading ---
@st.cache_resource # Cache the CSS content to avoid reloading on every script run
def load_global_css_styles(css_file_path_str: str):
    # Resolve path relative to project root if it's not absolute
    css_file_path_obj = Path(css_file_path_str)
    if not css_file_path_obj.is_absolute():
        css_file_path_obj = (_project_root_dir / css_file_path_str).resolve()
    
    if css_file_path_obj.exists() and css_file_path_obj.is_file():
        try:
            with open(css_file_path_obj, "r", encoding="utf-8") as f_css:
                st.markdown(f'<style>{f_css.read()}</style>', unsafe_allow_html=True)
            logger.debug(f"Global CSS styles loaded successfully from: {css_file_path_obj}")
        except Exception as e_css_load_app:
            logger.error(f"Error reading or applying CSS file {css_file_path_obj}: {e_css_load_app}", exc_info=True)
            st.error("Critical error: Application styles could not be loaded. UI may be broken.")
    else:
        logger.warning(f"Global CSS file not found at resolved path: {css_file_path_obj}")
        st.warning("Application stylesheet missing. UI might not render as intended.")

if settings.STYLE_CSS_PATH_WEB:
    load_global_css_styles(settings.STYLE_CSS_PATH_WEB)
else:
    logger.debug("No global CSS path configured in settings. Skipping custom CSS load.")


# --- Main Application Header ---
header_cols_main_app = st.columns([0.12, 0.88]) # Adjust ratio for logo size and title space
with header_cols_main_app[0]:
    large_logo_path = Path(settings.APP_LOGO_LARGE_PATH)
    if not large_logo_path.is_absolute(): large_logo_path = (_project_root_dir / settings.APP_LOGO_LARGE_PATH).resolve()
    
    small_logo_path_header = Path(settings.APP_LOGO_SMALL_PATH)
    if not small_logo_path_header.is_absolute(): small_logo_path_header = (_project_root_dir / settings.APP_LOGO_SMALL_PATH).resolve()

    if large_logo_path.exists() and large_logo_path.is_file():
        st.image(str(large_logo_path), width=100)
    elif small_logo_path_header.exists() and small_logo_path_header.is_file():
        st.image(str(small_logo_path_header), width=80) # Fallback to small logo
    else:
        logger.warning(f"App logos not found. Large: '{large_logo_path}', Small: '{small_logo_path_header}'. Using placeholder.")
        st.markdown("### 🌍", unsafe_allow_html=True)

with header_cols_main_app[1]:
    st.title(settings.APP_NAME)
    st.subheader("Transforming Data into Lifesaving Action at the Edge")
st.divider()

# --- Welcome & System Description ---
st.markdown(f"""
    ## Welcome to the {settings.APP_NAME} Demonstrator
    
    Sentinel is an **edge-first health intelligence system** designed for **maximum clinical and 
    operational actionability** in resource-limited, high-risk environments. It aims to convert 
    diverse data sources into life-saving, workflow-integrated decisions, even with 
    **minimal or intermittent internet connectivity.**
""")

st.markdown("#### Core Design Principles:")
core_principles_data_app = [
    ("📶 **Offline-First Operations**", "On-device Edge AI ensures critical functionality without continuous connectivity."),
    ("🎯 **Action-Oriented Intelligence**", "Insights aim to trigger clear, targeted responses relevant to frontline workflows."),
    ("🧑‍🤝‍🧑 **Human-Centered Design**", "Interfaces optimized for low-literacy, high-stress users, prioritizing immediate understanding."),
    ("🔗 **Resilience & Scalability**", "Modular design for scaling from personal devices to regional views with robust data sync.")
]
num_cols_principles = min(len(core_principles_data_app), 2) # Max 2 columns
if num_cols_principles > 0:
    cols_principles_ui_app = st.columns(num_cols_principles)
    for idx_principle_app, (title_p, desc_p) in enumerate(core_principles_data_app):
        with cols_principles_ui_app[idx_principle_app % num_cols_principles]:
            st.markdown(f"##### {title_p}")
            st.markdown(f"<small>{desc_p}</small>", unsafe_allow_html=True)
            st.markdown("<div style='margin-bottom: 1rem;'></div>", unsafe_allow_html=True)

st.markdown("---")
st.markdown(
    "👈 **Navigate via the sidebar** to explore simulated web dashboards for various operational tiers. "
    "These views represent perspectives of Supervisors, Clinic Managers, or District Health Officers (DHOs). "
    "Frontline workers (e.g., CHWs) would use dedicated native applications on their Personal Edge Devices (PEDs)."
)
st.info(
    "💡 **Note:** This web application serves as a high-level demonstrator of the Sentinel system's "
    "data processing capabilities and the types of aggregated views available to management and strategic personnel."
)
st.divider()

# --- Simulated Role-Specific Views Navigation ---
st.header("Explore Simulated Role-Specific Dashboards")
st.caption("These views demonstrate information available at higher tiers. Frontline workers use dedicated PED apps.")

pages_dir = _project_root_dir / "pages" # Define pages directory path
role_navigation_config_app = [
    {"title": "🧑‍⚕️ CHW Operations & Field Support", "desc": "Supervisor view: team performance, CHW support, local epi signals.", "page_filename": "chw_dashboard.py"},
    {"title": "🏥 Clinic Operations & Environment", "desc": "Clinic Manager view: service efficiency, care quality, resources, facility safety.", "page_filename": "clinic_dashboard.py"},
    {"title": "🗺️ District Health Strategic Overview", "desc": "DHO view: population health, resource allocation, strategic interventions.", "page_filename": "district_dashboard.py"},
    {"title": "📊 Population Health Analytics", "desc": "Analyst view: in-depth epi/systems analysis, SDOH impacts, equity.", "page_filename": "population_dashboard.py"},
]

num_nav_cols_main = min(len(role_navigation_config_app), 2)
if num_nav_cols_main > 0:
    nav_cols_ui_app = st.columns(num_nav_cols_main)
    current_col_idx_nav = 0
    for nav_item_main in role_navigation_config_app:
        page_file = pages_dir / nav_item_main["page_filename"]
        if not page_file.exists():
            logger.warning(f"Navigation page file not found: {page_file}")
            continue
        
        # For st.page_link, path is relative to the directory of the main script (app.py)
        # So, if app.py is at project root, path is "pages/filename.py"
        page_link_str_app = f"pages/{nav_item_main['page_filename']}"
        
        with nav_cols_ui_app[current_col_idx_nav % num_nav_cols_main]:
            try: # Use st.container with border if available
                with st.container(border=True):
                    st.subheader(nav_item_main["title"])
                    st.markdown(f"<small>{nav_item_main['desc']}</small>", unsafe_allow_html=True)
                    st.page_link(page_link_str_app, label=f"Explore {nav_item_main['title'].split('(')[0].strip()}", icon="➡️", use_container_width=True)
            except TypeError: # Fallback if border argument not supported
                st.subheader(nav_item_main["title"])
                st.markdown(f"<small>{nav_item_main['desc']}</small>", unsafe_allow_html=True)
                st.page_link(page_link_str_app, label=f"Explore {nav_item_main['title'].split('(')[0].strip()}", icon="➡️")
            st.markdown("<div style='margin-bottom: 0.5rem;'></div>", unsafe_allow_html=True) # Spacer
        current_col_idx_nav += 1
st.divider()

# --- Key Capabilities Section ---
st.header(f"{settings.APP_NAME} - Key Capabilities Reimagined")
capabilities_data_app = [
    ("🛡️ Frontline Worker Safety & Support", "Real-time vitals/environmental monitoring, fatigue detection, safety nudges on PEDs."),
    ("🌍 Offline-First Edge AI", "On-device intelligence for alerts, prioritization, guidance without continuous connectivity."),
    ("⚡ Actionable, Contextual Insights", "Raw data to clear, role-specific recommendations integrated into field workflows."),
    ("🤝 Human-Centered & Accessible UX", "Pictogram UIs, voice/tap commands, local language support for low-literacy, high-stress users on PEDs."),
    ("📡 Resilient Data Synchronization", "Flexible data sharing (Bluetooth, QR, SD card, SMS, opportunistic IP) across devices/tiers."),
    ("🌱 Scalable & Interoperable Architecture", "Modular design (personal to national), FHIR/HL7 considerations for integration.")
]
num_cap_cols_main = min(len(capabilities_data_app), 3) # Max 3 columns for capabilities
if num_cap_cols_main > 0:
    cap_cols_ui_app = st.columns(num_cap_cols_main)
    current_col_idx_cap = 0
    for cap_title_item, cap_desc_item in capabilities_data_app:
        with cap_cols_ui_app[current_col_idx_cap % num_cap_cols_main]:
            st.markdown(f"##### {cap_title_item}")
            st.markdown(f"<small>{cap_desc_item}</small>", unsafe_allow_html=True)
            st.markdown("<div style='margin-bottom: 1.2rem;'></div>", unsafe_allow_html=True) # Spacer
        current_col_idx_cap += 1
st.divider()

# --- Link to the Glossary Page ---
glossary_filename_app = "glossary_page.py"
glossary_page_full_path_app = pages_dir / glossary_filename_app
glossary_page_link_str_app = f"pages/{glossary_filename_app}"

if glossary_page_full_path_app.exists():
    with st.expander("📜 **System Glossary & Terminology**", expanded=False):
        st.markdown(f"Explore definitions for key terms, metrics, and system components specific to {settings.APP_NAME}.")
        st.page_link(glossary_page_link_str_app, label="Go to Glossary", icon="📚")
else:
    logger.warning(f"Glossary page file not found at expected location: {glossary_page_full_path_app}")
    st.caption("Glossary page is currently unavailable.")

# --- Sidebar Content ---
st.sidebar.header(f"{settings.APP_NAME} v{settings.APP_VERSION}")
st.sidebar.divider()
st.sidebar.markdown("#### About This Demonstrator:")
st.sidebar.info(
    "This web app simulates higher-level dashboards. "
    "Frontline worker interaction occurs on dedicated Personal Edge Devices (PEDs)."
)
st.sidebar.divider()
if glossary_page_full_path_app.exists():
    st.sidebar.page_link(glossary_page_link_str_app, label="📜 System Glossary", icon="📚")
st.sidebar.divider()
st.sidebar.markdown(f"**{settings.ORGANIZATION_NAME}**")
st.sidebar.markdown(f"Support: [{settings.SUPPORT_CONTACT_INFO}](mailto:{settings.SUPPORT_CONTACT_INFO})")
st.sidebar.caption(settings.APP_FOOTER_TEXT)

logger.info(f"{settings.APP_NAME} (v{settings.APP_VERSION}) - System Overview page loaded successfully.")
