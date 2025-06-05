# sentinel_project_root/app.py
# Main Streamlit application for Sentinel Health Co-Pilot Demonstrator.

import streamlit as st
import sys
import logging
from pathlib import Path

# --- Robust Path Setup ---
# Resolve the directory of this app.py file
_current_app_file_dir = Path(__file__).parent.resolve()
# Assume project root is one level up from where app.py is (e.g., sentinel_project_root/app.py)
# If app.py is in sentinel_project_root/main/app.py, then parent.parent
# Adjust this if your app.py is located differently relative to the project root.
_project_root_dir = _current_app_file_dir # Assuming app.py is in sentinel_project_root

if str(_project_root_dir) not in sys.path:
    sys.path.insert(0, str(_project_root_dir))
    print(f"INFO: Added project root to sys.path: {_project_root_dir}", file=sys.stderr)

# --- Import Settings ---
try:
    from config import settings
    print(f"INFO: Successfully imported config.settings. APP_NAME: {settings.APP_NAME}", file=sys.stderr)
except ImportError as e_cfg_app:
    print(f"FATAL: Failed to import config.settings in app.py: {e_cfg_app}", file=sys.stderr)
    print(f"PYTHONPATH for app.py: {sys.path}", file=sys.stderr)
    print(f"Attempted project root: {_project_root_dir}", file=sys.stderr)
    sys.exit(1) # Critical error, cannot proceed
except Exception as e_generic_cfg:
    print(f"FATAL: Generic error during config.settings import in app.py: {e_generic_cfg}", file=sys.stderr)
    sys.exit(1)


# --- Global Logging Configuration ---
# Use log level from settings.py, ensure it's valid
valid_log_levels_app = {"DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"}
log_level_app_str = str(settings.LOG_LEVEL).upper()
if log_level_app_str not in valid_log_levels_app:
    print(f"WARN: Invalid LOG_LEVEL '{log_level_app_str}' in settings. Using INFO for app logging.", file=sys.stderr)
    log_level_app_str = "INFO"

# Configure root logger for the application
# This will affect all loggers unless they are specifically configured otherwise.
logging.basicConfig(
    level=getattr(logging, log_level_app_str, logging.INFO),
    format=settings.LOG_FORMAT,
    datefmt=settings.LOG_DATE_FORMAT,
    handlers=[logging.StreamHandler(sys.stdout)], # Output to stdout for Streamlit compatibility
    force=True # Override any previous basicConfig by Streamlit or other modules
)
logger = logging.getLogger(__name__) # Logger for this app.py file

# --- Streamlit Version Check (Optional but good practice) ---
try:
    import streamlit
    # Example: Check for a minimum version if specific features are used
    major, minor, patch = map(int, streamlit.__version__.split('.'))
    if not (major >= 1 and minor >= 30): # Require Streamlit 1.30.0 or newer
        warn_msg_st_ver = f"Streamlit version {streamlit.__version__} is older than recommended (1.30.0+). Some UI features might not work as expected."
        logger.warning(warn_msg_st_ver)
        st.sidebar.warning(warn_msg_st_ver) # Show warning in UI if possible
except ImportError:
    logger.critical("Streamlit library not found. Cannot run the application.")
    sys.exit("Streamlit library not found. Please install it.")


# --- Page Configuration (Single Call, at the very top of script execution for Streamlit) ---
# Ensure page_icon path is absolute and exists
page_icon_path_obj = Path(settings.APP_LOGO_SMALL_PATH)
if not page_icon_path_obj.is_absolute(): # If relative, make it absolute from project root
    page_icon_path_obj = (PROJECT_ROOT_DIR / settings.APP_LOGO_SMALL_PATH).resolve()

final_page_icon: str
if page_icon_path_obj.exists() and page_icon_path_obj.is_file():
    final_page_icon = str(page_icon_path_obj)
else:
    logger.warning(f"Page icon not found at '{page_icon_path_obj}'. Using fallback emoji '🌍'.")
    final_page_icon = "🌍"

st.set_page_config(
    page_title=f"{settings.APP_NAME} - Overview", # Page title for the main page
    page_icon=final_page_icon,
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
except ImportError as e_theme_imp:
    logger.error(f"Failed to import set_sentinel_plotly_theme: {e_theme_imp}", exc_info=True)
    st.error("Critical error: Plotting theme could not be applied. Visualizations may be affected.")
except Exception as e_theme_apply:
    logger.error(f"Error applying Plotly theme in app.py: {e_theme_apply}", exc_info=True)
    st.error("Error applying visualization theme. Charts might not display correctly.")

# --- Global CSS Loading ---
# Use @st.cache_resource for CSS loading as it's a global resource.
@st.cache_resource
def load_global_css_styles(css_file_path_str: str):
    css_file_path_obj = Path(css_file_path_str)
    if not css_file_path_obj.is_absolute(): # If relative path from settings
        css_file_path_obj = (PROJECT_ROOT_DIR / css_file_path_str).resolve()
    
    if css_file_path_obj.exists() and css_file_path_obj.is_file():
        try:
            with open(css_file_path_obj, "r", encoding="utf-8") as f_css:
                st.markdown(f'<style>{f_css.read()}</style>', unsafe_allow_html=True)
            logger.debug(f"Global CSS styles loaded successfully from: {css_file_path_obj}")
        except Exception as e_css_load:
            logger.error(f"Error reading or applying CSS file {css_file_path_obj}: {e_css_load}", exc_info=True)
            st.error("Critical error: Application styles could not be loaded. UI may be broken.")
    else:
        logger.warning(f"Global CSS file not found at resolved path: {css_file_path_obj}")
        st.warning("Application stylesheet missing. UI might not render as intended.")

if settings.STYLE_CSS_PATH_WEB:
    load_global_css_styles(settings.STYLE_CSS_PATH_WEB)
else:
    logger.debug("No global CSS path configured in settings. Skipping custom CSS load.")


# --- Main Application Header ---
# Using columns for layout flexibility
header_cols_app = st.columns([0.12, 0.88]) # Adjust ratio for logo and title
with header_cols_app[0]:
    # Resolve logo path relative to project root if it's not absolute
    large_logo_path_obj = Path(settings.APP_LOGO_LARGE_PATH)
    if not large_logo_path_obj.is_absolute():
        large_logo_path_obj = (PROJECT_ROOT_DIR / settings.APP_LOGO_LARGE_PATH).resolve()
    
    small_logo_path_obj_header = Path(settings.APP_LOGO_SMALL_PATH) # Already string in settings
    if not small_logo_path_obj_header.is_absolute():
         small_logo_path_obj_header = (PROJECT_ROOT_DIR / settings.APP_LOGO_SMALL_PATH).resolve()


    if large_logo_path_obj.exists() and large_logo_path_obj.is_file():
        st.image(str(large_logo_path_obj), width=100) # Increased width slightly
    elif small_logo_path_obj_header.exists() and small_logo_path_obj_header.is_file(): # Fallback to small logo
        st.image(str(small_logo_path_obj_header), width=80)
    else:
        logger.warning(f"App logos not found (Large: {large_logo_path_obj}, Small: {small_logo_path_obj_header}). Using placeholder.")
        st.markdown("### 🌍", unsafe_allow_html=True) # Placeholder if no logo

with header_cols_app[1]:
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
# More robust column creation for varying numbers of principles
core_principles_list = [
    ("📶 **Offline-First Operations**", "On-device Edge AI ensures critical functionality without continuous connectivity."),
    ("🎯 **Action-Oriented Intelligence**", "Insights aim to trigger clear, targeted responses relevant to frontline workflows."),
    ("🧑‍🤝‍🧑 **Human-Centered Design**", "Interfaces optimized for low-literacy, high-stress users, prioritizing immediate understanding."),
    ("🔗 **Resilience & Scalability**", "Modular design for scaling from personal devices to regional views with robust data sync.")
]
num_principles_cols = min(len(core_principles_list), 2) # Max 2 columns for principles
if num_principles_cols > 0:
    principle_cols_ui = st.columns(num_principles_cols)
    for idx_principle, (title_principle, desc_principle) in enumerate(core_principles_list):
        with principle_cols_ui[idx_principle % num_principles_cols]:
            st.markdown(f"##### {title_principle}")
            st.markdown(f"<small>{desc_principle}</small>", unsafe_allow_html=True)
            st.markdown("<div style='margin-bottom: 1rem;'></div>", unsafe_allow_html=True) # Spacer

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

# Ensure paths to pages are correctly formed relative to _project_root_dir
# Pages are expected to be in a 'pages' subdirectory of the project root.
pages_dir_path = _project_root_dir / "pages"
role_nav_cfg = [
    {"title": "🧑‍⚕️ CHW Operations & Field Support", "desc": "Supervisor view: team performance, CHW support, local epi signals.", "page_filename": "chw_dashboard.py"},
    {"title": "🏥 Clinic Operations & Environment", "desc": "Clinic Manager view: service efficiency, care quality, resources, facility safety.", "page_filename": "clinic_dashboard.py"},
    {"title": "🗺️ District Health Strategic Overview", "desc": "DHO view: population health, resource allocation, strategic interventions.", "page_filename": "district_dashboard.py"},
    {"title": "📊 Population Health Analytics", "desc": "Analyst view: in-depth epi/systems analysis, SDOH impacts, equity.", "page_filename": "population_dashboard.py"},
]

num_nav_cols = min(len(role_nav_cfg), 2) # Max 2 columns for navigation items
if num_nav_cols > 0:
    nav_cols_ui = st.columns(num_nav_cols)
    col_idx_nav = 0
    for nav_item_cfg in role_nav_cfg:
        page_full_path = pages_dir_path / nav_item_cfg["page_filename"]
        if not page_full_path.exists():
            logger.warning(f"Navigation page file not found: {page_full_path}")
            continue # Skip if page file doesn't exist
        
        with nav_cols_ui[col_idx_nav % num_nav_cols]:
            with st.container(border=True): # Use border for visual grouping
                st.subheader(nav_item_cfg["title"])
                st.markdown(f"<small>{nav_item_cfg['desc']}</small>", unsafe_allow_html=True)
                # Construct page path for st.page_link relative to the directory containing app.py
                # Streamlit's page_link expects paths relative to the main app script's directory,
                # or an absolute path within the "pages" folder structure.
                # Example: "pages/chw_dashboard.py" if app.py is in project root.
                page_link_path_str = f"pages/{nav_item_cfg['page_filename']}"
                st.page_link(page_link_path_str, label=f"Explore {nav_item_cfg['title'].split('(')[0].strip()}", icon="➡️", use_container_width=True)
                st.markdown("<div style='margin-bottom: 0.5rem;'></div>", unsafe_allow_html=True)
        col_idx_nav += 1
st.divider()

# --- Key Capabilities Section ---
st.header(f"{settings.APP_NAME} - Key Capabilities Reimagined")
key_capabilities_list = [
    ("🛡️ Frontline Worker Safety & Support", "Real-time vitals/environmental monitoring, fatigue detection, safety nudges on PEDs."),
    ("🌍 Offline-First Edge AI", "On-device intelligence for alerts, prioritization, guidance without continuous connectivity."),
    ("⚡ Actionable, Contextual Insights", "Raw data to clear, role-specific recommendations integrated into field workflows."),
    ("🤝 Human-Centered & Accessible UX", "Pictogram UIs, voice/tap commands, local language support for low-literacy, high-stress users on PEDs."),
    ("📡 Resilient Data Synchronization", "Flexible data sharing (Bluetooth, QR, SD card, SMS, opportunistic IP) across devices/tiers."),
    ("🌱 Scalable & Interoperable Architecture", "Modular design (personal to national), FHIR/HL7 considerations for integration.")
]
num_cap_cols = min(len(key_capabilities_list), 3) # Max 3 columns for capabilities
if num_cap_cols > 0:
    cap_cols_ui = st.columns(num_cap_cols)
    col_idx_cap = 0
    for cap_title_val, cap_desc_val in key_capabilities_list:
        with cap_cols_ui[col_idx_cap % num_cap_cols]:
            st.markdown(f"##### {cap_title_val}")
            st.markdown(f"<small>{cap_desc_val}</small>", unsafe_allow_html=True)
            st.markdown("<div style='margin-bottom: 1.2rem;'></div>", unsafe_allow_html=True)
        col_idx_cap += 1
st.divider()

# --- Link to the Glossary Page ---
glossary_page_filename = "glossary_page.py" # Filename in 'pages' directory
glossary_page_link_path = f"pages/{glossary_page_filename}"
if (pages_dir_path / glossary_page_filename).exists():
    with st.expander("📜 **System Glossary & Terminology**", expanded=False):
        st.markdown(f"Explore definitions for key terms, metrics, and system components specific to {settings.APP_NAME}.")
        st.page_link(glossary_page_link_path, label="Go to Glossary", icon="📚")
else:
    logger.warning(f"Glossary page file not found at expected location: {pages_dir_path / glossary_page_filename}")
    st.caption("Glossary page is currently unavailable.")

# --- Sidebar Content (Static, so less prone to errors) ---
st.sidebar.header(f"{settings.APP_NAME} v{settings.APP_VERSION}") # Add version to sidebar header
st.sidebar.divider()
st.sidebar.markdown("#### About This Demonstrator:")
st.sidebar.info(
    "This web app simulates higher-level dashboards. "
    "Frontline worker interaction occurs on dedicated Personal Edge Devices (PEDs)."
)
st.sidebar.divider()
if (pages_dir_path / glossary_page_filename).exists(): # Check again for link in sidebar
    st.sidebar.page_link(glossary_page_link_path, label="📜 System Glossary", icon="📚")
st.sidebar.divider()
st.sidebar.markdown(f"**{settings.ORGANIZATION_NAME}**")
st.sidebar.markdown(f"Support: [{settings.SUPPORT_CONTACT_INFO}](mailto:{settings.SUPPORT_CONTACT_INFO})")
st.sidebar.caption(settings.APP_FOOTER_TEXT)

logger.info(f"{settings.APP_NAME} (v{settings.APP_VERSION}) - System Overview page loaded successfully.")
