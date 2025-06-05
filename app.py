# sentinel_project_root/app.py
# Main Streamlit application for Sentinel Health Co-Pilot Demonstrator.

import streamlit as st
import sys
import logging
from pathlib import Path

# --- Robust Path Setup ---
_current_app_file_dir = Path(__file__).parent.resolve()
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
    print(f"Calculated project root: {_project_root_dir}", file=sys.stderr)
    sys.exit(1)
except Exception as e_generic_cfg:
    print(f"FATAL: Generic error during config.settings import in app.py: {e_generic_cfg}", file=sys.stderr)
    sys.exit(1)

# --- Global Logging Configuration ---
valid_log_levels_app = {"DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"}
log_level_app_str = str(settings.LOG_LEVEL).upper()
if log_level_app_str not in valid_log_levels_app:
    print(f"WARN: Invalid LOG_LEVEL '{log_level_app_str}' in settings. Using INFO for app logging.", file=sys.stderr)
    log_level_app_str = "INFO"
logging.basicConfig(
    level=getattr(logging, log_level_app_str, logging.INFO),
    format=settings.LOG_FORMAT, datefmt=settings.LOG_DATE_FORMAT,
    handlers=[logging.StreamHandler(sys.stdout)], force=True
)
logger = logging.getLogger(__name__)

# --- Streamlit Version Check ---
try:
    import streamlit
    major, minor, patch_str = streamlit.__version__.split('.')
    patch = int(patch_str.split('-')[0])
    if not (int(major) >= 1 and int(minor) >= 30):
        warn_msg_st_ver = f"Streamlit version {streamlit.__version__} older than 1.30.0+. UI features might not work."
        logger.warning(warn_msg_st_ver)
        try: st.sidebar.warning(warn_msg_st_ver)
        except: pass
except ImportError:
    logger.critical("Streamlit library not found. Cannot run application.")
    sys.exit("Streamlit library not found. Please install it.")

# --- Page Configuration ---
page_icon_path_obj = Path(settings.APP_LOGO_SMALL_PATH)
if not page_icon_path_obj.is_absolute(): page_icon_path_obj = (_project_root_dir / settings.APP_LOGO_SMALL_PATH).resolve()
final_page_icon_str: str
if page_icon_path_obj.exists() and page_icon_path_obj.is_file(): final_page_icon_str = str(page_icon_path_obj)
else: logger.warning(f"Page icon not found at '{page_icon_path_obj}'. Using '🌍'."); final_page_icon_str = "🌍"

st.set_page_config(
    page_title=f"{settings.APP_NAME} - System Overview", page_icon=final_page_icon_str,
    layout="wide", initial_sidebar_state="expanded",
    menu_items={
        "Get Help": f"mailto:{settings.SUPPORT_CONTACT_INFO}?subject=Help Request - {settings.APP_NAME}",
        "Report a bug": f"mailto:{settings.SUPPORT_CONTACT_INFO}?subject=Bug Report - {settings.APP_NAME} v{settings.APP_VERSION}",
        "About": f"### {settings.APP_NAME} (v{settings.APP_VERSION})\n{settings.APP_FOOTER_TEXT}\n\nEdge-First Health Intelligence Co-Pilot for LMICs."
    }
)

# --- Apply Plotly Theme Globally ---
try:
    from visualization.plots import set_sentinel_plotly_theme
    set_sentinel_plotly_theme()
    logger.debug("Sentinel Plotly theme applied globally from app.py.")
except Exception as e_theme_app: logger.error(f"Error applying Plotly theme in app.py: {e_theme_app}", exc_info=True); st.error("Error applying visualization theme.")

# --- Global CSS Loading ---
@st.cache_resource
def load_global_css_styles(css_file_path_str: str):
    css_file_path_obj = Path(css_file_path_str)
    if not css_file_path_obj.is_absolute(): css_file_path_obj = (_project_root_dir / css_file_path_str).resolve()
    if css_file_path_obj.exists() and css_file_path_obj.is_file():
        try:
            with open(css_file_path_obj, "r", encoding="utf-8") as f_css: st.markdown(f'<style>{f_css.read()}</style>', unsafe_allow_html=True)
            logger.debug(f"Global CSS styles loaded from: {css_file_path_obj}")
        except Exception as e_css_app: logger.error(f"Error reading/applying CSS {css_file_path_obj}: {e_css_app}", exc_info=True); st.error("Critical error: Application styles failed to load.")
    else: logger.warning(f"Global CSS file not found: {css_file_path_obj}"); st.warning("Application stylesheet missing.")
if settings.STYLE_CSS_PATH_WEB: load_global_css_styles(settings.STYLE_CSS_PATH_WEB)
else: logger.debug("No global CSS path in settings. Skipping custom CSS.")

# --- Main Application Header ---
header_cols_app_main = st.columns([0.12, 0.88])
with header_cols_app_main[0]:
    large_logo_path_main = Path(settings.APP_LOGO_LARGE_PATH)
    if not large_logo_path_main.is_absolute(): large_logo_path_main = (_project_root_dir / settings.APP_LOGO_LARGE_PATH).resolve()
    small_logo_path_main_header = Path(settings.APP_LOGO_SMALL_PATH)
    if not small_logo_path_main_header.is_absolute(): small_logo_path_main_header = (_project_root_dir / settings.APP_LOGO_SMALL_PATH).resolve()
    if large_logo_path_main.exists() and large_logo_path_main.is_file(): st.image(str(large_logo_path_main), width=100)
    elif small_logo_path_main_header.exists() and small_logo_path_main_header.is_file(): st.image(str(small_logo_path_main_header), width=80)
    else: logger.warning(f"App logos not found. Large: '{large_logo_path_main}', Small: '{small_logo_path_main_header}'."); st.markdown("### 🌍", unsafe_allow_html=True)
with header_cols_app_main[1]:
    st.title(settings.APP_NAME)
    st.subheader("Transforming Data into Lifesaving Action at the Edge")
st.divider()

# --- INTEGRATED CONTENT: Welcome & System Description, Navigation ---
st.markdown(f"""
    ## Welcome to the {settings.APP_NAME} Demonstrator
    
    Sentinel is an **edge-first health intelligence system** designed for **maximum clinical and 
    operational actionability** in resource-limited, high-risk environments. It aims to convert 
    diverse data sources into life-saving, workflow-integrated decisions, even with 
    **minimal or intermittent internet connectivity.**
""")
st.markdown("#### Core Design Principles:")
core_principles_data_integrated = [
    ("📶 **Offline-First Operations**", "On-device Edge AI ensures critical functionality without continuous connectivity."),
    ("🎯 **Action-Oriented Intelligence**", "Insights aim to trigger clear, targeted responses relevant to frontline workflows."),
    ("🧑‍🤝‍🧑 **Human-Centered Design**", "Interfaces optimized for low-literacy, high-stress users, prioritizing immediate understanding."),
    ("🔗 **Resilience & Scalability**", "Modular design for scaling from personal devices to regional views with robust data sync.")
]
num_cols_principles_integrated = min(len(core_principles_data_integrated), 2)
if num_cols_principles_integrated > 0:
    cols_principles_ui_integrated = st.columns(num_cols_principles_integrated)
    for idx_integrated, (title_int, desc_int) in enumerate(core_principles_data_integrated):
        with cols_principles_ui_integrated[idx_integrated % num_cols_principles_integrated]:
            st.markdown(f"##### {title_int}")
            st.markdown(f"<small>{desc_int}</small>", unsafe_allow_html=True)
            st.markdown("<div style='margin-bottom: 1rem;'></div>", unsafe_allow_html=True)

st.markdown("---")
st.markdown("""
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

st.header("Explore Simulated Role-Specific Dashboards")
st.caption("These views demonstrate the information available at higher tiers (Facility/Regional Nodes).")

pages_base_dir = _project_root_dir / "pages" # Define base 'pages' directory

# Corrected page configuration with new descriptions
# Order: CHW, Clinic, District, Population (Glossary will be handled separately for sidebar)
role_navigation_config_updated = [
    {"title": "🧑‍⚕️ CHW Operations Summary & Field Support View (Supervisor/Hub Level)", 
     "desc": "This view simulates how a CHW Supervisor or a Hub coordinator might access summarized data from CHW Personal Edge Devices (PEDs).<br><br><b>Focus (Tier 1-2):</b> Team performance monitoring, targeted support for CHWs, localized outbreak signal detection based on aggregated CHW reports.<br><b>Key Data Points:</b> CHW activity summaries (visits, tasks completed), patient alert escalations, critical supply needs for CHW kits, early epidemiological signals from specific zones.<br><b>Objective:</b> Enable supervisors to manage CHW teams effectively, provide timely support, identify emerging health issues quickly, and coordinate local responses. The CHW's primary tool is their offline-first native app on their PED, providing real-time alerts & task management.", 
     "page_filename": "chw_dashboard.py", "icon": "🧑‍⚕️"},
    {"title": "🏥 Clinic Operations & Environmental Safety View (Facility Node Level)", 
     "desc": "Simulates a dashboard for Clinic Managers at a Facility Node (Tier 2), providing insights into service efficiency, care quality, resource management, and environmental conditions.<br><br><b>Focus (Tier 2):</b> Optimizing clinic workflows, ensuring quality patient care, managing supplies and testing backlogs, monitoring clinic environment for safety and infection control.<br><b>Key Data Points:</b> Clinic performance KPIs (e.g., test TAT, patient throughput), supply stock forecasts, IoT sensor data summaries (CO2, PM2.5, occupancy), clinic-level epidemiological trends, flagged patient cases for review.<br><b>Objective:</b> Enhance operational efficiency, support clinical decision-making, maintain resource availability, and ensure a safe clinic environment.", 
     "page_filename": "clinic_dashboard.py", "icon": "🏥"},
    {"title": "🗺️ District Health Strategic Overview (DHO at Facility/Regional Node Level)", 
     "desc": "Presents a strategic dashboard for District Health Officers (DHOs), typically accessed at a Facility Node (Tier 2) or a Regional/Cloud Node (Tier 3).<br><br><b>Focus (Tier 2-3):</b> Population health insights, resource allocation across zones, monitoring environmental well-being, and planning targeted interventions.<br><b>Key Data Points:</b> District-wide health KPIs, interactive maps for zonal comparisons (risk, disease burden, resources), trend analyses, intervention planning tools based on aggregated data.<br><b>Objective:</b> Support evidence-based strategic planning, public health interventions, program monitoring, and policy development for the district.", 
     "page_filename": "district_dashboard.py", "icon": "🗺️"},
    {"title": "📊 Population Health Analytics Deep Dive (Epidemiologist/Analyst View - Tier 3)", 
     "desc": "A view designed for detailed epidemiological and health systems analysis, typically used by analysts or program managers at a Regional/Cloud Node (Tier 3) with access to more comprehensive, aggregated datasets.<br><br><b>Focus (Tier 3):</b> In-depth analysis of demographic patterns, SDOH impacts, clinical trends, health system performance, and equity across broader populations.<br><b>Key Data Points:</b> Stratified disease burden, AI risk distributions by various factors, aggregated test positivity trends, comorbidity analysis, referral pathway performance, health equity metrics.<br><b>Objective:</b> Provide robust analytical capabilities to understand population health dynamics, evaluate interventions, identify areas for research, and inform large-scale public health strategy.", 
     "page_filename": "population_dashboard.py", "icon": "📊"},
]

num_nav_cols_updated = min(len(role_navigation_config_updated), 2)
if num_nav_cols_updated > 0:
    nav_cols_ui_updated = st.columns(num_nav_cols_updated)
    current_col_idx_nav_updated = 0
    for nav_item_updated in role_navigation_config_updated:
        # Corrected path for st.page_link: it should be relative to the `pages` directory itself.
        # If app.py is at project_root, and pages are in project_root/pages/
        # then the link is "pages/filename.py"
        page_link_path_str_updated = f"pages/{nav_item_updated['page_filename']}"
        
        # Verify if the actual file exists to prevent broken links
        physical_page_path = pages_base_dir / nav_item_updated["page_filename"]
        if not physical_page_path.exists():
            logger.warning(f"Navigation page file for '{nav_item_updated['title']}' not found at: {physical_page_path}")
            continue # Skip creating a link for a non-existent page

        with nav_cols_ui_updated[current_col_idx_nav_updated % num_nav_cols_updated]:
            try:
                with st.container(border=True):
                    st.subheader(f"{nav_item_updated['icon']} {nav_item_updated['title']}") # Add icon to subheader
                    st.markdown(f"<small>{nav_item_updated['desc']}</small>", unsafe_allow_html=True)
                    st.page_link(page_link_str_updated, label=f"Explore this View", icon="➡️", use_container_width=True) # Simplified label
            except TypeError: # Fallback for older Streamlit versions
                st.subheader(f"{nav_item_updated['icon']} {nav_item_updated['title']}")
                st.markdown(f"<small>{nav_item_updated['desc']}</small>", unsafe_allow_html=True)
                st.page_link(page_link_str_updated, label=f"Explore this View", icon="➡️")
            st.markdown("<div style='margin-bottom: 0.5rem;'></div>", unsafe_allow_html=True)
        current_col_idx_nav_updated += 1
st.divider()

# --- Key Capabilities Section (remains similar) ---
st.header(f"{settings.APP_NAME} - Key Capabilities Reimagined")
# ... (content for capabilities remains the same as your previous version, no changes needed here) ...
capabilities_data_app_main = [
    ("🛡️ Frontline Worker Safety & Support", "Real-time vitals/environmental monitoring, fatigue detection, safety nudges on PEDs."),
    ("🌍 Offline-First Edge AI", "On-device intelligence for alerts, prioritization, guidance without continuous connectivity."),
    ("⚡ Actionable, Contextual Insights", "Raw data to clear, role-specific recommendations integrated into field workflows."),
    ("🤝 Human-Centered & Accessible UX", "Pictogram UIs, voice/tap commands, local language support for low-literacy, high-stress users on PEDs."),
    ("📡 Resilient Data Synchronization", "Flexible data sharing (Bluetooth, QR, SD card, SMS, opportunistic IP) across devices/tiers."),
    ("🌱 Scalable & Interoperable Architecture", "Modular design (personal to national), FHIR/HL7 considerations for integration.")
]
num_cap_cols_main_app = min(len(capabilities_data_app_main), 3)
if num_cap_cols_main_app > 0:
    cap_cols_ui_main_app = st.columns(num_cap_cols_main_app)
    current_col_idx_cap_main = 0
    for cap_title_item_main, cap_desc_item_main in capabilities_data_app_main:
        with cap_cols_ui_main_app[current_col_idx_cap_main % num_cap_cols_main_app]:
            st.markdown(f"##### {cap_title_item_main}")
            st.markdown(f"<small>{cap_desc_item_main}</small>", unsafe_allow_html=True)
            st.markdown("<div style='margin-bottom: 1.2rem;'></div>", unsafe_allow_html=True)
        current_col_idx_cap_main += 1
st.divider()

# --- Sidebar Content (Order of pages for st.Page needs to be managed by filename prefix if not using st.navigation) ---
# Streamlit's default multi-page app feature sorts pages in the sidebar alphabetically by filename.
# To control order, prefix filenames in the `pages` directory with numbers (e.g., `1_chw_dashboard.py`, `2_clinic_dashboard.py`, etc.)
# The Glossary page link can be added manually at the end of the sidebar.

st.sidebar.header(f"{settings.APP_NAME} v{settings.APP_VERSION}")
st.sidebar.divider()
st.sidebar.markdown("#### About This Demonstrator:")
st.sidebar.info(
    "This web app simulates higher-level dashboards. "
    "Frontline worker interaction occurs on dedicated Personal Edge Devices (PEDs)."
)
st.sidebar.divider()

# Manually add Glossary link at the end of the sidebar
glossary_page_filename_app_final = "glossary_page.py"
glossary_page_link_str_app_final = f"pages/{glossary_page_filename_app_final}"
glossary_physical_path_app_final = pages_base_dir / glossary_page_filename_app_final

if glossary_physical_path_app_final.exists():
    # To place it last using st.Page, we'd need a more complex setup or rely on filename sorting
    # A simple st.page_link here will add it based on where this code runs in app.py flow,
    # which might not be last if other st.sidebar.page_link calls are made by Streamlit for pages dir.
    # For a guaranteed last position, it's often better to structure the "pages" directory with numbered prefixes.
    # However, adding it explicitly here is fine for now.
    st.sidebar.page_link(glossary_page_link_str_app_final, label="📜 System Glossary", icon="📚")
else:
    logger.warning(f"Glossary page file not found for sidebar link: {glossary_physical_path_app_final}")

st.sidebar.divider()
st.sidebar.markdown(f"**{settings.ORGANIZATION_NAME}**")
st.sidebar.markdown(f"Support: [{settings.SUPPORT_CONTACT_INFO}](mailto:{settings.SUPPORT_CONTACT_INFO})")
st.sidebar.caption(settings.APP_FOOTER_TEXT)

logger.info(f"{settings.APP_NAME} (v{settings.APP_VERSION}) - System Overview page loaded successfully.")
