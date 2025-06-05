# sentinel_project_root/app.py
# Main Streamlit application for Sentinel Health Co-Pilot Demonstrator.

import streamlit as st
import sys
import logging
from pathlib import Path
import html 

# --- Robust Path Setup ---
# app.py is in the project root (e.g., ssentinel_project_root/app.py)
# Path(__file__) is the path to this app.py file.
# .parent gives the directory containing app.py, which IS the project root.
_project_root_dir = Path(__file__).resolve().parent 

# This is the crucial part for imports.
# It ensures that Python can find modules like 'config', 'pages', 'data_processing' etc.
if str(_project_root_dir) not in sys.path:
    sys.path.insert(0, str(_project_root_dir))
    # This print goes to stderr, which Streamlit might show in server logs.
    print(f"DEBUG (app.py): Added project root to sys.path: {_project_root_dir}", file=sys.stderr)
else:
    print(f"DEBUG (app.py): Project root '{_project_root_dir}' was already in sys.path.", file=sys.stderr)


# --- Import Settings ---
try:
    from config import settings # This should now work if project_root_dir is correct
    print(f"INFO (app.py): Successfully imported config.settings. APP_NAME: {settings.APP_NAME}", file=sys.stderr)
except ImportError as e_cfg_app:
    print(f"FATAL (app.py): Failed to import config.settings: {e_cfg_app}", file=sys.stderr)
    print(f"PYTHONPATH for app.py: {sys.path}", file=sys.stderr)
    print(f"Calculated project root for sys.path: {_project_root_dir}", file=sys.stderr)
    sys.exit(1) 
except AttributeError as e_attr_settings: # Catch if settings was imported but APP_NAME is missing (circular import symptom)
    print(f"FATAL (app.py): AttributeError during config.settings access (likely circular import): {e_attr_settings}", file=sys.stderr)
    print(f"PYTHONPATH for app.py: {sys.path}", file=sys.stderr)
    print(f"Calculated project root for sys.path: {_project_root_dir}", file=sys.stderr)
    sys.exit(1)
except Exception as e_generic_cfg:
    print(f"FATAL (app.py): Generic error during config.settings import: {e_generic_cfg}", file=sys.stderr)
    sys.exit(1)

# --- Global Logging Configuration ---
valid_log_levels_app = {"DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"}
log_level_app_str = str(settings.LOG_LEVEL).upper()
if log_level_app_str not in valid_log_levels_app:
    print(f"WARN (app.py): Invalid LOG_LEVEL '{log_level_app_str}'. Using INFO.", file=sys.stderr); log_level_app_str = "INFO"
logging.basicConfig(level=getattr(logging, log_level_app_str, logging.INFO), format=settings.LOG_FORMAT,
                    datefmt=settings.LOG_DATE_FORMAT, handlers=[logging.StreamHandler(sys.stdout)], force=True)
logger = logging.getLogger(__name__) # Logger for this app.py

# --- Streamlit Version Check & Feature Availability ---
STREAMLIT_VERSION_GE_1_30 = False 
STREAMLIT_PAGE_LINK_AVAILABLE = False
try:
    import streamlit
    major, minor, patch_str_list = streamlit.__version__.split('.')
    patch_str = patch_str_list.split('-')[0] # Handle versions like "1.30.0-dev"
    patch = int(patch_str) 
    STREAMLIT_VERSION_GE_1_30 = (int(major) >= 1 and int(minor) >= 30)
    if hasattr(st, 'page_link'): STREAMLIT_PAGE_LINK_AVAILABLE = True
    if not STREAMLIT_VERSION_GE_1_30: logger.warning(f"Streamlit version {streamlit.__version__} < 1.30.0. Some UI features might use fallbacks.")
except ImportError: logger.critical("Streamlit library not found."); sys.exit("Streamlit library not found.")
except Exception as e_st_ver: logger.warning(f"Could not accurately determine Streamlit version/features: {e_st_ver}")

# --- Page Configuration ---
page_icon_path_obj_app = Path(settings.APP_LOGO_SMALL_PATH) # settings paths are already absolute strings
final_page_icon_str_app: str = str(page_icon_path_obj_app) if page_icon_path_obj_app.exists() and page_icon_path_obj_app.is_file() else "🌍"
if final_page_icon_str_app == "🌍": logger.warning(f"Page icon not found at '{page_icon_path_obj_app}'. Using '🌍'.")
st.set_page_config(
    page_title=f"{settings.APP_NAME} - System Overview", page_icon=final_page_icon_str_app,
    layout="wide", initial_sidebar_state="expanded", # initialSidebarState is correct here for page_config
    menu_items={
        "Get Help": f"mailto:{settings.SUPPORT_CONTACT_INFO}?subject=Help Request - {settings.APP_NAME}",
        "Report a bug": f"mailto:{settings.SUPPORT_CONTACT_INFO}?subject=Bug Report - {settings.APP_NAME} v{settings.APP_VERSION}",
        "About": f"### {settings.APP_NAME} (v{settings.APP_VERSION})\n{settings.APP_FOOTER_TEXT}\n\nEdge-First Health Intelligence Co-Pilot."
    }
)

# --- Apply Plotly Theme & CSS ---
try:
    from visualization.plots import set_sentinel_plotly_theme
    set_sentinel_plotly_theme(); logger.debug("Sentinel Plotly theme applied.")
except Exception as e_theme_main_app: logger.error(f"Error applying Plotly theme: {e_theme_main_app}", exc_info=True); st.error("Error applying visualization theme.")

@st.cache_resource
def load_global_css_styles_app(css_path_str_app: str): # Renamed func for clarity
    css_path_app = Path(css_path_str_app) # settings.STYLE_CSS_PATH_WEB is already absolute
    if css_path_app.exists() and css_path_app.is_file():
        try:
            with open(css_path_app, "r", encoding="utf-8") as f_css_app: st.markdown(f'<style>{f_css_app.read()}</style>', unsafe_allow_html=True)
            logger.debug(f"Global CSS loaded: {css_path_app}")
        except Exception as e_css_main_app: logger.error(f"Error applying CSS {css_path_app}: {e_css_main_app}", exc_info=True); st.error("Styles could not be loaded.")
    else: logger.warning(f"CSS file not found: {css_path_app}"); st.warning("Application stylesheet missing.")
if settings.STYLE_CSS_PATH_WEB: load_global_css_styles_app(settings.STYLE_CSS_PATH_WEB)

# --- Main Application Header ---
header_cols_app_ui = st.columns([0.12, 0.88])
with header_cols_app_ui[0]:
    l_logo_path_app = Path(settings.APP_LOGO_LARGE_PATH)
    s_logo_path_app = Path(settings.APP_LOGO_SMALL_PATH)
    # Paths from settings are absolute
    if l_logo_path_app.is_file(): st.image(str(l_logo_path_app), width=100)
    elif s_logo_path_app.is_file(): st.image(str(s_logo_path_app), width=80)
    else: logger.warning(f"App logos not found. L: '{l_logo_path_app}', S: '{s_logo_path_app}'."); st.markdown("### 🌍", unsafe_allow_html=True)
with header_cols_app_ui[1]: st.title(settings.APP_NAME); st.subheader("Transforming Data into Lifesaving Action at the Edge")
st.divider()

# --- Welcome & System Description ---
st.markdown(f"""## Welcome to the {settings.APP_NAME} Demonstrator...""") # Full content from prompt
st.markdown("#### Core Design Principles:")
core_principles_main_app_v3 = [
    ("📶 **Offline-First Operations**", "On-device Edge AI ensures critical functionality without continuous connectivity."),
    ("🎯 **Action-Oriented Intelligence**", "Insights aim to trigger clear, targeted responses relevant to frontline workflows."),
    ("🧑‍🤝‍🧑 **Human-Centered Design**", "Interfaces optimized for low-literacy, high-stress users, prioritizing immediate understanding."),
    ("🔗 **Resilience & Scalability**", "Modular design for scaling from personal devices to regional views with robust data sync.")
]
num_cols_core_principles_v3 = min(len(core_principles_main_app_v3), 2)
if num_cols_core_principles_v3 > 0:
    cols_core_principles_ui_v3 = st.columns(num_cols_core_principles_v3)
    for idx_core_v3, (title_core_v3, desc_core_v3) in enumerate(core_principles_main_app_v3):
        with cols_core_principles_ui_v3[idx_core_v3 % num_cols_core_principles_v3]:
            st.markdown(f"##### {title_core_v3}"); st.markdown(f"<small>{html.escape(desc_core_v3)}</small>", unsafe_allow_html=True)
            st.markdown("<div style='margin-bottom:1rem;'></div>", unsafe_allow_html=True)
st.markdown("---")
st.markdown("👈 **Navigate via the sidebar** to explore simulated web dashboards...") # Full content
st.info("💡 **Note:** This web application serves as a high-level demonstrator...") # Full content
st.divider()

st.header("Explore Simulated Role-Specific Dashboards")
st.caption("These views demonstrate information available at higher tiers (Facility/Regional Nodes).")

# This is the base directory where Streamlit looks for page files.
# `app.py` should be in `ssentinel_project_root/`
# Page files should be in `ssentinel_project_root/pages/`
# Example: `ssentinel_project_root/pages/01_chw_dashboard.py`
pages_directory_obj = _project_root_dir / "pages" 

role_navigation_config_app_final_v2 = [
    {"title": "🧑‍⚕️ CHW Operations Summary & Field Support View (Supervisor/Hub Level)", "desc": "This view simulates how a CHW Supervisor or a Hub coordinator might access summarized data from CHW Personal Edge Devices (PEDs).<br><br><b>Focus (Tier 1-2):</b> Team performance monitoring, targeted support for CHWs, localized outbreak signal detection based on aggregated CHW reports.<br><b>Key Data Points:</b> CHW activity summaries (visits, tasks completed), patient alert escalations, critical supply needs for CHW kits, early epidemiological signals from specific zones.<br><b>Objective:</b> Enable supervisors to manage CHW teams effectively, provide timely support, identify emerging health issues quickly, and coordinate local responses. The CHW's primary tool is their offline-first native app on their PED, providing real-time alerts & task management.", 
     "page_filename": "01_chw_dashboard.py", "icon": "🧑‍⚕️"},
    {"title": "🏥 Clinic Operations & Environmental Safety View (Facility Node Level)", "desc": "Simulates a dashboard for Clinic Managers at a Facility Node (Tier 2), providing insights into service efficiency, care quality, resource management, and environmental conditions.<br><br><b>Focus (Tier 2):</b> Optimizing clinic workflows, ensuring quality patient care, managing supplies and testing backlogs, monitoring clinic environment for safety and infection control.<br><b>Key Data Points:</b> Clinic performance KPIs (e.g., test TAT, patient throughput), supply stock forecasts, IoT sensor data summaries (CO2, PM2.5, occupancy), clinic-level epidemiological trends, flagged patient cases for review.<br><b>Objective:</b> Enhance operational efficiency, support clinical decision-making, maintain resource availability, and ensure a safe clinic environment.", 
     "page_filename": "02_clinic_dashboard.py", "icon": "🏥"},
    {"title": "🗺️ District Health Strategic Overview (DHO at Facility/Regional Node Level)", "desc": "Presents a strategic dashboard for District Health Officers (DHOs), typically accessed at a Facility Node (Tier 2) or a Regional/Cloud Node (Tier 3).<br><br><b>Focus (Tier 2-3):</b> Population health insights, resource allocation across zones, monitoring environmental well-being, and planning targeted interventions.<br><b>Key Data Points:</b> District-wide health KPIs, interactive maps for zonal comparisons (risk, disease burden, resources), trend analyses, intervention planning tools based on aggregated data.<br><b>Objective:</b> Support evidence-based strategic planning, public health interventions, program monitoring, and policy development for the district.", 
     "page_filename": "03_district_dashboard.py", "icon": "🗺️"},
    {"title": "📊 Population Health Analytics Deep Dive (Epidemiologist/Analyst View - Tier 3)", "desc": "A view designed for detailed epidemiological and health systems analysis, typically used by analysts or program managers at a Regional/Cloud Node (Tier 3) with access to more comprehensive, aggregated datasets.<br><br><b>Focus (Tier 3):</b> In-depth analysis of demographic patterns, SDOH impacts, clinical trends, health system performance, and equity across broader populations.<br><b>Key Data Points:</b> Stratified disease burden, AI risk distributions by various factors, aggregated test positivity trends, comorbidity analysis, referral pathway performance, health equity metrics.<br><b>Objective:</b> Provide robust analytical capabilities to understand population health dynamics, evaluate interventions, identify areas for research, and inform large-scale public health strategy.", 
     "page_filename": "04_population_dashboard.py", "icon": "📊"},
]

num_nav_cols_final_app_v2 = min(len(role_navigation_config_app_final_v2), 2)
if num_nav_cols_final_app_v2 > 0:
    nav_cols_ui_final_app_v2 = st.columns(num_nav_cols_final_app_v2)
    current_col_idx_nav_final_v2 = 0
    for nav_item_final_app_v2 in role_navigation_config_app_final_v2:
        # Path for st.page_link should be relative to the *pages* directory itself for Streamlit >= 1.23
        # e.g., "01_chw_dashboard.py" if the file is `pages/01_chw_dashboard.py`
        page_link_target_app = nav_item_final_app_v2['page_filename']
        physical_page_full_path_app = pages_directory_obj / nav_item_final_app_v2["page_filename"]
        
        if not physical_page_full_path_app.exists():
            logger.warning(f"Navigation page file for '{nav_item_final_app_v2['title']}' not found: {physical_page_full_path_app}")
            continue

        with nav_cols_ui_final_app_v2[current_col_idx_nav_final_v2 % num_nav_cols_final_app_v2]:
            container_args_final_app = {"border": True} if STREAMLIT_VERSION_GE_1_30 else {}
            with st.container(**container_args_final_app):
                st.subheader(f"{nav_item_final_app_v2['icon']} {html.escape(nav_item_final_app_v2['title'])}")
                st.markdown(f"<small>{nav_item_final_app_v2['desc']}</small>", unsafe_allow_html=True)
                link_label_final_app = f"Explore {nav_item_final_app_v2['title'].split('(')[0].split('View')[0].strip()} View"
                if STREAMLIT_PAGE_LINK_AVAILABLE:
                    link_kwargs_final_app = {"use_container_width": True} if STREAMLIT_VERSION_GE_1_30 else {}
                    st.page_link(page_link_target_app, label=link_label_final_app, icon="➡️", **link_kwargs_final_app)
                else: 
                    st.markdown(f'<a href="{nav_item_final_app_v2["page_filename"]}" target="_self" style="display:block;text-align:center;padding:0.5em;background-color:var(--sentinel-color-action-primary);color:white;border-radius:4px;text-decoration:none;">{link_label_final_app} ➡️</a>', unsafe_allow_html=True)
            st.markdown("<div style='margin-bottom:0.5rem;'></div>", unsafe_allow_html=True)
        current_col_idx_nav_final_v2 += 1
st.divider()

# --- Key Capabilities Section ---
st.header(f"{settings.APP_NAME} - Key Capabilities Reimagined")
# ... (Full capabilities descriptions from prompt go here) ...
capabilities_data_app_final_v2_full = [
    ("🛡️ Frontline Worker Safety & Support", "Real-time vitals/environmental monitoring, fatigue detection, safety nudges on PEDs."),
    ("🌍 Offline-First Edge AI", "On-device intelligence for alerts, prioritization, guidance without continuous connectivity."),
    ("⚡ Actionable, Contextual Insights", "Raw data to clear, role-specific recommendations integrated into field workflows."),
    ("🤝 Human-Centered & Accessible UX", "Pictogram UIs, voice/tap commands, local language support for low-literacy, high-stress users on PEDs."),
    ("📡 Resilient Data Synchronization", "Flexible data sharing (Bluetooth, QR, SD card, SMS, opportunistic IP) across devices/tiers."),
    ("🌱 Scalable & Interoperable Architecture", "Modular design (personal to national), FHIR/HL7 considerations for integration.")
]
num_cap_cols_final_app_v2 = min(len(capabilities_data_app_final_v2_full), 3)
if num_cap_cols_final_app_v2 > 0:
    cap_cols_ui_final_app_v2 = st.columns(num_cap_cols_final_app_v2)
    for i_cap_final, (cap_t_final, cap_d_final) in enumerate(capabilities_data_app_final_v2_full):
        with cap_cols_ui_final_app_v2[i_cap_final % num_cap_cols_final_app_v2]: 
            st.markdown(f"##### {html.escape(cap_t_final)}")
            st.markdown(f"<small>{html.escape(cap_d_final)}</small>", unsafe_allow_html=True)
            st.markdown("<div style='margin-bottom:1.2rem;'></div>", unsafe_allow_html=True)
st.divider()

# --- Sidebar Content & Glossary Link ---
st.sidebar.header(f"{settings.APP_NAME} v{settings.APP_VERSION}")
st.sidebar.divider()
st.sidebar.markdown("#### About This Demonstrator:")
st.sidebar.info("Web app simulates higher-level dashboards. Frontline workers use PED apps.")
st.sidebar.divider()

# The glossary link is now expected to be auto-discovered if named e.g., "05_glossary_page.py"
# No manual link is needed here for sidebar ordering if filenames are prefixed.
# If you need a specific link style or text, you can add it, but it might duplicate.
glossary_filename_sidebar_final_app = "05_glossary_page.py" # Assumed prefixed name
glossary_physical_path_final_app_sb = pages_directory_obj / glossary_filename_sidebar_final_app
if not glossary_physical_path_final_app_sb.exists():
    logger.warning(f"Glossary page file for sidebar (expected: {glossary_physical_path_final_app_sb}) not found. Auto-discovery may fail.")

st.sidebar.divider()
st.sidebar.markdown(f"**{settings.ORGANIZATION_NAME}**")
st.sidebar.markdown(f"Support: [{settings.SUPPORT_CONTACT_INFO}](mailto:{settings.SUPPORT_CONTACT_INFO})")
st.sidebar.caption(settings.APP_FOOTER_TEXT)

logger.info(f"{settings.APP_NAME} (v{settings.APP_VERSION}) - System Overview page loaded.")
