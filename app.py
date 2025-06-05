# sentinel_project_root/app.py
# Main Streamlit application for Sentinel Health Co-Pilot Demonstrator.

import streamlit as st
import sys
import logging
from pathlib import Path
import html 

# --- Robust Path Setup: This MUST be correct for all imports to work ---
# This script (app.py) is assumed to be in the project root: ssentinel_project_root/app.py
# Therefore, Path(__file__).resolve().parent IS the project root.
_this_file_path = Path(__file__).resolve()
_project_root_dir = _this_file_path.parent 

# Ensure the project root is the first thing in sys.path
if str(_project_root_dir) not in sys.path:
    sys.path.insert(0, str(_project_root_dir))
    print(f"DEBUG (app.py): Added project root to sys.path: {_project_root_dir}", file=sys.stderr)
else:
    # If it's already there, ensure it's at the beginning for precedence
    if sys.path[0] != str(_project_root_dir):
        try: # It might not be in the list if sys.path was manipulated elsewhere
            sys.path.remove(str(_project_root_dir))
        except ValueError:
            pass # Not found, so insert will work fine
        sys.path.insert(0, str(_project_root_dir))
        print(f"DEBUG (app.py): Moved project root to start of sys.path: {_project_root_dir}", file=sys.stderr)
    else:
        print(f"DEBUG (app.py): Project root '{_project_root_dir}' was already at start of sys.path.", file=sys.stderr)

# --- Import Settings (NOW that sys.path is correctly set up) ---
try:
    from config import settings 
except ImportError as e_cfg_app:
    print(f"FATAL (app.py): Failed to import config.settings: {e_cfg_app}", file=sys.stderr)
    print(f"PYTHONPATH for app.py: {sys.path}", file=sys.stderr)
    print(f"Calculated project root for sys.path: {_project_root_dir}", file=sys.stderr)
    sys.exit(1) 
except AttributeError as e_attr_settings: # Catch if settings was imported but APP_NAME is missing (circular import symptom)
    print(f"FATAL (app.py): AttributeError during config.settings access (likely circular import or settings.py error): {e_attr_settings}", file=sys.stderr)
    print(f"PYTHONPATH for app.py: {sys.path}", file=sys.stderr)
    print(f"Calculated project root for sys.path: {_project_root_dir}", file=sys.stderr)
    sys.exit(1)
except Exception as e_generic_cfg:
    print(f"FATAL (app.py): Generic error during config.settings import: {e_generic_cfg}", file=sys.stderr)
    sys.exit(1)

# --- Global Logging Configuration (Now that settings is imported) ---
valid_log_levels_app = {"DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"}
log_level_app_str = str(settings.LOG_LEVEL).upper()
if log_level_app_str not in valid_log_levels_app:
    print(f"WARN (app.py): Invalid LOG_LEVEL '{log_level_app_str}' in settings. Defaulting to INFO.", file=sys.stderr)
    log_level_app_str = "INFO"

logging.basicConfig(level=getattr(logging, log_level_app_str, logging.INFO), format=settings.LOG_FORMAT,
                    datefmt=settings.LOG_DATE_FORMAT, handlers=[logging.StreamHandler(sys.stdout)], force=True)
logger = logging.getLogger(__name__) # Logger for this app.py

logger.info(f"INFO (app.py): Successfully imported config.settings. APP_NAME: {settings.APP_NAME}")

# --- Streamlit Version Check & Feature Availability ---
STREAMLIT_VERSION_GE_1_30 = False 
STREAMLIT_PAGE_LINK_AVAILABLE = False
try:
    import streamlit # Ensure streamlit is imported before using st alias
    major_str, minor_str, patch_full_str = streamlit.__version__.split('.')
    major, minor = int(major_str), int(minor_str)
    # Extract only the numeric part of patch string
    patch_str_numeric = "".join(filter(str.isdigit, patch_full_str.split('-')[0]))
    patch = int(patch_str_numeric) if patch_str_numeric else 0
    
    STREAMLIT_VERSION_GE_1_30 = (major >= 1 and minor >= 30)
    if hasattr(st, 'page_link'): STREAMLIT_PAGE_LINK_AVAILABLE = True
    if not STREAMLIT_VERSION_GE_1_30: logger.warning(f"Streamlit version {streamlit.__version__} < 1.30.0. Some UI features might use fallbacks.")
except ImportError: logger.critical("Streamlit library not found."); sys.exit("Streamlit library not found.")
except Exception as e_st_ver: logger.warning(f"Could not accurately determine Streamlit version/features: {e_st_ver}")

# --- Page Configuration ---
page_icon_path_obj_app_cfg = Path(settings.APP_LOGO_SMALL_PATH) 
final_page_icon_str_app_cfg: str = str(page_icon_path_obj_app_cfg) if page_icon_path_obj_app_cfg.is_file() else "🌍"
if final_page_icon_str_app_cfg == "🌍": logger.warning(f"Page icon not found at '{page_icon_path_obj_app_cfg}'. Using '🌍'.")
st.set_page_config(
    page_title=f"{settings.APP_NAME} - System Overview", page_icon=final_page_icon_str_app_cfg,
    layout="wide", initial_sidebar_state="expanded",
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
except Exception as e_theme_main_app_cfg: logger.error(f"Error applying Plotly theme: {e_theme_main_app_cfg}", exc_info=True); st.error("Error applying visualization theme.")

@st.cache_resource
def load_global_css_styles_app_cfg(css_path_str_app_cfg: str):
    css_path_app_cfg = Path(css_path_str_app_cfg)
    if css_path_app_cfg.is_file():
        try:
            with open(css_path_app_cfg, "r", encoding="utf-8") as f_css_app_cfg: st.markdown(f'<style>{f_css_app_cfg.read()}</style>', unsafe_allow_html=True)
            logger.debug(f"Global CSS loaded: {css_path_app_cfg}")
        except Exception as e_css_main_app_cfg: logger.error(f"Error applying CSS {css_path_app_cfg}: {e_css_main_app_cfg}", exc_info=True); st.error("Styles could not be loaded.")
    else: logger.warning(f"CSS file not found: {css_path_app_cfg}"); st.warning("Application stylesheet missing.")
if settings.STYLE_CSS_PATH_WEB: load_global_css_styles_app_cfg(settings.STYLE_CSS_PATH_WEB)

# --- Main Application Header ---
header_cols_app_ui_cfg = st.columns([0.12, 0.88])
with header_cols_app_ui_cfg[0]:
    l_logo_path_app_cfg = Path(settings.APP_LOGO_LARGE_PATH)
    s_logo_path_app_cfg = Path(settings.APP_LOGO_SMALL_PATH)
    if l_logo_path_app_cfg.is_file(): st.image(str(l_logo_path_app_cfg), width=100)
    elif s_logo_path_app_cfg.is_file(): st.image(str(s_logo_path_app_cfg), width=80)
    else: logger.warning(f"App logos not found. L: '{l_logo_path_app_cfg}', S: '{s_logo_path_app_cfg}'."); st.markdown("### 🌍", unsafe_allow_html=True)
with header_cols_app_ui_cfg[1]: st.title(settings.APP_NAME); st.subheader("Transforming Data into Lifesaving Action at the Edge")
st.divider()

# --- Welcome & System Description ---
st.markdown(f"""## Welcome to the {settings.APP_NAME} Demonstrator
Sentinel is an **edge-first health intelligence system** designed for **maximum clinical and 
operational actionability** in resource-limited, high-risk environments. It aims to convert 
diverse data sources into life-saving, workflow-integrated decisions, even with 
**minimal or intermittent internet connectivity.**""")
st.markdown("#### Core Design Principles:")
core_principles_main_app_v4 = [
    ("📶 **Offline-First Operations**", "On-device Edge AI ensures critical functionality without continuous connectivity."),
    ("🎯 **Action-Oriented Intelligence**", "Insights aim to trigger clear, targeted responses relevant to frontline workflows."),
    ("🧑‍🤝‍🧑 **Human-Centered Design**", "Interfaces optimized for low-literacy, high-stress users, prioritizing immediate understanding."),
    ("🔗 **Resilience & Scalability**", "Modular design for scaling from personal devices to regional views with robust data sync.")
]
num_cols_core_principles_v4 = min(len(core_principles_main_app_v4), 2)
if num_cols_core_principles_v4 > 0:
    cols_core_principles_ui_v4 = st.columns(num_cols_core_principles_v4)
    for idx_core_v4, (title_core_v4, desc_core_v4) in enumerate(core_principles_main_app_v4):
        with cols_core_principles_ui_v4[idx_core_v4 % num_cols_core_principles_v4]:
            st.markdown(f"##### {title_core_v4}"); st.markdown(f"<small>{html.escape(desc_core_v4)}</small>", unsafe_allow_html=True)
            st.markdown("<div style='margin-bottom:1rem;'></div>", unsafe_allow_html=True)
st.markdown("---")
st.markdown("👈 **Navigate via the sidebar** to explore simulated web dashboards for various operational tiers. These views represent perspectives of **Supervisors, Clinic Managers, or District Health Officers (DHOs)**. The primary interface for frontline workers (e.g., CHWs) is a dedicated native application on their Personal Edge Device (PED), tailored for their specific operational context.")
st.info("💡 **Note:** This web application serves as a high-level demonstrator for the Sentinel system's data processing capabilities and the types of aggregated views available to management and strategic personnel.")
st.divider()

st.header("Explore Simulated Role-Specific Dashboards")
st.caption("These views demonstrate information available at higher tiers (Facility/Regional Nodes).")

pages_directory_obj_app_cfg = _project_root_dir / "pages" 
role_navigation_config_app_cfg = [
    {"title": "🧑‍⚕️ CHW Operations Summary & Field Support View (Supervisor/Hub Level)", "desc": "This view simulates how a CHW Supervisor or a Hub coordinator might access summarized data from CHW Personal Edge Devices (PEDs).<br><br><b>Focus (Tier 1-2):</b> Team performance monitoring, targeted support for CHWs, localized outbreak signal detection based on aggregated CHW reports.<br><b>Key Data Points:</b> CHW activity summaries (visits, tasks completed), patient alert escalations, critical supply needs for CHW kits, early epidemiological signals from specific zones.<br><b>Objective:</b> Enable supervisors to manage CHW teams effectively, provide timely support, identify emerging health issues quickly, and coordinate local responses. The CHW's primary tool is their offline-first native app on their PED, providing real-time alerts & task management.", 
     "page_filename": "01_chw_dashboard.py", "icon": "🧑‍⚕️"},
    {"title": "🏥 Clinic Operations & Environmental Safety View (Facility Node Level)", "desc": "Simulates a dashboard for Clinic Managers at a Facility Node (Tier 2), providing insights into service efficiency, care quality, resource management, and environmental conditions.<br><br><b>Focus (Tier 2):</b> Optimizing clinic workflows, ensuring quality patient care, managing supplies and testing backlogs, monitoring clinic environment for safety and infection control.<br><b>Key Data Points:</b> Clinic performance KPIs (e.g., test TAT, patient throughput), supply stock forecasts, IoT sensor data summaries (CO2, PM2.5, occupancy), clinic-level epidemiological trends, flagged patient cases for review.<br><b>Objective:</b> Enhance operational efficiency, support clinical decision-making, maintain resource availability, and ensure a safe clinic environment.", 
     "page_filename": "02_clinic_dashboard.py", "icon": "🏥"},
    {"title": "🗺️ District Health Strategic Overview (DHO at Facility/Regional Node Level)", "desc": "Presents a strategic dashboard for District Health Officers (DHOs), typically accessed at a Facility Node (Tier 2) or a Regional/Cloud Node (Tier 3).<br><br><b>Focus (Tier 2-3):</b> Population health insights, resource allocation across zones, monitoring environmental well-being, and planning targeted interventions.<br><b>Key Data Points:</b> District-wide health KPIs, interactive maps for zonal comparisons (risk, disease burden, resources), trend analyses, intervention planning tools based on aggregated data.<br><b>Objective:</b> Support evidence-based strategic planning, public health interventions, program monitoring, and policy development for the district.", 
     "page_filename": "03_district_dashboard.py", "icon": "🗺️"},
    {"title": "📊 Population Health Analytics Deep Dive (Epidemiologist/Analyst View - Tier 3)", "desc": "A view designed for detailed epidemiological and health systems analysis, typically used by analysts or program managers at a Regional/Cloud Node (Tier 3) with access to more comprehensive, aggregated datasets.<br><br><b>Focus (Tier 3):</b> In-depth analysis of demographic patterns, SDOH impacts, clinical trends, health system performance, and equity across broader populations.<br><b>Key Data Points:</b> Stratified disease burden, AI risk distributions by various factors, aggregated test positivity trends, comorbidity analysis, referral pathway performance, health equity metrics.<br><b>Objective:</b> Provide robust analytical capabilities to understand population health dynamics, evaluate interventions, identify areas for research, and inform large-scale public health strategy.", 
     "page_filename": "04_population_dashboard.py", "icon": "📊"},
]

num_nav_cols_final_app_cfg_val = min(len(role_navigation_config_app_cfg), 2)
if num_nav_cols_final_app_cfg_val > 0:
    nav_cols_ui_final_app_cfg_val = st.columns(num_nav_cols_final_app_cfg_val)
    current_col_idx_nav_final_cfg_val = 0
    for nav_item_final_app_cfg_item in role_navigation_config_app_cfg:
        page_link_target_app_cfg_item = nav_item_final_app_cfg_item['page_filename'] # Path relative to 'pages/' dir
        physical_page_full_path_app_cfg_item = pages_directory_obj_app_cfg / nav_item_final_app_cfg_item["page_filename"]
        
        if not physical_page_full_path_app_cfg_item.exists():
            logger.warning(f"Navigation page file for '{nav_item_final_app_cfg_item['title']}' not found: {physical_page_full_path_app_cfg_item}")
            continue

        with nav_cols_ui_final_app_cfg_val[current_col_idx_nav_final_cfg_val % num_nav_cols_final_app_cfg_val]:
            container_args_final_app_cfg = {"border": True} if STREAMLIT_VERSION_GE_1_30 else {}
            with st.container(**container_args_final_app_cfg):
                st.subheader(f"{nav_item_final_app_cfg_item['icon']} {html.escape(nav_item_final_app_cfg_item['title'])}")
                st.markdown(f"<small>{nav_item_final_app_cfg_item['desc']}</small>", unsafe_allow_html=True)
                link_label_final_app_cfg = f"Explore {nav_item_final_app_cfg_item['title'].split('(')[0].split('View')[0].strip()} View"
                if STREAMLIT_PAGE_LINK_AVAILABLE:
                    link_kwargs_final_app_cfg = {"use_container_width": True} if STREAMLIT_VERSION_GE_1_30 else {}
                    st.page_link(page_link_target_app_cfg_item, label=link_label_final_app_cfg, icon="➡️", **link_kwargs_final_app_cfg)
                else: 
                    st.markdown(f'<a href="{nav_item_final_app_cfg_item["page_filename"]}" target="_self" style="display:block;text-align:center;padding:0.5em;background-color:var(--sentinel-color-action-primary);color:white;border-radius:4px;text-decoration:none;">{link_label_final_app_cfg} ➡️</a>', unsafe_allow_html=True)
            st.markdown("<div style='margin-bottom:0.5rem;'></div>", unsafe_allow_html=True)
        current_col_idx_nav_final_cfg_val += 1
st.divider()

# --- Key Capabilities Section ---
st.header(f"{settings.APP_NAME} - Key Capabilities Reimagined")
capabilities_data_app_final_cfg = [
    ("🛡️ Frontline Worker Safety & Support", "Real-time vitals/environmental monitoring, fatigue detection, safety nudges on PEDs."),
    ("🌍 Offline-First Edge AI", "On-device intelligence for alerts, prioritization, guidance without continuous connectivity."),
    ("⚡ Actionable, Contextual Insights", "Raw data to clear, role-specific recommendations integrated into field workflows."),
    ("🤝 Human-Centered & Accessible UX", "Pictogram UIs, voice/tap commands, local language support for low-literacy, high-stress users on PEDs."),
    ("📡 Resilient Data Synchronization", "Flexible data sharing (Bluetooth, QR, SD card, SMS, opportunistic IP) across devices/tiers."),
    ("🌱 Scalable & Interoperable Architecture", "Modular design (personal to national), FHIR/HL7 considerations for integration.")
]
num_cap_cols_final_app_cfg_val = min(len(capabilities_data_app_final_cfg), 3)
if num_cap_cols_final_app_cfg_val > 0:
    cap_cols_ui_final_app_cfg_val = st.columns(num_cap_cols_final_app_cfg_val)
    for i_cap_final_cfg, (cap_t_final_cfg, cap_d_final_cfg) in enumerate(capabilities_data_app_final_cfg):
        with cap_cols_ui_final_app_cfg_val[i_cap_final_cfg % num_cap_cols_final_app_cfg_val]: 
            st.markdown(f"##### {html.escape(cap_t_final_cfg)}")
            st.markdown(f"<small>{html.escape(cap_d_final_cfg)}</small>", unsafe_allow_html=True)
            st.markdown("<div style='margin-bottom:1.2rem;'></div>", unsafe_allow_html=True)
st.divider()

# --- Sidebar Content ---
st.sidebar.header(f"{settings.APP_NAME} v{settings.APP_VERSION}")
st.sidebar.divider()
st.sidebar.markdown("#### About This Demonstrator:"); st.sidebar.info("Web app simulates higher-level dashboards...")
st.sidebar.divider()

glossary_filename_sidebar_cfg = "05_glossary_page.py" 
glossary_link_target_sidebar_cfg = glossary_filename_sidebar_cfg # For st.page_link, path is relative to pages dir
glossary_physical_path_final_sb_cfg = pages_directory_obj_app_cfg / glossary_filename_sidebar_cfg

if glossary_physical_path_final_sb_cfg.exists():
    if STREAMLIT_PAGE_LINK_AVAILABLE: st.sidebar.page_link(glossary_link_target_sidebar_cfg, label="📜 System Glossary", icon="📚")
    else: st.sidebar.markdown(f'<a href="{glossary_filename_sidebar_cfg}" target="_self">📜 System Glossary</a>', unsafe_allow_html=True)
else: logger.warning(f"Glossary page file for sidebar (expected: {glossary_physical_path_final_sb_cfg}) not found.")

st.sidebar.divider()
st.sidebar.markdown(f"**{settings.ORGANIZATION_NAME}**")
st.sidebar.markdown(f"Support: [{settings.SUPPORT_CONTACT_INFO}](mailto:{settings.SUPPORT_CONTACT_INFO})")
st.sidebar.caption(settings.APP_FOOTER_TEXT)

logger.info(f"{settings.APP_NAME} (v{settings.APP_VERSION}) - System Overview page loaded.")
